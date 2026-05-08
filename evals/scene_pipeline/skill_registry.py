"""Load and manage predefined skills from the YAML resource file.

Provides:
- YAML skill loading and caching
- scene_type → skill-category mapping  (e.g. living_room → home)
- Deduplication against BASE_SERVER_TEMPLATE tools
- Automatic MCP @mcp.tool() Python code generation from skill definitions
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .config import BASE_TOOLS

logger = logging.getLogger(__name__)

RESOURCE_PATH = Path(__file__).parent / "resource" / "refinev1_updated_with_io.yaml"

# Maps scene_type values produced by Vision LLM to YAML skill categories.
SCENE_TYPE_TO_CATEGORY: dict[str, str] = {
    # Home
    "living_room": "home",
    "bedroom": "home",
    "kitchen": "home",
    "bathroom": "home",
    "dining_room": "home",
    "home": "home",
    # Office
    "office": "office",
    "workspace": "office",
    "meeting_room": "office",
    # Retail
    "supermarket": "retail",
    "store": "retail",
    "shop": "retail",
    "retail": "retail",
    # Industrial
    "factory": "industrial",
    "warehouse": "industrial",
    "workshop": "industrial",
    "industrial": "industrial",
}

# Skills in `common` whose semantics are already covered by BASE_SERVER_TEMPLATE
# tools.  These are skipped when generating extra MCP tool code so we don't
# produce duplicate definitions.
_BASE_TOOL_COVERED_SKILLS: set[str] = {
    # --- Navigation / Posture: covered by move, move_to, adjust_body_height ---
    "navigate",       # covered by move / move_to
    "move",           # covered by move / move_to
    "stop",           # covered by move (implicit)
    "turn around",    # covered by move (yaw)
    "set height",     # covered by adjust_body_height
    "set_posture",    # covered by adjust_body_height
    "lean",           # covered by adjust_body_height
    "reset",          # covered by adjust_body_height / back_station
    "orient",         # covered by move (yaw)
    # --- Perception: covered by scene_recognition / spatial_memory_query_vec ---
    "scan",           # covered by scene_recognition
    "look_at",        # covered by scene_recognition
    # --- Grasp / Release: covered by grasp_start / release_hand ---
    "grasp",          # covered by grasp_start
    "release",        # covered by release_hand
    "regrasp",        # covered by grasp_start (retry)
    "suction",        # covered by grasp_start
    "close_effector", # covered by grasp_start
    "open_effector",  # covered by release_hand
    "hold",           # covered by grasp_start (implicit)
    "pick",           # covered by grasp_start
    # --- Place / Drop: covered by place / place_down ---
    "drop",           # covered by place / place_down
    "place",          # covered by place / place_down
    # --- Transport: covered by move_to (while holding) ---
    "carry",          # covered by move_to (while holding)
    "lift",           # covered by grasp_start + adjust_body_height
    # --- Handover: covered by handover ---
    "handout",        # covered by handover
    "takeover",       # covered by handover (reverse)
    # --- Box: covered by fetch_box / put_box ---
    "open_box",       # covered by fetch_box
    "close_box",      # covered by put_box
}

# Also skip skills whose name collides with a BASE_TOOLS function name.
_BASE_TOOL_NAMES: set[str] = set(BASE_TOOLS)


# Standard capabilities that the pipeline's task categories may require.
# Each entry maps a capability role to a list of skill-name candidates
# (checked against predefined skills and BASE tools in order).
# If none match, the capability is reported as a "gap" for LLM generation.
STANDARD_CAPABILITIES: dict[str, list[str]] = {
    "fold":  ["fold", "fold_item"],
    "tidy":  ["tidy_surface", "tidy", "sort"],
    "list":  ["list_objects", "scan", "look_at"],
}


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_predefined_skills() -> dict[str, Any]:
    """Load all predefined skills from the YAML resource (cached)."""
    with open(RESOURCE_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    logger.debug("Loaded predefined skills from %s", RESOURCE_PATH)
    return data


def get_available_categories() -> list[str]:
    """Return the list of scene categories present in the YAML."""
    data = load_predefined_skills()
    return [k for k in data if k not in ("overview",)]


# ---------------------------------------------------------------------------
# Scene skill resolution
# ---------------------------------------------------------------------------

def resolve_scene_category(scene_type: str) -> str:
    """Map a scene_type string to a YAML skill category.

    Falls back to ``"home"`` when no mapping is found.
    """
    category = SCENE_TYPE_TO_CATEGORY.get(scene_type)
    if category is None:
        for key, cat in SCENE_TYPE_TO_CATEGORY.items():
            if key in scene_type or scene_type in key:
                category = cat
                break
    if category is None:
        logger.warning(
            "Unknown scene_type %r, falling back to 'home'", scene_type,
        )
        category = "home"
    return category


def get_scene_skills(
    scene_type: str,
    *,
    category_override: str | None = None,
) -> tuple[list[dict], str]:
    """Return (common + scene-specific) skills and the resolved category.

    Args:
        scene_type: The ``scene_type`` string from scene analysis.
        category_override: Explicit category (from ``--scene-category``).

    Returns:
        A tuple of (skills_list, category_name).
    """
    category = category_override or resolve_scene_category(scene_type)
    data = load_predefined_skills()

    common = data.get("common", [])
    scene_specific = data.get(category, [])

    all_skills = common + scene_specific
    logger.info(
        "Resolved skills for scene_type=%r → category=%r: "
        "%d common + %d scene-specific = %d total",
        scene_type, category, len(common), len(scene_specific), len(all_skills),
    )
    return all_skills, category


def get_scene_skill_names(
    scene_type: str,
    *,
    category_override: str | None = None,
) -> list[str]:
    """Return just the skill name strings for the given scene."""
    skills, _ = get_scene_skills(
        scene_type, category_override=category_override,
    )
    return [s["name"] for s in skills]


# ---------------------------------------------------------------------------
# Deduplication: which predefined skills need new MCP tool definitions?
# ---------------------------------------------------------------------------

def _normalise_name(name: str) -> str:
    """Normalise a skill name to a valid Python identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name.strip()).strip("_")


def filter_skills_needing_code(skills: list[dict]) -> list[dict]:
    """Return skills that are NOT already covered by BASE_SERVER_TEMPLATE.

    A skill is considered "covered" if:
    - its name (or normalised name) appears in ``BASE_TOOLS``, or
    - it is in the ``_BASE_TOOL_COVERED_SKILLS`` semantic mapping.
    """
    result = []
    for skill in skills:
        name = skill["name"]
        norm = _normalise_name(name)
        if name in _BASE_TOOL_COVERED_SKILLS:
            continue
        if name in _BASE_TOOL_NAMES or norm in _BASE_TOOL_NAMES:
            continue
        result.append(skill)
    return result


# ---------------------------------------------------------------------------
# MCP tool code generation from skill definitions
# ---------------------------------------------------------------------------

def _infer_param_type(description: str) -> str:
    """Heuristic: infer Python type hint from a parameter description."""
    desc_lower = description.lower()
    if any(kw in desc_lower for kw in ("true/false", "bool")):
        return "bool"
    if any(kw in desc_lower for kw in ("数量", "count", "次数")):
        return "int"
    if any(kw in desc_lower for kw in ("距离", "高度", "角度", "速度", "float")):
        return "float"
    return "str"


def _is_optional(description: str) -> bool:
    """Detect if a parameter is optional from its description."""
    return "可选" in description or "optional" in description.lower()


def generate_tool_code(skill: dict) -> str:
    """Generate ``@mcp.tool()`` Python source from a single skill definition.

    The generated function is a mock stub (like BASE_SERVER_TEMPLATE tools)
    with ``time.sleep`` and a simple return dict.
    """
    raw_name = skill["name"]
    func_name = _normalise_name(raw_name)
    description = skill.get("description", raw_name)
    input_schema = skill.get("input", {})
    output_schema = skill.get("output", {})

    # Build parameter list
    required_params: list[str] = []
    optional_params: list[str] = []
    param_docs: list[str] = []
    first_param_name: str | None = None

    if isinstance(input_schema, dict) and input_schema:
        for pname, pdesc in input_schema.items():
            pdesc_str = str(pdesc) if pdesc else ""
            ptype = _infer_param_type(pdesc_str)
            safe_pname = _normalise_name(pname)
            if not first_param_name:
                first_param_name = safe_pname

            if _is_optional(pdesc_str):
                optional_params.append(f"{safe_pname}: Optional[{ptype}] = None")
            else:
                required_params.append(f"{safe_pname}: {ptype}")
            param_docs.append(f"        {safe_pname}: {pdesc_str}")

    params = required_params + optional_params

    # Build function signature
    if params:
        sig = f"def {func_name}(\n    " + ",\n    ".join(params) + ",\n)"
    else:
        sig = f"def {func_name}()"

    # Build docstring
    doc_lines = [f'    """{description}']
    if param_docs:
        doc_lines.append("")
        doc_lines.append("    Args:")
        doc_lines.extend(param_docs)
    doc_lines.append('    """')

    # Build return value from output schema
    return_dict = _build_return_dict(output_schema, description, first_param_name)

    lines = [
        "@mcp.tool()",
        f"{sig} -> dict:",
        *doc_lines,
        "    time.sleep(10)",
        f"    return {return_dict}",
    ]

    return "\n".join(lines)


def _build_return_dict(
    output_schema: dict | Any,
    description: str,
    first_param: str | None,
) -> str:
    """Build a Python dict literal string for the mock return value."""
    if not isinstance(output_schema, dict) or not output_schema:
        if first_param:
            return (
                '{"status": "success", '
                f'"message": f"已完成 {{{first_param}}} 的{description}操作"}}'
            )
        return f'{{"status": "success", "message": "已完成{description}操作"}}'

    has_status = "status" in output_schema
    has_state = "state" in output_schema

    if has_status:
        if first_param:
            return (
                '{"status": "success", '
                f'"message": f"已完成 {{{first_param}}} 的{description}操作"}}'
            )
        return f'{{"status": "success", "message": "已完成{description}操作"}}'
    elif has_state:
        if first_param:
            return (
                '{"state": "succeed", '
                f'"msg": f"已完成 {{{first_param}}} 的{description}操作"}}'
            )
        return f'{{"state": "succeed", "msg": "已完成{description}操作"}}'
    else:
        return '{"status": "success", "message": "操作完成"}'


def generate_all_tool_code(skills: list[dict]) -> str:
    """Generate MCP tool code for a list of skills (already filtered)."""
    parts = []
    for skill in skills:
        try:
            code = generate_tool_code(skill)
            parts.append(code)
        except Exception:
            logger.warning(
                "Failed to generate code for skill %r, skipping",
                skill.get("name"), exc_info=True,
            )
    return "\n\n\n".join(parts)


# ---------------------------------------------------------------------------
# High-level helpers for the pipeline
# ---------------------------------------------------------------------------

def get_predefined_tool_names(skills: list[dict]) -> list[str]:
    """Return the normalised function names for skills that need code gen."""
    filtered = filter_skills_needing_code(skills)
    return [_normalise_name(s["name"]) for s in filtered]


def format_skills_for_prompt(skills: list[dict]) -> str:
    """Format skill list as a concise text block for LLM prompts.

    Includes name, description, skill_type, input and output for each skill.
    """
    lines = []
    for s in skills:
        name = s["name"]
        desc = s.get("description", "")
        stype = s.get("skill_type", "")
        inp = s.get("input", {})
        out = s.get("output", {})

        inp_str = _format_schema(inp)
        out_str = _format_schema(out)
        lines.append(
            f"- {name}（{desc}，{stype}）"
            f"  输入: {inp_str}  输出: {out_str}"
        )
    return "\n".join(lines)


def _format_schema(schema: dict | Any) -> str:
    if not isinstance(schema, dict) or not schema:
        return "无"
    parts = []
    for k, v in schema.items():
        parts.append(f"{k}={v}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Capability gap detection
# ---------------------------------------------------------------------------

def detect_capability_gaps(all_tool_names: list[str]) -> list[str]:
    """Detect standard capabilities missing from the current tool set.

    Checks each role in ``STANDARD_CAPABILITIES`` against ``all_tool_names``.
    If none of the candidate names for a role exist, the **first** candidate
    is returned as the gap name (to be generated by LLM).

    Returns:
        List of tool names that need to be dynamically generated.
    """
    tool_set = set(all_tool_names)
    gaps: list[str] = []
    for role, candidates in STANDARD_CAPABILITIES.items():
        if not any(c in tool_set for c in candidates):
            gap_name = candidates[0]
            gaps.append(gap_name)
            logger.info(
                "Capability gap [%s]: none of %s found in tools, "
                "will request LLM to generate '%s'",
                role, candidates, gap_name,
            )
    return gaps


def resolve_tool_for_role(role: str, all_tool_names: list[str]) -> str | None:
    """Find the actual tool name that fills a standard capability role.

    Args:
        role: A key from ``STANDARD_CAPABILITIES`` (e.g. ``"fold"``).
        all_tool_names: Current full tool list.

    Returns:
        The matching tool name, or *None* if the role has no tool.
    """
    candidates = STANDARD_CAPABILITIES.get(role, [])
    tool_set = set(all_tool_names)
    for c in candidates:
        if c in tool_set:
            return c
    return None
