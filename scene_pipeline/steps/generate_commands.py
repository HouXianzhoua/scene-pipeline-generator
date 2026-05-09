"""Step 3: Generate user voice commands based on scene analysis."""

from __future__ import annotations

import logging
import re
import os
from pathlib import Path

from ..config import COMMAND_CATEGORIES
from ..llm_client import LLMClient
from ..prompts.command_gen import COMMAND_GEN_PROMPT
from ..skill_registry import resolve_tool_for_role

logger = logging.getLogger(__name__)

_AMBIGUOUS_POINTER_PATTERNS = (
    "那个",
    "这个",
    "那边那个",
    "这边那个",
    "那个东西",
    "这个东西",
    "那个物品",
    "这个物品",
)
_RELATION_HINTS = ("里", "中的", "上", "旁边", "左边", "右边", "前面", "后面", "中间")


def generate_commands(
    client: LLMClient,
    scene_data: dict,
    *,
    adaptive_categories: dict[str, dict] | None = None,
    all_tools: list[str] | None = None,
) -> dict:
    """Generate user voice commands and return structured command data.

    Args:
        client: LLM client instance.
        scene_data: Scene analysis dict from Step 2.
        adaptive_categories: Overridden category config from
            ``compute_adaptive_categories()``.  Falls back to the static
            ``COMMAND_CATEGORIES`` when *None*.
        all_tools: Full list of available tool names from server.py.
            When provided, the prompt constrains ``expected_tools`` to
            only use names from this list.
    """
    categories = adaptive_categories or COMMAND_CATEGORIES

    furniture_list = ", ".join(
        f"{f['name']}({f['name_en']}, {f['position']})"
        for f in scene_data.get("furniture", [])
    )
    objects_list = ", ".join(
        f"{o['name']}({o['name_en']}, {o.get('color', '')}, {o['position']})"
        for o in scene_data.get("objects", [])
    )

    categories_requirements = _format_categories(categories)
    total_min = sum(c["min_count"] for c in categories.values())

    tools_list = ", ".join(all_tools) if all_tools else "（未提供工具列表）"

    all_tool_names = list(all_tools or [])
    fold_tool = resolve_tool_for_role("fold", all_tool_names) or "fold"
    tidy_tool = resolve_tool_for_role("tidy", all_tool_names) or "tidy_surface"
    list_tool = resolve_tool_for_role("list", all_tool_names) or "list_objects"

    prompt = COMMAND_GEN_PROMPT.format(
        scene_type=scene_data["scene_type"],
        scene_display_name=scene_data["scene_display_name"],
        furniture_list=furniture_list,
        objects_list=objects_list,
        layout_description=scene_data.get("layout_description", ""),
        categories_requirements=categories_requirements,
        total_min=total_min,
        tools_list=tools_list,
        fold_tool=fold_tool,
        tidy_tool=tidy_tool,
        list_tool=list_tool,
    )

    messages = [{"role": "user", "content": prompt}]
    result = client.chat_json(messages)

    commands = result.get("commands", [])
    failures = result.get("failure_scenarios", [])

    commands = _filter_generated_commands(commands, scene_data)
    failures = _filter_failure_scenarios(failures, scene_data)
    result["commands"] = commands
    result["failure_scenarios"] = failures

    logger.info("Generated %d commands + %d failure scenarios", len(commands), len(failures))

    logger.info(
        "Generated %d commands (target reference: >= %d)", len(commands), total_min,
    )

    return result


def _filter_generated_commands(commands: list[dict], scene_data: dict) -> list[dict]:
    kept: list[dict] = []
    dropped = 0
    for command in commands:
        if _command_is_closed_and_unique(command, scene_data):
            kept.append(command)
        else:
            dropped += 1
            logger.info("Dropped fragile command: %s", command.get("text", ""))
    if dropped:
        logger.info("Filtered out %d fragile generated commands", dropped)
    return kept


def _filter_failure_scenarios(failures: list[dict], scene_data: dict) -> list[dict]:
    kept: list[dict] = []
    dropped = 0
    for failure in failures:
        pseudo_command = {
            "text": failure.get("text", ""),
            "target_object": failure.get("target_object", ""),
            "target_furniture": "",
        }
        if _command_is_closed_and_unique(pseudo_command, scene_data):
            kept.append(failure)
        else:
            dropped += 1
            logger.info("Dropped fragile failure scenario: %s", failure.get("text", ""))
    if dropped:
        logger.info("Filtered out %d fragile failure scenarios", dropped)
    return kept


def _command_is_closed_and_unique(command: dict, scene_data: dict) -> bool:
    text = (command.get("text") or "").strip()
    target_object = (command.get("target_object") or "").strip()
    if not text or not target_object:
        return False

    obj = _find_scene_object(scene_data, target_object)
    if not obj:
        return False

    if _contains_ambiguous_pointer(text) and not _text_mentions_object_or_anchor(text, obj, scene_data):
        return False

    if _requires_anchor_context(text) and not _text_mentions_object_or_anchor(text, obj, scene_data):
        return False

    return True


def _find_scene_object(scene_data: dict, target_object: str) -> dict | None:
    for obj in scene_data.get("objects", []):
        if obj.get("name_en") == target_object:
            return obj
    return None


def _find_furniture(scene_data: dict, name_en: str) -> dict | None:
    for furn in scene_data.get("furniture", []):
        if furn.get("name_en") == name_en:
            return furn
    return None


def _contains_ambiguous_pointer(text: str) -> bool:
    return any(token in text for token in _AMBIGUOUS_POINTER_PATTERNS)


def _requires_anchor_context(text: str) -> bool:
    return any(token in text for token in _RELATION_HINTS)


def _text_mentions_object_or_anchor(text: str, obj: dict, scene_data: dict) -> bool:
    haystack = _normalize_text(text)
    object_aliases = _build_text_aliases(obj.get("name", ""), obj.get("color", ""), obj.get("position", ""))
    if any(alias and alias in haystack for alias in object_aliases):
        return True

    furniture_name = obj.get("on_furniture", "")
    if not furniture_name:
        return False
    furn = _find_furniture(scene_data, furniture_name)
    if not furn:
        return False
    furniture_aliases = _build_text_aliases(furn.get("name", ""), furn.get("color", ""), furn.get("position", ""))
    return any(alias and alias in haystack for alias in furniture_aliases)


def _build_text_aliases(name: str, color: str, position: str) -> set[str]:
    aliases = {
        _normalize_text(name),
        _normalize_text(position),
    }
    core = name
    if color:
        core = core.replace(color, "")
    core = core.strip()
    aliases.add(_normalize_text(core))
    if color and core:
        aliases.add(_normalize_text(f"{color}{core}"))
    return {item for item in aliases if item}


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", lowered)


def _format_categories(categories: dict[str, dict]) -> str:
    """Build the numbered category requirements block for the prompt."""
    lines = []
    for idx, (cat, cfg) in enumerate(categories.items(), 1):
        desc = cfg["description"]
        count = cfg["min_count"]
        lines.append(f'{idx}. **{cat}**\uff08\u81f3\u5c11 {count} \u6761\uff09\uff1a\u5982\u201c{desc}\u201d')
    return "\n".join(lines)


def write_user_command_md(commands_data: dict, output_path: Path) -> None:
    """Write user_command.md from generated commands."""
    lines = ["以下是针对当前场景的用户语音指令测试集：\n"]

    categories: dict[str, list[str]] = {}
    for cmd in commands_data.get("commands", []):
        cat = cmd.get("category", "其他")
        categories.setdefault(cat, []).append(cmd["text"])

    for cat, texts in categories.items():
        lines.append(f"\n## {cat}\n")
        for text in texts:
            lines.append(f'"{text}"\n')

    failures = commands_data.get("failure_scenarios", [])
    if failures:
        lines.append("\n## 故障恢复场景\n")
        for f in failures:
            lines.append(f'- {f["description"]}："{f["text"]}"\n')

    _atomic_write_text(output_path, "".join(lines))
    logger.info("Wrote %s", output_path)


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace *path* with UTF-8 text written in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            logger.debug("Failed to remove temporary command artifact %s", tmp_path, exc_info=True)
