"""Pipeline orchestrator: runs all steps from scene photo to test report."""

from __future__ import annotations

import ast
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import (
    CODE_REPAIR_RETRIES,
    apply_default_case_volume,
    compute_adaptive_categories,
    default_param_case_limit,
)
from .image_utils import compress_image
from .llm_client import LLMClient
from .paths import EVAL_GENERATED_DIR
from .skill_registry import detect_capability_gaps, get_predefined_tool_names, get_scene_skills
from .steps.analyze_scene import analyze_scene
from .steps.generate_commands import generate_commands, write_user_command_md
from .steps.generate_config import generate_config
from .steps.generate_mocks import generate_mocks
from .steps.generate_server import generate_server
from .steps.generate_support import generate_support_files
from .steps.generate_tests import EVAL_PROTOCOL_VERSION, generate_tests
from .validation import validate_consistency

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step event — lightweight progress notification for UI consumers
# ---------------------------------------------------------------------------

@dataclass
class StepEvent:
    """Emitted before / after each pipeline step so that UIs can track progress."""

    step_id: str
    step_name: str
    status: str  # "running" | "done" | "skipped" | "error"
    timing: float | None = None
    data: Any = None
    error: str | None = None


StepCallback = Callable[[StepEvent], None]


def _emit(
    callback: StepCallback | None,
    step_id: str,
    step_name: str,
    status: str,
    *,
    timing: float | None = None,
    data: Any = None,
    error: str | None = None,
) -> None:
    """Fire *callback* if it is not ``None``; never raises."""
    if callback is None:
        return
    try:
        callback(StepEvent(
            step_id=step_id,
            step_name=step_name,
            status=status,
            timing=timing,
            data=data,
            error=error,
        ))
    except Exception:
        logger.debug("on_step callback raised; ignoring", exc_info=True)

_ARTIFACT_NAMES = {
    "scene_analysis": "scene_analysis.json",
    "commands_data": "commands_data.json",
    "all_tools": "all_tools.json",
    "mock_meta": "mock_meta.json",
}


class PipelineResult:
    """Holds the result of a pipeline run."""

    def __init__(self):
        self.scene_data: dict = {}
        self.commands_data: dict = {}
        self.all_tools: list[str] = []
        self.scene_dir: Path | None = None
        self.tests_dir: Path | None = None
        self.system_prompt: str = ""
        self.mock_meta: dict = {}
        self.errors: list[str] = []
        self.timings: dict[str, float] = {}
        self.consistency_warnings: list[str] = []
        self.steps_skipped: list[str] = []


def _ensure_mock_meta_ready(mock_meta: dict, commands_data: dict) -> None:
    """Fail fast when mock generation returned a structurally empty result."""
    command_count = len(commands_data.get("commands", []))
    factory_count = len(mock_meta.get("mock_factories", []))
    memory_count = len(mock_meta.get("memory_items", []))

    if command_count <= 0:
        return

    if not mock_meta:
        raise ValueError("mock_meta 为空，Mock 生成结果无效")
    if factory_count <= 0:
        raise ValueError("mock_meta 未生成任何 mock_factories，无法产出可执行测试")
    if memory_count <= 0:
        raise ValueError("mock_meta 未生成任何 memory_items，无法支撑查询类测试")


def _is_nonempty_file(path: Path) -> bool:
    """Return True when *path* exists and has non-zero size."""
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _is_valid_python_file(path: Path) -> bool:
    """Return True when *path* exists, is non-empty, and parses as Python."""
    if not _is_nonempty_file(path):
        return False
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        return True
    except (OSError, SyntaxError):
        return False


def _is_valid_scene_analysis(data: Any) -> bool:
    """Return True when scene analysis contains the minimum required fields."""
    return (
        isinstance(data, dict)
        and bool(data.get("scene_name"))
        and isinstance(data.get("objects", []), list)
        and isinstance(data.get("furniture", []), list)
    )


def _is_valid_all_tools(data: Any, scene_dir: Path) -> bool:
    """Return True when the tool list and generated server file are both usable."""
    return (
        isinstance(data, list)
        and len(data) > 0
        and _is_valid_python_file(scene_dir / "server.py")
    )


def _is_valid_commands_data(data: Any, scene_dir: Path) -> bool:
    """Return True when commands JSON and markdown companion are both usable."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("commands", []), list)
        and len(data.get("commands", [])) > 0
        and _is_nonempty_file(scene_dir / "user_command.md")
    )


def _is_valid_config_artifact(path: Path) -> bool:
    """Return True when the generated YAML config exists and is non-empty."""
    return _is_nonempty_file(path)


def _is_valid_mock_artifact(data: Any, tests_dir: Path, commands_data: dict) -> bool:
    """Return True when mock metadata and companion Python files are usable."""
    if not isinstance(data, dict):
        return False
    try:
        _ensure_mock_meta_ready(data, commands_data)
    except ValueError:
        return False
    return (
        _is_valid_python_file(tests_dir / "mock_tool_results.py")
        and _is_valid_python_file(tests_dir / "mock_server.py")
    )


def _is_valid_tests_artifact(tests_dir: Path) -> bool:
    """Return True when all generated test modules are present and structurally usable."""
    test_files = [
        tests_dir / "test_task_planning.py",
        tests_dir / "test_function_calling.py",
        tests_dir / "test_response_quality.py",
    ]
    if not all(_is_valid_python_file(path) for path in test_files):
        return False
    try:
        planning_source = (tests_dir / "test_task_planning.py").read_text(encoding="utf-8")
    except OSError:
        return False
    if f'EVAL_PROTOCOL_VERSION = "{EVAL_PROTOCOL_VERSION}"' not in planning_source:
        return False
    try:
        _ensure_generated_tests_nonempty(tests_dir)
    except ValueError:
        return False
    for path in test_files:
        try:
            _assert_generated_test_structure(path)
        except ValueError:
            return False
    return True


def _cleanup_stage_outputs(stage_id: str, scene_dir: Path, tests_dir: Path, *, scene_name: str | None = None) -> None:
    """Delete outputs for a single stage so resume can regenerate them safely."""
    targets = _stage_output_paths(stage_id, scene_dir, tests_dir, scene_name=scene_name)

    for target in targets:
        try:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        except OSError as exc:
            logger.warning("Failed to remove stale stage output %s: %s", target, exc)


def _stage_output_paths(
    stage_id: str,
    scene_dir: Path,
    tests_dir: Path,
    *,
    scene_name: str | None = None,
) -> list[Path]:
    """Return the generated files owned by one pipeline stage."""
    targets: list[Path] = []
    if stage_id == "analyze":
        targets.append(scene_dir / _ARTIFACT_NAMES["scene_analysis"])
    elif stage_id == "server":
        targets.extend([scene_dir / "server.py", scene_dir / _ARTIFACT_NAMES["all_tools"]])
    elif stage_id == "commands":
        targets.extend([scene_dir / "user_command.md", scene_dir / _ARTIFACT_NAMES["commands_data"]])
    elif stage_id == "config":
        if scene_name:
            targets.append(scene_dir / f"{scene_name}.yaml")
    elif stage_id == "mocks":
        targets.extend([
            scene_dir / _ARTIFACT_NAMES["mock_meta"],
            tests_dir / "mock_tool_results.py",
            tests_dir / "mock_server.py",
        ])
    elif stage_id == "tests":
        targets.extend([
            tests_dir / "test_task_planning.py",
            tests_dir / "test_function_calling.py",
            tests_dir / "test_response_quality.py",
        ])

    return targets


_STAGE_REGEN_ORDER = ("analyze", "server", "commands", "config", "mocks", "tests")


def _cleanup_stage_and_downstream(
    stage_id: str,
    scene_dir: Path,
    tests_dir: Path,
    *,
    scene_name: str | None = None,
) -> None:
    """Delete outputs for *stage_id* and every downstream stage."""
    try:
        start = _STAGE_REGEN_ORDER.index(stage_id)
    except ValueError:
        logger.warning("Unknown stage for cleanup cascade: %s", stage_id)
        return
    for downstream_stage in _STAGE_REGEN_ORDER[start:]:
        _cleanup_stage_outputs(
            downstream_stage,
            scene_dir,
            tests_dir,
            scene_name=scene_name,
        )


def _stage_or_downstream_outputs_exist(
    stage_id: str,
    scene_dir: Path,
    tests_dir: Path,
    *,
    scene_name: str | None = None,
) -> bool:
    """Return True when a stage or later stage left any materialized output."""
    try:
        start = _STAGE_REGEN_ORDER.index(stage_id)
    except ValueError:
        return False
    for downstream_stage in _STAGE_REGEN_ORDER[start:]:
        for path in _stage_output_paths(
            downstream_stage,
            scene_dir,
            tests_dir,
            scene_name=scene_name,
        ):
            if path.exists():
                return True
    return False


def _count_pytest_tests_in_file(path: Path) -> int:
    """Count top-level pytest test methods/functions in a generated test file."""
    if not path.exists():
        return 0

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return 0

    total = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            total += 1
    return total


def _assert_generated_test_structure(path: Path) -> None:
    """Fail fast when generated test modules contain duplicate classes/tests."""
    if not path.exists():
        return

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return

    class_names: list[str] = []
    duplicate_class_names: set[str] = set()
    duplicate_test_names: list[str] = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name in class_names:
            duplicate_class_names.add(node.name)
        class_names.append(node.name)

        seen_test_names: set[str] = set()
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not item.name.startswith("test_"):
                continue
            if item.name in seen_test_names:
                duplicate_test_names.append(f"{node.name}.{item.name}")
            seen_test_names.add(item.name)

    problems: list[str] = []
    if duplicate_class_names:
        problems.append(f"重复测试类: {sorted(duplicate_class_names)}")
    if duplicate_test_names:
        problems.append(f"重复测试方法: {sorted(duplicate_test_names)}")
    if problems:
        raise ValueError(f"{path.name} 结构校验失败，" + "；".join(problems))


def _ensure_generated_tests_nonempty(tests_dir: Path) -> None:
    """Fail fast when generated test files contain zero runnable pytest tests."""
    test_files = (
        tests_dir / "test_task_planning.py",
        tests_dir / "test_function_calling.py",
        tests_dir / "test_response_quality.py",
    )
    for path in test_files:
        _assert_generated_test_structure(path)
    total = sum(_count_pytest_tests_in_file(path) for path in test_files)
    if total <= 0:
        raise ValueError("生成的测试文件中未发现任何 pytest 用例")


def run_pipeline(
    image_path: str | Path,
    client: LLMClient,
    output_dir: str | Path | None = None,
    *,
    resume: bool = False,
    scene_category: str | None = None,
    on_step: StepCallback | None = None,
) -> PipelineResult:
    """Execute the full generation pipeline.

    Args:
        image_path: Path to the scene photo.
        client: LLM client for generation.
        output_dir: Explicit output directory (required when *resume=True*).
        resume: When True, skip steps whose output files already exist.
            Intermediate artifacts are saved/loaded as JSON alongside the
            generated code so that later steps can pick up where a previous
            run left off.
        scene_category: Explicit skill-category override (e.g. ``"home"``).
            When *None*, the category is auto-resolved from the analysed
            ``scene_type`` via ``SCENE_TYPE_TO_CATEGORY``.

    Steps:
        1. Compress image (if needed)
        2. Analyze scene via Vision LLM
        2b. Compute adaptive category config
        2c. Load predefined skills from YAML (common + scene-specific)
        3. Generate server.py  (+ LLM repair on syntax error) → all_tools
        4. Generate user commands (with adaptive counts, constrained by all_tools)
        5. Generate scene.yaml
        6. Generate mock test system (+ LLM repair on syntax error)
        7. Generate test files   (+ LLM repair on syntax error)
        8. Cross-file consistency validation
    """
    result = PipelineResult()
    image_path = Path(image_path)
    total_start = time.time()

    # When resuming we need the output directory first to look for artifacts.
    scene_dir: Path | None = Path(output_dir) if output_dir else None

    # Check if we can skip Steps 1-2 entirely (resume with existing analysis)
    cached_analysis = (
        _try_load(scene_dir, "scene_analysis")
        if resume and scene_dir
        else None
    )
    if resume and scene_dir and cached_analysis is not None and not _is_valid_scene_analysis(cached_analysis):
        logger.warning("scene_analysis artifact is incomplete; deleting stale analyze+downstream outputs before resume")
        _cleanup_stage_and_downstream("analyze", scene_dir, scene_dir / "tests")
        cached_analysis = None

    # ------------------------------------------------------------------
    # Step 1: Compress image
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1: Image compression")
    logger.info("=" * 60)
    t0 = time.time()
    if cached_analysis is not None:
        compressed = image_path
        logger.info("[SKIP] scene_analysis.json exists, image compression not needed")
        result.steps_skipped.append("compress")
        _emit(on_step, "compress", "图像压缩", "skipped")
    else:
        _emit(on_step, "compress", "图像压缩", "running")
        compressed = compress_image(image_path)
        result.timings["compress"] = time.time() - t0
        _emit(on_step, "compress", "图像压缩", "done",
              timing=result.timings["compress"],
              data={"compressed_path": str(compressed)})
    if "compress" not in result.timings:
        result.timings["compress"] = time.time() - t0

    # ------------------------------------------------------------------
    # Step 2: Analyze scene
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: Scene analysis")
    logger.info("=" * 60)
    t0 = time.time()
    if cached_analysis is not None:
        result.scene_data = cached_analysis
        logger.info("[SKIP] Loaded scene_analysis.json from previous run")
        result.steps_skipped.append("analyze")
        _emit(on_step, "analyze", "场景分析", "skipped", data=result.scene_data)
    else:
        _emit(on_step, "analyze", "场景分析", "running")
        try:
            result.scene_data = analyze_scene(client, compressed)
        except Exception as e:
            result.errors.append(f"Scene analysis failed: {e}")
            logger.error("Scene analysis failed: %s", e, exc_info=True)
            _emit(on_step, "analyze", "场景分析", "error", error=str(e))
            return result
    result.timings["analyze"] = time.time() - t0
    if "analyze" not in result.steps_skipped:
        _emit(on_step, "analyze", "场景分析", "done",
              timing=result.timings["analyze"], data=result.scene_data)

    scene_name = result.scene_data["scene_name"]

    # Step 2b: Adaptive category config (always computed, no I/O)
    _emit(on_step, "adaptive", "自适应配置", "running")
    adaptive_cats = apply_default_case_volume(
        compute_adaptive_categories(result.scene_data),
    )
    total_min = sum(c["min_count"] for c in adaptive_cats.values())
    logger.info(
        "Adaptive config: %d objects, %d furniture → total min commands = %d",
        len(result.scene_data.get("objects", [])),
        len(result.scene_data.get("furniture", [])),
        total_min,
    )

    # Step 2c: Load predefined skills from YAML (common + scene-specific)
    scene_type = result.scene_data.get("scene_type", "")
    predefined_skills, resolved_category = get_scene_skills(
        scene_type, category_override=scene_category,
    )
    logger.info(
        "Predefined skills: category=%s, %d skills loaded",
        resolved_category, len(predefined_skills),
    )

    # Step 2d: Detect capability gaps — standard capabilities not covered
    # by predefined skills or base tools.  Gap names are appended to
    # extra_tools so generate_server's LLM path creates them dynamically.
    from .config import BASE_TOOLS
    available_names = list(BASE_TOOLS) + get_predefined_tool_names(predefined_skills)
    gaps = detect_capability_gaps(available_names)
    if gaps:
        existing_extra = result.scene_data.get("extra_tools", [])
        merged = list(dict.fromkeys(existing_extra + gaps))
        result.scene_data["extra_tools"] = merged
        logger.info(
            "Capability gaps detected: %s → merged extra_tools: %s",
            gaps, merged,
        )

    _emit(on_step, "adaptive", "自适应配置 + 技能加载", "done", data={
        "adaptive_categories": adaptive_cats,
        "target_min_commands": total_min,
        "predefined_skills_count": len(predefined_skills),
        "resolved_category": resolved_category,
        "skill_names": [s.get("name", "") for s in predefined_skills],
        "capability_gaps": gaps if gaps else [],
    })

    # Setup output directory
    if scene_dir is None:
        scene_dir = EVAL_GENERATED_DIR / scene_name

    scene_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = scene_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "report").mkdir(parents=True, exist_ok=True)
    (tests_dir / "doc").mkdir(parents=True, exist_ok=True)

    result.scene_dir = scene_dir
    result.tests_dir = tests_dir

    # Save scene analysis
    _save_artifact(scene_dir, "scene_analysis", result.scene_data)

    # ------------------------------------------------------------------
    # Step 3: Generate server.py (with repair) → all_tools
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3: Generate server.py")
    logger.info("=" * 60)
    t0 = time.time()
    cached = _try_load(scene_dir, "all_tools") if resume else None
    if resume and cached is not None and not _is_valid_all_tools(cached, scene_dir):
        logger.warning("server artifacts are incomplete; deleting stale server+downstream outputs before resume")
        _cleanup_stage_and_downstream("server", scene_dir, tests_dir, scene_name=scene_name)
        cached = None
    if cached is not None:
        result.all_tools = cached
        logger.info("[SKIP] Loaded all_tools.json + server.py from previous run")
        result.steps_skipped.append("server")
        _emit(on_step, "server", "Server 生成", "skipped", data=result.all_tools)
    else:
        _emit(on_step, "server", "Server 生成", "running")
        try:
            result.all_tools = generate_server(
                client, result.scene_data, scene_dir / "server.py",
                predefined_skills=predefined_skills,
            )
            _validate_and_repair(scene_dir / "server.py", "server.py", client)
            _save_artifact(scene_dir, "all_tools", result.all_tools)
        except Exception as e:
            result.errors.append(f"Server generation failed: {e}")
            logger.error("Server generation failed: %s", e, exc_info=True)
            _emit(on_step, "server", "Server 生成", "error", error=str(e))
            return result
    result.timings["server"] = time.time() - t0
    if "server" not in result.steps_skipped:
        _emit(on_step, "server", "Server 生成", "done",
              timing=result.timings["server"], data=result.all_tools)

    # ------------------------------------------------------------------
    # Step 4: Generate user commands (adaptive, constrained by all_tools)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4: Generate user commands")
    logger.info("=" * 60)
    t0 = time.time()
    cached = _try_load(scene_dir, "commands_data") if resume else None
    commands_json_unusable = (
        resume
        and not _is_valid_commands_data(cached, scene_dir)
        and _stage_or_downstream_outputs_exist(
            "commands", scene_dir, tests_dir, scene_name=scene_name,
        )
    )
    if commands_json_unusable:
        logger.warning("command artifacts are incomplete; deleting stale commands+downstream outputs before resume")
        _cleanup_stage_and_downstream("commands", scene_dir, tests_dir, scene_name=scene_name)
        cached = None
    if cached is not None:
        result.commands_data = cached
        logger.info("[SKIP] Loaded commands_data.json from previous run")
        result.steps_skipped.append("commands")
        _emit(on_step, "commands", "指令生成", "skipped", data=result.commands_data)
    else:
        _emit(on_step, "commands", "指令生成", "running")
        try:
            result.commands_data = generate_commands(
                client, result.scene_data,
                adaptive_categories=adaptive_cats,
                all_tools=result.all_tools,
            )
            write_user_command_md(result.commands_data, scene_dir / "user_command.md")
            _save_artifact(scene_dir, "commands_data", result.commands_data)
        except Exception as e:
            result.errors.append(f"Command generation failed: {e}")
            logger.error("Command generation failed: %s", e, exc_info=True)
            _emit(on_step, "commands", "指令生成", "error", error=str(e))
            return result
    result.timings["commands"] = time.time() - t0
    if "commands" not in result.steps_skipped:
        _emit(on_step, "commands", "指令生成", "done",
              timing=result.timings["commands"], data=result.commands_data)

    # ------------------------------------------------------------------
    # Step 5: Generate scene.yaml
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 5: Generate scene YAML config")
    logger.info("=" * 60)
    t0 = time.time()
    yaml_path = scene_dir / f"{scene_name}.yaml"
    if resume and yaml_path.exists() and not _is_valid_config_artifact(yaml_path):
        logger.warning("config artifact is incomplete; deleting stale config+downstream outputs before resume")
        _cleanup_stage_and_downstream("config", scene_dir, tests_dir, scene_name=scene_name)
    if resume and _is_valid_config_artifact(yaml_path):
        logger.info("[SKIP] %s already exists", yaml_path.name)
        result.steps_skipped.append("config")
        _emit(on_step, "config", "配置生成", "skipped")
    else:
        _emit(on_step, "config", "配置生成", "running")
        try:
            result.system_prompt = generate_config(
                client, result.scene_data, result.all_tools, yaml_path,
                predefined_skills=predefined_skills,
                scene_category=resolved_category,
            )
        except Exception as e:
            result.errors.append(f"Config generation failed: {e}")
            logger.error("Config generation failed: %s", e, exc_info=True)
            _emit(on_step, "config", "配置生成", "error", error=str(e))
            return result
    result.timings["config"] = time.time() - t0
    if "config" not in result.steps_skipped:
        _emit(on_step, "config", "配置生成", "done",
              timing=result.timings["config"],
              data={"yaml_path": str(yaml_path)})

    # ------------------------------------------------------------------
    # Step 6: Generate mock test system (with repair)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 6: Generate mock test system")
    logger.info("=" * 60)
    t0 = time.time()
    cached = _try_load(scene_dir, "mock_meta") if resume else None
    if resume and cached is not None and not _is_valid_mock_artifact(cached, tests_dir, result.commands_data):
        logger.warning("mock artifacts are incomplete; deleting stale mocks+downstream outputs before resume")
        _cleanup_stage_and_downstream("mocks", scene_dir, tests_dir, scene_name=scene_name)
        cached = None
    if cached is not None:
        result.mock_meta = cached
        logger.info("[SKIP] Loaded mock_meta.json + mock files from previous run")
        result.steps_skipped.append("mocks")
        _emit(on_step, "mocks", "Mock 生成", "skipped", data=result.mock_meta)
    else:
        _emit(on_step, "mocks", "Mock 生成", "running")
        try:
            result.mock_meta = generate_mocks(
                client, result.scene_data, result.commands_data,
                result.all_tools, tests_dir,
            )
            _ensure_mock_meta_ready(result.mock_meta, result.commands_data)
            _validate_and_repair(
                tests_dir / "mock_tool_results.py", "mock_tool_results.py", client,
            )
            _save_artifact(scene_dir, "mock_meta", result.mock_meta)
        except Exception as e:
            result.errors.append(f"Mock generation failed: {e}")
            logger.error("Mock generation failed: %s", e, exc_info=True)
            _emit(on_step, "mocks", "Mock 生成", "error", error=str(e))
            return result
    result.timings["mocks"] = time.time() - t0
    if "mocks" not in result.steps_skipped:
        _emit(on_step, "mocks", "Mock 生成", "done",
              timing=result.timings["mocks"], data=result.mock_meta)

    # ------------------------------------------------------------------
    # Step 7: Generate test files (with repair) + support files
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 7: Generate test files")
    logger.info("=" * 60)
    t0 = time.time()
    tests_ready = _is_valid_tests_artifact(tests_dir)
    if resume and not tests_ready:
        stale_test_files = [
            tests_dir / "test_task_planning.py",
            tests_dir / "test_function_calling.py",
            tests_dir / "test_response_quality.py",
        ]
        if any(path.exists() for path in stale_test_files):
            logger.warning("test artifacts are incomplete; deleting stale tests before resume")
            _cleanup_stage_and_downstream("tests", scene_dir, tests_dir, scene_name=scene_name)
        tests_ready = False
    if resume and tests_ready:
        logger.info("[SKIP] Test files already exist")
        result.steps_skipped.append("tests")
        _emit(on_step, "tests", "测试生成", "skipped")
    else:
        _emit(on_step, "tests", "测试生成", "running")
        try:
            generate_tests(
                client,
                result.commands_data,
                result.mock_meta,
                tests_dir,
                param_case_limit=default_param_case_limit(),
            )
            for tf in (
                "test_task_planning.py",
                "test_function_calling.py",
                "test_response_quality.py",
            ):
                p = tests_dir / tf
                if p.exists():
                    _validate_and_repair(p, tf, client)
            _ensure_generated_tests_nonempty(tests_dir)
        except Exception as e:
            result.errors.append(f"Test generation failed: {e}")
            logger.error("Test generation failed: %s", e, exc_info=True)
            _emit(on_step, "tests", "测试生成", "error", error=str(e))

    # Support files are cheap to regenerate; always refresh them.
    try:
        generate_support_files(
            result.scene_data, result.all_tools, result.commands_data,
            scene_dir, tests_dir,
        )
    except Exception as e:
        result.errors.append(f"Support file generation failed: {e}")
        logger.error("Support file generation failed: %s", e, exc_info=True)

    result.timings["tests"] = time.time() - t0
    if "tests" not in result.steps_skipped:
        _emit(on_step, "tests", "测试生成", "done",
              timing=result.timings["tests"],
              data={"tests_dir": str(tests_dir)})

    # ------------------------------------------------------------------
    # Step 8: Cross-file consistency validation (always runs)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 8: Cross-file consistency validation")
    logger.info("=" * 60)
    _emit(on_step, "validation", "一致性校验", "running")
    t0 = time.time()
    consistency = validate_consistency(
        scene_dir, result.all_tools, result.commands_data, result.mock_meta,
    )
    result.consistency_warnings = consistency.warnings
    if consistency.errors:
        for err in consistency.errors:
            result.errors.append(f"Consistency: {err}")
    result.timings["validation"] = time.time() - t0
    _emit(on_step, "validation", "一致性校验", "done",
          timing=result.timings["validation"],
          data={
              "errors": consistency.errors if consistency.errors else [],
              "warnings": consistency.warnings,
          })

    result.timings["total"] = time.time() - total_start

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("Output: %s", scene_dir)
    if result.steps_skipped:
        logger.info("Skipped steps (resume): %s", ", ".join(result.steps_skipped))
    logger.info("Timings: %s", {k: f"{v:.1f}s" for k, v in result.timings.items()})
    if result.errors:
        logger.warning("Errors: %s", result.errors)
    logger.info("=" * 60)

    _emit(on_step, "complete", "流水线完成", "done",
          timing=result.timings.get("total"),
          data={
              "scene_dir": str(scene_dir),
              "errors": result.errors,
              "warnings": result.consistency_warnings,
              "timings": {k: round(v, 1) for k, v in result.timings.items()},
          })

    return result


# ---------------------------------------------------------------------------
# Intermediate artifact save / load
# ---------------------------------------------------------------------------

def _save_artifact(scene_dir: Path, name: str, data: dict | list) -> None:
    path = scene_dir / _ARTIFACT_NAMES[name]
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
    logger.debug("Saved artifact: %s", path)


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
            logger.debug("Failed to remove temporary artifact %s", tmp_path, exc_info=True)


def _try_load(scene_dir: Path | None, name: str) -> dict | list | None:
    """Return parsed JSON if the artifact file exists, else None."""
    if scene_dir is None:
        return None
    path = scene_dir / _ARTIFACT_NAMES[name]
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load artifact %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Validate-and-repair: feed syntax errors back to the LLM for auto-fix
# ---------------------------------------------------------------------------

def _validate_and_repair(
    filepath: Path,
    label: str,
    client: LLMClient,
    max_retries: int = CODE_REPAIR_RETRIES,
) -> None:
    """Validate a Python file; on SyntaxError, ask the LLM to repair it."""
    for attempt in range(max_retries + 1):
        try:
            source = filepath.read_text(encoding="utf-8")
            ast.parse(source)
            logger.info("%s: syntax OK", label)
            return
        except SyntaxError as e:
            if attempt == max_retries:
                logger.warning(
                    "%s: syntax error persisted after %d repair attempt(s)",
                    label, max_retries,
                )
                raise

            logger.warning(
                "%s: syntax error at line %d: %s — requesting LLM repair (attempt %d/%d)",
                label, e.lineno or 0, e.msg, attempt + 1, max_retries,
            )
            repaired = _repair_code(client, source, e)
            filepath.write_text(repaired, encoding="utf-8")
            logger.info("%s: wrote repaired code (attempt %d)", label, attempt + 1)


_REPAIR_PROMPT = """\
以下 Python 代码存在语法错误，请修复。

## 错误信息
- 行号: {lineno}
- 错误: {msg}
- 问题代码行: {text}

## 完整代码
```python
{source}
```

## 要求
1. 仅修复语法错误，不改变代码逻辑和功能
2. 返回完整的修复后代码（不要省略任何部分）
3. 返回 JSON 格式: {{"code": "修复后的完整 Python 代码"}}
"""


def _repair_code(client: LLMClient, source: str, error: SyntaxError) -> str:
    """Ask the LLM to fix a syntax error and return the corrected source."""
    prompt = _REPAIR_PROMPT.format(
        lineno=error.lineno or "?",
        msg=error.msg,
        text=(error.text or "").strip(),
        source=source,
    )
    try:
        result = client.chat_json([{"role": "user", "content": prompt}])
        code = result.get("code", "")
        if code and len(code) > len(source) * 0.5:
            return code
        logger.warning("LLM repair returned suspiciously short code, keeping original")
        return source
    except Exception as exc:
        logger.warning("LLM repair call failed: %s", exc)
        return source
