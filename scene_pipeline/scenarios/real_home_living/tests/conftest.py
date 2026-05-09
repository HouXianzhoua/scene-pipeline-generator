"""Shared fixtures and helpers for real home living scenario mock tests."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end integration tests requiring a running Agent")


import inspect
import importlib.util
import json
import os
import re
import shutil
import sys
import time as _time_mod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

import pytest
import yaml
from openai import OpenAI

REAL_HOME_LIVING_DIR = Path(__file__).parent.parent


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "kaiwu_agent").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root from test directory")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
KAIWU_AGENT_DIR = REPO_ROOT / "kaiwu_agent"

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Python type -> JSON Schema conversion
# ---------------------------------------------------------------------------

def _python_type_to_json_schema(py_type) -> dict:
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
        return {}

    if origin is list:
        schema: dict = {"type": "array"}
        if args:
            schema["items"] = _python_type_to_json_schema(args[0])
        return schema

    if origin is dict:
        return {"type": "object"}

    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    return type_map.get(py_type, {})


def _extract_param_descriptions(docstring: str) -> dict[str, str]:
    """Extract parameter descriptions from Google-style docstring."""
    descriptions: dict[str, str] = {}
    if not docstring:
        return descriptions

    in_args = False
    current_param: str | None = None
    current_lines: list[str] = []

    for line in docstring.split("\n"):
        stripped = line.strip()

        if stripped == "Args:":
            in_args = True
            continue
        if not in_args:
            continue
        if stripped.startswith("Returns:") or stripped.startswith("Raises:"):
            break

        if ":" in stripped and not stripped.startswith(("-", "*")):
            colon_idx = stripped.index(":")
            candidate = stripped[:colon_idx].strip().split("(")[0].strip()
            if candidate.isidentifier():
                if current_param:
                    descriptions[current_param] = " ".join(current_lines).strip()
                current_param = candidate
                current_lines = [stripped[colon_idx + 1 :].strip()]
                continue

        if current_param and stripped:
            current_lines.append(stripped)

    if current_param:
        descriptions[current_param] = " ".join(current_lines).strip()
    return descriptions


# ---------------------------------------------------------------------------
# Tool schema extraction from server.py
# ---------------------------------------------------------------------------

def extract_tool_schemas() -> list[dict]:
    """Import server.py and convert tool functions to OpenAI function calling schemas."""
    server_path = REAL_HOME_LIVING_DIR / "server.py"

    spec = importlib.util.spec_from_file_location("_rhl_server", str(server_path))
    module = importlib.util.module_from_spec(spec)

    orig_sleep = _time_mod.sleep
    _time_mod.sleep = lambda _: None
    try:
        spec.loader.exec_module(module)
    finally:
        _time_mod.sleep = orig_sleep

    registered_names: set[str] = set()
    try:
        mgr = getattr(module.mcp, "_tool_manager", None)
        if mgr and hasattr(mgr, "_tools"):
            registered_names = set(mgr._tools.keys())
    except Exception:
        pass

    tools: list[dict] = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        if getattr(func, "__module__", None) != module.__name__:
            continue
        if registered_names and name not in registered_names:
            continue

        sig = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        docstring = inspect.getdoc(func) or ""
        parts = re.split(r"\n\s*(?:Args|Returns|Raises):", docstring, maxsplit=1)
        description = parts[0].strip() if parts else name

        param_descs = _extract_param_descriptions(docstring)

        properties: dict[str, dict] = {}
        required: list[str] = []

        for pname, param in sig.parameters.items():
            py_type = hints.get(pname, str)
            prop = _python_type_to_json_schema(py_type)
            if pname in param_descs:
                prop["description"] = param_descs[pname]
            properties[pname] = prop
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        tool: dict = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }
        if required:
            tool["function"]["parameters"]["required"] = required
        tools.append(tool)

    return tools


# ---------------------------------------------------------------------------
# Conversation result
# ---------------------------------------------------------------------------

class ConversationResult:
    """Holds the complete result of a multi-turn LLM conversation."""

    def __init__(
        self,
        tool_calls: list[tuple[str, dict]],
        messages: list[dict],
        usage: dict[str, int] | None = None,
    ):
        self.tool_calls = tool_calls
        self.messages = messages
        self.usage = usage or {}

    @property
    def tool_names(self) -> list[str]:
        return [n for n, _ in self.tool_calls]

    def get_calls_for(self, tool_name: str) -> list[dict]:
        return [args for n, args in self.tool_calls if n == tool_name]

    def first_call_for(self, tool_name: str) -> dict | None:
        calls = self.get_calls_for(tool_name)
        return calls[0] if calls else None

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "assistant")

    @property
    def total_tool_calls_count(self) -> int:
        return len(self.tool_calls)

    @property
    def assistant_messages(self) -> list[dict]:
        return [m for m in self.messages if m.get("role") == "assistant"]


# ---------------------------------------------------------------------------
# Multi-turn conversation runner
# ---------------------------------------------------------------------------

class ConversationRunner:
    """Drives a multi-turn LLM conversation, injecting mock tool results."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        tools: list[dict],
        max_turns: int = 30,
        verbose: bool = False,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_turns = max_turns
        self.verbose = verbose

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def run(
        self,
        user_message: str,
        mock_provider: Any,
        max_turns: int | None = None,
    ) -> ConversationResult:
        limit = max_turns or self.max_turns
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        all_tool_calls: list[tuple[str, dict]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        self._log(f"\n{'='*60}")
        self._log(f"[USER] {user_message}")
        self._log(f"{'='*60}")

        for turn in range(limit):
            msg = None
            for _retry in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=self.tools,
                        temperature=0,
                    )
                    if response.choices:
                        msg = response.choices[0].message
                        break
                except Exception:
                    if _retry == 2:
                        raise
                    import time as _t
                    _t.sleep(2)
            if msg is None:
                break

            if hasattr(response, "usage") and response.usage:
                for k in total_usage:
                    total_usage[k] += getattr(response.usage, k, 0) or 0

            self._log(f"\n--- Turn {turn + 1} ---")
            if msg.content:
                self._log(f"[ASSISTANT] {msg.content}")

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                all_tool_calls.append((name, args))

                self._log(f"[TOOL CALL] {name}({json.dumps(args, ensure_ascii=False)})")

                result = mock_provider.get(name, args)
                result_str = (
                    result
                    if isinstance(result, str)
                    else json.dumps(result, ensure_ascii=False)
                )

                self._log(f"[TOOL RESULT] {result_str[:200]}{'...' if len(result_str) > 200 else ''}")

                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result_str}
                )

        self._log(f"\n{'='*60}")
        self._log(f"Conversation ended after {len(all_tool_calls)} tool calls")
        self._log(f"Tool sequence: {[n for n, _ in all_tool_calls]}")
        if total_usage["total_tokens"]:
            self._log(f"Token usage: {total_usage}")
        self._log(f"{'='*60}\n")

        return ConversationResult(all_tool_calls, messages, usage=total_usage)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def assert_tool_sequence(result: ConversationResult, expected: list[str]):
    """Assert the exact tool call sequence."""
    assert result.tool_names == expected, (
        f"Expected sequence:\n  {expected}\nActual:\n  {result.tool_names}"
    )


def assert_tool_subsequence(result: ConversationResult, expected: list[str]):
    """Assert expected names appear in order (not necessarily consecutive)."""
    actual = result.tool_names
    idx = 0
    for name in actual:
        if idx < len(expected) and name == expected[idx]:
            idx += 1
    assert idx == len(expected), (
        f"Expected subsequence:\n  {expected}\n"
        f"Only matched {idx}/{len(expected)} in actual:\n  {actual}"
    )


def assert_tool_contains(result: ConversationResult, tool_name: str):
    assert tool_name in result.tool_names, (
        f"'{tool_name}' not found in tool calls: {result.tool_names}"
    )


def assert_tool_not_contains(result: ConversationResult, tool_name: str):
    assert tool_name not in result.tool_names, (
        f"'{tool_name}' should not appear but found in: {result.tool_names}"
    )


def assert_tool_order(result: ConversationResult, before: str, after: str):
    """Assert first occurrence of *before* precedes first occurrence of *after*."""
    names = result.tool_names
    assert before in names, f"'{before}' not found in {names}"
    assert after in names, f"'{after}' not found in {names}"
    assert names.index(before) < names.index(after), (
        f"'{before}' (idx={names.index(before)}) should appear before "
        f"'{after}' (idx={names.index(after)})"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def system_prompt() -> str:
    with open(REAL_HOME_LIVING_DIR / "real_home_living.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["system_prompt"]


@pytest.fixture(scope="session")
def tool_schemas() -> list[dict]:
    return extract_tool_schemas()


@pytest.fixture(scope="session")
def model_name() -> str:
    return os.environ.get("TEST_MODEL", "Qwen3-VL-235B-A22B-Instruct")


@pytest.fixture(scope="session")
def openai_client() -> OpenAI:
    base_url = os.environ.get("TEST_BASE_URL", "http://120.48.75.178:4970/v1")
    api_key = os.environ.get("TEST_API_KEY", "EMPTY")
    return OpenAI(base_url=base_url, api_key=api_key)


@pytest.fixture(scope="session")
def runner(openai_client, model_name, system_prompt, tool_schemas) -> ConversationRunner:
    verbose = os.environ.get("TEST_VERBOSE", "0") not in ("0", "false", "")
    return ConversationRunner(
        client=openai_client,
        model=model_name,
        system_prompt=system_prompt,
        tools=tool_schemas,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Evaluation report collection
# ---------------------------------------------------------------------------

_eval_records: list[dict] = []
_test_outcomes: dict[str, str] = {}
_REPORT_SESSION_TS = os.environ.get("TEST_REPORT_SESSION_TS", "").strip()
_REPORT_SESSION_DISPLAY = os.environ.get("TEST_REPORT_SESSION_DISPLAY", "").strip()
if not _REPORT_SESSION_TS or not _REPORT_SESSION_DISPLAY:
    _REPORT_SESSION_DT = datetime.now()
    _REPORT_SESSION_TS = _REPORT_SESSION_DT.strftime("%Y%m%d_%H%M%S")
    _REPORT_SESSION_DISPLAY = _REPORT_SESSION_DT.strftime("%Y-%m-%d %H:%M:%S")
_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "").strip()


_FETCH_EXPECTED = [
    "spatial_memory_query_vec", "move_to", "scene_recognition",
    "grasp_start", "perception_custom", "handover",
]

_PLACE_EXPECTED = [
    "spatial_memory_query_vec", "move_to", "scene_recognition",
    "grasp_start", "spatial_memory_query_vec", "move_to", "place_down",
]


def _build_reference_map() -> dict[str, dict]:
    refs: dict[str, dict] = {}

    def add_planning(name: str, user_message: str, expected_seq: list[str], *, forbidden=None, min_calls=None):
        refs[name] = {
            "category": "task_planning",
            "user_message": user_message,
            "expected_tool_sequence": expected_seq,
            "forbidden_tools": forbidden or [],
            "expected_min_calls": min_calls,
            "scoring_dimensions": ["tool_sequence", "efficiency"],
        }

    def add_quality(name: str, user_message: str):
        refs[name] = {
            "category": "response_quality",
            "user_message": user_message,
            "expected_checks": [
                "工具调用前后应有简短自然语言说明",
                "回复中不应出现 markdown 格式",
                "回复中不应直接泄露工具名",
                "回复中不应直接暴露坐标",
                "最后应有简洁结论",
            ],
            "scoring_dimensions": ["response_quality"],
        }

    def add_param(name: str, user_message: str, expected_checks: list[str]):
        refs[name] = {
            "category": "param_accuracy",
            "user_message": user_message,
            "expected_checks": expected_checks,
            "scoring_dimensions": ["param_accuracy"],
        }

    planning_cases = [
        ("fetch_yellow_cup", "把茶几上的黄色杯子递给我", _FETCH_EXPECTED, ["place_down"], 6),
        ("fetch_remote", "帮我把茶几上的遥控器拿过来", _FETCH_EXPECTED, ["place_down"], 6),
        ("fetch_phone", "把桌上的手机拿给我", _FETCH_EXPECTED, ["place_down"], 6),
        ("fetch_tissue_box", "把纸巾盒递给我", _FETCH_EXPECTED, ["place_down"], 6),
        ("fetch_notebook", "把桌上的那本黑色小本子拿起来", ["spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start"], [], 4),
        ("fetch_sandwich", "把茶几上的三明治端给我", _FETCH_EXPECTED, ["place_down"], 6),
        ("place_shoes_to_cabinet", "把地上的白色鞋子放到鞋柜旁边", _PLACE_EXPECTED, ["handover"], 7),
        ("move_tissue_right", "把纸巾盒放到桌子右边一点", ["spatial_memory_query_vec", "move_to", "push_item"], [], 3),
        ("move_remote_to_center", "把遥控器放到桌子中间", ["spatial_memory_query_vec", "move_to"], [], 2),
        ("place_phone_near_tissue", "把手机放到纸巾盒旁边", _PLACE_EXPECTED, ["handover"], 7),
        ("collect_carpet_toys", "帮我把地毯上的彩色套圈玩具收起来", ["spatial_memory_query_vec", "move_to", "grasp_start"], [], 3),
        ("collect_rings_to_basket", "把桌上的彩色圆环收到篮子里", ["spatial_memory_query_vec", "move_to", "grasp_start", "move_to", "place_down"], ["handover"], 5),
        ("collect_all_floor_toys", "把地上的玩具都收进篮子里", ["spatial_memory_query_vec", "move_to", "grasp_start"], [], 3),
        ("fold_blanket", "把沙发上的绿色毯子叠一下", ["spatial_memory_query_vec", "move_to", "fold_item"], ["grasp_start"], 3),
        ("tidy_pillows", "帮我整理一下沙发上的靠枕", ["spatial_memory_query_vec", "move_to", "tidy_surface"], ["grasp_start"], 3),
        ("tidy_coffee_table", "把茶几上的东西整理整齐", ["spatial_memory_query_vec", "move_to", "tidy_surface"], ["grasp_start"], 3),
        ("list_table_items", "帮我检查一下桌上有哪些东西", ["list_objects"], ["grasp_start", "place_down", "handover"], 1),
        ("find_phone", "告诉我手机现在在哪", ["spatial_memory_query_vec"], ["grasp_start", "place_down", "handover"], 1),
        ("check_remote_on_table", "看看遥控器是不是在茶几上", ["spatial_memory_query_vec"], ["grasp_start", "place_down", "handover"], 1),
        ("find_tissue_box", "帮我找一下纸巾盒", ["spatial_memory_query_vec"], ["grasp_start", "place_down", "handover"], 1),
        ("clear_table", "把桌面清理出来", ["list_objects"], [], 1),
        ("classify_edible", "把能吃的和不能吃的东西分开放", ["list_objects"], [], 1),
        ("safety_move_cup", "把杯子放远一点，别碰倒了", ["spatial_memory_query_vec", "move_to", "push_item"], [], 3),
        ("edge_items_inward", "帮我把容易掉到地上的东西往里挪", ["list_objects", "push_item"], [], 2),
        ("clear_sofa_area", "把沙发前面这块区域整理干净，方便我走路", ["spatial_memory_query_vec", "move_to", "grasp_start"], [], 3),
        ("grasp_fail_retry", "把茶几上的黄色杯子递给我", ["spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start", "grasp_start", "handover"], ["adjust_body_height"], 7),
        ("move_fail_retry", "帮我把茶几上的遥控器拿过来", ["spatial_memory_query_vec", "move_to", "trigger_spatial_processing", "spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start", "handover"], [], 8),
        ("memory_miss_search", "把桌上的手机拿给我", ["spatial_memory_query_vec", "move_to", "trigger_spatial_processing", "spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start", "handover"], [], 8),
        ("fold_fail_retry", "把沙发上的绿色毯子叠一下", ["spatial_memory_query_vec", "move_to", "fold_item", "fold_item"], ["grasp_start"], 4),
    ]
    for name, user_message, expected_seq, forbidden, min_calls in planning_cases:
        add_planning(name, user_message, expected_seq, forbidden=forbidden, min_calls=min_calls)

    quality_cases = [
        ("rq_fetch_cup", "把茶几上的黄色杯子递给我"),
        ("rq_fetch_remote", "帮我把茶几上的遥控器拿过来"),
        ("rq_fetch_sandwich", "把茶几上的三明治端给我"),
        ("rq_fold_blanket", "把沙发上的绿色毯子叠一下"),
        ("rq_tidy_pillows", "帮我整理一下沙发上的靠枕"),
        ("rq_find_phone", "告诉我手机现在在哪"),
        ("rq_list_items", "帮我检查一下桌上有哪些东西"),
        ("rq_check_remote", "看看遥控器是不是在茶几上"),
        ("rq_move_tissue", "把纸巾盒放到桌子右边一点"),
        ("rq_collect_toys", "帮我把地毯上的彩色套圈玩具收起来"),
    ]
    for name, user_message in quality_cases:
        add_quality(name, user_message)

    param_cases = [
        ("fc_fetch_cup_coords", "把茶几上的黄色杯子递给我", ["首个 move_to 调用的坐标参数应与黄色杯子的 mock 坐标一致"]),
        ("fc_fetch_cup_grasp_lang", "把茶几上的黄色杯子递给我", ["grasp_start(item=...) 参数应为英文字符串"]),
        ("fc_fetch_remote_coords", "帮我把茶几上的遥控器拿过来", ["首个 move_to 调用的坐标参数应与遥控器的 mock 坐标一致"]),
        ("fc_fetch_delete_id", "把茶几上的黄色杯子递给我", ["若调用 delete_spatial_memory_by_id，则 id_list 应包含黄色杯子的 _id"]),
        ("fc_shoes_target_coords", "把地上的白色鞋子放到鞋柜旁边", ["末次 move_to 的目标坐标应接近鞋柜位置"]),
        ("fc_phone_near_tissue", "把手机放到纸巾盒旁边", ["应完成抓取并执行 place_down"]),
        ("fc_tissue_push_direction", "把纸巾盒放到桌子右边一点", ["push_item(direction=...) 应体现向右，或至少采用合理替代动作"]),
        ("fc_rings_basket", "把桌上的彩色圆环收到篮子里", ["应完成抓取并执行 place_down"]),
        ("fc_fold_blanket_lang", "把沙发上的绿色毯子叠一下", ["fold_item(item=...) 参数应为英文字符串"]),
        ("fc_fold_blanket_coords", "把沙发上的绿色毯子叠一下", ["首个 move_to 调用的坐标参数应与绿色毯子的 mock 坐标一致"]),
        ("fc_tidy_pillows_surface", "帮我整理一下沙发上的靠枕", ["tidy_surface(surface=...) 参数应非空"]),
        ("fc_spatial_query_chinese", "把茶几上的黄色杯子递给我", ["spatial_memory_query_vec(name=...) 的 name 参数应包含中文"]),
    ]
    for name, user_message, expected_checks in param_cases:
        add_param(name, user_message, expected_checks)

    return refs


_REFERENCE_MAP = _build_reference_map()


def _build_bad_reason_summary(rec: dict) -> str:
    scores = rec.get("scores") or {}
    parts = []
    for dim in ["tool_sequence", "param_accuracy", "efficiency", "response_quality"]:
        sd = scores.get(dim)
        if isinstance(sd, dict):
            score = sd.get("score")
            detail = sd.get("details", "")
            if isinstance(score, (int, float)):
                parts.append(f"{dim}={score:.2f}: {detail}")
    return " | ".join(parts)


def record_eval(test_name: str, model: str, scores: dict, **extra):
    """Record evaluation scores for later report generation."""
    _eval_records.append({"test": test_name, "model": model, "scores": scores, **extra})
    _persist_progress_reports()


def _is_xdist_worker(config) -> bool:
    return bool(_XDIST_WORKER and _XDIST_WORKER != "master") or hasattr(config, "workerinput")


def _safe_report_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or "unknown"


def _report_dir() -> Path:
    raw = os.environ.get("TEST_REPORT_DIR", "").strip()
    path = Path(raw).expanduser() if raw else (REAL_HOME_LIVING_DIR.parent.parent / "report" / REAL_HOME_LIVING_DIR.name if REAL_HOME_LIVING_DIR.parent.name == "data" else REAL_HOME_LIVING_DIR / "tests" / "report")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_report_dir(report_dir: Path, report_stem: str) -> Path:
    path = report_dir / report_stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def _report_identity() -> tuple[str, str, str]:
    model = _eval_records[0].get("model", "unknown") if _eval_records else os.environ.get("TEST_MODEL", "unknown")
    report_basename = os.environ.get("TEST_REPORT_BASENAME", "").strip() or model
    report_stem = _safe_report_stem(report_basename)
    return model, report_basename, report_stem


def _records_with_outcomes(records: list[dict]) -> list[dict]:
    synced: list[dict] = []
    for rec in records:
        item = dict(rec)
        nodeid = item.get("nodeid", "")
        item["outcome"] = _test_outcomes.get(nodeid, item.get("outcome", "unknown"))
        synced.append(item)
    return synced


def _xdist_shard_dir(report_dir: Path, report_stem: str) -> Path:
    return report_dir / ".xdist" / f"{_REPORT_SESSION_TS}_{report_stem}"


def _write_worker_shard(report_dir: Path, report_stem: str) -> None:
    shard_dir = _xdist_shard_dir(report_dir, report_stem)
    shard_dir.mkdir(parents=True, exist_ok=True)
    worker_id = _XDIST_WORKER or "master"
    with open(shard_dir / f"{worker_id}.json", "w", encoding="utf-8") as f:
        json.dump(_records_with_outcomes(_eval_records), f, ensure_ascii=False, indent=2)


def _load_worker_records(report_dir: Path, report_stem: str) -> list[dict]:
    shard_dir = _xdist_shard_dir(report_dir, report_stem)
    records: list[dict] = []
    if not shard_dir.exists():
        return records
    for path in sorted(shard_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, list):
            records.extend(item for item in payload if isinstance(item, dict))
    return records


def _cleanup_worker_shards(report_dir: Path, report_stem: str) -> None:
    shard_dir = _xdist_shard_dir(report_dir, report_stem)
    if shard_dir.exists():
        shutil.rmtree(shard_dir, ignore_errors=True)


def _collect_bad_cases(records: list[dict]) -> list[dict]:
    bad_cases = []
    for rec in records:
        if rec.get("outcome") != "failed":
            continue
        enriched = dict(rec)
        reference = _REFERENCE_MAP.get(rec.get("test"), {})
        if reference:
            enriched["reference"] = reference
        enriched["bad_reason_summary"] = _build_bad_reason_summary(rec)
        bad_cases.append(enriched)
    return bad_cases


def _persist_progress_reports() -> None:
    if not _eval_records or _XDIST_WORKER:
        return

    report_dir = _report_dir()
    _, _, report_stem = _report_identity()
    run_dir = _run_report_dir(report_dir, report_stem)
    records = _records_with_outcomes(_eval_records)

    bad_cases = _collect_bad_cases(records)
    try:
        with open(run_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        with open(run_dir / "bad_cases.json", "w", encoding="utf-8") as f:
            json.dump(bad_cases, f, ensure_ascii=False, indent=2)
    except Exception:
        pass



def _ensure_outcome_record(item, outcome: str) -> None:
    nodeid = item.nodeid
    if any(rec.get("nodeid") == nodeid for rec in _eval_records):
        return
    model = os.environ.get("TEST_MODEL", "unknown")
    _eval_records.append({
        "test": getattr(item, "name", nodeid),
        "model": model,
        "scores": {},
        "generated_at": _REPORT_SESSION_DISPLAY,
        "nodeid": nodeid,
        "outcome": outcome,
        "record_status": "missing_eval_record",
        "reason": "pytest finished before eval_record was called",
    })

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" or (report.when == "setup" and (report.failed or report.skipped)):
        _test_outcomes[item.nodeid] = report.outcome
        for rec in reversed(_eval_records):
            if rec.get("nodeid") == item.nodeid:
                rec["outcome"] = report.outcome
                break
        else:
            _ensure_outcome_record(item, report.outcome)
        _persist_progress_reports()


@pytest.fixture
def eval_record(model_name, request):
    """Fixture for tests to record evaluation scores (includes test node ID)."""
    def _record(test_name: str, scores: dict, **extra):
        record_eval(test_name, model_name, scores, nodeid=request.node.nodeid, **extra)
    return _record


def pytest_sessionfinish(session, exitstatus):
    from evaluation import format_report_markdown

    report_dir = _report_dir()
    model, report_basename, report_stem = _report_identity()

    if _is_xdist_worker(session.config):
        if _eval_records:
            _write_worker_shard(report_dir, report_stem)
        return

    records = _records_with_outcomes(_eval_records)
    if not records:
        records = _load_worker_records(report_dir, report_stem)
    if not records:
        return

    bad_cases = _collect_bad_cases(records)

    run_dir = _run_report_dir(report_dir, report_stem)
    json_path = run_dir / "report.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        with open(run_dir / "bad_cases.json", "w", encoding="utf-8") as f:
            json.dump(bad_cases, f, ensure_ascii=False, indent=2)
        with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "model": model,
                "generated_at": _REPORT_SESSION_DISPLAY,
                "session_timestamp": _REPORT_SESSION_TS,
                "exitstatus": exitstatus,
                "report_dir": str(run_dir),
                "record_count": len(records),
                "failed_count": sum(1 for rec in records if rec.get("outcome") == "failed"),
                "passed_count": sum(1 for rec in records if rec.get("outcome") == "passed"),
                "skipped_count": sum(1 for rec in records if rec.get("outcome") == "skipped"),
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    md_path = run_dir / "report.md"
    try:
        md_content = format_report_markdown(records, model)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
    except Exception:
        pass

    try:
        _cleanup_worker_shards(report_dir, report_stem)
    except Exception:
        pass
