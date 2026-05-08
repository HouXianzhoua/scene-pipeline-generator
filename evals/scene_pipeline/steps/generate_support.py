"""Step: Generate support files — conftest.py, evaluation.py, cmd.sh, README."""

import logging
import shutil
from pathlib import Path

from ..paths import EVAL_SCENARIOS_DIR

logger = logging.getLogger(__name__)

REAL_HOME_LIVING_TESTS = EVAL_SCENARIOS_DIR / "real_home_living" / "tests"


def generate_support_files(
    scene_data: dict,
    all_tools: list[str],
    commands_data: dict,
    scene_dir: Path,
    tests_dir: Path,
) -> None:
    """Generate executable pytest support files.

    Documentation is intentionally not generated here. Scene-eval docs live at
    the repo-level evals directory so generated test folders stay focused on
    executable assets.
    """
    scene_name = scene_data["scene_name"]

    _copy_evaluation(tests_dir)
    _generate_conftest(scene_name, tests_dir)
    _generate_cmd_sh(scene_name, tests_dir)


def _copy_evaluation(tests_dir: Path) -> None:
    src = REAL_HOME_LIVING_TESTS / "evaluation.py"
    dst = tests_dir / "evaluation.py"
    if src.exists():
        shutil.copy2(src, dst)
        logger.info("Copied evaluation.py from real_home_living")
    else:
        logger.warning("evaluation.py not found at %s, skipping copy", src)


def _generate_conftest(scene_name: str, tests_dir: Path) -> None:
    scene_var = scene_name.upper()
    content = f'''\
"""Shared fixtures and helpers for {scene_name} scenario mock tests."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end integration tests requiring a running Agent")


import inspect
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time as _time_mod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

import pytest
import yaml
from openai import OpenAI

{scene_var}_DIR = Path(__file__).parent.parent
SCENE_DIR = {scene_var}_DIR

sys.path.insert(0, str(Path(__file__).parent))


def _python_type_to_json_schema(py_type) -> dict:
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
        return {{}}

    if origin is list:
        schema: dict = {{"type": "array"}}
        if args:
            schema["items"] = _python_type_to_json_schema(args[0])
        return schema

    if origin is dict:
        return {{"type": "object"}}

    type_map = {{
        str: {{"type": "string"}},
        int: {{"type": "integer"}},
        float: {{"type": "number"}},
        bool: {{"type": "boolean"}},
        list: {{"type": "array"}},
        dict: {{"type": "object"}},
    }}
    return type_map.get(py_type, {{}})


def _extract_param_descriptions(docstring: str) -> dict[str, str]:
    """Extract parameter descriptions from Google-style docstring."""
    descriptions: dict[str, str] = {{}}
    if not docstring:
        return descriptions

    in_args = False
    current_param: str | None = None
    current_lines: list[str] = []

    for line in docstring.split("\\n"):
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


def extract_tool_schemas() -> list[dict]:
    """Import server.py and convert tool functions to OpenAI function calling schemas."""
    server_path = {scene_var}_DIR / "server.py"

    # Mock fastmcp before importing server.py to avoid dependency issues
    if "fastmcp" not in sys.modules:
        class _MockFastMCP:
            def __init__(self, name: str = "mock"):
                self.name = name
                self._tool_manager = None

            def tool(self):
                def decorator(func):
                    return func
                return decorator

        sys.modules["fastmcp"] = type(sys)("fastmcp")
        sys.modules["fastmcp"].FastMCP = _MockFastMCP

    spec = importlib.util.spec_from_file_location("_scene_server", str(server_path))
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
            hints = {{}}

        docstring = inspect.getdoc(func) or ""
        parts = re.split(r"\\n\\s*(?:Args|Returns|Raises):", docstring, maxsplit=1)
        description = parts[0].strip() if parts else name

        param_descs = _extract_param_descriptions(docstring)

        properties: dict[str, dict] = {{}}
        required: list[str] = []

        for pname, param in sig.parameters.items():
            py_type = hints.get(pname, str)
            prop = _python_type_to_json_schema(py_type)
            if pname in param_descs:
                prop["description"] = param_descs[pname]
            properties[pname] = prop
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        tool: dict = {{
            "type": "function",
            "function": {{
                "name": name,
                "description": description,
                "parameters": {{
                    "type": "object",
                    "properties": properties,
                }},
            }},
        }}
        if required:
            tool["function"]["parameters"]["required"] = required
        tools.append(tool)

    return tools


class ConversationResult:
    """Holds the complete result of a multi-turn LLM conversation."""

    def __init__(self, tool_calls, messages, usage=None):
        self.tool_calls = tool_calls
        self.messages = messages
        self.usage = usage or {{}}

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


class ConversationRunner:
    """Drives a multi-turn LLM conversation, injecting mock tool results."""

    def __init__(self, client, model, system_prompt, tools, max_turns=30, verbose=False):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_turns = max_turns
        self.verbose = verbose

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def run(self, user_message, mock_provider, max_turns=None):
        limit = max_turns or self.max_turns
        messages = [
            {{"role": "system", "content": self.system_prompt}},
            {{"role": "user", "content": user_message}},
        ]
        all_tool_calls = []
        total_usage = {{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

        self._log(f"\\n{{'='*60}}")
        self._log(f"[USER] {{user_message}}")
        self._log(f"{{'='*60}}")

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
                        # import time as _t
                        # _t.sleep(1.5)  # 避免触发 API 限流
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

            self._log(f"\\n--- Turn {{turn + 1}} ---")
            if msg.content:
                self._log(f"[ASSISTANT] {{msg.content}}")

            assistant_msg: dict[str, Any] = {{
                "role": "assistant",
                "content": msg.content or "",
            }}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {{
                        "id": tc.id,
                        "type": "function",
                        "function": {{
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }},
                    }}
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
                    args = {{}}
                all_tool_calls.append((name, args))

                self._log(f"[TOOL CALL] {{name}}({{json.dumps(args, ensure_ascii=False)}})")

                result = mock_provider.get(name, args)
                result_str = (
                    result if isinstance(result, str)
                    else json.dumps(result, ensure_ascii=False)
                )

                self._log(f"[TOOL RESULT] {{result_str[:200]}}{{'...' if len(result_str) > 200 else ''}}")

                messages.append(
                    {{"role": "tool", "tool_call_id": tc.id, "content": result_str}}
                )

        self._log(f"\\n{{'='*60}}")
        self._log(f"Conversation ended after {{len(all_tool_calls)}} tool calls")
        self._log(f"Tool sequence: {{[n for n, _ in all_tool_calls]}}")
        self._log(f"{{'='*60}}\\n")

        return ConversationResult(all_tool_calls, messages, usage=total_usage)


def assert_tool_sequence(result, expected):
    assert result.tool_names == expected, (
        f"Expected sequence:\\n  {{expected}}\\nActual:\\n  {{result.tool_names}}"
    )

def assert_tool_subsequence(result, expected):
    actual = result.tool_names
    idx = 0
    for name in actual:
        if idx < len(expected) and name == expected[idx]:
            idx += 1
    assert idx == len(expected), (
        f"Expected subsequence:\\n  {{expected}}\\n"
        f"Only matched {{idx}}/{{len(expected)}} in actual:\\n  {{actual}}"
    )

def assert_tool_contains(result, tool_name):
    assert tool_name in result.tool_names, (
        f"'{{tool_name}}' not found in tool calls: {{result.tool_names}}"
    )

def assert_tool_not_contains(result, tool_name):
    assert tool_name not in result.tool_names, (
        f"'{{tool_name}}' should not appear but found in: {{result.tool_names}}"
    )

def assert_tool_order(result, before, after):
    names = result.tool_names
    assert before in names, f"'{{before}}' not found in {{names}}"
    assert after in names, f"'{{after}}' not found in {{names}}"
    assert names.index(before) < names.index(after)


@pytest.fixture(scope="session")
def system_prompt() -> str:
    with open({scene_var}_DIR / "{scene_name}.yaml") as f:
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
def runner(openai_client, model_name, system_prompt, tool_schemas):
    verbose = os.environ.get("TEST_VERBOSE", "0") not in ("0", "false", "")
    return ConversationRunner(
        client=openai_client,
        model=model_name,
        system_prompt=system_prompt,
        tools=tool_schemas,
        verbose=verbose,
    )


_eval_records: list[dict] = []
_test_outcomes: dict[str, str] = {{}}
_REPORT_SESSION_TS = os.environ.get("TEST_REPORT_SESSION_TS", "").strip()
_REPORT_SESSION_DISPLAY = os.environ.get("TEST_REPORT_SESSION_DISPLAY", "").strip()
if not _REPORT_SESSION_TS or not _REPORT_SESSION_DISPLAY:
    _REPORT_SESSION_DT = datetime.now()
    _REPORT_SESSION_TS = _REPORT_SESSION_DT.strftime("%Y%m%d_%H%M%S")
    _REPORT_SESSION_DISPLAY = _REPORT_SESSION_DT.strftime("%Y-%m-%d %H:%M:%S")
_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "").strip()


def _safe_report_stem(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return safe or "unknown"


def _serialize_conversation_result(result):
    if result is None:
        return {{}}
    return {{
        "tool_calls": [
            {{"name": name, "arguments": args}}
            for name, args in result.tool_calls
        ],
        "messages": result.messages,
        "usage": result.usage,
        "turn_count": result.turn_count,
        "total_tool_calls_count": result.total_tool_calls_count,
    }}


def _collect_bad_cases(records: list[dict]) -> list[dict]:
    return [rec for rec in records if rec.get("outcome") == "failed"]


def _get_git_commit() -> str | None:
    for candidate in [SCENE_DIR, *SCENE_DIR.parents]:
        if not (candidate / ".git").exists():
            continue
        git_dir = candidate / ".git"
        try:
            return subprocess.check_output(
                ["git", "--git-dir", str(git_dir), "--work-tree", str(candidate), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except Exception:
            return None
    return None


def _get_source_revision() -> str | None:
    meta_path = SCENE_DIR / "batch_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            value = meta.get("source_revision") or meta.get("git_commit")
            if value:
                return str(value)
        except Exception:
            pass
    try:
        return _get_git_commit()
    except Exception:
        return None


def _build_run_meta(model: str, exitstatus: int, report_dir: Path) -> dict:
    env_keys = ["TEST_MODEL", "TEST_BASE_URL", "TEST_VERBOSE", "TEST_REPORT_BASENAME"]
    env = {{key: os.environ.get(key) for key in env_keys if os.environ.get(key) is not None}}
    return {{
        "model": model,
        "generated_at": _REPORT_SESSION_DISPLAY,
        "session_timestamp": _REPORT_SESSION_TS,
        "exitstatus": exitstatus,
        "report_dir": str(report_dir),
        "scene_dir": str({scene_var}_DIR),
        "git_commit": _get_git_commit(),
        "source_revision": _get_source_revision(),
        "env": env,
        "record_count": len(_eval_records),
        "failed_count": sum(1 for rec in _eval_records if rec.get("outcome") == "failed"),
        "passed_count": sum(1 for rec in _eval_records if rec.get("outcome") == "passed"),
        "skipped_count": sum(1 for rec in _eval_records if rec.get("outcome") == "skipped"),
    }}


def _write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_text(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _is_xdist_worker(config) -> bool:
    return bool(_XDIST_WORKER and _XDIST_WORKER != "master") or hasattr(config, "workerinput")


def _report_dir() -> Path:
    raw = os.environ.get("TEST_REPORT_DIR", "").strip()
    path = Path(raw).expanduser() if raw else (
        {scene_var}_DIR.parent.parent / "report" / {scene_var}_DIR.name
        if {scene_var}_DIR.parent.name == "data"
        else {scene_var}_DIR / "tests" / "report"
    )
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
    return report_dir / ".xdist" / f"{{_REPORT_SESSION_TS}}_{{report_stem}}"


def _write_worker_shard(report_dir: Path, report_stem: str) -> None:
    shard_dir = _xdist_shard_dir(report_dir, report_stem)
    shard_dir.mkdir(parents=True, exist_ok=True)
    worker_id = _XDIST_WORKER or "master"
    _write_json(shard_dir / f"{{worker_id}}.json", _records_with_outcomes(_eval_records))


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


def record_eval(test_name, model, scores, **extra):
    result = extra.pop("result", None)
    _eval_records.append({{
        "test": test_name,
        "model": model,
        "scores": scores,
        "generated_at": _REPORT_SESSION_DISPLAY,
        **_serialize_conversation_result(result),
        **extra,
    }})


def _ensure_outcome_record(item, outcome: str) -> None:
    nodeid = item.nodeid
    if any(rec.get("nodeid") == nodeid for rec in _eval_records):
        return
    model = os.environ.get("TEST_MODEL", "unknown")
    _eval_records.append({{
        "test": getattr(item, "name", nodeid),
        "model": model,
        "scores": {{}},
        "generated_at": _REPORT_SESSION_DISPLAY,
        "nodeid": nodeid,
        "outcome": outcome,
        "record_status": "missing_eval_record",
        "reason": "pytest finished before eval_record was called",
    }})


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

@pytest.fixture
def eval_record(model_name, request):
    def _record(test_name, scores, **extra):
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

    run_dir = _run_report_dir(report_dir, report_stem)
    bad_cases = _collect_bad_cases(records)
    meta = _build_run_meta(model, exitstatus, run_dir)
    meta["record_count"] = len(records)
    meta["failed_count"] = sum(1 for rec in records if rec.get("outcome") == "failed")
    meta["passed_count"] = sum(1 for rec in records if rec.get("outcome") == "passed")
    meta["skipped_count"] = sum(1 for rec in records if rec.get("outcome") == "skipped")

    json_path = run_dir / "report.json"
    md_path = run_dir / "report.md"

    try:
        _write_json(json_path, records)
        _write_json(run_dir / "bad_cases.json", bad_cases)
        _write_json(run_dir / "meta.json", meta)
    except Exception:
        pass

    try:
        try:
            md_content = format_report_markdown(
                records,
                model,
                generated_at=_REPORT_SESSION_DISPLAY,
            )
        except TypeError:
            md_content = format_report_markdown(records, model)
        _write_text(md_path, md_content)
    except Exception:
        pass

    try:
        _cleanup_worker_shards(report_dir, report_stem)
    except Exception:
        pass
'''
    (tests_dir / "conftest.py").write_text(content, encoding="utf-8")
    logger.info("Wrote conftest.py")


def _generate_cmd_sh(scene_name: str, tests_dir: Path) -> None:
    content = f'''\
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
SCENE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 默认使用环境变量中的 TEST_MODEL/TEST_BASE_URL/TEST_API_KEY。
python -m pytest "$SCENE_DIR/tests/test_task_planning.py" -v

# 运行全部 Layer 1 mock 测试。
python -m pytest "$SCENE_DIR/tests" -v --ignore="$SCENE_DIR/tests/test_e2e.py"

# 使用指定模型
TEST_MODEL=gpt-4.1 \\
TEST_BASE_URL=https://api.openai.com/v1 \\
TEST_API_KEY=$OPENAI_API_KEY \\
python -m pytest "$SCENE_DIR/tests" -v --ignore="$SCENE_DIR/tests/test_e2e.py"

# 仅运行查询类测试
python -m pytest "$SCENE_DIR/tests/test_task_planning.py" -v -k "TestQuery"
'''
    (tests_dir / "cmd.sh").write_text(content, encoding="utf-8")
    (tests_dir / "cmd.sh").chmod(0o755)
    logger.info("Wrote cmd.sh")


def _generate_readme(
    scene_data: dict,
    all_tools: list[str],
    commands_data: dict,
    tests_dir: Path,
) -> None:
    doc_dir = tests_dir / "doc"
    doc_dir.mkdir(parents=True, exist_ok=True)

    scene_name = scene_data["scene_name"]
    scene_display_name = scene_data["scene_display_name"]

    furniture_list = "\n".join(
        f"- {f['name']}（{f.get('position', '')}，{f.get('color', '')}）"
        for f in scene_data.get("furniture", [])
    )
    objects_list = "\n".join(
        f"- {o['name']}（{o.get('color', '')}，{o.get('position', '')}）"
        for o in scene_data.get("objects", [])
    )

    commands_count = len(commands_data.get("commands", []))

    content = f"""\
# {scene_display_name}测试体系

## 场景说明

{scene_data.get("layout_description", "")}

### 家具
{furniture_list}

### 物品
{objects_list}

## 目录结构

```
{scene_name}/
├── server.py              # MCP Server
├── {scene_name}.yaml      # Agent 配置
├── user_command.md        # 用户指令集（{commands_count} 条）
├── scene_analysis.json    # 场景分析结果
└── tests/
    ├── conftest.py        # 共享 fixture
    ├── evaluation.py      # 规划/参数/效率综合评分，回复质量单列
    ├── mock_tool_results.py  # Mock 数据
    ├── mock_server.py     # Mock MCP Server
    ├── test_task_planning.py    # 任务规划测试
    ├── test_function_calling.py # 参数校验测试
    ├── test_response_quality.py # 回复质量测试
    ├── cmd.sh             # 运行命令
    ├── doc/
    │   └── README.md      # 本文件
    └── report/
        └── eval_report_*.json/md  # 测试报告
```

## 两层测试架构

| 层级 | 做法 | 工具/Mock |
|------|------|-----------|
| Layer 1 | 直接 OpenAI API + Mock 工具返回 | ConversationRunner + MockResultProvider |
| Layer 2 | subprocess 启动 Mock MCP + Agent | FastMCP + 控制端口切换场景 |

## MCP Tools 说明

| 工具名 | 功能 |
|--------|------|
"""

    tool_descriptions = {
        "move": "移动到指定目标点",
        "move_to": "导航到目标点位",
        "grasp_start": "抓取物品",
        "place_start": "放置物品到指定位置",
        "place": "放置物品",
        "reach_out": "将物品递送到指定位置",
        "release_hand": "松开手指",
        "adjust_body_height": "调整机器人高度",
        "handover": "执行递送流程",
        "spatial_memory_query_vec": "查询空间记忆",
        "delete_spatial_memory_by_id": "删除空间记忆",
        "scene_recognition": "场景识别",
        "trigger_spatial_processing": "触发空间感知",
        "get_robot_pose": "获取机器人位置",
        "place_down": "放下物品",
        "back_station": "返回充电桩",
        "parameter_store": "存储/获取参数",
        "perception_custom": "自定义感知",
        "fetch_box": "搬起箱子",
        "put_box": "放下箱子",
        "fold_item": "折叠物品",
        "tidy_surface": "整理表面",
        "list_objects": "列出区域物品",
        "push_item": "推移物品",
    }

    for tool in all_tools:
        desc = tool_descriptions.get(tool, tool)
        content += f"| `{tool}` | {desc} |\n"

    content += f"""
## 快速开始

```bash
cd /path/to/{scene_name}

# 运行任务规划测试
python -m pytest tests/test_task_planning.py -v

# 运行全部 Layer 1 测试
python -m pytest tests -v

# 使用指定模型
TEST_MODEL=gpt-4.1 TEST_BASE_URL=https://api.openai.com/v1 TEST_API_KEY=$KEY \\
python -m pytest tests/test_task_planning.py -v
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| TEST_MODEL | Qwen3-VL-235B-A22B-Instruct | 被测模型名 |
| TEST_BASE_URL | http://120.48.75.178:4970/v1 | 模型 API 地址 |
| TEST_API_KEY | EMPTY | API 密钥 |
| TEST_VERBOSE | 0 | 是否输出详细日志 |
"""

    (doc_dir / "README.md").write_text(content, encoding="utf-8")
    logger.info("Wrote doc/README.md")
