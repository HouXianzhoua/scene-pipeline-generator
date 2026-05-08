"""Layer 2: End-to-end integration tests.

These tests start the Mock MCP Server and Agent process, send user
instructions over HTTP, and validate the output JSONL conversation logs.

The mock server (mock_server.py) exposes the same tool interfaces as
server.py but routes all calls through MockResultProvider, giving each
test full control over tool return values via runtime scenario switching.

Requirements:
  - httpx  (pip install httpx)
  - A reachable LLM model server
  - The kaiwu_agent.so runtime
"""

import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

REAL_HOME_LIVING_DIR = Path(__file__).parent.parent


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "kaiwu_agent").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root from test directory")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
KAIWU_AGENT_DIR = REPO_ROOT / "kaiwu_agent"
MOCK_SERVER_PATH = Path(__file__).parent / "mock_server.py"
CONTROL_PORT = 8001

pytestmark = pytest.mark.skipif(
    not HAS_HTTPX, reason="httpx required for e2e tests: pip install httpx"
)

_HAS_CHINESE = re.compile(r"[\u4e00-\u9fff]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _get_latest_output(after_ts: float = 0) -> list[dict] | None:
    output_root = KAIWU_AGENT_DIR / "output"
    if not output_root.exists():
        return None
    files: list[Path] = []
    for data_dir in sorted(output_root.glob("data*")):
        if data_dir.is_dir():
            files.extend(data_dir.glob("BrainAgent_*.jsonl"))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    for f in files:
        if f.stat().st_mtime < after_ts:
            continue
        lines = f.read_text().splitlines()
        if lines:
            return json.loads(lines[-1])
    return None


def _extract_tool_names(messages: list[dict]) -> list[str]:
    names: list[str] = []
    for msg in messages:
        if msg.get("role") != "ai":
            continue
        for tc in msg.get("additional_kwargs", {}).get("tool_calls", []):
            names.append(tc["function"]["name"])
    return names


def _extract_tool_calls(messages: list[dict]) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    for msg in messages:
        if msg.get("role") != "ai":
            continue
        for tc in msg.get("additional_kwargs", {}).get("tool_calls", []):
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            calls.append((name, args))
    return calls


def _get_calls_for(
    tool_calls: list[tuple[str, dict]], tool_name: str
) -> list[dict]:
    return [args for name, args in tool_calls if name == tool_name]


def _assert_subsequence(actual: list[str], expected: list[str]):
    idx = 0
    for name in actual:
        if idx < len(expected) and name == expected[idx]:
            idx += 1
    assert idx == len(expected), (
        f"Expected subsequence:\n  {expected}\n"
        f"Only matched {idx}/{len(expected)} in actual:\n  {actual}"
    )


def _set_scenario(scenario: str):
    with httpx.Client(timeout=5) as client:
        resp = client.post(
            f"http://localhost:{CONTROL_PORT}/scenario",
            json={"scenario": scenario},
        )
        resp.raise_for_status()


def _send_message(instruction: str) -> dict:
    import uuid as _uuid
    with httpx.Client(timeout=300) as client:
        resp = client.post(
            "http://localhost:8164/agent/instruction",
            json={
                "userUuid": "test_user",
                "task": {
                    "commandUuid": str(_uuid.uuid4()),
                    "command": instruction,
                    "mediaType": "text",
                },
                "agentName": "tiangong",
                "agentUuid": "agent_real_home_living",
            },
        )
        resp.raise_for_status()
        return resp.json()


def _wait_for_output(
    after_ts: float, timeout: float = 180, poll: float = 5
) -> list[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = _get_latest_output(after_ts)
        if result is not None:
            return result
        time.sleep(poll)
    return None


def _dump_agent_log(proc: subprocess.Popen) -> str:
    log_path = getattr(proc, "_log_path", None)
    if not log_path or not Path(log_path).exists():
        return "(no log file)"
    full = Path(log_path).read_text(errors="replace")
    if len(full) > 4000:
        return full[:2000] + "\n... [truncated] ...\n" + full[-2000:]
    return full


def _run_instruction(agent_process, instruction: str, timeout: float = 180) -> dict:
    ts = time.time()
    _send_message(instruction)

    messages = _wait_for_output(ts, timeout=timeout)
    assert messages is not None, (
        f"No JSONL output found for '{instruction}'.\n"
        f"Agent log: {_dump_agent_log(agent_process)}"
    )

    return {
        "tool_names": _extract_tool_names(messages),
        "tool_calls": _extract_tool_calls(messages),
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mcp_server():
    """Start the mock MCP server (with control endpoint)."""
    log = tempfile.NamedTemporaryFile(prefix="mcp_mock_", suffix=".log", delete=False)
    proc = subprocess.Popen(
        [
            "python", str(MOCK_SERVER_PATH),
            "--port", "8000",
            "--control-port", str(CONTROL_PORT),
            "--scenario", "fetch_cup",
        ],
        cwd=str(KAIWU_AGENT_DIR),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    if not _wait_for_port("localhost", 8000, timeout=15):
        proc.kill()
        log.flush()
        out = Path(log.name).read_text(errors="replace")[-4096:]
        pytest.fail(f"Mock MCP server did not start within 15 s.\nOutput:\n{out}")
    if not _wait_for_port("localhost", CONTROL_PORT, timeout=5):
        proc.kill()
        pytest.fail("Mock MCP control server did not start within 5 s.")
    yield proc
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    os.unlink(log.name)


@pytest.fixture(scope="module")
def agent_process(mcp_server):
    model_server = os.environ.get("TEST_MODEL_SERVER", "qwen3vl_235b")
    model_config = os.environ.get(
        "TEST_MODEL_CONFIG", str(KAIWU_AGENT_DIR / "model.example.yaml")
    )
    env_config = str(KAIWU_AGENT_DIR / ".example.env")

    env = os.environ.copy()
    env["NEED_AUTH"] = "false"

    log = tempfile.NamedTemporaryFile(prefix="agent_", suffix=".log", delete=False)
    proc = subprocess.Popen(
        [
            "python", "main.py",
            "-c", str(REAL_HOME_LIVING_DIR / "real_home_living.yaml"),
            "-mc", model_config,
            "-ms", model_server,
            "-ec", env_config,
        ],
        cwd=str(KAIWU_AGENT_DIR),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    proc._log_path = log.name
    if not _wait_for_port("localhost", 8164, timeout=120):
        proc.kill()
        log.flush()
        out = Path(log.name).read_text(errors="replace")
        if len(out) > 4000:
            out = out[:2000] + "\n... [truncated] ...\n" + out[-2000:]
        pytest.fail(f"Agent did not start within 120 s.\nOutput:\n{out}")
    time.sleep(3)
    yield proc
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    os.unlink(log.name)


# ===========================================================================
# Fetch & Deliver E2E Tests
# ===========================================================================

@pytest.mark.e2e
class TestFetchE2E:

    def test_e2e_fetch_yellow_cup(self, agent_process):
        _set_scenario("fetch_cup")
        r = _run_instruction(agent_process, "把茶几上的黄色杯子递给我")
        _assert_subsequence(r["tool_names"], [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "perception_custom", "handover",
        ])
        assert "place_down" not in r["tool_names"]

    def test_e2e_fetch_remote(self, agent_process):
        _set_scenario("fetch_remote")
        r = _run_instruction(agent_process, "帮我把茶几上的遥控器拿过来")
        _assert_subsequence(r["tool_names"], [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "perception_custom", "handover",
        ])

    def test_e2e_fetch_sandwich(self, agent_process):
        _set_scenario("fetch_sandwich")
        r = _run_instruction(agent_process, "把茶几上的三明治端给我")
        _assert_subsequence(r["tool_names"], [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "perception_custom", "handover",
        ])

    def test_e2e_implicit_hungry(self, agent_process):
        _set_scenario("fetch_sandwich")
        r = _run_instruction(agent_process, "我饿了")
        assert "spatial_memory_query_vec" in r["tool_names"]


# ===========================================================================
# Place & Move E2E Tests
# ===========================================================================

@pytest.mark.e2e
class TestPlaceE2E:

    def test_e2e_place_shoes_to_cabinet(self, agent_process):
        _set_scenario("place_shoes_to_cabinet")
        r = _run_instruction(agent_process, "把地上的白色鞋子放到鞋柜旁边")
        _assert_subsequence(r["tool_names"], [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "spatial_memory_query_vec", "move_to", "place_down",
        ])
        assert "handover" not in r["tool_names"]

    def test_e2e_move_tissue_right(self, agent_process):
        _set_scenario("move_tissue_right")
        r = _run_instruction(agent_process, "把纸巾盒放到桌子右边一点")
        assert "push_item" in r["tool_names"] or "grasp_start" in r["tool_names"]

    def test_e2e_place_phone_near_tissue(self, agent_process):
        _set_scenario("place_phone_near_tissue")
        r = _run_instruction(agent_process, "把手机放到纸巾盒旁边")
        assert "grasp_start" in r["tool_names"]
        assert "place_down" in r["tool_names"]

    def test_e2e_collect_rings_to_basket(self, agent_process):
        _set_scenario("collect_rings_to_basket")
        r = _run_instruction(agent_process, "把桌上的彩色圆环收到篮子里")
        assert "grasp_start" in r["tool_names"]
        assert "place_down" in r["tool_names"]

    def test_e2e_collect_toys_to_basket(self, agent_process):
        _set_scenario("collect_toys_to_basket")
        r = _run_instruction(agent_process, "把地上的玩具都收进篮子里")
        assert "grasp_start" in r["tool_names"]


# ===========================================================================
# Tidy & Fold E2E Tests
# ===========================================================================

@pytest.mark.e2e
class TestTidyE2E:

    def test_e2e_fold_blanket(self, agent_process):
        _set_scenario("fold_blanket")
        r = _run_instruction(agent_process, "把沙发上的绿色毯子叠一下")
        _assert_subsequence(r["tool_names"], [
            "spatial_memory_query_vec", "move_to", "fold_item",
        ])
        assert "grasp_start" not in r["tool_names"]

    def test_e2e_tidy_pillows(self, agent_process):
        _set_scenario("tidy_pillows")
        r = _run_instruction(agent_process, "帮我整理一下沙发上的靠枕")
        assert "tidy_surface" in r["tool_names"]

    def test_e2e_tidy_coffee_table(self, agent_process):
        _set_scenario("tidy_coffee_table")
        r = _run_instruction(agent_process, "把茶几上的东西整理整齐")
        assert "tidy_surface" in r["tool_names"]


# ===========================================================================
# Query E2E Tests
# ===========================================================================

@pytest.mark.e2e
class TestQueryE2E:

    def test_e2e_list_table_items(self, agent_process):
        _set_scenario("list_table_items")
        r = _run_instruction(agent_process, "帮我检查一下桌上有哪些东西")
        assert "list_objects" in r["tool_names"]
        assert "grasp_start" not in r["tool_names"]

    def test_e2e_find_phone(self, agent_process):
        _set_scenario("find_phone")
        r = _run_instruction(agent_process, "告诉我手机现在在哪")
        assert "spatial_memory_query_vec" in r["tool_names"]
        assert "grasp_start" not in r["tool_names"]

    def test_e2e_check_remote_on_table(self, agent_process):
        _set_scenario("check_remote_on_table")
        r = _run_instruction(agent_process, "看看遥控器是不是在茶几上")
        assert "spatial_memory_query_vec" in r["tool_names"]
        assert "grasp_start" not in r["tool_names"]


# ===========================================================================
# Complex E2E Tests
# ===========================================================================

@pytest.mark.e2e
class TestComplexE2E:

    def test_e2e_clear_table(self, agent_process):
        _set_scenario("clear_table")
        r = _run_instruction(agent_process, "把桌面清理出来", timeout=300)
        assert "list_objects" in r["tool_names"]
        assert "grasp_start" in r["tool_names"], (
            f"clear_table should grasp items, got: {r['tool_names']}"
        )
        assert "place_down" in r["tool_names"], (
            f"clear_table should place items elsewhere, got: {r['tool_names']}"
        )

    def test_e2e_classify_edible(self, agent_process):
        _set_scenario("classify_edible")
        r = _run_instruction(agent_process, "把能吃的和不能吃的东西分开放", timeout=300)
        assert "list_objects" in r["tool_names"]
        assert "grasp_start" in r["tool_names"], (
            f"classify_edible should grasp items, got: {r['tool_names']}"
        )


# ===========================================================================
# Failure Recovery E2E Tests
# ===========================================================================

@pytest.mark.e2e
class TestFailureRecoveryE2E:

    def test_e2e_grasp_fail_retry(self, agent_process):
        _set_scenario("grasp_fail_retry")
        r = _run_instruction(agent_process, "把茶几上的黄色杯子递给我", timeout=240)
        grasp_calls = _get_calls_for(r["tool_calls"], "grasp_start")
        assert len(grasp_calls) >= 2, (
            f"Expected >=2 grasp attempts, got {len(grasp_calls)}"
        )
        assert "adjust_body_height" not in r["tool_names"]

    def test_e2e_move_fail_retry(self, agent_process):
        _set_scenario("move_fail_retry")
        r = _run_instruction(agent_process, "帮我把茶几上的遥控器拿过来", timeout=240)
        move_calls = _get_calls_for(r["tool_calls"], "move_to")
        assert len(move_calls) >= 2, (
            f"Expected >=2 move_to attempts, got {len(move_calls)}"
        )

    def test_e2e_memory_miss_search(self, agent_process):
        _set_scenario("memory_miss_search")
        r = _run_instruction(agent_process, "把桌上的手机拿给我", timeout=240)
        assert "spatial_memory_query_vec" in r["tool_names"]
        assert "move" in r["tool_names"]
        assert "trigger_spatial_processing" in r["tool_names"]

    def test_e2e_fold_fail_retry(self, agent_process):
        _set_scenario("fold_fail_retry")
        r = _run_instruction(agent_process, "把沙发上的绿色毯子叠一下", timeout=240)
        fold_calls = _get_calls_for(r["tool_calls"], "fold_item")
        assert len(fold_calls) >= 2, (
            f"Expected >=2 fold_item attempts, got {len(fold_calls)}"
        )
