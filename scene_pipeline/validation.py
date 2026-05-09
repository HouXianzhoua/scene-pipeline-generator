"""Cross-file consistency validation between generated artifacts."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConsistencyReport:
    """Holds results of cross-file consistency checks."""

    def __init__(self):
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.fixes_applied: list[str] = []

    @property
    def is_clean(self) -> bool:
        return not self.errors


def validate_consistency(
    scene_dir: Path,
    all_tools: list[str],
    commands_data: dict,
    mock_meta: dict,
) -> ConsistencyReport:
    """Run all cross-file consistency checks and return a report.

    Checks performed:
        1. server.py tool definitions match the all_tools list
        2. expected_tools in commands only reference known tools
        3. Mock factory memory_query variables reference existing MEMORY_ITEM constants
        4. mock_server.py has stubs for every server.py tool
        5. All mock factories have at least one test referencing them
    """
    report = ConsistencyReport()
    tests_dir = scene_dir / "tests"

    _check_server_tools(report, scene_dir / "server.py", all_tools)
    _check_command_tool_refs(report, commands_data, all_tools)
    _check_memory_item_refs(report, mock_meta)
    _check_mock_server_coverage(report, tests_dir / "mock_server.py", all_tools)
    _check_mock_defaults_coverage(report, tests_dir / "mock_tool_results.py", all_tools)

    for w in report.warnings:
        logger.warning("Consistency: %s", w)
    for e in report.errors:
        logger.error("Consistency: %s", e)

    if report.is_clean:
        logger.info("Cross-file consistency check: PASSED (%d warnings)", len(report.warnings))
    else:
        logger.warning(
            "Cross-file consistency check: %d errors, %d warnings",
            len(report.errors), len(report.warnings),
        )

    return report


def _check_server_tools(report: ConsistencyReport, server_path: Path, all_tools: list[str]) -> None:
    if not server_path.exists():
        report.errors.append("server.py not found")
        return

    server_tools = _extract_decorated_functions(server_path)
    missing = set(all_tools) - set(server_tools)
    extra = set(server_tools) - set(all_tools)

    if missing:
        report.errors.append(f"Tools declared but missing from server.py: {sorted(missing)}")
    if extra:
        report.warnings.append(f"Tools in server.py not tracked in all_tools: {sorted(extra)}")


def _check_command_tool_refs(report: ConsistencyReport, commands_data: dict, all_tools: list[str]) -> None:
    tool_set = set(all_tools)
    for cmd in commands_data.get("commands", []):
        for tool in cmd.get("expected_tools", []):
            if tool not in tool_set:
                report.warnings.append(
                    f"Command '{cmd.get('text', '')[:30]}...' expects unknown tool '{tool}'"
                )


def _check_memory_item_refs(report: ConsistencyReport, mock_meta: dict) -> None:
    defined_vars = {mi["var_name"] for mi in mock_meta.get("memory_items", [])}

    for factory in [*mock_meta.get("mock_factories", []), *mock_meta.get("failure_factories", [])]:
        for _key, var_names in factory.get("memory_queries", {}).items():
            for var in var_names:
                if var not in defined_vars:
                    report.errors.append(
                        f"Factory '{factory['name']}' references undefined memory item '{var}'"
                    )


def _check_mock_server_coverage(report: ConsistencyReport, mock_server_path: Path, all_tools: list[str]) -> None:
    if not mock_server_path.exists():
        report.errors.append("mock_server.py not found")
        return

    mock_tools = _extract_decorated_functions(mock_server_path)
    missing = set(all_tools) - set(mock_tools)
    if missing:
        report.errors.append(f"mock_server.py missing stubs for: {sorted(missing)}")


def _check_mock_defaults_coverage(report: ConsistencyReport, mock_results_path: Path, all_tools: list[str]) -> None:
    """Warn if any tool has zero references in mock_tool_results.py."""
    if not mock_results_path.exists():
        report.errors.append("mock_tool_results.py not found")
        return

    try:
        content = mock_results_path.read_text(encoding="utf-8")
    except OSError:
        return

    for tool in all_tools:
        if tool not in content:
            report.warnings.append(f"Tool '{tool}' not referenced in mock_tool_results.py")


def _extract_decorated_functions(filepath: Path) -> list[str]:
    """Extract function names decorated with @mcp.tool() from a Python file."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return []

    tools = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for deco in node.decorator_list:
            if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute):
                if deco.func.attr == "tool":
                    tools.append(node.name)
                    break
    return tools
