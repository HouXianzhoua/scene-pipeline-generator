"""Quantitative evaluation scoring for real home living mock tests.

Provides scoring functions across four dimensions:
  1. Tool sequence correctness (ordered match plus penalties for extra/forbidden tools)
  2. Response text quality (step explanation, no forbidden patterns)
  3. Parameter value accuracy (coordinates, IDs match mock data)
  4. Efficiency (redundant calls, token usage)

Each scorer returns a ScoreDetail with a 0.0–1.0 score, human-readable
details, and optional sub-scores for fine-grained analysis.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from conftest import ConversationResult

TOOL_FUNCTION_NAMES = {
    "spatial_memory_query_vec", "move_to", "move", "grasp_start",
    "scene_recognition", "delete_spatial_memory_by_id",
    "trigger_spatial_processing", "perception_custom", "handover",
    "place_down", "place_start", "place", "reach_out", "release_hand",
    "adjust_body_height", "get_robot_pose", "back_station",
    "parameter_store", "fetch_box", "put_box",
    "fold_item", "tidy_surface", "list_objects", "push_item",
    "fold", "unfold", "tuck", "sort", "align", "clean", "wipe", "push",
    "insert", "hang", "stretch",
}

OPTIONAL_SUPPORT_TOOLS = {
    "spatial_memory_query_vec",
    "scene_recognition",
    "delete_spatial_memory_by_id",
    "trigger_spatial_processing",
    "perception_custom",
}
FREE_SUPPORT_CALL_ALLOWANCE = 4

TOOL_EQUIVALENCE_GROUPS = [
    {"place_down", "place_start", "place", "insert", "hang"},
    {"push_item", "push"},
    {"fold_item", "fold"},
    {"tidy_surface", "align", "sort", "clean", "wipe", "tuck"},
    {"move_to", "move"},
]

TOOL_EQUIVALENTS: dict[str, set[str]] = {}
for _group in TOOL_EQUIVALENCE_GROUPS:
    for _tool in _group:
        TOOL_EQUIVALENTS[_tool] = set(_group)


def _tools_match(actual: str, expected: str) -> bool:
    return actual == expected or actual in TOOL_EQUIVALENTS.get(expected, {expected})


def _contains_tool(actual: list[str], expected: str) -> bool:
    return any(_tools_match(name, expected) for name in actual)

_MARKDOWN_PATTERNS = [
    re.compile(r"^\s*[-*]\s", re.MULTILINE),
    re.compile(r"^\s*#{1,6}\s", re.MULTILINE),
    re.compile(r"^\s*\d+\.\s", re.MULTILINE),
    re.compile(r"```"),
]

_COORDINATE_PATTERN = re.compile(r"-?\d+\.\d{2,}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoreDetail:
    """Score for one evaluation dimension."""
    dimension: str
    score: float
    details: str
    sub_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "score": round(self.score, 4),
            "details": self.details,
            "sub_scores": {k: round(v, 4) for k, v in self.sub_scores.items()},
        }


# ---------------------------------------------------------------------------
# 1. Tool sequence scoring
# ---------------------------------------------------------------------------

def score_tool_sequence(
    result: ConversationResult,
    expected_subsequence: list[str],
    forbidden_tools: list[str] | None = None,
) -> ScoreDetail:
    """Score how well the actual sequence matches the expected ordered sequence.

    The previous scorer only rewarded subsequence matching. That was too permissive:
    a model could insert many unrelated tool calls and still receive 1.0. This scorer
    keeps ordered partial credit but penalizes unaligned extra calls and forbidden
    tools so the score better reflects plan fidelity.
    """
    actual = result.tool_names

    idx = 0
    aligned_actual_indexes: set[int] = set()
    for actual_idx, name in enumerate(actual):
        if idx < len(expected_subsequence) and _tools_match(name, expected_subsequence[idx]):
            aligned_actual_indexes.add(actual_idx)
            idx += 1
    matched = idx
    total = len(expected_subsequence)
    seq_score = matched / total if total > 0 else 1.0

    forbidden_penalty = 0.0
    forbidden_found = []
    if forbidden_tools:
        for ft in forbidden_tools:
            if _contains_tool(actual, ft):
                forbidden_found.append(ft)
                forbidden_penalty += 0.25

    neutral_extra_calls = [
        name for i, name in enumerate(actual)
        if (
            i not in aligned_actual_indexes
            and name in TOOL_FUNCTION_NAMES
            and name in OPTIONAL_SUPPORT_TOOLS
        )
    ]
    extra_calls = [
        name for i, name in enumerate(actual)
        if (
            i not in aligned_actual_indexes
            and name in TOOL_FUNCTION_NAMES
            and name not in OPTIONAL_SUPPORT_TOOLS
        )
    ]

    extra_penalty = min(0.4, len(extra_calls) * 0.1)
    final_score = max(0.0, min(1.0, seq_score - forbidden_penalty - extra_penalty))

    parts = [f"匹配 {matched}/{total} 个预期工具"]
    if extra_calls:
        parts.append(f"额外工具调用: {extra_calls}")
    if neutral_extra_calls:
        parts.append(f"中性辅助调用: {neutral_extra_calls}")
    if forbidden_found:
        parts.append(f"出现禁止工具: {forbidden_found}")
    details = "；".join(parts)

    return ScoreDetail(
        dimension="tool_sequence",
        score=final_score,
        details=details,
        sub_scores={
            "subsequence_match": round(seq_score, 4),
            "forbidden_penalty": round(forbidden_penalty, 4),
            "extra_penalty": round(extra_penalty, 4),
            "matched": matched,
            "expected": total,
            "actual_total": len(actual),
            "extra_total": len(extra_calls),
            "neutral_extra_total": len(neutral_extra_calls),
        },
    )


# ---------------------------------------------------------------------------
# 2. Response quality scoring
# ---------------------------------------------------------------------------

def score_response_quality(result: ConversationResult) -> ScoreDetail:
    """Score the quality of assistant text responses."""
    assistant_msgs = result.assistant_messages
    if not assistant_msgs:
        return ScoreDetail("response_quality", 0.0, "没有助手回复消息")

    checks: dict[str, float] = {}

    tc_msgs = [m for m in assistant_msgs if m.get("tool_calls")]
    if tc_msgs:
        explained = sum(1 for m in tc_msgs if m.get("content", "").strip())
        checks["step_explanation"] = explained / len(tc_msgs)
    else:
        checks["step_explanation"] = 1.0

    all_text = "\n".join(m.get("content", "") for m in assistant_msgs)
    md_hits = sum(1 for p in _MARKDOWN_PATTERNS if p.search(all_text))
    checks["no_markdown"] = max(0.0, 1.0 - md_hits / len(_MARKDOWN_PATTERNS))

    tool_hits = sum(1 for name in TOOL_FUNCTION_NAMES if name in all_text)
    checks["no_tool_names"] = max(0.0, 1.0 - tool_hits * 0.15)

    coord_matches = _COORDINATE_PATTERN.findall(all_text)
    checks["no_coordinates"] = max(0.0, 1.0 - len(coord_matches) * 0.05)

    last = assistant_msgs[-1]
    last_text = last.get("content", "").strip()
    checks["has_conclusion"] = 1.0 if len(last_text) >= 4 else 0.0

    avg = sum(checks.values()) / len(checks)
    detail_parts = [f"{k}={v:.2f}" for k, v in checks.items()]

    return ScoreDetail(
        dimension="response_quality",
        score=round(avg, 4),
        details="；".join(detail_parts),
        sub_scores=checks,
    )


# ---------------------------------------------------------------------------
# 3. Parameter value accuracy scoring
# ---------------------------------------------------------------------------

def score_param_accuracy(
    result: ConversationResult,
    tool_name: str,
    expected_params: dict[str, Any],
    call_index: int = 0,
    tolerance: float = 0.01,
) -> ScoreDetail:
    """Score whether a tool call's parameter VALUES match expected data."""
    calls = result.get_calls_for(tool_name)
    if call_index >= len(calls):
        return ScoreDetail(
            "param_accuracy", 0.0,
            f"{tool_name} 调用索引 {call_index} 不存在（共 {len(calls)} 次调用）",
        )

    actual = calls[call_index]
    matched = 0
    total = len(expected_params)
    mismatches: list[str] = []

    for key, expected_val in expected_params.items():
        if key not in actual:
            mismatches.append(f"{key}: 缺失")
            continue
        actual_val = actual[key]
        if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
            if abs(expected_val - actual_val) <= tolerance:
                matched += 1
            else:
                mismatches.append(f"{key}: 期望={expected_val}, 实际={actual_val}")
        elif expected_val == actual_val:
            matched += 1
        else:
            mismatches.append(f"{key}: 期望={expected_val}, 实际={actual_val}")

    score = matched / total if total > 0 else 1.0
    parts = [f"匹配 {matched}/{total} 个参数"]
    if mismatches:
        parts.append(f"不匹配: {mismatches}")

    return ScoreDetail(
        "param_accuracy", round(score, 4), "；".join(parts),
        sub_scores={"matched": matched, "total": total},
    )


def score_param_accuracy_multi(
    result: ConversationResult,
    checks: list[tuple[str, dict[str, Any], int]],
    tolerance: float = 0.01,
) -> ScoreDetail:
    """Score parameter accuracy across multiple tool calls."""
    if not checks:
        return ScoreDetail("param_accuracy", 1.0, "无参数检查项")

    total_matched = 0
    total_params = 0
    all_mismatches: list[str] = []

    for tool_name, expected, idx in checks:
        sd = score_param_accuracy(result, tool_name, expected, idx, tolerance)
        total_matched += sd.sub_scores.get("matched", 0)
        total_params += sd.sub_scores.get("total", 0)
        if sd.score < 1.0:
            all_mismatches.append(f"{tool_name}[{idx}]: {sd.details}")

    score = total_matched / total_params if total_params > 0 else 1.0
    parts = [f"总计匹配 {total_matched}/{total_params} 个参数"]
    if all_mismatches:
        parts.append(f"问题: {'; '.join(all_mismatches)}")

    return ScoreDetail(
        "param_accuracy", round(score, 4), "；".join(parts),
        sub_scores={"matched": total_matched, "total": total_params},
    )


def score_efficiency(
    result: ConversationResult,
    expected_min_calls: int,
) -> ScoreDetail:
    """Score efficiency based on tool call count vs expected minimum."""
    actual = result.total_tool_calls_count
    neutral_calls = sum(1 for name in result.tool_names if name in OPTIONAL_SUPPORT_TOOLS)
    free_support_calls = min(neutral_calls, FREE_SUPPORT_CALL_ALLOWANCE)
    billable_calls = max(0, actual - free_support_calls)
    if billable_calls <= expected_min_calls:
        call_score = 1.0
    else:
        excess = billable_calls - expected_min_calls
        call_score = max(0.0, 1.0 - excess * 0.05)

    sub = {
        "actual_calls": actual,
        "billable_calls": billable_calls,
        "neutral_support_calls": neutral_calls,
        "free_support_calls": free_support_calls,
        "expected_min_calls": expected_min_calls,
        "expected_max_calls": expected_min_calls,
        "turns": result.turn_count,
    }
    parts = [
        f"实际 {actual} 次调用",
        f"计分调用 {billable_calls} 次",
        f"建议最多 {expected_min_calls} 次",
    ]

    if result.usage.get("total_tokens"):
        sub["prompt_tokens"] = result.usage["prompt_tokens"]
        sub["completion_tokens"] = result.usage["completion_tokens"]
        sub["total_tokens"] = result.usage["total_tokens"]
        parts.append(f"token: {result.usage['total_tokens']}")

    if billable_calls > expected_min_calls:
        parts.append(f"超出建议 {billable_calls - expected_min_calls} 次计分调用")
    if neutral_calls:
        parts.append(f"中性辅助调用 {neutral_calls} 次，免罚 {free_support_calls} 次")

    return ScoreDetail(
        "efficiency", round(call_score, 4), "；".join(parts), sub_scores=sub,
    )


_DEFAULT_WEIGHTS = {
    "tool_sequence": 0.35,
    "response_quality": 0.25,
    "param_accuracy": 0.25,
    "efficiency": 0.15,
}


def composite_score(scores: list[ScoreDetail], weights: dict[str, float] | None = None) -> float:
    """Compute weighted composite score from a list of ScoreDetails."""
    w = weights or _DEFAULT_WEIGHTS
    total = 0.0
    weight_sum = 0.0
    for sd in scores:
        if sd.dimension in w:
            total += sd.score * w[sd.dimension]
            weight_sum += w[sd.dimension]
    return round(total / weight_sum, 4) if weight_sum > 0 else 0.0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

_OUTCOME_LABELS = {
    "passed": "PASS",
    "failed": "FAIL",
    "skipped": "SKIP",
}

_DIMS = ["tool_sequence", "response_quality", "param_accuracy", "efficiency"]


def _compute_composite_from_record(scores: dict) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for dim in _DIMS:
        val = scores.get(dim, {}).get("score")
        if isinstance(val, (int, float)):
            total += val * _DEFAULT_WEIGHTS[dim]
            weight_sum += _DEFAULT_WEIGHTS[dim]
    if weight_sum == 0:
        return None
    return round(total / weight_sum, 4)


def format_report_markdown(records: list[dict], model: str) -> str:
    """Generate a Markdown evaluation report from collected records."""
    total = len(records)
    passed = sum(1 for r in records if r.get("outcome") == "passed")
    failed = sum(1 for r in records if r.get("outcome") == "failed")
    skipped = sum(1 for r in records if r.get("outcome") == "skipped")
    pass_rate = f"{passed / total:.1%}" if total else "N/A"

    all_composites: list[float] = []
    for rec in records:
        comp = _compute_composite_from_record(rec.get("scores", {}))
        if comp is not None:
            all_composites.append(comp)
    overall_composite = (
        f"{sum(all_composites) / len(all_composites):.2%}"
        if all_composites else "N/A"
    )
    lines = [
        f"# 真实居家场景 Mock 测试评估报告",
        f"",
        f"**模型**: `{model}`",
        f"",
        f"## 测试结果汇总",
        f"",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 总用例数 | {total} |",
        f"| 通过 | {passed} |",
        f"| 失败 | {failed} |",
        f"| 跳过 | {skipped} |",
        f"| **通过率** | **{pass_rate}** |",
        f"| **综合得分** | **{overall_composite}** |",
        f"",
        f"> 综合得分 = 各用例加权得分的平均值（权重: 工具序列 35%, 回复质量 25%, 参数准确 25%, 效率 15%）",
        f"",
        f"## 分维度得分",
        f"",
        f"| 测试用例 | 结果 | 综合分 | 工具序列 | 回复质量 | 参数准确 | 效率 |",
        f"|---------|------|-------|---------|---------|---------|------|",
    ]

    totals: dict[str, list[float]] = {
        "tool_sequence": [], "response_quality": [], "param_accuracy": [], "efficiency": [],
    }

    for rec in records:
        scores = rec.get("scores", {})
        outcome = rec.get("outcome", "unknown")
        label = _OUTCOME_LABELS.get(outcome, outcome.upper())
        if outcome == "failed":
            label = f"**{label}**"
        comp = _compute_composite_from_record(scores)
        comp_str = f"{comp:.2%}" if comp is not None else "-"
        row = [rec["test"], label, comp_str]
        for dim in _DIMS:
            val = scores.get(dim, {}).get("score", "-")
            if isinstance(val, (int, float)):
                totals[dim].append(val)
                row.append(f"{val:.2%}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    avg_row = ["**平均**", "", f"**{overall_composite}**"]
    for dim in _DIMS:
        vals = totals[dim]
        if vals:
            avg_row.append(f"**{sum(vals)/len(vals):.2%}**")
        else:
            avg_row.append("-")
    lines.append("| " + " | ".join(avg_row) + " |")

    lines.append("")
    lines.append("## 详细维度")
    lines.append("")
    for rec in records:
        outcome = rec.get("outcome", "unknown")
        label = _OUTCOME_LABELS.get(outcome, outcome.upper())
        lines.append(f"### {rec['test']}  [{label}]")
        for dim, sd in rec.get("scores", {}).items():
            if isinstance(sd, dict):
                lines.append(f"- **{dim}** ({sd.get('score', '-')}): {sd.get('details', '')}")
        lines.append("")

    return "\n".join(lines)
