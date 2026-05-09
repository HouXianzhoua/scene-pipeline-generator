"""Layer 1: Stability tests — run key scenarios N times to measure consistency.

Each scenario is executed STABILITY_REPEATS times (env var, default 3).
Collects per-run scores and computes statistics: mean, std, min, max, pass_rate.

The aggregated results are recorded to the evaluation report for cross-model
comparison of reliability.
"""

import os
import statistics

import pytest

from conftest import ConversationResult, ConversationRunner
from evaluation import (
    composite_score,
    score_efficiency,
    score_response_quality,
    score_tool_sequence,
)
from mock_tool_results import (
    fetch_cup_mock,
    fold_blanket_mock,
    grasp_fail_retry_mock,
    list_table_items_mock,
    place_shoes_to_cabinet_mock,
)

REPEATS = int(os.environ.get("STABILITY_REPEATS", "3"))


def _aggregate_stats(values: list[float]) -> dict:
    """Compute statistics for a list of scores."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "count": len(values),
    }


def _run_repeated(
    runner: ConversationRunner,
    instruction: str,
    mock_factory,
    expected_seq: list[str],
    forbidden: list[str] | None = None,
    min_calls: int = 6,
    max_turns: int | None = None,
) -> dict:
    """Run a scenario REPEATS times and collect per-run scores."""
    seq_scores: list[float] = []
    quality_scores: list[float] = []
    efficiency_scores: list[float] = []
    composite_scores: list[float] = []
    pass_count = 0
    run_details: list[dict] = []

    for i in range(REPEATS):
        mock = mock_factory()
        kwargs = {}
        if max_turns:
            kwargs["max_turns"] = max_turns

        result = runner.run(instruction, mock, **kwargs)

        ts = score_tool_sequence(result, expected_seq, forbidden)
        rq = score_response_quality(result)
        eff = score_efficiency(result, min_calls)
        comp = composite_score([ts, rq, eff])

        seq_scores.append(ts.score)
        quality_scores.append(rq.score)
        efficiency_scores.append(eff.score)
        composite_scores.append(comp)

        passed = _check_pass(result, expected_seq)
        if passed:
            pass_count += 1

        run_details.append({
            "run": i + 1,
            "tool_sequence": ts.score,
            "response_quality": rq.score,
            "efficiency": eff.score,
            "composite": comp,
            "passed": passed,
            "tool_names": result.tool_names,
            "total_tokens": result.usage.get("total_tokens", 0),
        })

    return {
        "repeats": REPEATS,
        "pass_rate": round(pass_count / REPEATS, 4),
        "tool_sequence": _aggregate_stats(seq_scores),
        "response_quality": _aggregate_stats(quality_scores),
        "efficiency": _aggregate_stats(efficiency_scores),
        "composite": _aggregate_stats(composite_scores),
        "runs": run_details,
    }


def _check_pass(result: ConversationResult, expected_seq: list[str]) -> bool:
    """Check if the result matches the expected subsequence (non-asserting)."""
    actual = result.tool_names
    idx = 0
    for name in actual:
        if idx < len(expected_seq) and name == expected_seq[idx]:
            idx += 1
    return idx == len(expected_seq)


class TestStability:
    """Run key scenarios multiple times to measure model consistency."""

    def test_stability_fetch_cup(self, runner, eval_record):
        """Stability test: fetch yellow cup and hand to user."""
        expected = [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "perception_custom", "handover",
        ]
        stats = _run_repeated(
            runner, "把茶几上的黄色杯子递给我", fetch_cup_mock,
            expected_seq=expected, forbidden=["place_down"], min_calls=6,
        )

        eval_record("stability_fetch_cup", {
            "stability": stats,
            "tool_sequence": {"score": stats["tool_sequence"]["mean"], "details": f"pass_rate={stats['pass_rate']:.0%}"},
        })

        assert stats["pass_rate"] >= 0.5, (
            f"稳定性不足: {REPEATS} 次运行中仅 {stats['pass_rate']:.0%} 通过\n"
            f"各轮详情: {stats['runs']}"
        )

    def test_stability_place_shoes(self, runner, eval_record):
        """Stability test: pick up shoes and place near cabinet."""
        expected = [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "spatial_memory_query_vec", "move_to", "place_down",
        ]
        stats = _run_repeated(
            runner, "把地上的白色鞋子放到鞋柜旁边", place_shoes_to_cabinet_mock,
            expected_seq=expected, forbidden=["handover"], min_calls=7,
        )

        eval_record("stability_place_shoes", {
            "stability": stats,
            "tool_sequence": {"score": stats["tool_sequence"]["mean"], "details": f"pass_rate={stats['pass_rate']:.0%}"},
        })

        assert stats["pass_rate"] >= 0.5, (
            f"稳定性不足: {REPEATS} 次运行中仅 {stats['pass_rate']:.0%} 通过\n"
            f"各轮详情: {stats['runs']}"
        )

    def test_stability_fold_blanket(self, runner, eval_record):
        """Stability test: fold blanket on sofa."""
        expected = [
            "spatial_memory_query_vec", "move_to", "fold_item",
        ]
        stats = _run_repeated(
            runner, "把沙发上的绿色毯子叠一下", fold_blanket_mock,
            expected_seq=expected, forbidden=["grasp_start"], min_calls=3,
        )

        eval_record("stability_fold_blanket", {
            "stability": stats,
            "tool_sequence": {"score": stats["tool_sequence"]["mean"], "details": f"pass_rate={stats['pass_rate']:.0%}"},
        })

        assert stats["pass_rate"] >= 0.5, (
            f"稳定性不足: {REPEATS} 次运行中仅 {stats['pass_rate']:.0%} 通过\n"
            f"各轮详情: {stats['runs']}"
        )

    def test_stability_list_items(self, runner, eval_record):
        """Stability test: list items on coffee table."""
        expected = ["list_objects"]
        stats = _run_repeated(
            runner, "帮我检查一下桌上有哪些东西", list_table_items_mock,
            expected_seq=expected, forbidden=["grasp_start"], min_calls=1,
        )

        eval_record("stability_list_items", {
            "stability": stats,
            "tool_sequence": {"score": stats["tool_sequence"]["mean"], "details": f"pass_rate={stats['pass_rate']:.0%}"},
        })

        assert stats["pass_rate"] >= 0.5, (
            f"稳定性不足: {REPEATS} 次运行中仅 {stats['pass_rate']:.0%} 通过\n"
            f"各轮详情: {stats['runs']}"
        )

    def test_stability_grasp_fail_retry(self, runner, eval_record):
        """Stability test: grasp failure and retry."""
        expected = [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "grasp_start", "handover",
        ]
        stats = _run_repeated(
            runner, "把茶几上的黄色杯子递给我", grasp_fail_retry_mock,
            expected_seq=expected, forbidden=["adjust_body_height"],
            min_calls=7, max_turns=25,
        )

        eval_record("stability_grasp_fail_retry", {
            "stability": stats,
            "tool_sequence": {"score": stats["tool_sequence"]["mean"], "details": f"pass_rate={stats['pass_rate']:.0%}"},
        })

        assert stats["pass_rate"] >= 0.3, (
            f"稳定性不足: {REPEATS} 次运行中仅 {stats['pass_rate']:.0%} 通过\n"
            f"各轮详情: {stats['runs']}"
        )
