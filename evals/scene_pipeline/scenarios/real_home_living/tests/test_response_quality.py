"""Layer 1: Response text quality tests.

Verify the assistant's responses follow the quality requirements:
  - Step-by-step explanation before/after tool calls
  - No markdown formatting symbols
  - No raw tool function names
  - No raw coordinate numbers
  - Meaningful conclusion message

Owns the **response_quality** scoring dimension.
"""

import pytest

from evaluation import score_response_quality
from mock_tool_results import (
    check_remote_on_table_mock,
    collect_toys_to_basket_mock,
    fetch_cup_mock,
    fetch_remote_mock,
    fetch_sandwich_mock,
    find_phone_mock,
    fold_blanket_mock,
    list_table_items_mock,
    move_tissue_right_mock,
    tidy_pillows_mock,
)


def _record_quality_score(result, eval_record, test_name):
    rq = score_response_quality(result)
    eval_record(test_name, {"response_quality": rq.to_dict()})
    return rq


class TestResponseQuality:
    """Verify assistant response text quality across different instruction types."""

    def test_fetch_cup_quality(self, runner, eval_record):
        """Response quality for simple fetch task."""
        result = runner.run("把茶几上的黄色杯子递给我", fetch_cup_mock())
        rq = _record_quality_score(result, eval_record, "rq_fetch_cup")
        assert rq.score >= 0.5, f"Response quality too low: {rq.details}"

    def test_fetch_remote_quality(self, runner, eval_record):
        """Response quality for fetch remote task."""
        result = runner.run("帮我把茶几上的遥控器拿过来", fetch_remote_mock())
        rq = _record_quality_score(result, eval_record, "rq_fetch_remote")
        assert rq.score >= 0.5, f"Response quality too low: {rq.details}"

    def test_fetch_sandwich_quality(self, runner, eval_record):
        """Response quality for fetch sandwich task."""
        result = runner.run("把茶几上的三明治端给我", fetch_sandwich_mock())
        rq = _record_quality_score(result, eval_record, "rq_fetch_sandwich")
        assert rq.score >= 0.5, f"Response quality too low: {rq.details}"

    def test_fold_blanket_quality(self, runner, eval_record):
        """Response quality for fold blanket task."""
        result = runner.run("把沙发上的绿色毯子叠一下", fold_blanket_mock())
        rq = _record_quality_score(result, eval_record, "rq_fold_blanket")
        assert rq.score >= 0.5, f"Response quality too low: {rq.details}"

    def test_tidy_pillows_quality(self, runner, eval_record):
        """Response quality for tidy pillows task."""
        result = runner.run("帮我整理一下沙发上的靠枕", tidy_pillows_mock())
        rq = _record_quality_score(result, eval_record, "rq_tidy_pillows")
        assert rq.score >= 0.5, f"Response quality too low: {rq.details}"

    def test_find_phone_quality(self, runner, eval_record):
        """Response quality for query task — should reply in natural language."""
        result = runner.run("告诉我手机现在在哪", find_phone_mock())
        rq = _record_quality_score(result, eval_record, "rq_find_phone")
        assert rq.score >= 0.5, f"Response quality too low: {rq.details}"

    def test_list_items_quality(self, runner, eval_record):
        """Response quality for list table items query."""
        result = runner.run("帮我检查一下桌上有哪些东西", list_table_items_mock())
        rq = _record_quality_score(result, eval_record, "rq_list_items")
        assert rq.score >= 0.4, f"Response quality too low: {rq.details}"

    def test_check_remote_quality(self, runner, eval_record):
        """Response quality for confirmation query."""
        result = runner.run("看看遥控器是不是在茶几上", check_remote_on_table_mock())
        rq = _record_quality_score(result, eval_record, "rq_check_remote")
        assert rq.score >= 0.5, f"Response quality too low: {rq.details}"

    def test_move_tissue_quality(self, runner, eval_record):
        """Response quality for push/move task."""
        result = runner.run("把纸巾盒放到桌子右边一点", move_tissue_right_mock())
        rq = _record_quality_score(result, eval_record, "rq_move_tissue")
        assert rq.score >= 0.5, f"Response quality too low: {rq.details}"

    def test_collect_toys_quality(self, runner, eval_record):
        """Response quality for batch collect task."""
        result = runner.run("帮我把地毯上的彩色套圈玩具收起来", collect_toys_to_basket_mock())
        rq = _record_quality_score(result, eval_record, "rq_collect_toys")
        assert rq.score >= 0.5, f"Response quality too low: {rq.details}"
