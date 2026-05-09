"""Layer 1: Task planning tests.

Verify the model produces the correct tool-call sequences for different user
instructions, following the task flows defined in the system prompt.

Owns the **tool_sequence** and **efficiency** scoring dimensions.
"""

import pytest

from conftest import (
    assert_tool_contains,
    assert_tool_not_contains,
    assert_tool_order,
    assert_tool_subsequence,
)
from evaluation import score_efficiency, score_tool_sequence
from mock_tool_results import (
    check_remote_on_table_mock,
    classify_edible_mock,
    clear_table_mock,
    collect_rings_to_basket_mock,
    collect_toys_to_basket_mock,
    fetch_cup_mock,
    fetch_remote_mock,
    fetch_sandwich_mock,
    find_phone_mock,
    fold_blanket_mock,
    fold_fail_retry_mock,
    grasp_fail_retry_mock,
    list_table_items_mock,
    memory_miss_search_mock,
    move_fail_retry_mock,
    move_tissue_right_mock,
    place_phone_near_tissue_mock,
    place_shoes_to_cabinet_mock,
    tidy_coffee_table_mock,
    tidy_pillows_mock,
)


def _record_planning_scores(
    result, eval_record, test_name,
    expected_seq=None, forbidden=None, min_calls=6,
):
    """Compute tool_sequence + efficiency scores and record to eval report."""
    scores = {}
    if expected_seq:
        ts = score_tool_sequence(result, expected_seq, forbidden)
        scores["tool_sequence"] = ts.to_dict()
    eff = score_efficiency(result, min_calls)
    scores["efficiency"] = eff.to_dict()
    eval_record(test_name, scores)


class TestFetchAndDeliver:
    """Verify fetch-and-handover task planning for 拿取递送 commands."""

    def test_fetch_yellow_cup(self, runner, eval_record):
        """把茶几上的黄色杯子递给我"""
        result = runner.run("把茶几上的黄色杯子递给我", fetch_cup_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "perception_custom", "handover",
        ]
        _record_planning_scores(
            result, eval_record, "fetch_yellow_cup",
            expected_seq=expected, forbidden=["place_down"], min_calls=6,
        )

        assert_tool_subsequence(result, expected)
        assert_tool_not_contains(result, "place_down")

    def test_fetch_remote(self, runner, eval_record):
        """帮我把茶几上的遥控器拿过来"""
        result = runner.run("帮我把茶几上的遥控器拿过来", fetch_remote_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "perception_custom", "handover",
        ]
        _record_planning_scores(
            result, eval_record, "fetch_remote",
            expected_seq=expected, forbidden=["place_down"], min_calls=6,
        )

        assert_tool_subsequence(result, expected)
        assert_tool_not_contains(result, "place_down")

    def test_fetch_phone(self, runner, eval_record):
        """把桌上的手机拿给我"""
        result = runner.run("把桌上的手机拿给我", fetch_cup_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "perception_custom", "handover",
        ]
        _record_planning_scores(
            result, eval_record, "fetch_phone",
            expected_seq=expected, forbidden=["place_down"], min_calls=6,
        )

        assert_tool_subsequence(result, expected)

    def test_fetch_tissue_box(self, runner, eval_record):
        """把纸巾盒递给我"""
        result = runner.run("把纸巾盒递给我", fetch_cup_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "perception_custom", "handover",
        ]
        _record_planning_scores(
            result, eval_record, "fetch_tissue_box",
            expected_seq=expected, min_calls=6,
        )

        assert_tool_subsequence(result, expected)

    def test_fetch_notebook(self, runner, eval_record):
        """把桌上的那本黑色小本子拿起来"""
        result = runner.run("把桌上的那本黑色小本子拿起来", fetch_cup_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start",
        ]
        _record_planning_scores(
            result, eval_record, "fetch_notebook",
            expected_seq=expected, min_calls=4,
        )

        assert_tool_subsequence(result, expected)

    def test_fetch_sandwich(self, runner, eval_record):
        """把茶几上的三明治端给我"""
        result = runner.run("把茶几上的三明治端给我", fetch_sandwich_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "perception_custom", "handover",
        ]
        _record_planning_scores(
            result, eval_record, "fetch_sandwich",
            expected_seq=expected, forbidden=["place_down"], min_calls=6,
        )

        assert_tool_subsequence(result, expected)


class TestPlaceAndMove:
    """Verify place/move task planning for 放置移动 commands."""

    def test_place_shoes_to_cabinet(self, runner, eval_record):
        """把地上的白色鞋子放到鞋柜旁边"""
        result = runner.run("把地上的白色鞋子放到鞋柜旁边", place_shoes_to_cabinet_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "spatial_memory_query_vec", "move_to", "place_down",
        ]
        _record_planning_scores(
            result, eval_record, "place_shoes_to_cabinet",
            expected_seq=expected, forbidden=["handover"], min_calls=7,
        )

        assert_tool_subsequence(result, expected)
        assert_tool_not_contains(result, "handover")

    def test_move_tissue_right(self, runner, eval_record):
        """把纸巾盒放到桌子右边一点"""
        result = runner.run("把纸巾盒放到桌子右边一点", move_tissue_right_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "push_item",
        ]
        _record_planning_scores(
            result, eval_record, "move_tissue_right",
            expected_seq=expected, min_calls=3,
        )

        assert_tool_contains(result, "push_item")

    def test_move_remote_to_center(self, runner, eval_record):
        """把遥控器放到桌子中间"""
        result = runner.run("把遥控器放到桌子中间", fetch_remote_mock())

        expected = [
            "spatial_memory_query_vec", "move_to",
        ]
        _record_planning_scores(
            result, eval_record, "move_remote_to_center",
            expected_seq=expected, min_calls=2,
        )

        assert_tool_subsequence(result, expected)

    def test_place_phone_near_tissue(self, runner, eval_record):
        """把手机放到纸巾盒旁边"""
        result = runner.run("把手机放到纸巾盒旁边", place_phone_near_tissue_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "spatial_memory_query_vec", "move_to", "place_down",
        ]
        _record_planning_scores(
            result, eval_record, "place_phone_near_tissue",
            expected_seq=expected, forbidden=["handover"], min_calls=7,
        )

        assert_tool_subsequence(result, expected)


class TestCollectAndOrganize:
    """Verify collect/organize task planning for 收纳归集 commands."""

    def test_collect_carpet_toys(self, runner, eval_record):
        """帮我把地毯上的彩色套圈玩具收起来"""
        result = runner.run("帮我把地毯上的彩色套圈玩具收起来", collect_toys_to_basket_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "grasp_start",
        ]
        _record_planning_scores(
            result, eval_record, "collect_carpet_toys",
            expected_seq=expected, min_calls=3,
        )

        assert_tool_subsequence(result, expected)
        assert_tool_contains(result, "grasp_start")

    def test_collect_rings_to_basket(self, runner, eval_record):
        """把桌上的彩色圆环收到篮子里"""
        result = runner.run("把桌上的彩色圆环收到篮子里", collect_rings_to_basket_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "grasp_start",
            "move_to", "place_down",
        ]
        _record_planning_scores(
            result, eval_record, "collect_rings_to_basket",
            expected_seq=expected, forbidden=["handover"], min_calls=5,
        )

        assert_tool_subsequence(result, expected)
        assert_tool_not_contains(result, "handover")

    def test_collect_all_floor_toys(self, runner, eval_record):
        """把地上的玩具都收进篮子里"""
        result = runner.run("把地上的玩具都收进篮子里", collect_toys_to_basket_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "grasp_start",
        ]
        _record_planning_scores(
            result, eval_record, "collect_all_floor_toys",
            expected_seq=expected, min_calls=3,
        )

        assert_tool_subsequence(result, expected)


class TestTidyAndFold:
    """Verify tidy/fold task planning for 整理折叠 commands."""

    def test_fold_blanket(self, runner, eval_record):
        """把沙发上的绿色毯子叠一下"""
        result = runner.run("把沙发上的绿色毯子叠一下", fold_blanket_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "fold_item",
        ]
        _record_planning_scores(
            result, eval_record, "fold_blanket",
            expected_seq=expected, forbidden=["grasp_start"], min_calls=3,
        )

        assert_tool_subsequence(result, expected)
        assert_tool_contains(result, "fold_item")
        assert_tool_not_contains(result, "grasp_start")

    def test_tidy_pillows(self, runner, eval_record):
        """帮我整理一下沙发上的靠枕"""
        result = runner.run("帮我整理一下沙发上的靠枕", tidy_pillows_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "tidy_surface",
        ]
        _record_planning_scores(
            result, eval_record, "tidy_pillows",
            expected_seq=expected, forbidden=["grasp_start"], min_calls=3,
        )

        assert_tool_subsequence(result, expected)
        assert_tool_contains(result, "tidy_surface")

    def test_tidy_coffee_table(self, runner, eval_record):
        """把茶几上的东西整理整齐"""
        result = runner.run("把茶几上的东西整理整齐", tidy_coffee_table_mock())

        expected = [
            "spatial_memory_query_vec", "move_to", "tidy_surface",
        ]
        _record_planning_scores(
            result, eval_record, "tidy_coffee_table",
            expected_seq=expected, min_calls=3,
        )

        assert_tool_subsequence(result, expected)
        assert_tool_contains(result, "tidy_surface")


class TestQuery:
    """Verify query task planning for 查询检索 commands."""

    def test_list_table_items(self, runner, eval_record):
        """帮我检查一下桌上有哪些东西"""
        result = runner.run("帮我检查一下桌上有哪些东西", list_table_items_mock())

        expected = ["list_objects"]
        _record_planning_scores(
            result, eval_record, "list_table_items",
            expected_seq=expected, forbidden=["grasp_start", "handover"], min_calls=1,
        )

        assert_tool_contains(result, "list_objects")
        assert_tool_not_contains(result, "grasp_start")
        assert_tool_not_contains(result, "handover")

    def test_find_phone(self, runner, eval_record):
        """告诉我手机现在在哪"""
        result = runner.run("告诉我手机现在在哪", find_phone_mock())

        expected = ["spatial_memory_query_vec"]
        _record_planning_scores(
            result, eval_record, "find_phone",
            expected_seq=expected, forbidden=["grasp_start", "handover"], min_calls=1,
        )

        assert_tool_contains(result, "spatial_memory_query_vec")
        assert_tool_not_contains(result, "grasp_start")

    def test_check_remote_on_table(self, runner, eval_record):
        """看看遥控器是不是在茶几上"""
        result = runner.run("看看遥控器是不是在茶几上", check_remote_on_table_mock())

        expected = ["spatial_memory_query_vec"]
        _record_planning_scores(
            result, eval_record, "check_remote_on_table",
            expected_seq=expected, forbidden=["grasp_start"], min_calls=1,
        )

        assert_tool_contains(result, "spatial_memory_query_vec")
        assert_tool_not_contains(result, "grasp_start")

    def test_find_tissue_box(self, runner, eval_record):
        """帮我找一下纸巾盒"""
        result = runner.run("帮我找一下纸巾盒", fetch_cup_mock())

        expected = ["spatial_memory_query_vec"]
        _record_planning_scores(
            result, eval_record, "find_tissue_box",
            expected_seq=expected, min_calls=1,
        )

        assert_tool_contains(result, "spatial_memory_query_vec")


class TestComplex:
    """Verify complex/composite task planning."""

    def test_clear_table(self, runner, eval_record):
        """把桌面清理出来"""
        result = runner.run("把桌面清理出来", clear_table_mock())

        expected = [
            "list_objects",
            "move_to", "grasp_start", "move_to", "place_down",
        ]
        _record_planning_scores(
            result, eval_record, "clear_table",
            expected_seq=expected, min_calls=5,
        )

        assert_tool_contains(result, "list_objects")
        assert_tool_contains(result, "grasp_start")
        assert_tool_contains(result, "place_down")

    def test_classify_edible(self, runner, eval_record):
        """把能吃的和不能吃的东西分开放"""
        result = runner.run("把能吃的和不能吃的东西分开放", classify_edible_mock())

        expected = [
            "list_objects",
            "move_to", "grasp_start", "move_to", "place_down",
        ]
        _record_planning_scores(
            result, eval_record, "classify_edible",
            expected_seq=expected, min_calls=5,
        )

        assert_tool_contains(result, "list_objects")
        assert_tool_contains(result, "grasp_start")

    def test_safety_move_cup(self, runner, eval_record):
        """把杯子放远一点，别碰倒了"""
        result = runner.run("把杯子放远一点，别碰倒了", fetch_cup_mock())

        expected = ["spatial_memory_query_vec", "move_to", "push_item"]
        _record_planning_scores(
            result, eval_record, "safety_move_cup",
            expected_seq=expected, min_calls=3,
        )

        assert_tool_contains(result, "spatial_memory_query_vec")

    def test_edge_items_inward(self, runner, eval_record):
        """帮我把容易掉到地上的东西往里挪"""
        result = runner.run("帮我把容易掉到地上的东西往里挪", clear_table_mock())

        expected = ["spatial_memory_query_vec", "move_to", "push_item"]
        _record_planning_scores(
            result, eval_record, "edge_items_inward",
            expected_seq=expected, min_calls=1,
        )

        has_push = "push_item" in result.tool_names
        has_grasp = "grasp_start" in result.tool_names
        assert has_push or has_grasp, (
            f"Expected push_item or grasp_start, got: {result.tool_names}"
        )

    def test_clear_sofa_area(self, runner, eval_record):
        """把沙发前面这块区域整理干净，方便我走路"""
        result = runner.run("把沙发前面这块区域整理干净，方便我走路", collect_toys_to_basket_mock())

        expected = ["spatial_memory_query_vec", "move_to", "grasp_start"]
        _record_planning_scores(
            result, eval_record, "clear_sofa_area",
            expected_seq=expected, min_calls=1,
        )

        has_action = any(
            t in result.tool_names
            for t in ["grasp_start", "tidy_surface", "push_item", "list_objects"]
        )
        assert has_action, f"Expected some action tool, got: {result.tool_names}"


class TestFailureRecovery:
    """Verify failure-recovery task planning."""

    def test_grasp_fail_retry(self, runner, eval_record):
        """Grasp fails out of range on first attempt, then succeeds."""
        result = runner.run("把茶几上的黄色杯子递给我", grasp_fail_retry_mock())

        grasp_calls = result.get_calls_for("grasp_start")
        expected = [
            "spatial_memory_query_vec", "move_to", "scene_recognition",
            "grasp_start", "grasp_start", "perception_custom", "handover",
        ]
        _record_planning_scores(
            result, eval_record, "grasp_fail_retry",
            expected_seq=expected, forbidden=["adjust_body_height"], min_calls=6,
        )

        assert len(grasp_calls) >= 2, (
            f"Expected at least 2 grasp_start calls (retry), got {len(grasp_calls)}"
        )

    def test_move_fail_retry(self, runner, eval_record):
        """move_to fails on first attempt, then succeeds."""
        result = runner.run("帮我把茶几上的遥控器拿过来", move_fail_retry_mock())

        move_calls = result.get_calls_for("move_to")
        expected = [
            "spatial_memory_query_vec", "move_to", "move_to",
            "scene_recognition", "grasp_start", "perception_custom", "handover",
        ]
        _record_planning_scores(
            result, eval_record, "move_fail_retry",
            expected_seq=expected, min_calls=6,
        )

        assert len(move_calls) >= 2, (
            f"Expected at least 2 move_to calls (retry), got {len(move_calls)}"
        )

    def test_memory_miss_search(self, runner, eval_record):
        """Memory query returns empty, triggers search flow."""
        result = runner.run("把桌上的手机拿给我", memory_miss_search_mock())

        expected = [
            "spatial_memory_query_vec", "move",
            "trigger_spatial_processing", "spatial_memory_query_vec",
        ]
        _record_planning_scores(
            result, eval_record, "memory_miss_search",
            expected_seq=expected, min_calls=4,
        )

        assert_tool_subsequence(result, expected)

    def test_fold_fail_retry(self, runner, eval_record):
        """fold_item fails on first attempt, then succeeds on retry."""
        result = runner.run("把沙发上的绿色毯子叠一下", fold_fail_retry_mock())

        fold_calls = result.get_calls_for("fold_item")
        expected = [
            "spatial_memory_query_vec", "move_to", "fold_item", "fold_item",
        ]
        _record_planning_scores(
            result, eval_record, "fold_fail_retry",
            expected_seq=expected, min_calls=3,
        )

        assert len(fold_calls) >= 2, (
            f"Expected at least 2 fold_item calls (retry), got {len(fold_calls)}"
        )
