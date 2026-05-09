"""Layer 1: Function calling parameter validation tests.

Verify the model passes correct parameter VALUES to tool calls — coordinates,
item names, IDs, etc. — matching the data returned by mock tools.

Owns the **param_accuracy** scoring dimension.
"""

import pytest

from conftest import assert_tool_contains
from evaluation import score_param_accuracy, score_param_accuracy_multi, ScoreDetail
from mock_tool_results import (
    MEMORY_ITEM_BASKET,
    MEMORY_ITEM_COFFEE_TABLE,
    MEMORY_ITEM_GREEN_BLANKET,
    MEMORY_ITEM_PHONE,
    MEMORY_ITEM_PILLOW,
    MEMORY_ITEM_REMOTE,
    MEMORY_ITEM_RING_TOY_2,
    MEMORY_ITEM_SHOE_CABINET,
    MEMORY_ITEM_SOFA,
    MEMORY_ITEM_TISSUE_BOX,
    MEMORY_ITEM_WHITE_SHOES,
    MEMORY_ITEM_YELLOW_CUP,
    collect_rings_to_basket_mock,
    fetch_cup_mock,
    fetch_remote_mock,
    fold_blanket_mock,
    move_tissue_right_mock,
    place_phone_near_tissue_mock,
    place_shoes_to_cabinet_mock,
    tidy_pillows_mock,
)


def _record_fc_scores(result, eval_record, test_name, checks, tolerance=0.01):
    """Compute param_accuracy scores and record to eval report."""
    pa = score_param_accuracy_multi(result, checks, tolerance)
    eval_record(test_name, {"param_accuracy": pa.to_dict()})


def _bool_param_accuracy(passed: bool, detail: str) -> dict:
    """Build a param_accuracy score dict from a boolean check result."""
    return ScoreDetail(
        "param_accuracy",
        1.0 if passed else 0.0,
        detail,
        sub_scores={"matched": 1 if passed else 0, "total": 1},
    ).to_dict()


class TestFetchParamAccuracy:
    """Verify parameter values in fetch-and-handover flows."""

    def test_fetch_cup_move_to_coords(self, runner, eval_record):
        """move_to should use coordinates from spatial_memory_query_vec for yellow cup."""
        result = runner.run("把茶几上的黄色杯子递给我", fetch_cup_mock())
        item = MEMORY_ITEM_YELLOW_CUP

        checks = [(
            "move_to",
            {
                "obj_x": item["obj_x"],
                "obj_y": item["obj_y"],
                "coordinate_x": item["coordinate_x"],
                "coordinate_y": item["coordinate_y"],
                "coordinate_z": item["coordinate_z"],
                "coordinate_yaw": item["coordinate_yaw"],
            },
            0,
        )]

        _record_fc_scores(result, eval_record, "fc_fetch_cup_coords", checks)

        assert_tool_contains(result, "move_to")
        pa = score_param_accuracy_multi(result, checks)
        assert pa.score >= 0.5, f"Param accuracy too low: {pa.details}"

    def test_fetch_cup_grasp_english(self, runner, eval_record):
        """grasp_start should receive English item name."""
        result = runner.run("把茶几上的黄色杯子递给我", fetch_cup_mock())
        grasp_calls = result.get_calls_for("grasp_start")

        passed = bool(grasp_calls) and grasp_calls[0].get("item", "").isascii()
        item_arg = grasp_calls[0].get("item", "") if grasp_calls else "(未调用)"
        eval_record("fc_fetch_cup_grasp_lang", {
            "param_accuracy": _bool_param_accuracy(
                passed, f"grasp_start item='{item_arg}', 英文={'是' if passed else '否'}",
            ),
        })

        assert grasp_calls, "grasp_start was not called"
        assert grasp_calls[0].get("item", "").isascii(), (
            f"grasp_start item should be English, got: {item_arg}"
        )

    def test_fetch_remote_move_to_coords(self, runner, eval_record):
        """move_to should use coordinates from spatial_memory_query_vec for remote."""
        result = runner.run("帮我把茶几上的遥控器拿过来", fetch_remote_mock())
        item = MEMORY_ITEM_REMOTE

        checks = [(
            "move_to",
            {
                "obj_x": item["obj_x"],
                "obj_y": item["obj_y"],
                "coordinate_x": item["coordinate_x"],
                "coordinate_y": item["coordinate_y"],
                "coordinate_z": item["coordinate_z"],
                "coordinate_yaw": item["coordinate_yaw"],
            },
            0,
        )]

        _record_fc_scores(result, eval_record, "fc_fetch_remote_coords", checks)

        pa = score_param_accuracy_multi(result, checks)
        assert pa.score >= 0.5, f"Param accuracy too low: {pa.details}"

    def test_fetch_delete_memory_id(self, runner, eval_record):
        """delete_spatial_memory_by_id should use the _id from spatial_memory_query_vec."""
        result = runner.run("把茶几上的黄色杯子递给我", fetch_cup_mock())

        delete_calls = result.get_calls_for("delete_spatial_memory_by_id")
        expected_id = MEMORY_ITEM_YELLOW_CUP["_id"]

        if delete_calls:
            id_list = delete_calls[0].get("id_list", [])
            passed = expected_id in id_list
            detail = f"id_list 包含 {expected_id}: {'是' if passed else '否'} (实际: {id_list})"
        else:
            passed = False
            detail = "delete_spatial_memory_by_id 未被调用（可选步骤）"
            passed = True

        eval_record("fc_fetch_delete_id", {
            "param_accuracy": _bool_param_accuracy(passed, detail),
        })

        if delete_calls:
            id_list = delete_calls[0].get("id_list", [])
            assert expected_id in id_list, (
                f"Expected _id {expected_id} in id_list, got {id_list}"
            )


class TestPlaceParamAccuracy:
    """Verify parameter values in place/move flows."""

    def test_place_shoes_target_coords(self, runner, eval_record):
        """After grasping shoes, move_to should use shoe cabinet coordinates."""
        result = runner.run("把地上的白色鞋子放到鞋柜旁边", place_shoes_to_cabinet_mock())

        move_calls = result.get_calls_for("move_to")
        cabinet = MEMORY_ITEM_SHOE_CABINET
        tolerance = 0.5

        if len(move_calls) >= 2:
            last_move = move_calls[-1]
            diff = abs(last_move.get("coordinate_x", 0) - cabinet["coordinate_x"])
            passed = diff <= tolerance
            detail = f"末次 move_to coordinate_x 与鞋柜差值={diff:.2f} (容差={tolerance})"
        else:
            passed = False
            detail = f"move_to 仅调用 {len(move_calls)} 次，预期至少 2 次"

        eval_record("fc_shoes_target_coords", {
            "param_accuracy": _bool_param_accuracy(passed, detail),
        })

        assert len(move_calls) >= 2, f"Expected at least 2 move_to calls, got {len(move_calls)}"
        last_move = move_calls[-1]
        assert abs(last_move.get("coordinate_x", 0) - cabinet["coordinate_x"]) <= tolerance, (
            f"Last move_to coordinate_x should be near shoe cabinet"
        )

    def test_place_phone_near_tissue_coords(self, runner, eval_record):
        """Phone should be placed near tissue box location."""
        result = runner.run("把手机放到纸巾盒旁边", place_phone_near_tissue_mock())

        has_grasp = "grasp_start" in result.tool_names
        has_place = "place_down" in result.tool_names
        passed = has_grasp and has_place
        detail = f"grasp_start={'有' if has_grasp else '无'}, place_down={'有' if has_place else '无'}"

        eval_record("fc_phone_near_tissue", {
            "param_accuracy": _bool_param_accuracy(passed, detail),
        })

        assert_tool_contains(result, "grasp_start")
        assert_tool_contains(result, "place_down")

    def test_move_tissue_push_direction(self, runner, eval_record):
        """push_item should push tissue box to the right."""
        result = runner.run("把纸巾盒放到桌子右边一点", move_tissue_right_mock())

        push_calls = result.get_calls_for("push_item")

        if push_calls:
            direction = push_calls[0].get("direction", "")
            passed = "right" in direction.lower()
            detail = f"push_item direction='{direction}', 包含 right={'是' if passed else '否'}"
        else:
            grasp_calls = result.get_calls_for("grasp_start")
            passed = bool(grasp_calls)
            detail = f"push_item 未调用，使用 grasp_start 替代={'是' if passed else '否'}"

        eval_record("fc_tissue_push_direction", {
            "param_accuracy": _bool_param_accuracy(passed, detail),
        })

        if push_calls:
            direction = push_calls[0].get("direction", "")
            assert "right" in direction.lower(), (
                f"Expected push direction 'right', got: {direction}"
            )

    def test_collect_rings_basket_coords(self, runner, eval_record):
        """After grasping rings, should move to basket location and place_down."""
        result = runner.run("把桌上的彩色圆环收到篮子里", collect_rings_to_basket_mock())

        has_grasp = "grasp_start" in result.tool_names
        has_place = "place_down" in result.tool_names
        passed = has_grasp and has_place
        detail = f"grasp_start={'有' if has_grasp else '无'}, place_down={'有' if has_place else '无'}"

        eval_record("fc_rings_basket", {
            "param_accuracy": _bool_param_accuracy(passed, detail),
        })

        assert_tool_contains(result, "grasp_start")
        assert_tool_contains(result, "place_down")


class TestTidyParamAccuracy:
    """Verify parameter values in tidy/fold flows."""

    def test_fold_blanket_item_english(self, runner, eval_record):
        """fold_item should receive English item name."""
        result = runner.run("把沙发上的绿色毯子叠一下", fold_blanket_mock())

        fold_calls = result.get_calls_for("fold_item")

        passed = bool(fold_calls) and fold_calls[0].get("item", "").isascii()
        item_arg = fold_calls[0].get("item", "") if fold_calls else "(未调用)"
        eval_record("fc_fold_blanket_lang", {
            "param_accuracy": _bool_param_accuracy(
                passed, f"fold_item item='{item_arg}', 英文={'是' if passed else '否'}",
            ),
        })

        assert fold_calls, "fold_item was not called"
        assert fold_calls[0].get("item", "").isascii(), (
            f"fold_item item should be English, got: {item_arg}"
        )

    def test_fold_blanket_move_to_sofa(self, runner, eval_record):
        """Should move to blanket/sofa location before folding."""
        result = runner.run("把沙发上的绿色毯子叠一下", fold_blanket_mock())
        item = MEMORY_ITEM_GREEN_BLANKET

        checks = [(
            "move_to",
            {
                "obj_x": item["obj_x"],
                "obj_y": item["obj_y"],
                "coordinate_x": item["coordinate_x"],
                "coordinate_y": item["coordinate_y"],
                "coordinate_z": item["coordinate_z"],
                "coordinate_yaw": item["coordinate_yaw"],
            },
            0,
        )]

        _record_fc_scores(result, eval_record, "fc_fold_blanket_coords", checks)

        pa = score_param_accuracy_multi(result, checks)
        assert pa.score >= 0.5, f"Param accuracy too low: {pa.details}"

    def test_tidy_pillows_surface(self, runner, eval_record):
        """tidy_surface should receive 'sofa' as surface."""
        result = runner.run("帮我整理一下沙发上的靠枕", tidy_pillows_mock())

        tidy_calls = result.get_calls_for("tidy_surface")

        passed = bool(tidy_calls) and bool(tidy_calls[0].get("surface", ""))
        surface = tidy_calls[0].get("surface", "") if tidy_calls else "(未调用)"
        eval_record("fc_tidy_pillows_surface", {
            "param_accuracy": _bool_param_accuracy(
                passed, f"tidy_surface surface='{surface}', 非空={'是' if passed else '否'}",
            ),
        })

        assert tidy_calls, "tidy_surface was not called"
        assert tidy_calls[0].get("surface", ""), "tidy_surface surface param is empty"

    def test_spatial_query_chinese_name(self, runner, eval_record):
        """spatial_memory_query_vec name param should be in Chinese."""
        result = runner.run("把茶几上的黄色杯子递给我", fetch_cup_mock())

        query_calls = result.get_calls_for("spatial_memory_query_vec")

        if query_calls:
            name = query_calls[0].get("name", "")
            has_chinese = any("\u4e00" <= c <= "\u9fff" for c in name)
            detail = f"query name='{name}', 含中文={'是' if has_chinese else '否'}"
        else:
            has_chinese = False
            detail = "spatial_memory_query_vec 未调用"

        eval_record("fc_spatial_query_chinese", {
            "param_accuracy": _bool_param_accuracy(has_chinese, detail),
        })

        assert query_calls, "spatial_memory_query_vec was not called"
        name = query_calls[0].get("name", "")
        assert any("\u4e00" <= c <= "\u9fff" for c in name), (
            f"spatial_memory_query_vec name should be Chinese, got: {name}"
        )
