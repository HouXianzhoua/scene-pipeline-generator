"""Mock tool return values and providers for different test scenarios."""

import json
from collections import defaultdict
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Standard mock data — Scene objects
# ---------------------------------------------------------------------------

MEMORY_ITEM_YELLOW_CUP = {
    "_id": "test_id_yellow_cup_001",
    "type": "object",
    "name": "杯子",
    "obj_x": 1.35,
    "obj_y": -3.40,
    "coordinate_x": 1.75,
    "coordinate_y": -2.85,
    "coordinate_z": 0.45,
    "coordinate_yaw": -2.10,
    "property": "黄色杯子",
    "status": "放置在茶几上",
    "color": "黄色",
    "desp": "一个黄色的杯子",
    "similarity_score": 0.90,
}

MEMORY_ITEM_REMOTE = {
    "_id": "test_id_remote_001",
    "type": "object",
    "name": "遥控器",
    "obj_x": 1.50,
    "obj_y": -3.35,
    "coordinate_x": 1.90,
    "coordinate_y": -2.80,
    "coordinate_z": 0.42,
    "coordinate_yaw": -2.05,
    "property": "电视遥控器",
    "status": "放置在茶几上",
    "color": "黑色",
    "desp": "一个黑色的电视遥控器",
    "similarity_score": 0.88,
}

MEMORY_ITEM_PHONE = {
    "_id": "test_id_phone_001",
    "type": "object",
    "name": "手机",
    "obj_x": 1.60,
    "obj_y": -3.45,
    "coordinate_x": 2.00,
    "coordinate_y": -2.90,
    "coordinate_z": 0.43,
    "coordinate_yaw": -2.08,
    "property": "黑色手机",
    "status": "放置在茶几上",
    "color": "黑色",
    "desp": "一部黑色手机",
    "similarity_score": 0.92,
}

MEMORY_ITEM_TISSUE_BOX = {
    "_id": "test_id_tissue_001",
    "type": "object",
    "name": "纸巾盒",
    "obj_x": 1.70,
    "obj_y": -3.30,
    "coordinate_x": 2.10,
    "coordinate_y": -2.75,
    "coordinate_z": 0.44,
    "coordinate_yaw": -2.00,
    "property": "白色纸巾盒",
    "status": "放置在茶几上",
    "color": "白色",
    "desp": "一个白色纸巾盒",
    "similarity_score": 0.87,
}

MEMORY_ITEM_NOTEBOOK = {
    "_id": "test_id_notebook_001",
    "type": "object",
    "name": "本子",
    "obj_x": 1.45,
    "obj_y": -3.50,
    "coordinate_x": 1.85,
    "coordinate_y": -2.95,
    "coordinate_z": 0.41,
    "coordinate_yaw": -2.12,
    "property": "黑色小本子",
    "status": "放置在茶几上",
    "color": "黑色",
    "desp": "一本黑色的小本子",
    "similarity_score": 0.85,
}

MEMORY_ITEM_SANDWICH = {
    "_id": "test_id_sandwich_001",
    "type": "object",
    "name": "三明治",
    "obj_x": 1.55,
    "obj_y": -3.25,
    "coordinate_x": 1.95,
    "coordinate_y": -2.70,
    "coordinate_z": 0.43,
    "coordinate_yaw": -2.03,
    "property": "三明治",
    "status": "放置在茶几上",
    "color": "棕色",
    "desp": "一个三明治",
    "similarity_score": 0.86,
}

MEMORY_ITEM_RING_TOY = {
    "_id": "test_id_ring_toy_001",
    "type": "object",
    "name": "套圈玩具",
    "obj_x": 1.20,
    "obj_y": -4.20,
    "coordinate_x": 1.60,
    "coordinate_y": -3.65,
    "coordinate_z": 0.10,
    "coordinate_yaw": -2.50,
    "property": "彩色套圈玩具",
    "status": "放置在地毯上",
    "color": "彩色",
    "desp": "一组彩色套圈玩具",
    "similarity_score": 0.83,
}

MEMORY_ITEM_RING_TOY_2 = {
    "_id": "test_id_ring_toy_002",
    "type": "object",
    "name": "圆环",
    "obj_x": 1.40,
    "obj_y": -3.38,
    "coordinate_x": 1.80,
    "coordinate_y": -2.83,
    "coordinate_z": 0.42,
    "coordinate_yaw": -2.06,
    "property": "彩色圆环",
    "status": "放置在茶几上",
    "color": "彩色",
    "desp": "几个彩色圆环",
    "similarity_score": 0.81,
}

MEMORY_ITEM_WHITE_SHOES = {
    "_id": "test_id_shoes_001",
    "type": "object",
    "name": "鞋子",
    "obj_x": 0.80,
    "obj_y": -4.50,
    "coordinate_x": 1.20,
    "coordinate_y": -3.95,
    "coordinate_z": 0.05,
    "coordinate_yaw": -2.80,
    "property": "白色鞋子",
    "status": "放置在地上",
    "color": "白色",
    "desp": "一双白色鞋子",
    "similarity_score": 0.89,
}

MEMORY_ITEM_GREEN_BLANKET = {
    "_id": "test_id_blanket_001",
    "type": "object",
    "name": "毯子",
    "obj_x": 2.10,
    "obj_y": -2.30,
    "coordinate_x": 2.50,
    "coordinate_y": -1.75,
    "coordinate_z": 0.50,
    "coordinate_yaw": -1.80,
    "property": "绿色毯子",
    "status": "放置在沙发上",
    "color": "绿色",
    "desp": "一条绿色的毯子",
    "similarity_score": 0.91,
}

MEMORY_ITEM_PILLOW = {
    "_id": "test_id_pillow_001",
    "type": "object",
    "name": "靠枕",
    "obj_x": 2.00,
    "obj_y": -2.20,
    "coordinate_x": 2.40,
    "coordinate_y": -1.65,
    "coordinate_z": 0.55,
    "coordinate_yaw": -1.75,
    "property": "靠枕",
    "status": "放置在沙发上",
    "color": "米色",
    "desp": "沙发上的靠枕",
    "similarity_score": 0.84,
}

# ---------------------------------------------------------------------------
# Standard mock data — Furniture
# ---------------------------------------------------------------------------

MEMORY_ITEM_COFFEE_TABLE = {
    "_id": "test_id_coffee_table_001",
    "type": "furniture",
    "name": "茶几",
    "obj_x": 1.50,
    "obj_y": -3.40,
    "coordinate_x": 1.90,
    "coordinate_y": -2.85,
    "coordinate_z": 0.40,
    "coordinate_yaw": -2.05,
    "property": "茶几",
    "status": "固定家具",
    "color": "木色",
    "desp": "客厅中间的茶几",
    "similarity_score": 0.95,
}

MEMORY_ITEM_SOFA = {
    "_id": "test_id_sofa_001",
    "type": "furniture",
    "name": "沙发",
    "obj_x": 2.10,
    "obj_y": -2.10,
    "coordinate_x": 2.50,
    "coordinate_y": -1.55,
    "coordinate_z": 0.45,
    "coordinate_yaw": -1.70,
    "property": "沙发",
    "status": "固定家具",
    "color": "灰色",
    "desp": "客厅的沙发",
    "similarity_score": 0.94,
}

MEMORY_ITEM_SHOE_CABINET = {
    "_id": "test_id_shoe_cabinet_001",
    "type": "furniture",
    "name": "鞋柜",
    "obj_x": 0.40,
    "obj_y": -1.20,
    "coordinate_x": 0.80,
    "coordinate_y": -0.65,
    "coordinate_z": 0.50,
    "coordinate_yaw": -1.50,
    "property": "鞋柜",
    "status": "固定家具",
    "color": "白色",
    "desp": "门口的鞋柜",
    "similarity_score": 0.93,
}

MEMORY_ITEM_BASKET = {
    "_id": "test_id_basket_001",
    "type": "furniture",
    "name": "篮子",
    "obj_x": 1.80,
    "obj_y": -4.20,
    "coordinate_x": 2.20,
    "coordinate_y": -3.65,
    "coordinate_z": 0.15,
    "coordinate_yaw": -2.40,
    "property": "收纳篮子",
    "status": "放置在地上",
    "color": "棕色",
    "desp": "一个收纳篮子",
    "similarity_score": 0.88,
}

MEMORY_ITEM_CARPET = {
    "_id": "test_id_carpet_001",
    "type": "furniture",
    "name": "地毯",
    "obj_x": 1.20,
    "obj_y": -4.00,
    "coordinate_x": 1.60,
    "coordinate_y": -3.45,
    "coordinate_z": 0.02,
    "coordinate_yaw": -2.50,
    "property": "地毯",
    "status": "固定物品",
    "color": "米色",
    "desp": "沙发前的地毯",
    "similarity_score": 0.90,
}

# ---------------------------------------------------------------------------
# Standard result constants
# ---------------------------------------------------------------------------

MOVE_SUCCESS = {"state": "succeed", "msg": "已经移动到目的地"}
MOVE_FAILED = {"state": "failed", "msg": "导航失败，路径被阻挡"}
GRASP_SUCCESS_TEMPLATE = lambda item: {"status": "success", "message": f"执行任务: {item}"}
GRASP_OUT_OF_RANGE = {"status": "failed", "message": "超出操作范围"}
GRASP_FAILED = {"status": "failed", "message": "抓取失败，未能稳定抓取物品"}
PERCEPTION_CUSTOM_SUCCESS = {"state": "succeed", "msg": "已经移动到用户位置"}
PERCEPTION_CUSTOM_FAILED = {"state": "failed", "msg": "未检测到用户位置"}
HANDOVER_SUCCESS = "SUCCESS: auto hand – force-release detected, arm returned home."
HANDOVER_FAILED = "ERROR: force-release timeout – no pull detected within 30 s."
PLACE_DOWN_SUCCESS = {"state": "succeed", "msg": ""}
PLACE_DOWN_FAILED = {"state": "failed", "msg": "放置失败，目标位置不可达"}
FOLD_ITEM_SUCCESS = lambda item: {"state": "succeed", "msg": f"已完成 {item} 的折叠"}
FOLD_ITEM_FAILED = {"state": "failed", "msg": "折叠失败，物品不适合折叠"}
TIDY_SURFACE_SUCCESS = lambda s: {"state": "succeed", "msg": f"已整理 {s} 上的物品"}
PUSH_ITEM_SUCCESS = lambda item, d: {"state": "succeed", "msg": f"已将 {item} 向 {d} 推移"}
PUSH_ITEM_FAILED = {"state": "failed", "msg": "推移失败，物品太重"}
FETCH_BOX_SUCCESS = {"state": "succeed", "msg": "已从 table 搬起箱子"}
FETCH_BOX_FAILED = {"state": "failed", "msg": "未能搬起箱子，位置偏移"}
PUT_BOX_SUCCESS = {"state": "succeed", "msg": "已将箱子放到 stack"}
PUT_BOX_FAILED = {"state": "failed", "msg": "放置箱子失败"}

LIST_OBJECTS_COFFEE_TABLE = {
    "state": "succeed",
    "objects": [
        {"name": "杯子", "color": "黄色", "position": "左侧"},
        {"name": "遥控器", "color": "黑色", "position": "中间偏左"},
        {"name": "手机", "color": "黑色", "position": "中间偏右"},
        {"name": "纸巾盒", "color": "白色", "position": "右侧"},
        {"name": "本子", "color": "黑色", "position": "左侧偏后"},
        {"name": "三明治", "color": "棕色", "position": "中间"},
        {"name": "圆环", "color": "彩色", "position": "左侧"},
    ],
    "msg": "在 coffee_table 上检测到 7 个物品",
}

LIST_OBJECTS_CARPET = {
    "state": "succeed",
    "objects": [
        {"name": "套圈玩具", "color": "彩色", "position": "中间"},
    ],
    "msg": "在 carpet 上检测到 1 个物品",
}

LIST_OBJECTS_FLOOR = {
    "state": "succeed",
    "objects": [
        {"name": "套圈玩具", "color": "彩色", "position": "地毯上"},
        {"name": "鞋子", "color": "白色", "position": "门口"},
    ],
    "msg": "在 floor 上检测到 2 个物品",
}

LIST_OBJECTS_SOFA = {
    "state": "succeed",
    "objects": [
        {"name": "毯子", "color": "绿色", "position": "右侧"},
        {"name": "靠枕", "color": "米色", "position": "左侧"},
    ],
    "msg": "在 sofa 上检测到 2 个物品",
}


# ---------------------------------------------------------------------------
# MockResultProvider
# ---------------------------------------------------------------------------

class MockResultProvider:
    """Provides mock results for tool calls with support for sequencing and conditions."""

    def __init__(self):
        self._defaults: dict[str, Any] = {}
        self._sequences: dict[str, list] = {}
        self._conditional: dict[str, Callable] = {}
        self._call_counts: dict[str, int] = defaultdict(int)

    def set_default(self, tool_name: str, result: Any) -> "MockResultProvider":
        self._defaults[tool_name] = result
        return self

    def set_sequence(self, tool_name: str, results: list) -> "MockResultProvider":
        self._sequences[tool_name] = results
        self._conditional.pop(tool_name, None)
        return self

    def set_conditional(self, tool_name: str, fn: Callable[[dict], Any]) -> "MockResultProvider":
        self._conditional[tool_name] = fn
        self._sequences.pop(tool_name, None)
        return self

    def get(self, tool_name: str, arguments: dict | None = None) -> str:
        self._call_counts[tool_name] += 1
        count = self._call_counts[tool_name]

        if tool_name in self._sequences:
            seq = self._sequences[tool_name]
            idx = min(count - 1, len(seq) - 1)
            result = seq[idx]
        elif tool_name in self._conditional:
            result = self._conditional[tool_name](arguments or {})
        elif tool_name in self._defaults:
            result = self._defaults[tool_name]
        else:
            result = {"state": "succeed", "msg": ""}

        if callable(result):
            result = result(arguments or {})

        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Memory query handler builder
# ---------------------------------------------------------------------------

def _memory_query_handler(items_map: dict[str, list]) -> Callable:
    """Build a handler that matches the 'name' / 'query' argument against keys in items_map."""

    def handler(args: dict) -> list:
        name = args.get("name", "")
        query = args.get("query", "")
        color = args.get("color", "")
        combined = f"{name} {query} {color}"
        for key, items in items_map.items():
            if key in combined:
                return items
        first = list(items_map.values())
        return first[0] if first else []

    return handler


def _list_objects_handler(area_map: dict[str, dict]) -> Callable:
    """Build a handler that returns object list based on area argument."""

    def handler(args: dict) -> dict:
        area = args.get("area", "")
        for key, result in area_map.items():
            if key in area:
                return result
        return {"state": "succeed", "objects": [], "msg": f"在 {area} 上未检测到物品"}

    return handler


# ---------------------------------------------------------------------------
# Common defaults
# ---------------------------------------------------------------------------

def _apply_common_defaults(p: MockResultProvider) -> None:
    """Register default success results for all common tools."""
    p.set_default("move_to", MOVE_SUCCESS)
    p.set_default("move", MOVE_SUCCESS)
    p.set_conditional(
        "scene_recognition",
        lambda a: {"state": "succeed", "msg": f"当前位置有{a.get('target', '物品')}"},
    )
    p.set_conditional(
        "grasp_start",
        lambda a: {"status": "success", "message": f"执行任务: {a.get('item', 'item')}"},
    )
    p.set_default("perception_custom", PERCEPTION_CUSTOM_SUCCESS)
    p.set_default("handover", HANDOVER_SUCCESS)
    p.set_default("place_down", PLACE_DOWN_SUCCESS)
    p.set_conditional(
        "delete_spatial_memory_by_id",
        lambda a: {"deleted_count": len(a.get("id_list", []))},
    )
    p.set_conditional(
        "trigger_spatial_processing",
        lambda a: {"state": "succeed", "msg": f"已完成空间感知处理，检测到 {a.get('obj_name', '')} 相关物品"},
    )
    p.set_conditional(
        "fold_item",
        lambda a: {"state": "succeed", "msg": f"已完成 {a.get('item', 'item')} 的折叠"},
    )
    p.set_conditional(
        "tidy_surface",
        lambda a: {"state": "succeed", "msg": f"已整理 {a.get('surface', '')} 上的物品"},
    )
    p.set_conditional(
        "push_item",
        lambda a: {"state": "succeed", "msg": f"已将 {a.get('item', '')} 向 {a.get('direction', 'inward')} 推移"},
    )
    p.set_default("fetch_box", FETCH_BOX_SUCCESS)
    p.set_default("put_box", PUT_BOX_SUCCESS)


# ---------------------------------------------------------------------------
# Pre-configured mock providers — Fetch scenarios
# ---------------------------------------------------------------------------

def fetch_cup_mock() -> MockResultProvider:
    """Fetch yellow cup from coffee table and hand to user."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "杯子": [MEMORY_ITEM_YELLOW_CUP],
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
        }),
    )
    return p


def fetch_remote_mock() -> MockResultProvider:
    """Fetch remote control from coffee table and hand to user."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "遥控器": [MEMORY_ITEM_REMOTE],
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
        }),
    )
    return p


def fetch_sandwich_mock() -> MockResultProvider:
    """Fetch sandwich from coffee table and hand to user."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "三明治": [MEMORY_ITEM_SANDWICH],
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
        }),
    )
    return p


# ---------------------------------------------------------------------------
# Pre-configured mock providers — Place / Move scenarios
# ---------------------------------------------------------------------------

def place_shoes_to_cabinet_mock() -> MockResultProvider:
    """Pick up white shoes from floor and place near shoe cabinet."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "鞋子": [MEMORY_ITEM_WHITE_SHOES],
            "鞋柜": [MEMORY_ITEM_SHOE_CABINET],
        }),
    )
    return p


def move_tissue_right_mock() -> MockResultProvider:
    """Push tissue box to the right on coffee table."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "纸巾盒": [MEMORY_ITEM_TISSUE_BOX],
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
        }),
    )
    return p


def place_phone_near_tissue_mock() -> MockResultProvider:
    """Pick up phone and place it next to tissue box."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "手机": [MEMORY_ITEM_PHONE],
            "纸巾盒": [MEMORY_ITEM_TISSUE_BOX],
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
        }),
    )
    return p


# ---------------------------------------------------------------------------
# Pre-configured mock providers — Collect / Organize scenarios
# ---------------------------------------------------------------------------

def collect_toys_to_basket_mock() -> MockResultProvider:
    """Collect toys from carpet/floor into basket."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "玩具": [MEMORY_ITEM_RING_TOY],
            "套圈": [MEMORY_ITEM_RING_TOY],
            "篮子": [MEMORY_ITEM_BASKET],
            "地毯": [MEMORY_ITEM_CARPET],
        }),
    )
    p.set_conditional(
        "list_objects",
        _list_objects_handler({
            "carpet": LIST_OBJECTS_CARPET,
            "floor": LIST_OBJECTS_FLOOR,
            "地毯": LIST_OBJECTS_CARPET,
            "地": LIST_OBJECTS_FLOOR,
        }),
    )
    return p


def collect_rings_to_basket_mock() -> MockResultProvider:
    """Collect colorful rings from coffee table into basket."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "圆环": [MEMORY_ITEM_RING_TOY_2],
            "篮子": [MEMORY_ITEM_BASKET],
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
        }),
    )
    return p


# ---------------------------------------------------------------------------
# Pre-configured mock providers — Tidy / Fold scenarios
# ---------------------------------------------------------------------------

def fold_blanket_mock() -> MockResultProvider:
    """Fold the green blanket on sofa."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "毯子": [MEMORY_ITEM_GREEN_BLANKET],
            "沙发": [MEMORY_ITEM_SOFA],
        }),
    )
    return p


def tidy_pillows_mock() -> MockResultProvider:
    """Tidy up pillows on sofa."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "靠枕": [MEMORY_ITEM_PILLOW],
            "沙发": [MEMORY_ITEM_SOFA],
        }),
    )
    return p


def tidy_coffee_table_mock() -> MockResultProvider:
    """Tidy up all items on coffee table."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
        }),
    )
    p.set_conditional(
        "list_objects",
        _list_objects_handler({
            "coffee_table": LIST_OBJECTS_COFFEE_TABLE,
            "茶几": LIST_OBJECTS_COFFEE_TABLE,
        }),
    )
    return p


# ---------------------------------------------------------------------------
# Pre-configured mock providers — Query scenarios
# ---------------------------------------------------------------------------

def list_table_items_mock() -> MockResultProvider:
    """List all items on the coffee table."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
        }),
    )
    p.set_conditional(
        "list_objects",
        _list_objects_handler({
            "coffee_table": LIST_OBJECTS_COFFEE_TABLE,
            "茶几": LIST_OBJECTS_COFFEE_TABLE,
        }),
    )
    return p


def find_phone_mock() -> MockResultProvider:
    """Find phone location via spatial memory."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "手机": [MEMORY_ITEM_PHONE],
        }),
    )
    return p


def check_remote_on_table_mock() -> MockResultProvider:
    """Check if remote control is on coffee table."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "遥控器": [MEMORY_ITEM_REMOTE],
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
        }),
    )
    return p


# ---------------------------------------------------------------------------
# Pre-configured mock providers — Complex scenarios
# ---------------------------------------------------------------------------

def clear_table_mock() -> MockResultProvider:
    """Clear the coffee table — list items then move them off."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "杯子": [MEMORY_ITEM_YELLOW_CUP],
            "遥控器": [MEMORY_ITEM_REMOTE],
            "手机": [MEMORY_ITEM_PHONE],
            "纸巾盒": [MEMORY_ITEM_TISSUE_BOX],
            "三明治": [MEMORY_ITEM_SANDWICH],
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
            "沙发": [MEMORY_ITEM_SOFA],
        }),
    )
    p.set_conditional(
        "list_objects",
        _list_objects_handler({
            "coffee_table": LIST_OBJECTS_COFFEE_TABLE,
            "茶几": LIST_OBJECTS_COFFEE_TABLE,
        }),
    )
    return p


def classify_edible_mock() -> MockResultProvider:
    """Classify items into edible / non-edible and separate them."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "杯子": [MEMORY_ITEM_YELLOW_CUP],
            "三明治": [MEMORY_ITEM_SANDWICH],
            "遥控器": [MEMORY_ITEM_REMOTE],
            "手机": [MEMORY_ITEM_PHONE],
            "纸巾盒": [MEMORY_ITEM_TISSUE_BOX],
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
        }),
    )
    p.set_conditional(
        "list_objects",
        _list_objects_handler({
            "coffee_table": LIST_OBJECTS_COFFEE_TABLE,
            "茶几": LIST_OBJECTS_COFFEE_TABLE,
        }),
    )
    return p


# ---------------------------------------------------------------------------
# Pre-configured mock providers — Failure-recovery scenarios
# ---------------------------------------------------------------------------

def grasp_fail_retry_mock() -> MockResultProvider:
    """grasp_start fails on the first attempt (out of range), succeeds on retry."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "杯子": [MEMORY_ITEM_YELLOW_CUP],
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
        }),
    )
    p.set_sequence("grasp_start", [
        GRASP_OUT_OF_RANGE,
        {"status": "success", "message": "执行任务: yellow cup"},
    ])
    return p


def move_fail_retry_mock() -> MockResultProvider:
    """move_to fails on the first attempt, succeeds on subsequent calls."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "遥控器": [MEMORY_ITEM_REMOTE],
            "茶几": [MEMORY_ITEM_COFFEE_TABLE],
        }),
    )
    p.set_sequence("move_to", [MOVE_FAILED, MOVE_SUCCESS, MOVE_SUCCESS, MOVE_SUCCESS])
    return p


def memory_miss_search_mock() -> MockResultProvider:
    """First spatial_memory_query_vec returns empty; subsequent calls find the item."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_sequence(
        "spatial_memory_query_vec",
        [
            [],
            [MEMORY_ITEM_PHONE],
            [MEMORY_ITEM_PHONE],
            [MEMORY_ITEM_PHONE],
        ],
    )
    return p


def fold_fail_retry_mock() -> MockResultProvider:
    """fold_item fails on the first attempt, succeeds on retry."""
    p = MockResultProvider()
    _apply_common_defaults(p)
    p.set_conditional(
        "spatial_memory_query_vec",
        _memory_query_handler({
            "毯子": [MEMORY_ITEM_GREEN_BLANKET],
            "沙发": [MEMORY_ITEM_SOFA],
        }),
    )
    p.set_sequence("fold_item", [
        FOLD_ITEM_FAILED,
        {"state": "succeed", "msg": "已完成 blanket 的折叠"},
    ])
    return p
