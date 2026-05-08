"""Step 6: Generate mock_tool_results.py and mock_server.py."""

import json
import logging
import re
from pathlib import Path

from ..llm_client import LLMClient

logger = logging.getLogger(__name__)

_COLOR_WORDS = (
    "白色", "黑色", "灰色", "蓝色", "蓝灰色", "绿色", "黄色", "红色", "棕色",
    "紫色", "橙色", "粉色", "米白色", "浅色", "深色", "金色", "银色",
    "白金色", "木色", "浅木色", "深木色", "多色",
)
_COMMON_OBJECT_HEADS = (
    "那个", "这个", "那本", "这本", "那堆", "这堆", "那束", "这束", "那盒",
    "这盒", "那只", "这只", "那台", "这台", "那盆", "这盆", "那个东西",
    "这个东西", "那个物品", "这个物品",
)
_COMMON_OBJECT_NOUNS = (
    "书", "书本", "书籍", "书籍堆", "鲜花", "花", "花瓶", "托盘", "盒子", "杯子",
    "马克杯", "抱枕", "靠垫", "毛毯", "相机", "小相机", "手机", "平板", "笔记本",
    "纸巾盒", "收音机", "三明治", "盘子", "本子", "玩具", "塑料圈", "盆栽植物",
    "植物", "盆栽",
)


def generate_mocks(
    client: LLMClient,
    scene_data: dict,
    commands_data: dict,
    all_tools: list[str],
    tests_dir: Path,
) -> dict:
    """Generate mock_tool_results.py and mock_server.py, return mock metadata."""
    mock_meta = _generate_mock_data(client, scene_data, commands_data)

    _write_mock_tool_results(mock_meta, all_tools, tests_dir / "mock_tool_results.py")
    _write_mock_server(mock_meta, all_tools, scene_data, tests_dir / "mock_server.py")

    return mock_meta


def _generate_mock_data(
    client: LLMClient, scene_data: dict, commands_data: dict
) -> dict:
    mock_meta = _build_mock_meta_base(scene_data, commands_data)
    if not _mock_meta_is_usable(mock_meta, commands_data):
        raise ValueError("deterministic mock_meta generation produced invalid result")
    return mock_meta


def _mock_meta_is_usable(mock_meta: dict, commands_data: dict) -> bool:
    """Return True when mock_meta contains enough structure for test generation."""
    if not isinstance(mock_meta, dict) or not mock_meta:
        return False
    command_count = len(commands_data.get("commands", []))
    factory_count = len(mock_meta.get("mock_factories", []))
    memory_count = len(mock_meta.get("memory_items", []))
    if command_count <= 0:
        return True
    return factory_count > 0 and memory_count > 0


def _build_mock_meta_base(scene_data: dict, commands_data: dict) -> dict:
    """Build the deterministic mock_meta baseline."""
    furniture = scene_data.get("furniture", [])
    objects = scene_data.get("objects", [])
    memory_items = []
    list_objects_data = []
    mock_factories = []
    failure_factories = []

    furniture_by_en = {}
    object_by_en = {}
    object_names = set()

    for idx, item in enumerate(furniture, 1):
        entry = _make_memory_item(item, idx=idx, kind="furniture")
        memory_items.append(entry)
        furniture_by_en[item.get("name_en", "")] = entry["var_name"]

    for idx, item in enumerate(objects, 1):
        entry = _make_memory_item(item, idx=idx, kind="object")
        memory_items.append(entry)
        object_by_en[item.get("name_en", "")] = entry["var_name"]
        object_names.add(item.get("name_en", ""))

    list_objects_var_by_furniture: dict[str, str] = {}
    objects_by_furniture: dict[str, list[dict]] = {}
    for obj in objects:
        furniture_name = obj.get("on_furniture", "")
        if furniture_name:
            objects_by_furniture.setdefault(furniture_name, []).append(obj)

    for idx, furn in enumerate(furniture, 1):
        furniture_name = furn.get("name_en", "")
        attached = objects_by_furniture.get(furniture_name, [])
        if not attached:
            continue
        var_name = f"LIST_OBJECTS_{_slug_to_constant(furniture_name)}"
        list_objects_var_by_furniture[furniture_name] = var_name
        list_objects_data.append({
            "var_name": var_name,
            "area_keys": [furniture_name, furn.get("name", "")],
            "data": {
                "state": "succeed",
                "objects": [
                    {
                        "name": obj.get("name", obj.get("name_en", "")),
                        "color": obj.get("color", ""),
                        "position": obj.get("position", ""),
                    }
                    for obj in attached
                ],
                "msg": f"在 {furniture_name} 上检测到 {len(attached)} 个物品",
            },
        })

    for command in commands_data.get("commands", []):
        mock_factories.append(_build_command_factory(
            command,
            scene_data,
            furniture_by_en,
            object_by_en,
            list_objects_var_by_furniture,
            object_names,
        ))

    for failure in commands_data.get("failure_scenarios", []):
        failure_factories.append(_build_failure_factory(
            failure,
            scene_data,
            furniture_by_en,
            object_by_en,
            list_objects_var_by_furniture,
            object_names,
        ))

    return {
        "memory_items": memory_items,
        "list_objects_data": list_objects_data,
        "mock_factories": mock_factories,
        "failure_factories": failure_factories,
    }


def _make_memory_item(item: dict, *, idx: int, kind: str) -> dict:
    name_en = item.get("name_en", f"{kind}_{idx}")
    name_cn = item.get("name", name_en)
    slug = _slug_to_constant(name_en)
    coord_x, coord_y, coord_z, coord_yaw = _stable_pose(name_en, idx, kind=kind)
    obj_x = round(coord_x - 0.35, 2)
    obj_y = round(coord_y + 0.28, 2)
    var_name = f"MEMORY_ITEM_{slug}"
    data = {
        "_id": f"test_{kind}_{name_en}_{idx:03d}",
        "type": kind,
        "name": name_cn,
        "obj_x": obj_x,
        "obj_y": obj_y,
        "coordinate_x": coord_x,
        "coordinate_y": coord_y,
        "coordinate_z": coord_z,
        "coordinate_yaw": coord_yaw,
        "property": f"{item.get('color', '')}{name_cn}".strip() or name_cn,
        "status": item.get("position", ""),
        "color": item.get("color", ""),
        "desp": item.get("description", item.get("position", "")),
        "similarity_score": round(0.86 + (idx % 8) * 0.01, 2),
    }
    return {"var_name": var_name, "data": data}


def _stable_pose(name_en: str, idx: int, *, kind: str) -> tuple[float, float, float, float]:
    seed = sum(ord(ch) for ch in name_en) + idx * 37 + (17 if kind == "object" else 0)
    coord_x = round(((seed % 900) / 100.0) - 4.5, 2)
    coord_y = round((((seed // 7) % 900) / 100.0) - 4.5, 2)
    coord_z = round(0.42 if kind == "object" else 0.0, 2)
    coord_yaw = round((((seed // 13) % 628) / 100.0) - 3.14, 2)
    return coord_x, coord_y, coord_z, coord_yaw


def _slug_to_constant(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    return slug or "ITEM"


def _build_command_factory(
    command: dict,
    scene_data: dict,
    furniture_by_en: dict[str, str],
    object_by_en: dict[str, str],
    list_objects_var_by_furniture: dict[str, str],
    object_names: set[str],
) -> dict:
    target_object = command.get("target_object", "")
    target_furniture = command.get("target_furniture", "")
    factory_name = f"{command.get('test_name', 'command')}_mock"
    category = command.get("category", "")
    text = command.get("text", "")
    memory_queries = _build_memory_queries(
        text=text,
        target_object=target_object,
        target_furniture=target_furniture,
        scene_data=scene_data,
        furniture_by_en=furniture_by_en,
        object_by_en=object_by_en,
        include_target_furniture=True,
    )
    list_objects_areas = {}
    if category in {"收纳归集", "查询检索"} and target_furniture in list_objects_var_by_furniture:
        list_objects_areas[target_furniture] = list_objects_var_by_furniture[target_furniture]

    if category == "查询检索" and not target_furniture:
        obj = _find_scene_object(scene_data, target_object)
        furniture_name = obj.get("on_furniture", "") if obj else ""
        if furniture_name and furniture_name in list_objects_var_by_furniture:
            list_objects_areas[furniture_name] = list_objects_var_by_furniture[furniture_name]

    if target_object in object_names and target_object in object_by_en:
        obj_cn = _find_scene_object(scene_data, target_object).get("name", "")
        memory_queries.setdefault(obj_cn or target_object, [object_by_en[target_object]])

    return {
        "name": factory_name,
        "description": f"Mock for {command.get('test_name', factory_name.removesuffix('_mock'))}",
        "memory_queries": memory_queries,
        "list_objects_areas": list_objects_areas,
        "sequence_overrides": {},
        "for_command": command.get("test_name", ""),
    }


def _build_failure_factory(
    failure: dict,
    scene_data: dict,
    furniture_by_en: dict[str, str],
    object_by_en: dict[str, str],
    list_objects_var_by_furniture: dict[str, str],
    object_names: set[str],
) -> dict:
    failure_type = failure.get("type", "failure")
    text = failure.get("text", "")
    target_object = failure.get("target_object", "")
    obj = _find_scene_object(scene_data, target_object)
    target_furniture = obj.get("on_furniture", "") if obj else ""
    memory_queries = _build_memory_queries(
        text=text,
        target_object=target_object,
        target_furniture=target_furniture,
        scene_data=scene_data,
        furniture_by_en=furniture_by_en,
        object_by_en=object_by_en,
        include_target_furniture=True,
    )
    list_objects_areas = {}
    if failure_type == "memory_miss":
        furniture_name = target_furniture or ""
        if furniture_name and furniture_name in list_objects_var_by_furniture:
            list_objects_areas[furniture_name] = list_objects_var_by_furniture[furniture_name]

    base_name = failure.get("test_name", failure_type)
    factory = {
        "name": f"{base_name}_mock",
        "description": failure.get("description", base_name),
        "memory_queries": memory_queries,
        "list_objects_areas": list_objects_areas,
        "sequence_overrides": {},
        "for_failure": failure_type,
    }

    obj_cn = obj.get("name", target_object) if obj else target_object
    if failure_type == "grasp_fail":
        factory["sequence_overrides"] = {
            "grasp_start": [
                json.dumps({"status": "failed", "error": "out_of_range", "message": f"首次抓取{obj_cn}失败"}, ensure_ascii=False),
                json.dumps({"status": "success", "message": f"成功抓取{obj_cn}"}, ensure_ascii=False),
            ]
        }
    elif failure_type == "move_fail":
        factory["sequence_overrides"] = {
            "move_to": [
                json.dumps({"status": "failed", "error": "navigation_blocked", "message": f"前往{obj_cn}途中导航失败"}, ensure_ascii=False),
                json.dumps({"status": "success", "message": f"重新规划后到达{obj_cn}所在位置"}, ensure_ascii=False),
            ]
        }
    elif failure_type == "memory_miss":
        factory["sequence_overrides"] = {
            "spatial_memory_query_vec": ["[]"]
        }

    if target_object in object_names and target_object in object_by_en:
        factory["memory_queries"].setdefault(obj_cn or target_object, [object_by_en[target_object]])
    return factory


def _build_memory_queries(
    *,
    text: str,
    target_object: str,
    target_furniture: str,
    scene_data: dict,
    furniture_by_en: dict[str, str],
    object_by_en: dict[str, str],
    include_target_furniture: bool,
) -> dict[str, list[str]]:
    queries: dict[str, list[str]] = {}

    obj = _find_scene_object(scene_data, target_object)
    if obj and target_object in object_by_en:
        for alias in _build_object_query_aliases(obj, scene_data):
            queries.setdefault(alias, [object_by_en[target_object]])
        obj_furniture = obj.get("on_furniture", "")
        furn = _find_furniture(scene_data, obj_furniture)
        if furn and obj_furniture in furniture_by_en:
            for alias in _build_furniture_query_aliases(furn):
                queries.setdefault(alias, [furniture_by_en[obj_furniture]])

    if include_target_furniture and target_furniture:
        furn = _find_furniture(scene_data, target_furniture)
        if furn and target_furniture in furniture_by_en:
            for alias in _build_furniture_query_aliases(furn):
                queries.setdefault(alias, [furniture_by_en[target_furniture]])

    for furn in scene_data.get("furniture", []):
        cn = furn.get("name", "")
        en = furn.get("name_en", "")
        if (cn and cn in text) or (en and en in text):
            var_name = furniture_by_en.get(en)
            if var_name:
                for alias in _build_furniture_query_aliases(furn):
                    queries.setdefault(alias, [var_name])

    for obj_item in scene_data.get("objects", []):
        cn = obj_item.get("name", "")
        en = obj_item.get("name_en", "")
        if (cn and cn in text) or (en and en in text):
            var_name = object_by_en.get(en)
            if var_name:
                for alias in _build_object_query_aliases(obj_item, scene_data):
                    queries.setdefault(alias, [var_name])

    return queries


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = (value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _build_furniture_query_aliases(furniture: dict) -> list[str]:
    aliases = [
        furniture.get("name", ""),
        furniture.get("name_en", ""),
        furniture.get("position", ""),
        furniture.get("description", ""),
    ]
    return _ordered_unique(aliases)


def _build_object_query_aliases(obj: dict, scene_data: dict) -> list[str]:
    name_cn = obj.get("name", "")
    name_en = obj.get("name_en", "")
    color = obj.get("color", "")
    position = obj.get("position", "")
    description = obj.get("description", "")
    aliases = [
        name_cn,
        name_en,
        position,
        description,
    ]

    noun_aliases = _derive_object_core_aliases(name_cn)
    aliases.extend(noun_aliases)
    if color:
        aliases.extend(f"{color}{noun}" for noun in noun_aliases)
        aliases.extend(f"{noun}{color}" for noun in noun_aliases)

    furniture_name = obj.get("on_furniture", "")
    furniture = _find_furniture(scene_data, furniture_name) if furniture_name else None
    if furniture:
        furniture_aliases = _build_furniture_query_aliases(furniture)
        for furn_alias in furniture_aliases:
            for noun in noun_aliases:
                aliases.append(f"{furn_alias}{noun}")
                aliases.append(f"{furn_alias}上的{noun}")
            if color:
                for noun in noun_aliases:
                    aliases.append(f"{furn_alias}上的{color}{noun}")
                    aliases.append(f"{furn_alias}{color}{noun}")

    if position:
        for noun in noun_aliases:
            aliases.append(f"{position}{noun}")
            if color:
                aliases.append(f"{position}{color}{noun}")

    return _ordered_unique(aliases)


def _derive_object_core_aliases(name_cn: str) -> list[str]:
    aliases = [name_cn]
    core = name_cn
    for color in _COLOR_WORDS:
        core = core.replace(color, "")
    core = core.strip()
    for head in _COMMON_OBJECT_HEADS:
        if core.startswith(head):
            core = core[len(head):].strip()
    core = core.strip("的里上中前后左右边附近")
    if core:
        aliases.append(core)

    for noun in _COMMON_OBJECT_NOUNS:
        if noun in name_cn or noun in core:
            aliases.append(noun)

    if "平板" in name_cn:
        aliases.append("平板电脑")
        aliases.append("tablet")
    if "笔记本" in name_cn or "本子" in name_cn:
        aliases.append("笔记本")
        aliases.append("notebook")
    if "手机" in name_cn:
        aliases.append("phone")
    if "相机" in name_cn:
        aliases.append("camera")
    if "鲜花" in name_cn or name_cn.endswith("花"):
        aliases.append("flower")
    if "抱枕" in name_cn or "靠垫" in name_cn:
        aliases.append("pillow")
        aliases.append("cushion")
    if "杯子" in name_cn or "马克杯" in name_cn:
        aliases.append("cup")

    return _ordered_unique(aliases)


def _find_scene_object(scene_data: dict, name_en: str) -> dict | None:
    for obj in scene_data.get("objects", []):
        if obj.get("name_en") == name_en:
            return obj
    return None


def _find_furniture(scene_data: dict, name_en: str) -> dict | None:
    for furn in scene_data.get("furniture", []):
        if furn.get("name_en") == name_en:
            return furn
    return None


def _write_mock_tool_results(mock_meta: dict, all_tools: list[str], output_path: Path) -> None:
    lines = [
        '"""Mock tool return values and providers for different test scenarios."""',
        "",
        "import json",
        "import re",
        "from collections import defaultdict",
        "from typing import Any, Callable",
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# Standard mock data",
        "# ---------------------------------------------------------------------------",
        "",
    ]

    for item in mock_meta.get("memory_items", []):
        var_name = item["var_name"]
        data = item["data"]
        lines.append(f"{var_name} = {json.dumps(data, ensure_ascii=False, indent=4)}")
        lines.append("")

    lines.extend([
        "",
        "# ---------------------------------------------------------------------------",
        "# Standard result constants",
        "# ---------------------------------------------------------------------------",
        "",
        'MOVE_SUCCESS = {"state": "succeed", "msg": "已经移动到目的地"}',
        'MOVE_FAILED = {"state": "failed", "msg": "导航失败，路径被阻挡"}',
        'GRASP_SUCCESS_TEMPLATE = lambda item: {"status": "success", "message": f"执行任务: {item}"}',
        'GRASP_OUT_OF_RANGE = {"status": "failed", "message": "超出操作范围"}',
        'GRASP_FAILED = {"status": "failed", "message": "抓取失败，未能稳定抓取物品"}',
        'PERCEPTION_CUSTOM_SUCCESS = {"state": "succeed", "msg": "已经移动到用户位置"}',
        'PERCEPTION_CUSTOM_FAILED = {"state": "failed", "msg": "未检测到用户位置"}',
        'HANDOVER_SUCCESS = "SUCCESS: auto hand – force-release detected, arm returned home."',
        'HANDOVER_FAILED = "ERROR: force-release timeout – no pull detected within 30 s."',
        'PLACE_DOWN_SUCCESS = {"state": "succeed", "msg": ""}',
        'PLACE_DOWN_FAILED = {"state": "failed", "msg": "放置失败，目标位置不可达"}',
        'FOLD_ITEM_SUCCESS = lambda item: {"state": "succeed", "msg": f"已完成 {item} 的折叠"}',
        'FOLD_ITEM_FAILED = {"state": "failed", "msg": "折叠失败，物品不适合折叠"}',
        'TIDY_SURFACE_SUCCESS = lambda s: {"state": "succeed", "msg": f"已整理 {s} 上的物品"}',
        'PUSH_ITEM_SUCCESS = lambda item, d: {"state": "succeed", "msg": f"已将 {item} 向 {d} 推移"}',
        'PUSH_ITEM_FAILED = {"state": "failed", "msg": "推移失败，物品太重"}',
        'FETCH_BOX_SUCCESS = {"state": "succeed", "msg": "已从 table 搬起箱子"}',
        'PUT_BOX_SUCCESS = {"state": "succeed", "msg": "已将箱子放到 stack"}',
        "",
    ])

    for lo in mock_meta.get("list_objects_data", []):
        var_name = lo["var_name"]
        data = lo["data"]
        lines.append(f"{var_name} = {json.dumps(data, ensure_ascii=False, indent=4)}")
        lines.append("")

    lines.extend(_generate_provider_class())
    lines.extend(_generate_helper_functions())
    lines.extend(_generate_common_defaults(all_tools))

    for factory in mock_meta.get("mock_factories", []):
        lines.extend(_generate_mock_factory(factory, mock_meta))

    for factory in mock_meta.get("failure_factories", []):
        lines.extend(_generate_failure_factory(factory, mock_meta))

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", output_path)


def _generate_provider_class() -> list[str]:
    return [
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# MockResultProvider",
        "# ---------------------------------------------------------------------------",
        "",
        "class MockResultProvider:",
        '    """Provides mock results for tool calls with support for sequencing and conditions."""',
        "",
        "    def __init__(self):",
        '        self._defaults: dict[str, Any] = {}',
        "        self._sequences: dict[str, list] = {}",
        "        self._conditional: dict[str, Callable] = {}",
        "        self._call_counts: dict[str, int] = defaultdict(int)",
        "",
        '    def set_default(self, tool_name: str, result: Any) -> "MockResultProvider":',
        "        self._defaults[tool_name] = result",
        "        return self",
        "",
        '    def set_sequence(self, tool_name: str, results: list) -> "MockResultProvider":',
        "        self._sequences[tool_name] = results",
        "        self._conditional.pop(tool_name, None)",
        "        return self",
        "",
        '    def set_conditional(self, tool_name: str, fn: Callable[[dict], Any]) -> "MockResultProvider":',
        "        self._conditional[tool_name] = fn",
        "        self._sequences.pop(tool_name, None)",
        "        return self",
        "",
        "    def get(self, tool_name: str, arguments: dict | None = None) -> str:",
        "        self._call_counts[tool_name] += 1",
        "        count = self._call_counts[tool_name]",
        "",
        "        if tool_name in self._sequences:",
        "            seq = self._sequences[tool_name]",
        "            idx = min(count - 1, len(seq) - 1)",
        "            result = seq[idx]",
        "        elif tool_name in self._conditional:",
        "            result = self._conditional[tool_name](arguments or {})",
        "        elif tool_name in self._defaults:",
        "            result = self._defaults[tool_name]",
        "        else:",
        '            result = {"state": "succeed", "msg": ""}',
        "",
        "        if callable(result):",
        "            result = result(arguments or {})",
        "",
        "        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)",
        "",
    ]


def _generate_helper_functions() -> list[str]:
    return [
        "",
        "# ---------------------------------------------------------------------------",
        "# Memory query handler builder",
        "# ---------------------------------------------------------------------------",
        "",
        "def _normalize_memory_text(value: str) -> str:",
        '    value = (value or "").lower()',
        '    value = re.sub(r"[^\\w\\u4e00-\\u9fff]+", "", value)',
        "    return value",
        "",
        "def _memory_query_handler(items_map: dict[str, list]) -> Callable:",
        '    """Build a handler that matches the \'name\' / \'query\' argument against keys in items_map."""',
        "",
        "    def handler(args: dict) -> list:",
        '        name = args.get("name", "")',
        '        query = args.get("query", "")',
        '        color = args.get("color", "")',
        '        combined = f"{name} {query} {color}"',
        "        normalized = _normalize_memory_text(combined)",
        "        best_score = -1",
        "        best_items = []",
        "        for key, items in items_map.items():",
        "            norm_key = _normalize_memory_text(key)",
        "            if not norm_key or not normalized:",
        "                continue",
        "            if norm_key == normalized:",
        "                score = 1000 + len(norm_key)",
        "            elif norm_key in normalized:",
        "                score = 100 + len(norm_key)",
        "            else:",
        "                score = -1",
        "            if score > best_score:",
        "                best_score = score",
        "                best_items = items",
        "        return best_items if best_score >= 0 else []",
        "",
        "    return handler",
        "",
        "",
        "def _list_objects_handler(area_map: dict[str, dict]) -> Callable:",
        '    """Build a handler that returns object list based on area argument."""',
        "",
        "    def handler(args: dict) -> dict:",
        '        area = args.get("area", "")',
        "        for key, result in area_map.items():",
        "            if key in area:",
        "                return result",
        '        return {"state": "succeed", "objects": [], "msg": f"在 {area} 上未检测到物品"}',
        "",
        "    return handler",
        "",
    ]


def _generate_common_defaults(all_tools: list[str]) -> list[str]:
    registered: set[str] = set()

    def _default(tool: str, val: str) -> str:
        registered.add(tool)
        return f'    p.set_default("{tool}", {val})'

    def _cond(tool: str, expr: str) -> list[str]:
        registered.add(tool)
        return [
            f"    p.set_conditional(",
            f'        "{tool}",',
            f"        {expr},",
            f"    )",
        ]

    lines = [
        "",
        "# ---------------------------------------------------------------------------",
        "# Common defaults",
        "# ---------------------------------------------------------------------------",
        "",
        "def _apply_common_defaults(p: MockResultProvider) -> None:",
        '    """Register default success results for all tools."""',
        _default("move_to", "MOVE_SUCCESS"),
        _default("move", "MOVE_SUCCESS"),
        *_cond("scene_recognition",
               """lambda a: {"state": "succeed", "msg": f"当前位置有{a.get('target', '物品')}"}"""),
        *_cond("grasp_start",
               """lambda a: {"status": "success", "message": f"执行任务: {a.get('item', 'item')}"}"""),
        _default("perception_custom", "PERCEPTION_CUSTOM_SUCCESS"),
        _default("handover", "HANDOVER_SUCCESS"),
        _default("place_down", "PLACE_DOWN_SUCCESS"),
        *_cond("delete_spatial_memory_by_id",
               """lambda a: {"deleted_count": len(a.get("id_list", []))}"""),
        *_cond("trigger_spatial_processing",
               """lambda a: {"state": "succeed", "msg": f"已完成空间感知处理，检测到 {a.get('obj_name', '')} 相关物品"}"""),
        *_cond("fold_item",
               """lambda a: {"state": "succeed", "msg": f"已完成 {a.get('item', 'item')} 的折叠"}"""),
        *_cond("tidy_surface",
               """lambda a: {"state": "succeed", "msg": f"已整理 {a.get('surface', '')} 上的物品"}"""),
        *_cond("push_item",
               """lambda a: {"state": "succeed", "msg": f"已将 {a.get('item', '')} 向 {a.get('direction', 'inward')} 推移"}"""),
        _default("fetch_box", "FETCH_BOX_SUCCESS"),
        _default("put_box", "PUT_BOX_SUCCESS"),
        "",
        *_cond("list_objects",
               """lambda a: {"state": "succeed", "objects": [], "msg": f"在 {a.get('area', '')} 上未检测到物品"}"""),
    ]

    uncovered = [t for t in all_tools if t not in registered]
    if uncovered:
        lines.append("")
        lines.append("    # Auto-generated defaults for remaining tools")
        for tool_name in uncovered:
            lines.append(
                f'    p.set_default("{tool_name}", '
                f'{{"state": "succeed", "msg": ""}})'
            )

    lines.append("")
    return lines


def _generate_mock_factory(factory: dict, mock_meta: dict) -> list[str]:
    name = factory["name"]
    desc = factory.get("description", "")
    memory_queries = factory.get("memory_queries", {})
    list_objects_areas = factory.get("list_objects_areas", {})

    lines = [
        "",
        f"def {name}() -> MockResultProvider:",
        f'    """{desc}"""',
        "    p = MockResultProvider()",
        "    _apply_common_defaults(p)",
    ]

    if memory_queries:
        items_map = {}
        for key, var_names in memory_queries.items():
            items_map[key] = var_names
        lines.append("    p.set_conditional(")
        lines.append('        "spatial_memory_query_vec",')
        lines.append("        _memory_query_handler({")
        for key, var_names in memory_queries.items():
            vars_str = ", ".join(var_names)
            lines.append(f'            "{key}": [{vars_str}],')
        lines.append("        }),")
        lines.append("    )")

    if list_objects_areas:
        lines.append("    p.set_conditional(")
        lines.append('        "list_objects",')
        lines.append("        _list_objects_handler({")
        for key, var_name in list_objects_areas.items():
            lines.append(f'            "{key}": {var_name},')
        lines.append("        }),")
        lines.append("    )")

    lines.append("    return p")
    lines.append("")
    return lines


def _sanitize_json_booleans(val: str) -> str:
    """Convert JSON-style true/false/null to Python True/False/None in inline literals."""
    import re
    val = re.sub(r'\btrue\b', 'True', val)
    val = re.sub(r'\bfalse\b', 'False', val)
    val = re.sub(r'\bnull\b', 'None', val)
    return val


def _generate_failure_factory(factory: dict, mock_meta: dict) -> list[str]:
    name = factory["name"]
    desc = factory.get("description", "")
    memory_queries = factory.get("memory_queries", {})
    sequence_overrides = factory.get("sequence_overrides", {})

    lines = [
        "",
        f"def {name}() -> MockResultProvider:",
        f'    """{desc}"""',
        "    p = MockResultProvider()",
        "    _apply_common_defaults(p)",
    ]

    if memory_queries:
        lines.append("    p.set_conditional(")
        lines.append('        "spatial_memory_query_vec",')
        lines.append("        _memory_query_handler({")
        for key, var_names in memory_queries.items():
            vars_str = ", ".join(var_names)
            lines.append(f'            "{key}": [{vars_str}],')
        lines.append("        }),")
        lines.append("    )")

    for tool_name, seq_values in sequence_overrides.items():
        lines.append(f'    p.set_sequence("{tool_name}", [')
        for val in seq_values:
            if val in ("MOVE_FAILED", "MOVE_SUCCESS", "GRASP_OUT_OF_RANGE",
                       "GRASP_FAILED", "FOLD_ITEM_FAILED"):
                lines.append(f"        {val},")
            elif val.startswith("[") or val.startswith("{"):
                sanitized = _sanitize_json_booleans(val)
                lines.append(f"        {sanitized},")
            else:
                first_memory_item = None
                for mi in mock_meta.get("memory_items", []):
                    if mi["data"]["type"] == "object":
                        first_memory_item = mi["var_name"]
                        break
                if val == "first_object_item" and first_memory_item:
                    lines.append(f"        [{first_memory_item}],")
                else:
                    lines.append(f"        {val},")
        lines.append("    ])")

    lines.append("    return p")
    lines.append("")
    return lines


def _write_mock_server(
    mock_meta: dict, all_tools: list[str], scene_data: dict, output_path: Path
) -> None:
    """Generate mock_server.py that mirrors server.py tool interfaces."""
    all_factories = []
    for f in mock_meta.get("mock_factories", []):
        all_factories.append(f["name"])
    for f in mock_meta.get("failure_factories", []):
        all_factories.append(f["name"])

    scene_display_name = scene_data["scene_display_name"]
    imports = ", ".join(f"\n    {n}" for n in sorted(set(all_factories)))

    content = f'''\
"""Mock MCP Server for Layer 2 e2e tests.

Exposes the same tool interfaces as server.py but routes all calls through
MockResultProvider, giving tests full control over tool return values.

Usage:
    python mock_server.py [--port 8000] [--control-port 8001] [--scenario default]
"""

import argparse
import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent))

from mock_tool_results import (
    MockResultProvider,{imports}
)

logger = logging.getLogger(__name__)

SCENARIOS: dict[str, Any] = {{
'''

    for f in mock_meta.get("mock_factories", []):
        content += f'    "{f["name"].replace("_mock", "")}": {f["name"]},\n'
    for f in mock_meta.get("failure_factories", []):
        content += f'    "{f["name"].replace("_mock", "")}": {f["name"]},\n'

    first_factory = all_factories[0] if all_factories else "lambda: MockResultProvider()"
    first_scenario = mock_meta.get("mock_factories", [{}])[0].get("name", "").replace("_mock", "") or "default"

    content += f'''\
}}

_provider: MockResultProvider = {first_factory}()
_provider_lock = threading.Lock()
_scenario_name: str = "{first_scenario}"


def _set_provider(scenario: str) -> None:
    global _provider, _scenario_name
    factory = SCENARIOS.get(scenario)
    if not factory:
        raise ValueError(f"Unknown scenario: {{scenario}}. Available: {{list(SCENARIOS)}}")
    with _provider_lock:
        _provider = factory()
        _scenario_name = scenario


def _call(tool_name: str, args: dict) -> Any:
    with _provider_lock:
        result_str = _provider.get(tool_name, args)
    if isinstance(result_str, str):
        try:
            return json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            return result_str
    return result_str


class _ControlHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/scenario":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            try:
                _set_provider(body["scenario"])
                self._respond(200, {{"ok": True, "scenario": body["scenario"]}})
            except (ValueError, KeyError) as exc:
                self._respond(400, {{"error": str(exc)}})
        else:
            self._respond(404, {{"error": "not found"}})

    def do_GET(self):
        if self.path == "/scenario":
            self._respond(200, {{"scenario": _scenario_name}})
        else:
            self._respond(404, {{"error": "not found"}})

    def _respond(self, code: int, body: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, fmt, *args):
        pass


def _start_control_server(port: int) -> HTTPServer:
    server = HTTPServer(("0.0.0.0", port), _ControlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Control server listening on port %d", port)
    return server


mcp = FastMCP(name="TianGong {scene_display_name} Mock MCP Server")
'''

    # Add tool stubs
    extra_tools = scene_data.get("extra_tools", [])

    content += _generate_mock_server_tools(all_tools, extra_tools)

    content += f'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock MCP Server for e2e tests")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--control-port", type=int, default=8001)
    parser.add_argument("--scenario", default="{first_scenario}")
    args = parser.parse_args()

    _set_provider(args.scenario)
    _start_control_server(args.control_port)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=args.port, log_level="INFO")
'''

    output_path.write_text(content, encoding="utf-8")
    logger.info("Wrote %s", output_path)


def _generate_mock_server_tools(all_tools: list[str], extra_tools: list[str]) -> str:
    """Generate mock tool stubs that delegate to _call()."""
    base_tool_code = {
        "move": '''
@mcp.tool()
def move(dst_point: dict[str, float]) -> dict[str, str]:
    """Move robot to the specified destination.

    Args:
        dst_point: Destination point to move to, {x: float, y: float, yaw: float}.

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'failed reason or empty'}.
    """
    return _call("move", {"dst_point": dst_point})
''',
        "move_to": '''
@mcp.tool()
def move_to(obj_x: float, obj_y: float, coordinate_x: float, coordinate_y: float, coordinate_z: float, coordinate_yaw: float) -> dict[str, str]:
    """导航到目标点位

    Args:
        obj_x: x-coordinate of object
        obj_y: y-coordinate of object
        coordinate_x: x-coordinate of the target point
        coordinate_y: y-coordinate of the target point
        coordinate_z: z-coordinate of the target point
        coordinate_yaw: direction of the target point

    Returns:
        dict: Dict of "state" and "msg".
    """
    return _call("move_to", {"obj_x": obj_x, "obj_y": obj_y, "coordinate_x": coordinate_x, "coordinate_y": coordinate_y, "coordinate_z": coordinate_z, "coordinate_yaw": coordinate_yaw})
''',
        "grasp_start": '''
@mcp.tool()
def grasp_start(item: str) -> dict[str, str]:
    """Pick up the specified item.

    Args:
        item: The name of the item to pick up. Must be in English.

    Returns:
        dict: result message.
    """
    return _call("grasp_start", {"item": item})
''',
        "handover": '''
@mcp.tool()
def handover(hand: str = "auto", force_threshold: float = 5.0, timeout: float = 30.0, delay_after_release: float = 0.5, vel: float = 0.3, acc: float = 0.3) -> str:
    """Execute the full handover pipeline for the robot arm.

    Args:
        hand: Which hand to use.
        force_threshold: Force in Newtons to trigger release.
        timeout: Max seconds to wait.
        delay_after_release: Seconds to wait after release.
        vel: MoveIt velocity scaling factor.
        acc: MoveIt acceleration scaling factor.

    Returns:
        A human-readable status string.
    """
    return _call("handover", {"hand": hand, "force_threshold": force_threshold, "timeout": timeout, "delay_after_release": delay_after_release, "vel": vel, "acc": acc})
''',
        "spatial_memory_query_vec": '''
@mcp.tool()
def spatial_memory_query_vec(name: str, query: Optional[str] = None, color: Optional[str] = None, top_k: int = 10) -> list[dict]:
    """使用语义向量搜索查询空间记忆数据库。

    Args:
        name: 物品名称
        query: 查询语句
        color: 颜色
        top_k: 返回结果的最大数量

    Returns:
        list: 匹配的物品列表
    """
    return _call("spatial_memory_query_vec", {"name": name, "query": query, "color": color, "top_k": top_k})
''',
        "delete_spatial_memory_by_id": '''
@mcp.tool()
def delete_spatial_memory_by_id(id_list: list[str]) -> dict[str, int]:
    """删除空间记忆文档。

    Args:
        id_list: 要删除的文档 _id 列表。

    Returns:
        dict: 删除结果
    """
    return _call("delete_spatial_memory_by_id", {"id_list": id_list})
''',
        "scene_recognition": '''
@mcp.tool()
def scene_recognition(target: str) -> dict[str, str]:
    """判断当前图片是否包含指定名称

    Args:
        target: 物品名称

    Returns:
        dict: result message.
    """
    return _call("scene_recognition", {"target": target})
''',
        "trigger_spatial_processing": '''
@mcp.tool()
def trigger_spatial_processing(obj_name: str) -> dict[str, str]:
    """手动触发空间记忆数据处理。

    Args:
        obj_name: 待更新的物品类别

    Returns:
        dict: 处理结果
    """
    return _call("trigger_spatial_processing", {"obj_name": obj_name})
''',
        "perception_custom": '''
@mcp.tool()
def perception_custom() -> dict[str, str]:
    """自定义感知工具。

    Returns:
        dict: result message.
    """
    return _call("perception_custom", {})
''',
        "place_down": '''
@mcp.tool()
def place_down() -> dict[str, str]:
    """将物品放置在目标点位

    Returns:
        dict: result message.
    """
    return _call("place_down", {})
''',
        "fold_item": '''
@mcp.tool()
def fold_item(item: str) -> dict[str, str]:
    """折叠指定的软性物品。

    Args:
        item: 要折叠的物品名称，英文

    Returns:
        dict: result message.
    """
    return _call("fold_item", {"item": item})
''',
        "tidy_surface": '''
@mcp.tool()
def tidy_surface(surface: str, items: Optional[list[str]] = None) -> dict[str, str]:
    """整理指定表面或区域上的物品。

    Args:
        surface: 目标表面或区域名称
        items: 可选，需要整理的物品名称列表

    Returns:
        dict: result message.
    """
    return _call("tidy_surface", {"surface": surface, "items": items})
''',
        "list_objects": '''
@mcp.tool()
def list_objects(area: str) -> dict[str, Any]:
    """扫描并列出指定区域内的所有可见物品。

    Args:
        area: 要扫描的区域名称

    Returns:
        dict: 包含 'state' 和 'objects' 的字典。
    """
    return _call("list_objects", {"area": area})
''',
        "push_item": '''
@mcp.tool()
def push_item(item: str, direction: str, distance: float = 0.1) -> dict[str, str]:
    """在当前位置推移物品到指定方向。

    Args:
        item: 要推移的物品名称，英文
        direction: 推移方向
        distance: 推移距离（米）

    Returns:
        dict: result message.
    """
    return _call("push_item", {"item": item, "direction": direction, "distance": distance})
''',
    }

    # Simple stubs for tools not worth full docstrings
    simple_tools = {
        "place_start": ('position: str', '{"position": position}', 'dict[str, str]'),
        "place": ('position: Optional[list[float]] = None, object_name: Optional[str] = None, drop: bool = True', '{"position": position, "object_name": object_name, "drop": drop}', 'dict[str, str]'),
        "reach_out": ('position: list[float], object_name: Optional[str] = None', '{"position": position, "object_name": object_name}', 'dict[str, str]'),
        "release_hand": ('', '{}', 'dict[str, str]'),
        "adjust_body_height": ('expected_height: float', '{"expected_height": expected_height}', 'dict[str, str]'),
        "get_robot_pose": ('', '{}', 'dict[str, str]'),
        "back_station": ('', '{}', 'dict[str, str]'),
        "parameter_store": ('action: str, key: Optional[str] = None, value: Optional[Any] = None, value_type: Optional[str] = None, params: Optional[dict] = None, param_types: Optional[dict[str, str]] = None', '{"action": action, "key": key, "value": value, "value_type": value_type, "params": params, "param_types": param_types}', 'dict[str, str]'),
        "fetch_box": ('target: str', '{"target": target}', 'dict[str, str]'),
        "put_box": ('target: str', '{"target": target}', 'dict[str, str]'),
    }

    result = "\n"
    for tool_name in all_tools:
        if tool_name in base_tool_code:
            result += base_tool_code[tool_name]
        elif tool_name in simple_tools:
            params, args_dict, ret_type = simple_tools[tool_name]
            result += f'''
@mcp.tool()
def {tool_name}({params}) -> {ret_type}:
    """Mock {tool_name}."""
    return _call("{tool_name}", {args_dict})
'''
        else:
            result += f'''
@mcp.tool()
def {tool_name}(**kwargs) -> dict[str, str]:
    """Mock {tool_name}."""
    return _call("{tool_name}", kwargs)
'''
    return result
