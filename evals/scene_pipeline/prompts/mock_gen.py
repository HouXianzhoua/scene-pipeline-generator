"""Prompt template for generating mock tool results."""

MOCK_DATA_PROMPT = """\
你是一个测试数据生成专家。根据场景信息，生成 mock_tool_results.py 所需的测试数据。

## 场景信息
- 场景：{scene_display_name}
- 家具：{furniture_json}
- 物品：{objects_json}
- 用户指令：{commands_json}
- 故障场景：{failures_json}

## 要求

为每个物品和家具生成 MEMORY_ITEM 常量。为每条用户指令生成一个 mock 工厂函数。

## 标准工具序列参考

生成 mock 时，请确保 memory_queries 覆盖以下标准序列中所需的查询：

- 拿取递送: spatial_memory_query_vec → move_to → scene_recognition → grasp_start → perception_custom → handover
- 放置移动: spatial_memory_query_vec(物品) → move_to → scene_recognition → grasp_start → spatial_memory_query_vec(目标位置) → move_to → place_down
- 收纳归集: fetch_box → spatial_memory_query_vec(区域) → move_to → [list_objects →] grasp_start → put_box
- 整理折叠: spatial_memory_query_vec → move_to → fold_item / tidy_surface
- 查询检索: spatial_memory_query_vec 或 list_objects

对于放置移动类任务，memory_queries 必须同时包含物品和目标位置的查询条目。

## 返回 JSON 格式

{{
    "memory_items": [
        {{
            "var_name": "MEMORY_ITEM_YELLOW_CUP",
            "data": {{
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
                "similarity_score": 0.90
            }}
        }}
    ],
    "list_objects_data": [
        {{
            "var_name": "LIST_OBJECTS_COFFEE_TABLE",
            "area_keys": ["coffee_table", "茶几"],
            "data": {{
                "state": "succeed",
                "objects": [
                    {{"name": "杯子", "color": "黄色", "position": "左侧"}}
                ],
                "msg": "在 coffee_table 上检测到 N 个物品"
            }}
        }}
    ],
    "mock_factories": [
        {{
            "name": "fetch_cup_mock",
            "description": "Fetch yellow cup from coffee table and hand to user.",
            "memory_queries": {{
                "杯子": ["MEMORY_ITEM_YELLOW_CUP"],
                "茶几": ["MEMORY_ITEM_COFFEE_TABLE"]
            }},
            "list_objects_areas": {{}},
            "sequence_overrides": {{}},
            "for_command": "fetch_yellow_cup"
        }},
        {{
            "name": "place_remote_near_tissue_box_mock",
            "description": "Place remote near tissue box (requires memory queries for both remote and target).",
            "memory_queries": {{
                "遥控器": ["MEMORY_ITEM_REMOTE"],
                "纸巾盒": ["MEMORY_ITEM_TISSUE_BOX"],
                "茶几": ["MEMORY_ITEM_COFFEE_TABLE"]
            }},
            "list_objects_areas": {{}},
            "sequence_overrides": {{}},
            "for_command": "place_remote_near_tissue_box"
        }}
    ],
    "failure_factories": [
        {{
            "name": "grasp_fail_retry_mock",
            "description": "grasp_start fails on the first attempt (out of range), succeeds on retry.",
            "memory_queries": {{
                "杯子": ["MEMORY_ITEM_YELLOW_CUP"],
                "茶几": ["MEMORY_ITEM_COFFEE_TABLE"]
            }},
            "sequence_overrides": {{
                "grasp_start": ["GRASP_OUT_OF_RANGE", "{{\\"status\\": \\"success\\", \\"message\\": \\"执行任务: yellow cup\\"}}"]
            }},
            "for_failure": "grasp_fail"
        }},
        {{
            "name": "move_fail_retry_mock",
            "description": "move_to fails on the first attempt, succeeds on subsequent calls.",
            "memory_queries": {{
                "杯子": ["MEMORY_ITEM_YELLOW_CUP"],
                "茶几": ["MEMORY_ITEM_COFFEE_TABLE"]
            }},
            "sequence_overrides": {{
                "move_to": ["MOVE_FAILED", "MOVE_SUCCESS", "MOVE_SUCCESS"]
            }},
            "for_failure": "move_fail"
        }},
        {{
            "name": "memory_miss_search_mock",
            "description": "spatial_memory_query_vec returns empty first, then finds item.",
            "memory_queries": {{}},
            "sequence_overrides": {{
                "spatial_memory_query_vec": ["[]", "first_object_item", "first_object_item"]
            }},
            "for_failure": "memory_miss"
        }}
    ]
}}

注意：
- 坐标值在 -5.0 到 5.0 范围内，各物品间有区分度
- similarity_score 在 0.80 到 0.95 范围内
- 每条指令都需要对应一个 mock_factory
- memory_queries 的 key 是中文关键词，value 是 MEMORY_ITEM 变量名列表
- 家具的 type 是 "furniture"，物品的 type 是 "object"
- 故障场景至少 3 个：grasp_fail, move_fail, memory_miss
- 放置移动类的 mock_factory 的 memory_queries 必须同时包含物品和目标位置的条目
- move_fail_retry_mock 必须提供 memory_queries 以便模型查询物品位置
- 收纳归集类必须包含 list_objects_areas 以支持 list_objects 查询
"""
