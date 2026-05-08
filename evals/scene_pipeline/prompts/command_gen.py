"""Prompt template for generating user voice commands."""

COMMAND_GEN_PROMPT = """\
你是一个机器人语音交互测试专家。根据以下场景信息，生成用户可能对机器人下达的语音指令。

## 场景信息
- 场景类型：{scene_type}（{scene_display_name}）
- 家具列表：{furniture_list}
- 物品列表：{objects_list}
- 空间布局：{layout_description}

## 生成要求

按以下类别生成语音指令，总数不少于 {total_min} 条：

{categories_requirements}

另外添加 2-3 条**故障恢复场景**描述（抓取失败重试、导航失败、记忆未命中）。

## 标准工具序列（生成关键锚点，避免过拟合）

以下是每种任务类别对应的 expected_tools 锚点。expected_tools 用来描述任务完成的关键动作，不要把每个可接受的辅助步骤都写成唯一标准路径。`spatial_memory_query_vec`、`scene_recognition`、`delete_spatial_memory_by_id`、`trigger_spatial_processing`、`perception_custom` 属于常见辅助步骤，可在真实执行中出现；不要因为这些步骤多或少就改变任务语义。

### 拿取递送（fetch + handover）
expected_tools: ["spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start", "perception_custom", "handover"]
forbidden_tools: ["place_down"]
min_tool_calls: 6

### 放置移动（fetch + place）
expected_tools: ["spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start", "spatial_memory_query_vec", "move_to", "place_down"]
forbidden_tools: ["handover"]
min_tool_calls: 7

### 收纳归集
- 明确“收进盒子/收纳盒/箱子/盒中”时: expected_tools: ["fetch_box", "move_to", "grasp_start", "put_box"]
- 如果需要先了解区域物品: ["fetch_box", "move_to", "{list_tool}", "grasp_start", "put_box"]
- 仅表示“收在一起/归拢/集中好/摆到一起”且没有闭合容器时: expected_tools: ["spatial_memory_query_vec", "move_to", "{tidy_tool}"] 或 ["spatial_memory_query_vec", "move_to", "sort"]
forbidden_tools: ["handover"]
min_tool_calls: 4

### 整理折叠
- 折叠类: expected_tools: ["spatial_memory_query_vec", "move_to", "{fold_tool}"]
- 表面整理类: expected_tools: ["spatial_memory_query_vec", "move_to", "{tidy_tool}"]
- 铺展类: expected_tools: ["spatial_memory_query_vec", "move_to", "{fold_tool}", "place_down"]
forbidden_tools: ["handover", "grasp_start"]
min_tool_calls: 3

### 查询检索
- 查询物品位置: expected_tools: ["spatial_memory_query_vec"]
- 列举区域物品: expected_tools: ["{list_tool}"] 或 ["move_to", "{list_tool}"]
- 确认物品在某处: expected_tools: ["spatial_memory_query_vec"]
forbidden_tools: ["grasp_start", "place_down", "handover"]
min_tool_calls: 1

### 复合任务
根据任务子步骤组合上述标准序列。例如：
- "先拿XX再放到YY" → 放置移动序列
- "整理+折叠" → ["spatial_memory_query_vec", "move_to", "{fold_tool}", "{tidy_tool}"]
- "检查+收纳" → ["fetch_box", "move_to", "{list_tool}", "grasp_start", "put_box"]
- "检查+归拢/整理" 且没有闭合容器 → ["move_to", "{list_tool}", "{tidy_tool}"] 或 ["spatial_memory_query_vec", "move_to", "sort"]

## 返回 JSON 格式

{{
    "commands": [
        {{
            "category": "拿取递送",
            "text": "把茶几上的黄色杯子递给我",
            "target_object": "yellow_cup",
            "target_furniture": "coffee_table",
            "expected_tools": ["spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start", "perception_custom", "handover"],
            "min_tool_calls": 6,
            "forbidden_tools": ["place_down"],
            "test_name": "fetch_yellow_cup"
        }},
        {{
            "category": "放置移动",
            "text": "把遥控器放到纸巾盒旁边",
            "target_object": "remote_control",
            "target_furniture": "coffee_table",
            "expected_tools": ["spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start", "spatial_memory_query_vec", "move_to", "place_down"],
            "min_tool_calls": 7,
            "forbidden_tools": ["handover"],
            "test_name": "place_remote_near_tissue_box"
        }},
        {{
            "category": "收纳归集",
            "text": "把茶几上的纸张都收进收纳盒里",
            "target_object": "papers",
            "target_furniture": "coffee_table",
            "expected_tools": ["fetch_box", "move_to", "grasp_start", "put_box"],
            "min_tool_calls": 4,
            "forbidden_tools": ["handover"],
            "test_name": "store_papers_in_box"
        }},
        {{
            "category": "整理折叠",
            "text": "把沙发上的绿色毯子叠一下",
            "target_object": "green_blanket",
            "target_furniture": "sofa",
            "expected_tools": ["spatial_memory_query_vec", "move_to", "{fold_tool}"],
            "min_tool_calls": 3,
            "forbidden_tools": ["handover", "grasp_start"],
            "test_name": "fold_green_blanket"
        }},
        {{
            "category": "查询检索",
            "text": "手机在哪儿",
            "target_object": "smartphone",
            "target_furniture": "",
            "expected_tools": ["spatial_memory_query_vec"],
            "min_tool_calls": 1,
            "forbidden_tools": ["grasp_start", "place_down", "handover"],
            "test_name": "query_phone_location"
        }}
    ],
    "failure_scenarios": [
        {{
            "type": "grasp_fail",
            "text": "把茶几上的黄色杯子递给我",
            "description": "抓取失败后重试",
            "target_object": "yellow_cup",
            "expected_tools": ["spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start", "grasp_start", "perception_custom", "handover"],
            "test_name": "grasp_fail_retry"
        }},
        {{
            "type": "move_fail",
            "text": "把书架上的收音机递给我",
            "description": "导航失败后重新规划路径",
            "target_object": "radio",
            "expected_tools": ["spatial_memory_query_vec", "move_to", "move_to", "scene_recognition", "grasp_start", "perception_custom", "handover"],
            "test_name": "move_fail_replan"
        }},
        {{
            "type": "memory_miss",
            "text": "帮我找一下遥控器",
            "description": "空间记忆未命中，需要搜索",
            "target_object": "remote_control",
            "expected_tools": ["spatial_memory_query_vec"],
            "test_name": "memory_miss_search"
        }}
    ]
}}

## 可用工具列表（严格约束）

以下是机器人支持的全部工具，expected_tools 和 forbidden_tools 中只能使用这些名称：
{tools_list}

注意：
- 每条指令必须引用场景中实际存在的物品和家具
- 当总数较大时，必须通过不同物品、不同家具/区域、不同任务意图、不同口语表达扩展覆盖面；禁止只改同义词生成大量近重复指令
- 每个 category 内的 test_name 必须唯一，且尽量覆盖不同 target_object/target_furniture 组合
- expected_tools 和 forbidden_tools 只能从上方可用工具列表中选取，禁止自创工具名
- expected_tools 是预期的关键工具调用子序列（按调用顺序），不要把辅助查询/确认/删除记忆写成硬性唯一模板
- test_name 用英文小写下划线格式
- 拿取递送类必须包含 perception_custom 和 handover（在 grasp_start 之后）
- 放置移动类必须包含两次 spatial_memory_query_vec（抓取前和放置前各一次）和 place_down
- 收纳归集类只有在用户明确要求收进盒子/收纳盒/箱子时才必须包含 fetch_box 和 put_box；没有闭合容器的“收在一起/归拢/集中好”应按整理类处理
- 整理折叠类禁止使用 grasp_start，必须使用 {fold_tool} 或 {tidy_tool}
- 查询检索类禁止使用 grasp_start、place_down、handover
- 故障恢复场景至少 3 个：grasp_fail, move_fail, memory_miss
"""
