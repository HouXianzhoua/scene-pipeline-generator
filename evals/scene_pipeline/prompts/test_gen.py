"""Prompt template for generating test cases."""

TEST_GEN_PROMPT = """\
你是一个测试工程师。根据用户指令列表和 mock 工厂函数，生成 pytest 测试用例。

## 用户指令
{commands_json}

## 故障场景
{failures_json}

## 可用的 mock 工厂函数
{mock_factories_json}

## 工具序列生成原则（生成锚点，不要过拟合单一路径）

生成测试时，不要把 expected_subsequence 设计成过于僵硬的“唯一标准路径”。目标是让 bad case 更贴近模型能力问题，而不是链路模板问题。

### 常见锚点序列
- FETCH_EXPECTED = ["spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start", "perception_custom", "handover"]
- PLACE_EXPECTED = ["spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start", "spatial_memory_query_vec", "move_to", "place_down"]

### 各类别生成要求
- **拿取递送**：给出能代表任务完成的锚点序列，优先覆盖 `spatial_memory_query_vec / move_to / grasp_start / handover`；如果指令包含“先看看/确认/检查”，允许前置 `list_objects` 或额外查询，不要强制写死双查询或 `scene_recognition/perception_custom`
- **放置移动**：给出能代表任务完成的锚点序列，优先覆盖 `spatial_memory_query_vec / move_to / grasp_start / place_down`；如果有“先看看/确认”，允许前置 `list_objects` 或额外查询
- **放置移动动作别名**：对于“插进花瓶/挂到挂钩/挂到杆上”等目标，`insert`、`hang` 可视为完成放置动作，不要强制只认 `place_down`
- **放置移动改写约束**：任何改写都不能丢失目标位置/参照物；如果改写后只剩“这个东西按我说的位置放一下”这类缺目标位置的话术，这种 case 不要生成
- **收纳归集**：只有当用户明确要求“收进盒子/收纳盒/放入盒中”时，才强约束 `fetch_box` 和 `put_box`；如果是“集中好/归拢到盒子旁边/整理到旁边”，不要强行套盒子收纳链路
- **收纳归集改写约束**：原句如果明确是“进盒/进收纳盒/进箱子”，改写后也必须保留这个闭合容器约束，不能改写成仅“收整好/归拢好”
- **整理折叠**：
  - 折叠类：优先 `fold_item`，但不要把“展开铺好”误写成纯折叠
  - 铺展类：允许 `fold_item + place_down`、`fold + place_down`、`unfold + place_down` 一类序列，不要只认一种工具名
  - 表面整理类：允许 `tidy_surface`、`align`、`sort`、`clean`、`tuck` 等语义接近的整理动作，不要因为工具名不同而过拟合
  - “靠稳/靠墙/扶正/固定”类安全整理允许 `push_item` 或 `push`
- **查询检索**：允许 `["spatial_memory_query_vec"]`、`["list_objects"]`、`["move_to", "list_objects"]` 这类轻量链路，不要给查询类强加抓取/放置
- **复合任务**：只保留关键子任务锚点，不要因为多一步确认/列举方式不同就写成极窄路径
  - “检查/查看 + 收进盒子/收纳盒” 必须落到 `fetch_box / list_objects / grasp_start / put_box` 一类收纳链路，不能写成 `handover`
  - “整理/清理区域 + 再移动物体” 允许前半段是 `sort / tidy_surface / clean` 等语义接近动作
  - “取下来递给我/拿给我 + 不要放别处”仍是递送任务，必须允许 `handover`，不能把 `handover` 放进 forbidden_tools
  - “拿到/搬到某处 + 然后整理”是混合任务，前半段允许 `grasp_start` 和 `place_down`，不能按纯整理任务禁止抓取/放置

### 故障恢复标准序列
- grasp_fail: ["spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start", "grasp_start", "perception_custom", "handover"]
  assert_min_calls: {{"grasp_start": 2}}
- move_fail: ["spatial_memory_query_vec", "move_to", "move_to", "scene_recognition", "grasp_start", "perception_custom", "handover"]
  assert_min_calls: {{"move_to": 2}}
- memory_miss: ["spatial_memory_query_vec"]
  min_calls: 1

## 要求

1. 每条指令对应一个测试用例
2. expected_subsequence 输出“关键锚点序列”即可，不要机械复制唯一标准路径
3. 对于拿取递送和放置移动类，可以使用 "FETCH_EXPECTED" 或 "PLACE_EXPECTED" 字符串标记，也可以输出更短、更贴近语义的列表
4. forbidden_tools 必须正确设置：只在语义互斥时设置；复合任务不要把某个子任务的必要工具误标为 forbidden
5. 参数测试隐含要求：复合任务中不要假设“第一个 move_to 就是目标物 move_to”，因此测试设计不要依赖这种假设

## 返回 JSON 格式

{{
    "test_classes": [
        {{
            "name": "TestFetchAndDeliver",
            "description": "Verify fetch-and-handover task planning for 拿取递送 commands.",
            "tests": [
                {{
                    "method_name": "test_fetch_yellow_cup",
                    "docstring": "把茶几上的黄色杯子递给我",
                    "user_message": "把茶几上的黄色杯子递给我",
                    "mock_factory": "fetch_cup_mock",
                    "expected_subsequence": "FETCH_EXPECTED",
                    "forbidden_tools": ["place_down"],
                    "min_calls": 6
                }}
            ]
        }},
        {{
            "name": "TestPlaceAndMove",
            "description": "Verify place-and-move task planning for 放置移动 commands.",
            "tests": [
                {{
                    "method_name": "test_place_remote_near_tissue_box",
                    "docstring": "把遥控器放到茶几右侧靠近纸巾盒的位置",
                    "user_message": "把遥控器放到茶几右侧靠近纸巾盒的位置",
                    "mock_factory": "place_remote_near_tissue_box_mock",
                    "expected_subsequence": "PLACE_EXPECTED",
                    "forbidden_tools": ["handover"],
                    "min_calls": 7
                }}
            ]
        }},
        {{
            "name": "TestCollectAndOrganize",
            "description": "Verify collection task planning for 收纳归集 commands.",
            "tests": [
                {{
                    "method_name": "test_store_papers_in_box",
                    "docstring": "把茶几上的纸张都收进收纳盒里",
                    "user_message": "把茶几上的纸张都收进收纳盒里",
                    "mock_factory": "store_papers_in_box_mock",
                    "expected_subsequence": ["fetch_box", "move_to", "grasp_start", "put_box"],
                    "forbidden_tools": ["handover"],
                    "min_calls": 4
                }}
            ]
        }},
        {{
            "name": "TestTidyAndFold",
            "description": "Verify tidying and folding task planning for 整理折叠 commands.",
            "tests": [
                {{
                    "method_name": "test_fold_green_blanket",
                    "docstring": "把沙发上的绿色毯子叠一下",
                    "user_message": "把沙发上的绿色毯子叠一下",
                    "mock_factory": "fold_green_blanket_mock",
                    "expected_subsequence": ["spatial_memory_query_vec", "move_to", "fold_item"],
                    "forbidden_tools": ["handover", "grasp_start"],
                    "min_calls": 3
                }}
            ]
        }},
        {{
            "name": "TestQuery",
            "description": "Verify query task planning for 查询检索 commands.",
            "tests": [
                {{
                    "method_name": "test_query_phone_location",
                    "docstring": "手机在哪儿",
                    "user_message": "手机在哪儿",
                    "mock_factory": "query_phone_location_mock",
                    "expected_subsequence": ["spatial_memory_query_vec"],
                    "forbidden_tools": ["grasp_start", "place_down", "handover"],
                    "min_calls": 1
                }}
            ]
        }}
    ],
    "failure_tests": [
        {{
            "method_name": "test_grasp_fail_retry",
            "docstring": "Grasp fails out of range on first attempt, then succeeds.",
            "user_message": "把茶几上的黄色杯子递给我",
            "mock_factory": "grasp_fail_retry_mock",
            "expected_subsequence": ["spatial_memory_query_vec", "move_to", "scene_recognition", "grasp_start", "grasp_start", "perception_custom", "handover"],
            "min_calls": 7,
            "assert_min_calls": {{"grasp_start": 2}}
        }},
        {{
            "method_name": "test_move_fail_replan",
            "docstring": "Navigation fails on first attempt, replans and succeeds.",
            "user_message": "把书架上的收音机递给我",
            "mock_factory": "move_fail_replan_mock",
            "expected_subsequence": ["spatial_memory_query_vec", "move_to", "move_to", "scene_recognition", "grasp_start", "perception_custom", "handover"],
            "min_calls": 7,
            "assert_min_calls": {{"move_to": 2}}
        }},
        {{
            "method_name": "test_memory_miss_search",
            "docstring": "Spatial memory miss, triggers search.",
            "user_message": "帮我找一下遥控器",
            "mock_factory": "memory_miss_search_mock",
            "expected_subsequence": ["spatial_memory_query_vec"],
            "min_calls": 1,
            "assert_min_calls": {{}}
        }}
    ]
}}

注意：
- 测试类按指令类别分组：TestFetchAndDeliver, TestPlaceAndMove, TestCollectAndOrganize, TestTidyAndFold, TestQuery, TestComplex, TestFailureRecovery
- 不要为了“更完整”而额外加入双查询、固定 `scene_recognition`、固定 `perception_custom` 等硬步骤，除非指令语义明确要求
- 收纳归集类只有在“明确入盒”时才必须包含 fetch_box 和 put_box
- 整理折叠类禁止使用 grasp_start，但允许 fold_item / fold / unfold / tidy_surface / align / sort / clean / tuck / push_item / push 等语义接近工具
- 查询检索类禁止使用 grasp_start、place_down、handover
- 复合任务如果语义是“查看/检查后收纳”，禁止把 `handover` 写成目标动作
- 复合任务如果包含“递给我/拿给我/拿过来”，不要禁止 `handover`；如果包含“拿到/移到/放到某处”，不要禁止 `place_down`
- 改写 case 只保留信息完整、目标约束未丢失的自然表达；宁可少，不要生成语义残缺 case
- 故障恢复的 grasp_fail 必须有 grasp_start 至少 2 次
- 故障恢复的 move_fail 必须有 move_to 至少 2 次
"""
