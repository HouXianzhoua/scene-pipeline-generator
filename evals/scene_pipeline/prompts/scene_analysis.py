"""Prompt template for scene analysis via Vision LLM."""

SCENE_ANALYSIS_PROMPT = """\
你是一个场景分析专家。请仔细分析这张照片，提取以下信息并以 JSON 格式返回。

要求：
1. 识别场景类型（客厅/卧室/厨房/办公室等）
2. 列出所有家具（含位置描述和颜色）
3. 列出所有可见物品（含位置、颜色、放置状态）
4. 描述空间布局
5. 生成英文场景名（小写下划线格式）
6. 判断是否有预定义技能库无法覆盖的特殊能力需求

返回 JSON 格式如下（必须严格遵守）：
{
    "scene_type": "living_room",
    "scene_name": "real_living_room_01",
    "scene_display_name": "客厅",
    "layout_description": "这是一个典型的家庭客厅，中间有茶几，周围有沙发...",
    "furniture": [
        {
            "name": "茶几",
            "name_en": "coffee_table",
            "position": "房间中央",
            "color": "木色",
            "description": "一张木色的茶几，位于客厅中央"
        }
    ],
    "objects": [
        {
            "name": "黄色杯子",
            "name_en": "yellow_cup",
            "position": "茶几上",
            "color": "黄色",
            "on_furniture": "coffee_table",
            "graspable": true,
            "description": "一个黄色的杯子，放在茶几上"
        }
    ],
    "patrol_points": [
        {"name": "巡逻点一", "description": "茶几附近", "x": 1.0, "y": -2.0, "yaw": -1.57},
        {"name": "巡逻点二", "description": "沙发旁", "x": 2.0, "y": -1.0, "yaw": 0.0},
        {"name": "巡逻点三", "description": "门口", "x": 0.5, "y": -4.0, "yaw": -2.5}
    ],
    "extra_tools": []
}

注意：
- furniture 至少包含 2 个家具
- objects 至少包含 5 个物品
- 每个物品的 on_furniture 字段指向其所在家具的 name_en
- patrol_points 生成 3 个合理的巡逻点位，坐标在 -5.0 到 5.0 范围内
- scene_name 格式为 real_{场景英文}，如 real_living_room、real_kitchen
- scene_type 必须是以下之一：living_room, bedroom, kitchen, bathroom, dining_room, office, workspace, supermarket, store, factory, warehouse, workshop

关于 extra_tools：
- 系统已内置「通用技能 + 场景专用技能」预定义库，覆盖导航、抓取、放置、折叠、切割、开门、搅拌、扫描条码、焊接、组装等大量能力
- extra_tools 仅用于标注预定义技能库**无法覆盖**的特殊能力需求
- 如果场景中的所有操作都可以被常见的机器人技能覆盖，extra_tools 应为空数组 []
- 仅当照片中出现非常特殊的设备或操作需求时，才在 extra_tools 中填入工具名称（小写下划线格式）
"""
