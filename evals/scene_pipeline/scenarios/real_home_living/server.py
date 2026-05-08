#
# Copyright (c) 2023~2025 Beijing Innovation Center of Humanoid Robotics. All rights reserved.
#
import logging
import time
from typing import Any, Optional

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(name="TianGong Real Home Living MCP Server")


@mcp.tool()
def move(dst_point: dict[str, float]) -> dict[str, str]:
    """Move robot to the specified destination.

    Args:
        dst_point: Destination point to move to, {x: float, y: float, yaw: float}.

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'failed reason or empty'}.
    """
    time.sleep(10)
    return {"state": "succeed", "msg": "已经移动到目的地"}


@mcp.tool()
def move_to(
    obj_x: float,
    obj_y: float,
    coordinate_x: float,
    coordinate_y: float,
    coordinate_z: float,
    coordinate_yaw: float,
) -> dict[str, str]:
    """导航到目标点位

    Args:
        obj_x: x-coordinate of object
        obj_y: y-coordinate of object
        coordinate_x: x-coordinate of the target point
        coordinate_y: y-coordinate of the target point
        coordinate_z: z-coordinate of the target point
        coordinate_yaw: direction of the target point

    Returns:
        dict: Dict of "state" and "msg", where "state" is "succeed" or "failed", and "msg" is the error message.
    """
    time.sleep(10)
    return {"state": "succeed", "msg": "已经移动到目的地"}


@mcp.tool()
def grasp_start(item: str) -> dict[str, str]:
    """Pick up the specified item.

    Args:
        item: The name of the item to pick up. Must be in English, e.g. 'bottle', 'yellow bottle', 'red toy'.

    Returns:
        dict: result message, {'status': 'success or failed', 'message': 'execution info'}.
    """
    time.sleep(15)
    return {"status": "success", "message": f"执行任务: {item}"}


@mcp.tool()
def place_start(position: str) -> dict[str, str]:
    """place object at the given position.

    Args:
        position: The target position to place the item. Allowed values are
                  'top_right', 'bottom_right', 'top_left', 'bottom_left'.

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'failed reason or empty'}.
    """
    time.sleep(10)
    return {"state": "succeed", "msg": ""}


@mcp.tool()
def place(
    position: Optional[list[float]] = None,
    object_name: Optional[str] = None,
    drop: bool = True,
) -> dict[str, str]:
    """Place the grasped object on the given position.

    Args:
        position: The target position to place, with its three items representing x, y, z in space.
        object_name: the object name which would be dropped.
        drop: the flag which represents if releasing hand fingers to throw object away.
              here, please keep it always be True.

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'failed reason or empty'}.
    """
    time.sleep(10)
    return {"state": "succeed", "msg": ""}


@mcp.tool()
def reach_out(
    position: list[float],
    object_name: Optional[str] = None,
) -> dict[str, str]:
    """Deliver the grasped object to the given position, but without throwing it away.

    Args:
        position: The target position to deliver, with its three items representing x, y, z in space.
        object_name: the object name which would be delivered.

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'failed reason or empty'}.
    """
    time.sleep(10)
    return {"state": "succeed", "msg": ""}


@mcp.tool()
def release_hand() -> dict[str, str]:
    """Open the robot's finger fully

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'failed reason or empty'}.
    """
    time.sleep(3)
    return {"state": "succeed", "msg": ""}


@mcp.tool()
def adjust_body_height(expected_height: float) -> dict[str, str]:
    """Adjust the height of robot body vertically.

    Args:
        expected_height: one expected robot vertical height value based on navigation ground

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'failed reason or empty'}.
    """
    time.sleep(5)
    return {"state": "succeed", "msg": ""}


@mcp.tool()
def handover(
    hand: str = "auto",
    force_threshold: float = 5.0,
    timeout: float = 30.0,
    delay_after_release: float = 0.5,
    vel: float = 0.3,
    acc: float = 0.3,
) -> str:
    """Execute the full handover pipeline for the robot arm.

    Steps:
      1. Auto-detect which hand holds an object (or use the specified hand).
      2. Move the arm to the handover pose (MoveIt Cartesian planning).
      3. Wait for the human to pull the object (force-triggered release).
      4. Return the arm to the home position (MoveIt joint planning).
      5. Clean up all ROS2 resources.

    Args:
        hand: Which hand to use – 'auto' (default) reads from grasp context, or 'left' / 'right' to override.
        force_threshold: Force in Newtons to trigger release (default 5.0).
        timeout: Max seconds to wait for force release (default 30.0).
        delay_after_release: Seconds to wait after release before returning the arm (default 0.5).
        vel: MoveIt velocity scaling factor (0–1, default 0.3).
        acc: MoveIt acceleration scaling factor (0–1, default 0.3).

    Returns:
        A human-readable status string starting with SUCCESS or ERROR.
    """
    time.sleep(10)
    return f"SUCCESS: {hand} hand – force-release detected, arm returned home."


@mcp.tool()
def spatial_memory_query_vec(
    name: str,
    query: Optional[str] = None,
    color: Optional[str] = None,
    top_k: int = 10,
) -> list[dict]:
    """使用语义向量搜索查询空间记忆数据库，返回最匹配的物品信息。

    该工具使用 BGE 模型进行语义理解，支持三重匹配。

    输入要求：
    ** 输入参数必须是中文**
    - name (必需): 物品名称，例如：苹果、瓶子、手机、杯子
    - query (可选): 查询语句，例如：放在桌子上新的红色的杯子
    - color (可选): 颜色，精确匹配，例如：红色、蓝色

    注意事项：
    1、name不要出现出物体名称以外的其他属性， 如：颜色，材质等

    输出
    - 成功时返回匹配的物品列表
    - 失败时返回 {"error": "错误信息"}

    示例：
    - name="瓶子", query="放在桌子上" -> 查找桌子上的瓶子
    - name="苹果", color="红色" -> 查找红色苹果
    - name="手机" -> 查找所有手机

    Args:
        name: 物品名称，用于语义匹配。例如：苹果、瓶子、手机、杯子
        query: 查询语句，用于语义匹配。例如：放在桌子上、新的、红色的
        color: 颜色，用于精确匹配。例如：red、红色、blue、蓝色
        top_k: 返回结果的最大数量，默认 10

    Returns:
        list: 匹配的物品列表
    """
    return [
        {
            "_id": "693bcc798b147215d15f7a24",
            "type": "object",
            "name": name,
            "obj_x": 1.41,
            "obj_y": -4.33,
            "coordinate_x": 1.83,
            "coordinate_y": -3.77,
            "coordinate_z": 0.65,
            "coordinate_yaw": -2.22,
            "property": "示例物品",
            "status": "放置在桌面上",
            "color": color or "未知",
            "desp": f"一个{name}",
            "similarity_score": 0.85,
        }
    ]


@mcp.tool()
def delete_spatial_memory_by_id(id_list: list[str]) -> dict[str, int]:
    """根据提供的 _id 列表，从保存空间记忆的 MongoDB 集合中永久删除对应的文档。

    输入要求：
    - 输入参数名为 "id_list"，必须是 _id 字符串列表
    - 示例：id_list: ["693bcc798b147215d15f7a24", "693bcc798b147215d15f7a25"]

    输出：
    - 成功时返回 {"deleted_count": N}
    - 失败时返回 {"error": "错误信息"}

    Args:
        id_list: 要删除的文档 _id 列表。

    Returns:
        dict: 删除结果
    """
    return {"deleted_count": len(id_list)}


@mcp.tool()
def scene_recognition(target: str) -> dict[str, str]:
    """判断当前图片是否包含指定名称

    Args:
        target: 物品名称

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'result info'}.
    """
    time.sleep(5)
    return {"state": "succeed", "msg": f"当前位置有{target}"}


@mcp.tool()
def trigger_spatial_processing(obj_name: str) -> dict[str, str]:
    """手动触发空间记忆数据处理，从队列获取最新数据进行物体检测和空间记忆构建。

    功能：
    - 从空间记忆系统队列获取最新采集的图像数据
    - 对图像进行物体检测和坐标转换
    - 存储到空间记忆数据库

    注意事项：
    - 该工具从队列获取最新数据，无需手动指定图像路径
    - 确保空间记忆系统正在运行
    - 如果队列为空，会等待指定时间（5秒）后返回超时

    Args:
        obj_name: 待更新的物品类别, 例如apple, bottle, orange, table等, 仅支持英文

    Returns:
        dict: 处理结果
    """
    time.sleep(5)
    return {"state": "succeed", "msg": f"已完成空间感知处理，检测到 {obj_name} 相关物品"}


@mcp.tool()
def get_robot_pose() -> dict[str, str]:
    """获取当前机器人的位置

    Returns:
        dict: Dict of "state" and "msg", where "state" is "succeed" or "failed", and "msg" is the error message.
    """
    return {
        "state": "succeed",
        "msg": '{"x": 0.0, "y": 0.0, "yaw": 0.0}',
    }


@mcp.tool()
def place_down() -> dict[str, str]:
    """将物品放置在目标点位

    Returns:
        dict: Dict of "state" and "msg", where "state" is "succeed" or "failed", and "msg" is the error message.
    """
    time.sleep(10)
    return {"state": "succeed", "msg": ""}


@mcp.tool()
def back_station() -> dict[str, str]:
    """返回充电桩，触发回桩操作

    Returns:
        dict: Dict of "state" and "msg", where "state" is "succeed" or "failed", and "msg" is the error message.
    """
    time.sleep(15)
    return {"state": "succeed", "msg": ""}


@mcp.tool()
def parameter_store(
    action: str,
    key: Optional[str] = None,
    value: Optional[Any] = None,
    value_type: Optional[str] = None,
    params: Optional[dict] = None,
    param_types: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """存储或获取参数值

    支持两种存储方式：
    1. 单个参数存储：使用key和value参数
    2. 批量参数存储：使用params参数传入字典

    Args:
        action: 要执行的操作，"batch_set"表示批量存储，"set"表示存储参数，"get"表示获取参数，"list"表示列出所有参数
        key: 单个参数的键名，当action为"set"或"get"时需要
        value: 单个参数的值，当action为"set"时需要
        value_type: 单个值的数据类型，支持"str", "int", "float", "bool", "list", "dict"
        params: 批量参数存储的字典，格式如{"key1": value1, "key2": value2}
        param_types: 批量参数类型字典，格式如{"key1": "dict", "key2": "str"}

    Returns:
        dict: Dict of "state" and "msg", where "state" is "succeed" or "failed", and "msg" is the result or error message.
    """
    if action == "set":
        return {"state": "succeed", "msg": f"参数 {key} 已存储"}
    elif action == "get":
        return {"state": "succeed", "msg": f"参数 {key} 的值为 null"}
    elif action == "batch_set":
        return {"state": "succeed", "msg": f"已批量存储 {len(params or {})} 个参数"}
    elif action == "list":
        return {"state": "succeed", "msg": "{}"}
    return {"state": "failed", "msg": f"未知操作: {action}"}


@mcp.tool()
def perception_custom() -> dict[str, str]:
    """自定义感知工具，通过订阅 ROS2 图像和位姿消息获取 RGB-D 数据并进行处理。

    功能：
    - 订阅 RGB 图像、深度图像和机器人位姿话题
    - 同步获取多传感器数据
    - 自动保存图像到 captured_images 目录
    - 调用自定义处理函数 (custom_process)

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'result info'}.
    """
    time.sleep(5)
    return {"state": "succeed", "msg": "已经移动到用户位置"}


@mcp.tool()
def fetch_box(target: str) -> dict[str, str]:
    """搬起指定位置的箱子

    Args:
        target: 搬起箱子的目标位置，如 "table"

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'failed reason or empty'}.
    """
    time.sleep(15)
    return {"state": "succeed", "msg": f"已从 {target} 搬起箱子"}


@mcp.tool()
def put_box(target: str) -> dict[str, str]:
    """将箱子放下到指定位置

    Args:
        target: 放置箱子的目标位置，如 "stack"

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'failed reason or empty'}.
    """
    time.sleep(10)
    return {"state": "succeed", "msg": f"已将箱子放到 {target}"}


# ---------------------------------------------------------------------------
# New tools for real home living scene
# ---------------------------------------------------------------------------


@mcp.tool()
def fold_item(item: str) -> dict[str, str]:
    """折叠指定的软性物品，如毯子、衣物等。

    机器人会使用双臂对物品进行对折操作，适用于毯子、毛巾等柔性物品。

    Args:
        item: 要折叠的物品名称，必须为英文，例如 'blanket', 'towel'

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'execution info'}.
    """
    time.sleep(15)
    return {"state": "succeed", "msg": f"已完成 {item} 的折叠"}


@mcp.tool()
def tidy_surface(
    surface: str,
    items: Optional[list[str]] = None,
) -> dict[str, str]:
    """整理指定表面或区域上的物品，使其排列整齐有序。

    机器人会对目标区域的物品进行归置和排列，可指定需要整理的物品列表，
    或不指定则整理该区域所有物品。

    Args:
        surface: 目标表面或区域名称，例如 'coffee_table', 'sofa', 'carpet'
        items: 可选，需要整理的物品名称列表（英文），不指定则整理所有物品

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'execution info'}.
    """
    time.sleep(10)
    item_desc = ", ".join(items) if items else "所有物品"
    return {"state": "succeed", "msg": f"已整理 {surface} 上的 {item_desc}"}


@mcp.tool()
def list_objects(area: str) -> dict[str, Any]:
    """扫描并列出指定区域内的所有可见物品。

    机器人使用视觉系统扫描目标区域，返回检测到的物品清单及其属性信息。

    Args:
        area: 要扫描的区域名称，例如 'coffee_table', 'sofa', 'carpet', 'floor'

    Returns:
        dict: 包含 'state' 和 'objects' 的字典，'objects' 为物品列表。
    """
    time.sleep(5)
    return {
        "state": "succeed",
        "objects": [
            {"name": "杯子", "color": "黄色", "position": "左侧"},
            {"name": "遥控器", "color": "黑色", "position": "中间"},
            {"name": "手机", "color": "黑色", "position": "右侧"},
        ],
        "msg": f"在 {area} 上检测到 3 个物品",
    }


@mcp.tool()
def push_item(
    item: str,
    direction: str,
    distance: float = 0.1,
) -> dict[str, str]:
    """在当前位置推移物品到指定方向，无需抓取。

    机器人使用手臂轻推物品，使其沿指定方向滑动一段距离。
    适用于需要微调物品位置的场景，如将物品往里推避免掉落。

    Args:
        item: 要推移的物品名称，必须为英文，例如 'cup', 'remote'
        direction: 推移方向，如 'inward', 'left', 'right', 'forward', 'backward'
        distance: 推移距离（米），默认 0.1

    Returns:
        dict: result message, {'state': 'succeed or failed', 'msg': 'execution info'}.
    """
    time.sleep(5)
    return {"state": "succeed", "msg": f"已将 {item} 向 {direction} 推移 {distance} 米"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000, log_level="INFO")
