"""Prompt template for generating extra MCP tool definitions.

This prompt is only used as a **fallback** when the predefined skill registry
(refinev1_updated_with_io.yaml) does not cover a capability identified
during scene analysis.
"""

EXTRA_TOOLS_PROMPT = """\
你是一个 MCP Server 工具开发专家。根据场景需要，生成额外的 MCP 工具定义。

## 场景信息
- 场景名：{scene_display_name}
- 需要的额外工具：{extra_tools}
- 家具：{furniture_list}
- 物品：{objects_list}

## 已有工具（不需要生成，不要重复）
以下工具已经存在，包括平台基础工具和预定义场景技能，请勿重复生成：
{existing_tools}

## 需要生成的工具

对于以下每个工具，生成完整的 Python 函数定义（含 @mcp.tool() 装饰器、docstring、time.sleep、返回值）：
{extra_tools}

重要：这些工具是因为预定义技能库中没有覆盖才需要你来生成的。请确保：
1. 工具名不与已有工具重复
2. 函数签名使用类型标注
3. docstring 使用中文描述功能，Args 部分说明每个参数
4. 使用 time.sleep(10) 模拟执行耗时
5. 返回格式统一为 dict，包含 "status"/"state" 和 "message"/"msg"

## 参考格式
```python
@mcp.tool()
def fold_item(item: str) -> dict[str, str]:
    \"\"\"折叠指定的软性物品，如毯子、衣物等。

    机器人会使用双臂对物品进行对折操作，适用于毯子、毛巾等柔性物品。

    Args:
        item: 要折叠的物品名称，必须为英文，例如 'blanket', 'towel'

    Returns:
        dict: result message, {{'state': 'succeed or failed', 'msg': 'execution info'}}.
    \"\"\"
    time.sleep(15)
    return {{"state": "succeed", "msg": f"已完成 {{item}} 的折叠"}}
```

## 返回格式
返回一个 JSON 对象，包含 "tools" 数组，每个元素是一个工具定义：
{{
    "tools": [
        {{
            "name": "tool_name",
            "code": "@mcp.tool()\\ndef tool_name(...) -> dict[str, str]:\\n    ...(完整函数代码)"
        }}
    ]
}}
"""
