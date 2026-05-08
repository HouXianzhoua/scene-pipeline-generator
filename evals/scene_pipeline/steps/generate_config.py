"""Step 5: Generate scene YAML configuration."""

import logging
from pathlib import Path

import yaml

from ..llm_client import LLMClient
from ..prompts.config_gen import SYSTEM_PROMPT_GEN

logger = logging.getLogger(__name__)


def generate_config(
    client: LLMClient,
    scene_data: dict,
    all_tools: list[str],
    output_path: Path,
    *,
    predefined_skills: list[dict] | None = None,
    scene_category: str = "",
) -> str:
    """Generate scene YAML config and return the system_prompt text."""
    scene_name = scene_data["scene_name"]
    scene_display_name = scene_data["scene_display_name"]

    system_prompt = _generate_system_prompt(
        client, scene_data, all_tools,
        predefined_skills=predefined_skills,
        scene_category=scene_category,
    )

    config = {
        "version": "0.0.1",
        "msg_client": {
            "type": "ComposeClient",
            "maxsize": 1,
            "client_list": [
                {
                    "type": "VoiceSocketClient",
                    "uri": "ws://localhost:8765",
                    "voice_path": "/voice",
                    "tts_path": "/ttsplay",
                    "maxsize": 1,
                    "connect_timeout": 3.0,
                    "enable_input": True,
                    "enable_output": True,
                },
                {
                    "type": "HuisikaiwuMsgClient",
                    "host": "0.0.0.0",
                    "port": 8164,
                    "result_url": "http://localhost:9091/v1/api/chat/multiResult",
                    "maxsize": 1,
                    "enable_input": True,
                    "enable_output": True,
                    "agent_name": "tiangong",
                    "agent_uuid": f"agent_{scene_name}",
                },
            ],
        },
        "system_prompt": system_prompt,
        "tools": [
            {"type": "BoChaSearch"},
            {"type": "HeFengWeather"},
        ],
        "model": {
            "type": "ChatOpenAI",
            "model": "{{ model }}",
            "api_key": "{{ api_key }}",
            "base_url": "{{ base_url }}",
            "temperature": 0,
        },
        "mcp_cfg": {
            scene_name: {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
                "sse_read_timeout": 1800,
            },
        },
        "graph_creator": {
            "type": "ReactAgentCreator",
            "model": "{{ model_ref }}",
            "tools": "{{ tools_ref }}",
            "mcp_cfg": "{{ mcp_cfg_ref }}",
            "system_prompt": "{{ system_prompt_ref }}",
            "parallel_tool_calls": True,
            "enable_memory": True,
        },
        "brain_agent": {
            "type": "GraphBrainAgent",
            "msg_client": "{{ msg_client }}",
            "graph_creator": "{{ graph_creator_ref }}",
            "system_prompt": "{{ system_prompt_ref }}",
            "accept_msg_when_busy": True,
            "max_concurrency": 1,
        },
    }

    _write_yaml_with_anchors(config, system_prompt, scene_name, output_path)

    logger.info("Wrote %s", output_path)
    return system_prompt


def _generate_system_prompt(
    client: LLMClient,
    scene_data: dict,
    all_tools: list[str],
    *,
    predefined_skills: list[dict] | None = None,
    scene_category: str = "",
) -> str:
    from ..skill_registry import format_skills_for_prompt

    furniture_list = ", ".join(
        f"{f['name']}({f.get('position', '')})"
        for f in scene_data.get("furniture", [])
    )
    objects_list = ", ".join(
        f"{o['name']}({o.get('color', '')}, {o.get('position', '')})"
        for o in scene_data.get("objects", [])
    )
    patrol_points = "; ".join(
        f"{p['name']}: x={p['x']}, y={p['y']}, yaw={p['yaw']}"
        for p in scene_data.get("patrol_points", [])
    )

    skills_desc = "（无预定义技能）"
    if predefined_skills:
        skills_desc = format_skills_for_prompt(predefined_skills)

    prompt = SYSTEM_PROMPT_GEN.format(
        scene_type=scene_data["scene_type"],
        scene_display_name=scene_data["scene_display_name"],
        scene_category=scene_category or "通用",
        layout_description=scene_data.get("layout_description", ""),
        furniture_list=furniture_list,
        objects_list=objects_list,
        patrol_points=patrol_points,
        all_tools=", ".join(all_tools),
        predefined_skills_description=skills_desc,
    )

    messages = [{"role": "user", "content": prompt}]
    raw = client.chat(messages)

    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def _write_yaml_with_anchors(
    config: dict, system_prompt: str, scene_name: str, output_path: Path
) -> None:
    """Write YAML with proper anchors matching the real_home_living format."""
    lines = [
        "# Auto-generated by scene_pipeline",
        "version: 0.0.1",
        "",
        "# required",
        "msg_client:",
        "  type: ComposeClient",
        "  maxsize: 1",
        "  client_list:",
        "    - type: VoiceSocketClient",
        "      uri: ws://localhost:8765",
        "      voice_path: /voice",
        "      tts_path: /ttsplay",
        "      maxsize: 1",
        "      connect_timeout: 3.0",
        "      enable_input: true",
        "      enable_output: true",
        "    - type: HuisikaiwuMsgClient",
        "      host: 0.0.0.0",
        "      port: 8164",
        "      result_url: http://localhost:9091/v1/api/chat/multiResult",
        "      maxsize: 1",
        "      enable_input: true",
        "      enable_output: true",
        "      agent_name: tiangong",
        f"      agent_uuid: agent_{scene_name}",
        "",
        "system_prompt: &system_prompt |",
    ]

    for line in system_prompt.split("\n"):
        lines.append(f"  {line}" if line.strip() else "")

    lines.extend([
        "",
        "tools: &tools",
        "  - type: BoChaSearch",
        "  - type: HeFengWeather",
        "",
        "model: &model",
        "  type: ChatOpenAI",
        '  model: "{{ model }}"',
        '  api_key: "{{ api_key }}"',
        '  base_url: "{{ base_url }}"',
        "  temperature: 0",
        "",
        "mcp_cfg: &mcp_cfg",
        f"  {scene_name}:",
        "    url: http://localhost:8000/mcp",
        "    transport: streamable_http",
        "    sse_read_timeout: 1800",
        "",
        "graph_creator: &graph_creator",
        "  type: ReactAgentCreator",
        "  model: *model",
        "  tools: *tools",
        "  mcp_cfg: *mcp_cfg",
        "  system_prompt: *system_prompt",
        "  parallel_tool_calls: true",
        "  enable_memory: true",
        "",
        "# required",
        "brain_agent:",
        "  type: GraphBrainAgent",
        '  msg_client: "{{ msg_client }}"',
        "  graph_creator: *graph_creator",
        "  system_prompt: *system_prompt",
        "  accept_msg_when_busy: true",
        "  max_concurrency: 1",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
