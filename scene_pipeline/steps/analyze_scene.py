"""Step 2: Analyze scene photo using Vision LLM."""

import json
import logging
from pathlib import Path

from ..llm_client import LLMClient
from ..prompts.scene_analysis import SCENE_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


def analyze_scene(client: LLMClient, image_path: Path) -> dict:
    """Analyze a scene photo and return structured scene data.

    Returns a dict with keys: scene_type, scene_name, scene_display_name,
    layout_description, furniture, objects, patrol_points, extra_tools.

    ``extra_tools`` now only lists capabilities that are NOT covered by
    the predefined skills in refinev1_updated_with_io.yaml.  The bulk of
    scene-specific tools come from the YAML skill registry instead.
    """
    logger.info("Analyzing scene from %s ...", image_path.name)

    result = client.vision_json(SCENE_ANALYSIS_PROMPT, image_path)

    _validate_scene_data(result)

    logger.info(
        "Scene analysis complete: type=%s, name=%s, %d furniture, %d objects",
        result.get("scene_type"),
        result.get("scene_name"),
        len(result.get("furniture", [])),
        len(result.get("objects", [])),
    )
    return result


def _validate_scene_data(data: dict) -> None:
    required_keys = [
        "scene_type", "scene_name", "scene_display_name",
        "furniture", "objects",
    ]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Scene analysis missing required key: {key}")

    if len(data.get("furniture", [])) < 1:
        raise ValueError("Scene analysis must include at least 1 furniture item")
    if len(data.get("objects", [])) < 3:
        raise ValueError("Scene analysis must include at least 3 objects")

    if "patrol_points" not in data or len(data["patrol_points"]) < 2:
        data["patrol_points"] = _generate_default_patrol_points()

    # extra_tools now only lists capabilities NOT covered by predefined
    # skills.  Default to empty — the skill registry provides scene tools.
    if "extra_tools" not in data:
        data["extra_tools"] = []


def _generate_default_patrol_points() -> list[dict]:
    return [
        {"name": "巡逻点一", "description": "区域A", "x": 1.0, "y": -2.0, "yaw": -1.57},
        {"name": "巡逻点二", "description": "区域B", "x": 2.0, "y": -1.0, "yaw": 0.0},
        {"name": "巡逻点三", "description": "区域C", "x": 0.5, "y": -4.0, "yaw": -2.5},
    ]
