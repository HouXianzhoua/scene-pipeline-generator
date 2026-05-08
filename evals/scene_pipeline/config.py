"""Default configuration and constants for the scene pipeline."""

from __future__ import annotations

from enum import Enum


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


DEFAULT_GEN_MODEL = "gpt-5.5"
DEFAULT_GEN_BASE_URL = "https://api.openai.com/v1"

IMAGE_MAX_SIZE = 2048
IMAGE_QUALITY = 85
IMAGE_SIZE_THRESHOLD = 2 * 1024 * 1024

MAX_LLM_RETRIES = 2
LLM_REQUEST_TIMEOUT_SECONDS = 180
CODE_VALIDATION_RETRIES = 1
CODE_REPAIR_RETRIES = 2

BASE_TOOLS = [
    "move", "move_to", "grasp_start", "place_start", "place",
    "reach_out", "release_hand", "adjust_body_height", "handover",
    "spatial_memory_query_vec", "delete_spatial_memory_by_id",
    "scene_recognition", "trigger_spatial_processing",
    "get_robot_pose", "place_down", "back_station",
    "parameter_store", "perception_custom", "fetch_box", "put_box",
]

# Deprecated: extra tool candidates are now driven by the predefined skills
# in resource/refinev1_updated_with_io.yaml via skill_registry.py.
# Kept for backward compatibility only.
EXTRA_TOOL_CANDIDATES = [
    "fold_item", "tidy_surface", "list_objects", "push_item",
    "open_drawer", "close_drawer", "pour_water", "wipe_surface",
    "stack_items", "sort_items",
]


# SCENE_TYPE_TO_CATEGORY is defined in skill_registry.py.
# Import it from there directly: from .skill_registry import SCENE_TYPE_TO_CATEGORY

COMMAND_CATEGORIES = {
    "拿取递送": {"min_count": 5, "description": "把XX递给我、帮我拿XX"},
    "放置移动": {"min_count": 4, "description": "把XX放到YY"},
    "收纳归集": {"min_count": 3, "description": "把XX都收进YY"},
    "整理折叠": {"min_count": 3, "description": "叠一下XX、整理XX"},
    "查询检索": {"min_count": 4, "description": "XX在哪、桌上有什么"},
    "复合任务": {"min_count": 4, "description": "清理、分类、安全调整"},
}

DEFAULT_TARGET_CATEGORY_COUNTS = {
    # Designed for roughly 100-150 pytest cases after generated planning,
    # function-calling parameter, and response-quality tests are combined.
    # Keeping this under 100 commands avoids padding the suite with
    # low-diversity near-duplicates from one image.
    "拿取递送": 18,
    "放置移动": 16,
    "收纳归集": 12,
    "整理折叠": 10,
    "查询检索": 12,
    "复合任务": 12,
}

DEFAULT_PARAM_CASE_LIMIT = 18

# ---------------------------------------------------------------------------
# Scene complexity adaptive thresholds
# ---------------------------------------------------------------------------
SCENE_SIMPLE_THRESHOLD = 3
SCENE_COMPLEX_THRESHOLD = 10

FOLDABLE_KEYWORDS = {"毯子", "衣服", "衣物", "毛巾", "布", "衫", "裤", "被子", "围巾"}
CONTAINER_KEYWORDS = {"箱子", "盒子", "抽屉", "篮子", "收纳盒", "柜子", "箱", "桶"}


def compute_adaptive_categories(scene_data: dict) -> dict[str, dict]:
    """Return adjusted COMMAND_CATEGORIES based on scene object/furniture counts."""
    objects = scene_data.get("objects", [])
    furniture = scene_data.get("furniture", [])
    num_objects = len(objects)

    categories = {k: dict(v) for k, v in COMMAND_CATEGORIES.items()}

    if num_objects <= SCENE_SIMPLE_THRESHOLD:
        for cat in categories:
            categories[cat]["min_count"] = max(1, categories[cat]["min_count"] - 2)

    elif num_objects >= SCENE_COMPLEX_THRESHOLD:
        for cat in categories:
            categories[cat]["min_count"] += 2

    has_foldable = any(
        any(kw in o.get("name", "") for kw in FOLDABLE_KEYWORDS)
        for o in objects
    )
    if not has_foldable:
        categories["整理折叠"]["min_count"] = max(1, categories["整理折叠"]["min_count"] - 1)

    has_container = any(
        any(kw in item.get("name", "") for kw in CONTAINER_KEYWORDS)
        for item in [*furniture, *objects]
    )
    if not has_container:
        categories["收纳归集"]["min_count"] = max(1, categories["收纳归集"]["min_count"] - 1)

    return categories


def apply_default_case_volume(categories: dict[str, dict]) -> dict[str, dict]:
    """Return category counts for the default case suite."""
    scaled = {k: dict(v) for k, v in categories.items()}
    for name, count in DEFAULT_TARGET_CATEGORY_COUNTS.items():
        if name in scaled:
            scaled[name]["min_count"] = max(scaled[name].get("min_count", 0), count)
    return scaled


def default_param_case_limit() -> int:
    """Return generated function-calling parameter cases for the default suite."""
    return DEFAULT_PARAM_CASE_LIMIT
