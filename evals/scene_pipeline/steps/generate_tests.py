"""Step 7: Generate Layer 1 test files."""

import json
import logging
from collections import defaultdict
from pathlib import Path

from ..llm_client import LLMClient
from ..prompts.test_gen import TEST_GEN_PROMPT

logger = logging.getLogger(__name__)

EVAL_PROTOCOL_VERSION = "scene_pipeline_eval_v2_optional_support"

FETCH_EXPECTED = [
    "spatial_memory_query_vec", "move_to", "scene_recognition",
    "grasp_start", "perception_custom", "handover",
]

PLACE_EXPECTED = [
    "spatial_memory_query_vec", "move_to", "scene_recognition",
    "grasp_start", "spatial_memory_query_vec", "move_to", "place_down",
]


def _resolve_expected(raw_expected):
    """Resolve expected_subsequence: handle 'FETCH_EXPECTED'/'PLACE_EXPECTED' string markers."""
    if raw_expected == "FETCH_EXPECTED":
        return FETCH_EXPECTED, "FETCH_EXPECTED"
    if raw_expected == "PLACE_EXPECTED":
        return PLACE_EXPECTED, "PLACE_EXPECTED"
    if isinstance(raw_expected, list):
        return raw_expected, None
    return raw_expected, None


def generate_tests(
    client: LLMClient,
    commands_data: dict,
    mock_meta: dict,
    tests_dir: Path,
    *,
    param_case_limit: int = 40,
) -> None:
    """Generate all test files."""
    test_meta = _generate_test_metadata(client, commands_data, mock_meta)
    test_meta = _augment_test_meta_with_semantic_variants(test_meta, commands_data)
    test_meta = _stabilize_test_meta(test_meta, commands_data)

    _write_test_task_planning(test_meta, mock_meta, tests_dir / "test_task_planning.py")
    _write_test_function_calling(
        test_meta,
        commands_data,
        mock_meta,
        tests_dir / "test_function_calling.py",
        param_case_limit=param_case_limit,
    )
    _write_test_response_quality(test_meta, mock_meta, tests_dir / "test_response_quality.py")


def _generate_test_metadata(
    client: LLMClient, commands_data: dict, mock_meta: dict
) -> dict:
    commands_json = json.dumps(commands_data.get("commands", []), ensure_ascii=False, indent=2)
    failures_json = json.dumps(commands_data.get("failure_scenarios", []), ensure_ascii=False, indent=2)

    all_factories = [f["name"] for f in mock_meta.get("mock_factories", [])]
    all_factories += [f["name"] for f in mock_meta.get("failure_factories", [])]
    factories_json = json.dumps(all_factories, ensure_ascii=False, indent=2)

    prompt = TEST_GEN_PROMPT.format(
        commands_json=commands_json,
        failures_json=failures_json,
        mock_factories_json=factories_json,
    )

    messages = [{"role": "user", "content": prompt}]
    return client.chat_json(messages)


def _augment_test_meta_with_semantic_variants(test_meta: dict, commands_data: dict) -> dict:
    """Upgrade the eval set with deterministic semantic-coverage sampling.

    v1 only appended a few first-hit paraphrases. v2 keeps runtime behavior
    unchanged but selects a compact set of extra cases to cover more semantic
    dimensions: deixis, relative location, colloquial ellipsis, attribute
    composition and compound connectors.
    """
    commands_by_text = {
        cmd.get("text", ""): cmd
        for cmd in commands_data.get("commands", [])
        if cmd.get("text")
    }

    category_budget = {
        "拿取递送": 5,
        "放置移动": 5,
        "查询检索": 4,
        "收纳归集": 3,
        "整理折叠": 3,
        "复合任务": 3,
    }
    max_extra_cases = sum(category_budget.values())

    variant_candidates = []
    for cls in test_meta.get("test_classes", []):
        for test in cls.get("tests", []):
            command = commands_by_text.get(test.get("user_message", ""))
            if not command:
                continue
            for style_id, variant_text in _build_semantic_variant_candidates(
                test["user_message"], command
            ):
                if not variant_text or variant_text == test["user_message"]:
                    continue
                profile = _build_semantic_profile(test["user_message"], command, style_id)
                variant_candidates.append({
                    "class_ref": cls,
                    "base_test": test,
                    "command": command,
                    "style_id": style_id,
                    "variant_text": variant_text,
                    "profile": profile,
                })

    selected = _select_semantic_variant_candidates(
        variant_candidates,
        category_budget=category_budget,
        max_cases=max_extra_cases,
    )

    class_extra_tests: dict[int, list[dict]] = defaultdict(list)
    for idx, entry in enumerate(selected, start=1):
        base_test = entry["base_test"]
        style_id = entry["style_id"]
        variant = dict(base_test)
        variant["method_name"] = f'{base_test["method_name"]}_{style_id}_{idx}'
        variant["docstring"] = f'{base_test["docstring"]}（语义覆盖改写）'
        variant["user_message"] = entry["variant_text"]
        variant["_source_test_name"] = base_test.get("method_name", "").replace("test_", "")
        variant["_source_user_message"] = base_test.get("user_message", "")
        class_extra_tests[id(entry["class_ref"])].append(variant)

    for cls in test_meta.get("test_classes", []):
        cls["tests"].extend(class_extra_tests.get(id(cls), []))

    return test_meta


def _build_semantic_variant_candidates(text: str, command: dict) -> list[tuple[str, str]]:
    normalized = text.strip()
    if not normalized:
        return []
    if normalized.endswith("。"):
        normalized = normalized[:-1]
    category = command.get("category", "")
    object_hint = _extract_object_phrase(normalized, category)
    location_hint = _extract_location_phrase(normalized)
    relation_hint = _extract_relation_phrase(normalized)

    candidates: list[tuple[str, str]] = []
    if category == "拿取递送":
        candidates.append((
            "deixis",
            _rewrite_fetch_like(normalized, object_hint, location_hint, relation_hint),
        ))
        candidates.append((
            "colloquial",
            _rewrite_fetch_colloquial(normalized, object_hint, location_hint),
        ))
        candidates.append((
            "memory_reference",
            _rewrite_fetch_memory_reference(normalized, object_hint, location_hint, relation_hint),
        ))
    elif category == "放置移动":
        candidates.append((
            "relative_location",
            _rewrite_place_like(normalized, object_hint, location_hint, relation_hint),
        ))
        candidates.append((
            "constraint_focus",
            _rewrite_place_constraint_focus(normalized, object_hint, location_hint, relation_hint),
        ))
        candidates.append((
            "spoken_place",
            _rewrite_place_spoken(normalized, object_hint, location_hint, relation_hint),
        ))
    elif category == "查询检索":
        candidates.append((
            "ellipsis",
            _rewrite_query_like(normalized, object_hint, location_hint, relation_hint),
        ))
        candidates.append((
            "spoken_query",
            _rewrite_query_spoken(normalized, object_hint, location_hint),
        ))
        candidates.append((
            "memory_query",
            _rewrite_query_memory_reference(normalized, object_hint, location_hint, relation_hint),
        ))
    elif category == "收纳归集":
        candidates.append(("collect_rephrase", _rewrite_collect_like(normalized)))
        candidates.append(("batch_collect", _rewrite_collect_batch(normalized, object_hint, location_hint)))
        candidates.append(("cleanup_goal", _rewrite_collect_goal_first(normalized, object_hint, location_hint)))
    elif category == "整理折叠":
        candidates.append(("tidy_rephrase", _rewrite_tidy_like(normalized)))
        candidates.append(("result_focus", _rewrite_tidy_result_focus(normalized, object_hint, location_hint)))
        candidates.append(("spoken_tidy", _rewrite_tidy_spoken(normalized, object_hint, location_hint)))
    elif category == "复合任务":
        candidates.append(("compound_connector", _rewrite_compound_like(normalized, object_hint)))
        candidates.append(("compound_goal", _rewrite_compound_goal_first(normalized, object_hint)))
        candidates.append(("compound_user_like", _rewrite_compound_user_like(normalized, object_hint)))

    deduped: list[tuple[str, str]] = []
    seen_texts = {normalized}
    for style_id, variant_text in candidates:
        rewritten = (variant_text or "").strip()
        if not rewritten or rewritten in seen_texts:
            continue
        if not _variant_preserves_core_constraints(
            source_text=normalized,
            variant_text=rewritten,
            command=command,
            category=category,
            object_hint=object_hint,
            location_hint=location_hint,
            relation_hint=relation_hint,
        ):
            continue
        seen_texts.add(rewritten)
        deduped.append((style_id, rewritten))
    return deduped


def _build_semantic_profile(text: str, command: dict, style_id: str) -> dict:
    category = command.get("category", "")
    tags = {style_id}
    location_hint = _extract_location_phrase(text)
    relation_hint = _extract_relation_phrase(text)
    object_hint = _extract_object_phrase(text, category)

    if location_hint:
        tags.add("furniture_anchor")
    if relation_hint:
        tags.add("relative_relation")
    if object_hint and any(token in object_hint for token in ("色", "白", "黑", "粉", "绿", "黄", "蓝", "棕", "灰", "红", "橙")):
        tags.add("attribute_color")
    if object_hint and any(token in object_hint for token in ("那个", "那摞", "那束", "这", "那边")):
        tags.add("deixis")
    if "帮我" in text or "麻烦" in text or "一下" in text or "顺手" in text or "我刚才" in text:
        tags.add("spoken_style")
    if "刚才" in text or "之前" in text or "刚刚" in text:
        tags.add("memory_reference")
    if "规整" in text or "就行" in text or "别太乱" in text or "放稳" in text:
        tags.add("result_constraint")
    if category == "复合任务" or "然后" in text or "再" in text:
        tags.add("multi_step")
    if style_id in {
        "ellipsis", "spoken_query", "colloquial", "spoken_place",
        "spoken_tidy", "compound_user_like", "memory_query",
    }:
        tags.add("user_like")

    if {"relative_relation", "attribute_color", "spoken_style"} <= tags:
        complexity = "high"
    elif {"memory_reference", "spoken_style"} <= tags or len(tags) >= 4:
        complexity = "high"
    elif len(tags) >= 3:
        complexity = "mid"
    else:
        complexity = "base"

    return {
        "category": category,
        "furniture": command.get("target_furniture", ""),
        "object": command.get("target_object", ""),
        "style_id": style_id,
        "tags": tags,
        "complexity": complexity,
    }


def _stabilize_test_meta(test_meta: dict, commands_data: dict) -> dict:
    commands_by_test_name = {
        cmd.get("test_name", ""): cmd
        for cmd in commands_data.get("commands", [])
        if cmd.get("test_name")
    }
    commands_by_text = {
        cmd.get("text", ""): cmd
        for cmd in commands_data.get("commands", [])
        if cmd.get("text")
    }

    for cls in test_meta.get("test_classes", []):
        stabilized_tests = []
        for test in cls.get("tests", []):
            normalized = dict(test)
            test_name = normalized.get("method_name", "").replace("test_", "")
            source_test_name = normalized.get("_source_test_name", "")
            source_user_message = normalized.get("_source_user_message", "")
            command = (
                commands_by_test_name.get(test_name)
                or commands_by_text.get(normalized.get("user_message", ""))
                or commands_by_test_name.get(source_test_name)
                or commands_by_text.get(source_user_message)
                or {}
            )
            normalized = _stabilize_single_test(normalized, command)
            if not normalized.get("_drop_case"):
                stabilized_tests.append(normalized)
        cls["tests"] = stabilized_tests

    return test_meta


def _stabilize_single_test(test: dict, command: dict) -> dict:
    text = (test.get("user_message") or "").strip()
    category = command.get("category", "")
    if not text or not category:
        return test

    if category == "拿取递送":
        test["expected_subsequence"] = _fetch_expected_for_text(text)
        test["forbidden_tools"] = ["place_down"]
        test["min_calls"] = 4
        return test

    if category == "放置移动":
        if _lacks_place_destination(text):
            test["_drop_case"] = True
            return test
        test["expected_subsequence"] = _place_expected_for_text(text)
        test["forbidden_tools"] = ["handover"]
        test["min_calls"] = 4
        return test

    if category == "查询检索":
        test["expected_subsequence"] = _query_expected_for_text(text)
        test["forbidden_tools"] = ["grasp_start", "place_down", "handover"]
        test["min_calls"] = 1
        return test

    if category == "整理折叠":
        test["expected_subsequence"] = _tidy_expected_for_text(text)
        test["forbidden_tools"] = ["handover", "grasp_start"]
        test["min_calls"] = 3
        return test

    if category == "收纳归集":
        if _is_ambiguous_collect_case(text):
            test["expected_subsequence"] = None
            test["forbidden_tools"] = ["handover"]
            test["min_calls"] = 1
            return test
        test["expected_subsequence"] = _collect_expected_for_text(text)
        test["forbidden_tools"] = ["handover"]
        test["min_calls"] = 4
        return test

    if category == "复合任务":
        expected = _compound_expected_for_text(text)
        if expected:
            test["expected_subsequence"] = expected
            if _looks_like_box_collect_command(text):
                test["forbidden_tools"] = ["handover"]
                test["min_calls"] = 5
            elif _looks_like_fetch_command(text):
                test["forbidden_tools"] = ["place_down"]
                test["min_calls"] = 4
            elif _looks_like_manipulation_then_tidy_command(text):
                test["forbidden_tools"] = ["handover"]
                test["min_calls"] = 5
            elif _looks_like_transfer_or_place_command(text):
                test["forbidden_tools"] = ["handover"]
                test["min_calls"] = 4
            elif _looks_like_tidy_command(text):
                test["forbidden_tools"] = ["handover"]
                test["min_calls"] = 4
            else:
                test["forbidden_tools"] = []
                test["min_calls"] = 4
        return test

    return test


def _fetch_expected_for_text(text: str):
    if _mentions_preface_query(text):
        return [
            ["spatial_memory_query_vec", "move_to", "grasp_start", "handover"],
            ["list_objects", "spatial_memory_query_vec", "move_to", "grasp_start", "handover"],
            ["move_to", "list_objects", "spatial_memory_query_vec", "move_to", "grasp_start", "handover"],
        ]
    return ["spatial_memory_query_vec", "move_to", "grasp_start", "handover"]


def _place_expected_for_text(text: str):
    if _looks_like_push_adjustment(text):
        return [
            ["spatial_memory_query_vec", "move_to", "push_item"],
            ["spatial_memory_query_vec", "move_to", "push"],
            ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
        ]
    if _mentions_preface_query(text):
        return [
            ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
            ["list_objects", "spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
            ["move_to", "list_objects", "spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
            ["spatial_memory_query_vec", "spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
        ]
    return ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down"]


def _query_expected_for_text(text: str):
    if any(token in text for token in ("有什么", "哪些", "都有什么", "看看")):
        return [
            ["list_objects"],
            ["move_to", "list_objects"],
            ["spatial_memory_query_vec"],
        ]
    return [["spatial_memory_query_vec"], ["list_objects"]]


def _tidy_expected_for_text(text: str):
    if _looks_like_lean_or_secure_adjustment(text):
        return [
            ["spatial_memory_query_vec", "move_to", "push_item"],
            ["spatial_memory_query_vec", "move_to", "push"],
            ["spatial_memory_query_vec", "move_to", "tidy_surface"],
            ["spatial_memory_query_vec", "move_to", "align"],
            ["spatial_memory_query_vec", "move_to", "sort"],
        ]
    if _looks_like_surface_tidy(text):
        return [
            ["spatial_memory_query_vec", "move_to", "tidy_surface"],
            ["spatial_memory_query_vec", "move_to", "align"],
            ["spatial_memory_query_vec", "move_to", "sort"],
            ["spatial_memory_query_vec", "move_to", "clean"],
            ["spatial_memory_query_vec", "move_to", "wipe"],
            ["spatial_memory_query_vec", "move_to", "fold_item"],
            ["spatial_memory_query_vec", "move_to", "fold"],
            ["spatial_memory_query_vec", "move_to", "tuck"],
            ["list_objects", "spatial_memory_query_vec", "move_to", "tidy_surface"],
        ]
    if any(token in text for token in ("展开", "铺好", "铺平", "铺开")):
        return [
            ["spatial_memory_query_vec", "move_to", "fold_item", "place_down"],
            ["spatial_memory_query_vec", "move_to", "fold", "place_down"],
            ["spatial_memory_query_vec", "move_to", "unfold", "place_down"],
            ["spatial_memory_query_vec", "move_to", "tuck"],
            ["spatial_memory_query_vec", "move_to", "tidy_surface"],
        ]
    if any(token in text for token in (
        "摆齐", "对齐", "整理平整", "规整", "摆整齐", "整理一下",
        "整理好", "端正", "成一排", "摆正", "摆好", "收拾",
    )):
        return [
            ["spatial_memory_query_vec", "move_to", "tidy_surface"],
            ["spatial_memory_query_vec", "move_to", "align"],
            ["spatial_memory_query_vec", "move_to", "sort"],
            ["spatial_memory_query_vec", "move_to", "clean"],
            ["spatial_memory_query_vec", "move_to", "wipe"],
            ["spatial_memory_query_vec", "move_to", "fold_item"],
            ["spatial_memory_query_vec", "move_to", "fold"],
            ["spatial_memory_query_vec", "move_to", "fold_item", "place_down"],
            ["spatial_memory_query_vec", "move_to", "fold", "place_down"],
            ["spatial_memory_query_vec", "move_to", "tuck"],
            ["list_objects", "spatial_memory_query_vec", "move_to", "tidy_surface"],
        ]
    if any(token in text for token in ("叠", "折叠", "收拢", "折好")):
        return [
            ["spatial_memory_query_vec", "move_to", "fold_item"],
            ["spatial_memory_query_vec", "move_to", "fold"],
            ["spatial_memory_query_vec", "move_to", "fold_item", "place_down"],
            ["spatial_memory_query_vec", "move_to", "fold", "place_down"],
            ["spatial_memory_query_vec", "move_to", "tuck"],
        ]
    return [
        ["spatial_memory_query_vec", "move_to", "tidy_surface"],
        ["spatial_memory_query_vec", "move_to", "fold_item"],
        ["spatial_memory_query_vec", "move_to", "align"],
        ["spatial_memory_query_vec", "move_to", "sort"],
        ["spatial_memory_query_vec", "move_to", "clean"],
        ["spatial_memory_query_vec", "move_to", "wipe"],
        ["spatial_memory_query_vec", "move_to", "fold"],
        ["list_objects", "spatial_memory_query_vec", "move_to", "tidy_surface"],
    ]


def _collect_expected_for_text(text: str):
    if _has_closed_container_target(text):
        return [
            ["fetch_box", "move_to", "grasp_start", "put_box"],
            ["fetch_box", "move_to", "list_objects", "grasp_start", "put_box"],
            ["fetch_box", "spatial_memory_query_vec", "move_to", "grasp_start", "put_box"],
            ["spatial_memory_query_vec", "move_to", "grasp_start", "fetch_box", "put_box"],
            ["spatial_memory_query_vec", "move_to", "grasp_start", "put_box"],
        ]
    return [
        ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
        ["spatial_memory_query_vec", "move_to", "tidy_surface"],
        ["spatial_memory_query_vec", "move_to", "align"],
        ["spatial_memory_query_vec", "move_to", "sort"],
        ["spatial_memory_query_vec", "move_to", "clean"],
        ["spatial_memory_query_vec", "move_to", "wipe"],
        ["spatial_memory_query_vec", "move_to", "fold_item"],
        ["spatial_memory_query_vec", "move_to", "fold"],
        ["spatial_memory_query_vec", "move_to", "tuck"],
        ["spatial_memory_query_vec", "move_to", "list_objects", "tidy_surface"],
        ["list_objects", "spatial_memory_query_vec", "move_to", "tidy_surface"],
    ]


def _compound_expected_for_text(text: str):
    if _looks_like_box_collect_command(text):
        return [
            ["fetch_box", "move_to", "list_objects", "grasp_start", "put_box"],
            ["move_to", "list_objects", "fetch_box", "move_to", "grasp_start", "put_box"],
            ["fetch_box", "spatial_memory_query_vec", "move_to", "grasp_start", "put_box"],
            ["spatial_memory_query_vec", "move_to", "list_objects", "fetch_box", "grasp_start", "put_box"],
            ["spatial_memory_query_vec", "move_to", "grasp_start", "fetch_box", "put_box"],
        ]
    if _looks_like_fetch_command(text):
        return _fetch_expected_for_text(text)
    if _looks_like_manipulation_then_tidy_command(text):
        return [
            ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down", "tidy_surface"],
            ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down", "align"],
            ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down", "sort"],
            ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down", "clean"],
            ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down", "list_objects"],
            ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
        ]
    if _looks_like_transfer_or_place_command(text):
        return [
            ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
            ["list_objects", "spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
            ["move_to", "list_objects", "spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
            ["spatial_memory_query_vec", "spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
        ]
    if _looks_like_tidy_command(text):
        return [
            ["spatial_memory_query_vec", "move_to", "tidy_surface"],
            ["spatial_memory_query_vec", "move_to", "align"],
            ["spatial_memory_query_vec", "move_to", "sort"],
            ["spatial_memory_query_vec", "move_to", "clean"],
            ["spatial_memory_query_vec", "move_to", "fold", "sort"],
            ["spatial_memory_query_vec", "move_to", "fold_item", "sort"],
            ["spatial_memory_query_vec", "move_to", "fold_item", "tidy_surface"],
            ["spatial_memory_query_vec", "move_to", "fold", "tuck"],
            ["list_objects", "spatial_memory_query_vec", "move_to", "tidy_surface"],
        ]
    if _looks_like_collect_command(text):
        return _collect_expected_for_text(text)
    if _mentions_preface_query(text):
        return [
            ["spatial_memory_query_vec", "move_to", "grasp_start", "handover"],
            ["list_objects", "spatial_memory_query_vec", "move_to", "grasp_start", "handover"],
            ["move_to", "list_objects", "spatial_memory_query_vec", "move_to", "grasp_start", "handover"],
            ["spatial_memory_query_vec", "spatial_memory_query_vec", "move_to", "grasp_start", "handover"],
        ]
    return [
        ["spatial_memory_query_vec", "move_to"],
        ["spatial_memory_query_vec", "move_to", "tidy_surface"],
        ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
    ]


def _mentions_preface_query(text: str) -> bool:
    return any(token in text for token in ("确认", "看看", "看下", "先看", "先看看", "检查", "查一下"))


def _looks_like_fetch_command(text: str) -> bool:
    return any(token in text for token in (
        "递给我", "拿给我", "拿过来", "拿来", "取下来递给我", "交给我",
    ))


def _looks_like_place_command(text: str) -> bool:
    return any(token in text for token in ("放到", "放在", "移到", "挪到", "摆到"))


def _looks_like_transfer_or_place_command(text: str) -> bool:
    return _looks_like_place_command(text) or any(
        token in text for token in ("拿到", "带到", "送到", "搬到", "移去", "挪去")
    )


def _looks_like_manipulation_then_tidy_command(text: str) -> bool:
    return _looks_like_transfer_or_place_command(text) and _looks_like_tidy_command(text)


def _looks_like_push_adjustment(text: str) -> bool:
    return any(
        token in text
        for token in (
            "挪一点", "挪到", "移一点", "推一下", "推到", "放远一点",
            "往里挪", "往外挪", "靠里", "靠外", "别碰倒", "别掉",
            "左侧一点", "右侧一点", "前面一点", "后面一点",
            "别挡", "挡着", "挡住", "遮挡", "露出来", "让开", "移开",
        )
    )


def _looks_like_lean_or_secure_adjustment(text: str) -> bool:
    return any(token in text for token in (
        "靠稳", "靠墙", "靠住", "稳一点", "扶正", "固定", "别倒",
        "别挡", "挡着", "挡住", "遮挡", "露出来", "让开", "移开",
    ))


def _is_ambiguous_collect_case(text: str) -> bool:
    has_inspect = any(token in text for token in ("检查", "看看", "确认"))
    has_collect = any(token in text for token in ("收", "归", "集中", "整理"))
    lacks_closed_container = (not _has_closed_container_target(text)) or ("旁边" in text or "旁" in text)
    return has_inspect and has_collect and lacks_closed_container


def _looks_like_box_collect_command(text: str) -> bool:
    return _has_closed_container_target(text) and any(
        token in text for token in ("收", "归", "放进", "放入", "收到")
    )


def _looks_like_collect_command(text: str) -> bool:
    return any(token in text for token in ("收", "归", "集中", "归拢", "收纳"))


def _looks_like_tidy_command(text: str) -> bool:
    return any(token in text for token in (
        "整理", "叠", "折", "铺", "收拾", "规整", "摆齐",
        "摆正", "摆好", "端正", "成一排", "挂好", "排整齐", "排齐",
    ))


def _looks_like_surface_tidy(text: str) -> bool:
    return any(token in text for token in (
        "整理平整", "平码", "规整", "摆齐", "摆整齐",
        "整理干净", "干净一点", "清理干净", "擦干净",
    ))


def _has_closed_container_target(text: str) -> bool:
    # A token such as "书本收纳盒" can be an object, not a destination container.
    # Treat it as a closed-container target only when the instruction has an
    # explicit put-into/store-into relation.
    explicit_container_relations = (
        "收进收纳盒", "收进盒子", "收进箱子",
        "收进收纳箱", "收纳到收纳盒", "收纳到盒子", "收纳到箱子",
        "收纳到收纳箱", "收到收纳盒", "收到盒子", "收到箱子",
        "收到收纳箱", "放进收纳盒", "放进盒子", "放进箱子",
        "放进收纳箱", "放入收纳盒", "放入盒子", "放入箱子",
        "放入收纳箱", "装进收纳盒", "装进盒子", "装进箱子",
        "装进收纳箱", "装入收纳盒", "装入盒子", "装入箱子",
        "装入收纳箱",
        "放到盒中", "放到盒里", "放到箱中", "放到箱里",
        "盒中", "盒里", "箱中", "箱里",
    )
    return any(token in text for token in explicit_container_relations)


def _lacks_place_destination(text: str) -> bool:
    has_place_intent = any(token in text for token in ("放", "摆", "移", "挪"))
    if not has_place_intent:
        return False
    target_markers = ("到", "在", "旁边", "附近", "靠近", "左侧", "右侧", "前侧", "后侧", "中间")
    object_only_markers = ("这个", "这个白色", "这个绿色", "这个黄色", "这个蓝灰色")
    if any(marker in text for marker in target_markers):
        return False
    return any(marker in text for marker in object_only_markers)


def _select_semantic_variant_candidates(
    candidates: list[dict],
    *,
    category_budget: dict[str, int],
    max_cases: int,
) -> list[dict]:
    if not candidates or max_cases <= 0:
        return []

    remaining = sorted(
        candidates,
        key=lambda item: (
            item["profile"].get("category", ""),
            item["profile"].get("furniture", ""),
            item["profile"].get("object", ""),
            item.get("style_id", ""),
            item["base_test"].get("method_name", ""),
        ),
    )
    selected: list[dict] = []
    used_categories: dict[str, int] = defaultdict(int)
    seen_tags: set[str] = set()
    seen_styles: set[str] = set()
    seen_furniture: set[str] = set()
    seen_objects: set[str] = set()
    used_methods: set[tuple[str, str]] = set()

    while remaining and len(selected) < max_cases:
        best_idx = None
        best_score = None
        for idx, item in enumerate(remaining):
            profile = item["profile"]
            category = profile.get("category", "")
            budget = category_budget.get(category, 0)
            if budget <= 0 or used_categories[category] >= budget:
                continue

            method_key = (item["base_test"]["method_name"], item["style_id"])
            if method_key in used_methods:
                continue

            new_tags = profile.get("tags", set()) - seen_tags
            complexity = profile.get("complexity", "base")
            score = (
                4 if complexity == "high" else 2 if complexity == "mid" else 0,
                len(new_tags),
                2 if item["style_id"] not in seen_styles else 0,
                2 if profile.get("furniture") and profile["furniture"] not in seen_furniture else 0,
                1 if profile.get("object") and profile["object"] not in seen_objects else 0,
                1 if used_categories[category] == 0 else 0,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        profile = chosen["profile"]
        category = profile.get("category", "")
        used_categories[category] += 1
        seen_tags.update(profile.get("tags", set()))
        if chosen.get("style_id"):
            seen_styles.add(chosen["style_id"])
        if profile.get("furniture"):
            seen_furniture.add(profile["furniture"])
        if profile.get("object"):
            seen_objects.add(profile["object"])
        used_methods.add((chosen["base_test"]["method_name"], chosen["style_id"]))

    return selected


def _rewrite_fetch_like(text: str, object_hint: str, location_hint: str, relation_hint: str) -> str:
    if text.startswith("把") and ("递给我" in text or "拿给我" in text):
        core = text
        if core.startswith("把"):
            core = core[1:]
        core = core.replace("递给我", "", 1).replace("拿给我", "", 1)
        core = core.replace("拿过来给我", "", 1)
        if location_hint and object_hint and relation_hint:
            return f"帮我把{location_hint}{relation_hint}那个{object_hint}拿过来"
        if location_hint and object_hint:
            return f"帮我把{location_hint}那边的那个{object_hint}拿过来"
        return f"帮我把{core}拿过来给我一下"
    if text.startswith("帮我把"):
        return text.replace("帮我把", "麻烦把", 1).replace("拿给我", "拿过来给我", 1)
    if object_hint:
        if relation_hint:
            return f"帮我拿一下{relation_hint}那个{object_hint}"
        return f"帮我拿一下那个{object_hint}"
    return f"麻烦{text}"


def _rewrite_fetch_colloquial(text: str, object_hint: str, location_hint: str) -> str:
    if object_hint and location_hint:
        return f"{location_hint}那边那个{object_hint}，帮我顺手拿来"
    if object_hint:
        return f"那个{object_hint}帮我拿过来吧"
    return f"{text}，顺手帮我拿一下"


def _rewrite_fetch_memory_reference(
    text: str, object_hint: str, location_hint: str, relation_hint: str
) -> str:
    if object_hint and location_hint and relation_hint:
        return f"我刚才看到{location_hint}{relation_hint}有个{object_hint}，你把它拿给我"
    if object_hint and location_hint:
        return f"我刚才放在{location_hint}那边的{object_hint}，帮我拿过来"
    if object_hint:
        return f"之前提到的那个{object_hint}，你拿来给我一下"
    return f"刚才说的那个东西，按这个要求处理一下：{text}"


def _rewrite_place_like(text: str, object_hint: str, location_hint: str, relation_hint: str) -> str:
    rewritten = text
    if "放到" in rewritten:
        rewritten = rewritten.replace("放到", "摆到", 1)
    elif "放在" in rewritten:
        rewritten = rewritten.replace("放在", "摆在", 1)
    elif "放好" in rewritten:
        rewritten = rewritten.replace("放好", "摆好", 1)
    if relation_hint and "靠近" not in rewritten and "旁边" not in rewritten:
        if "摆到" in rewritten:
            rewritten = rewritten.replace("摆到", f"摆到{relation_hint}", 1)
        elif "摆在" in rewritten:
            rewritten = rewritten.replace("摆在", f"摆在{relation_hint}", 1)
    if rewritten.startswith("把"):
        if object_hint and object_hint not in rewritten:
            return f"麻烦把{object_hint}{rewritten[1:]}"
        return f"麻烦把{rewritten[1:]}"
    if object_hint and location_hint:
        return f"把那个{object_hint}摆到{location_hint}{relation_hint or '附近'}"
    return rewritten


def _rewrite_place_constraint_focus(
    text: str, object_hint: str, location_hint: str, relation_hint: str
) -> str:
    if object_hint and location_hint and relation_hint:
        return f"把{object_hint}放到{location_hint}{relation_hint}一点的位置"
    if object_hint and location_hint:
        return f"把{object_hint}放到{location_hint}那边就行"
    return f"按我说的位置把它放好：{text}"


def _rewrite_place_spoken(
    text: str, object_hint: str, location_hint: str, relation_hint: str
) -> str:
    if object_hint and location_hint and relation_hint:
        return f"麻烦把{object_hint}放到{location_hint}{relation_hint}，放稳一点"
    if object_hint and location_hint:
        return f"{object_hint}你给我放到{location_hint}那边，别放偏了"
    return f"{text}，放稳一点"


def _rewrite_query_like(text: str, object_hint: str, location_hint: str, relation_hint: str) -> str:
    if text.startswith("帮我找一下"):
        target = text.replace("帮我找一下", "", 1)
        return f"帮我看看{target}现在放哪了"
    if text.endswith("在哪儿"):
        return text.replace("在哪儿", "现在放哪了", 1)
    if text.endswith("在哪里"):
        return text.replace("在哪里", "现在放哪了", 1)
    if object_hint:
        if location_hint and relation_hint:
            return f"帮我看下{location_hint}{relation_hint}那个{object_hint}还在不在"
        if location_hint:
            return f"帮我看下{location_hint}那个{object_hint}在哪"
        return f"{object_hint}现在在哪，帮我看看"
    return f"帮我看看{text}"


def _rewrite_query_spoken(text: str, object_hint: str, location_hint: str) -> str:
    if object_hint and location_hint:
        return f"我刚才放在{location_hint}附近的{object_hint}现在还在吗"
    if object_hint:
        return f"那个{object_hint}你帮我确认下现在在哪"
    return f"{text}，你帮我确认一下"


def _rewrite_query_memory_reference(
    text: str, object_hint: str, location_hint: str, relation_hint: str
) -> str:
    if object_hint and location_hint and relation_hint:
        return f"我记得{location_hint}{relation_hint}有个{object_hint}，你看现在还在不在"
    if object_hint and location_hint:
        return f"之前放在{location_hint}那边的{object_hint}现在在哪"
    if object_hint:
        return f"我刚才提到的{object_hint}现在在哪里"
    return f"按刚才那个描述，帮我确认一下位置：{text}"


def _rewrite_collect_like(text: str) -> str:
    if text.startswith("把") and "收进" in text:
        return f"麻烦{text}"
    if text.startswith("把") and "都收" in text:
        return f"麻烦{ text.replace('都收', '都整理', 1) }"
    return f"麻烦{text}"


def _rewrite_collect_batch(text: str, object_hint: str, location_hint: str) -> str:
    if _has_closed_container_target(text):
        return text
    if object_hint and location_hint:
        return f"把{location_hint}上的{object_hint}这一类东西集中收好"
    return f"把这些零散的东西统一收整一下：{text}"


def _rewrite_collect_goal_first(text: str, object_hint: str, location_hint: str) -> str:
    if _has_closed_container_target(text):
        return f"我想让场面看起来利落一点，{text}"
    if object_hint and location_hint:
        return f"我想让{location_hint}看起来利落一点，把{object_hint}都收整好"
    if object_hint:
        return f"把{object_hint}统一归拢一下，别零零散散放着"
    return f"目标是收整干净一些：{text}"


def _rewrite_tidy_like(text: str) -> str:
    if text.startswith("把") and "叠一下" in text:
        return f"麻烦{ text.replace('叠一下', '整理好', 1) }"
    if text.startswith("整理"):
        return text.replace("整理", "帮我整理一下", 1)
    return f"麻烦{text}"


def _rewrite_tidy_result_focus(text: str, object_hint: str, location_hint: str) -> str:
    if object_hint and location_hint:
        return f"把{location_hint}那边的{object_hint}整理得规整一点"
    if object_hint:
        return f"把{object_hint}整理好，看起来别太乱"
    return f"整理完让它看起来规整一点：{text}"


def _rewrite_tidy_spoken(text: str, object_hint: str, location_hint: str) -> str:
    if object_hint and location_hint:
        return f"麻烦把{location_hint}那边的{object_hint}稍微收拾一下，整齐点"
    if object_hint:
        return f"这个{object_hint}你整理一下，别显得乱"
    return f"{text}，顺手整理规整一点"


def _rewrite_compound_like(text: str, object_hint: str) -> str:
    if "，再" in text:
        rewritten = text.replace("，再", "，然后再", 1)
    elif "然后" in text:
        rewritten = text.replace("然后", "接着", 1)
    else:
        rewritten = f"先{text}"
    if object_hint and rewritten.startswith("把"):
        return rewritten.replace("把", "先把", 1)
    return rewritten


def _rewrite_compound_goal_first(text: str, object_hint: str) -> str:
    if "，再" in text:
        first, second = text.split("，再", 1)
        return f"最终把它处理好，先{first}，再{second}"
    if object_hint:
        return f"围绕{object_hint}把这件事一步步做完：{text}"
    return f"按顺序把这件复合任务做完：{text}"


def _rewrite_compound_user_like(text: str, object_hint: str) -> str:
    if "然后" in text:
        return f"你先把前面这步做了，然后后面那步也接着完成：{text}"
    if "，再" in text:
        return text.replace("，再", "，弄完前一步再", 1)
    if object_hint:
        return f"围绕{object_hint}这件事你按顺序处理一下，别漏步骤"
    return f"这事分两步做，按顺序完成：{text}"


def _extract_object_phrase(text: str, category: str) -> str:
    trimmed = text
    for prefix in ("帮我把", "麻烦把", "把", "帮我找一下", "帮我看看", "整理", "先把"):
        if trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix):]
            break

    stop_markers = [
        "递给我", "拿给我", "拿过来给我", "拿过来", "放到", "放在", "摆到", "摆在",
        "收进", "归拢到", "都收", "都整理", "叠一下", "整理好", "在哪儿", "在哪里",
        "现在放哪了", "，", "然后", "再",
    ]
    end = len(trimmed)
    for marker in stop_markers:
        idx = trimmed.find(marker)
        if idx != -1:
            end = min(end, idx)
    phrase = trimmed[:end].strip("，、 ")

    if category == "查询检索" and not phrase:
        phrase = trimmed.strip("，、 ")
    return phrase


def _extract_location_phrase(text: str) -> str:
    location_markers = [
        "茶几", "边桌", "沙发", "书架", "脚凳", "休闲椅", "收纳盒", "桌上", "桌面",
        "木椅", "木椅前方地面", "地面", "沙发旁", "沙发边", "书架旁", "书架边",
    ]
    for marker in location_markers:
        if marker in text:
            return marker
    return ""


def _extract_relation_phrase(text: str) -> str:
    relation_markers = [
        "中间偏右", "中部偏左", "中部偏右", "前侧中央", "右侧", "左侧", "中部", "中间",
        "旁边", "附近", "前面", "后面", "右手边", "左手边", "靠近", "靠后", "右边", "左边",
        "靠里", "外侧", "内侧",
    ]
    for marker in relation_markers:
        if marker in text:
            return marker
    return ""


def _variant_preserves_core_constraints(
    *,
    source_text: str,
    variant_text: str,
    command: dict,
    category: str,
    object_hint: str,
    location_hint: str,
    relation_hint: str,
) -> bool:
    source = source_text.strip()
    variant = variant_text.strip()
    if not source or not variant:
        return False
    if category == "放置移动":
        if object_hint and object_hint not in variant:
            return False
        if relation_hint and relation_hint not in variant and relation_hint in source:
            return False
        if _lacks_place_destination(variant):
            return False
    elif category == "收纳归集":
        if object_hint and object_hint not in variant and object_hint not in source:
            return False
        if _has_closed_container_target(source) and not _has_closed_container_target(variant):
            return False
    elif category == "复合任务":
        if _looks_like_box_collect_command(source) and not _looks_like_box_collect_command(variant):
            return False
        if _looks_like_tidy_command(source) and not _looks_like_tidy_command(variant):
            return False
    return True


def _write_test_task_planning(test_meta: dict, mock_meta: dict, output_path: Path) -> None:
    all_factory_names = set()
    for f in mock_meta.get("mock_factories", []):
        all_factory_names.add(f["name"])
    for f in mock_meta.get("failure_factories", []):
        all_factory_names.add(f["name"])

    used_factories = set()
    for cls in test_meta.get("test_classes", []):
        for t in cls.get("tests", []):
            used_factories.add(t["mock_factory"])
    for t in test_meta.get("failure_tests", []):
        used_factories.add(t["mock_factory"])

    valid_imports = sorted(used_factories & all_factory_names)

    lines = [
        '"""Layer 1: Task planning tests.',
        "",
        "Verify the model produces the correct tool-call sequences for different user",
        "instructions, following the task flows defined in the system prompt.",
        "",
        'Owns the **tool_sequence** and **efficiency** scoring dimensions.',
        '"""',
        "",
        f'EVAL_PROTOCOL_VERSION = "{EVAL_PROTOCOL_VERSION}"',
        "",
        "import pytest",
        "from typing import Iterable",
        "",
        "from conftest import (",
        "    assert_tool_contains,",
        "    assert_tool_order,",
        ")",
        "from evaluation import score_efficiency, score_tool_sequence",
        "from mock_tool_results import (",
    ]

    for name in valid_imports:
        lines.append(f"    {name},")

    lines.extend([
        ")",
        "",
        "",
        "FETCH_EXPECTED = [",
        '    "spatial_memory_query_vec", "move_to", "scene_recognition",',
        '    "grasp_start", "perception_custom", "handover",',
        "]",
        "",
        "PLACE_EXPECTED = [",
        '    "spatial_memory_query_vec", "move_to", "scene_recognition",',
        '    "grasp_start", "spatial_memory_query_vec", "move_to", "place_down",',
        "]",
        "",
        "",
        "def _normalize_expected_options(expected_seq):",
        '    """Treat a flat list as one candidate, nested lists as multiple candidates."""',
        "    if not expected_seq:",
        "        return []",
        "    if isinstance(expected_seq, list) and expected_seq and isinstance(expected_seq[0], str):",
        "        return [expected_seq]",
        "    return list(expected_seq)",
        "",
        "",
        "def _assert_any_expected_subsequence(result, expected_seq):",
        "    options = _normalize_expected_options(expected_seq)",
        "    last_error = None",
        "    for option in options:",
        "        score = score_tool_sequence(result, option)",
        "        if score.sub_scores.get('matched') == score.sub_scores.get('expected'):",
        "            return",
        "        last_error = AssertionError(",
        "            f\"Expected semantic subsequence:\\n  {option}\\n\"",
        "            f\"Only matched {score.sub_scores.get('matched')}/{score.sub_scores.get('expected')} \"",
        "            f\"in actual:\\n  {result.tool_names}\\nDetails: {score.details}\"",
        "        )",
        "    if last_error is not None:",
        "        raise last_error",
        "",
        "",
        "def _assert_no_forbidden_tools(result, forbidden):",
        "    for tool_name in forbidden or []:",
        "        score = score_tool_sequence(result, [], [tool_name])",
        "        assert score.sub_scores.get('forbidden_penalty', 0) == 0, (",
        "            f\"'{tool_name}' should not appear semantically but found in: \"",
        "            f\"{result.tool_names}; details: {score.details}\"",
        "        )",
        "",
        "",
        "def _record_planning_scores(",
        "    result, eval_record, test_name,",
        "    expected_seq=None, forbidden=None, min_calls=6,",
        "):",
        '    """Compute tool_sequence + efficiency scores and record to eval report."""',
        "    scores = {}",
        "    if expected_seq:",
        "        options = _normalize_expected_options(expected_seq)",
        "        ts = None",
        "        best_score = None",
        "        for option in options:",
        "            candidate = score_tool_sequence(result, option, forbidden)",
        "            score = candidate.score",
        "            if best_score is None or score > best_score:",
        "                ts = candidate",
        "                best_score = score",
        '        scores["tool_sequence"] = ts.to_dict()',
        "    eff = score_efficiency(result, min_calls)",
        '    scores["efficiency"] = eff.to_dict()',
        "    eval_record(test_name, scores, result=result)",
        "",
    ])

    for cls in test_meta.get("test_classes", []):
        if cls.get("name") == "TestFailureRecovery":
            # Failure-recovery cases are rendered once below so assert_min_calls
            # can be included without duplicating the class.
            continue
        lines.append("")
        lines.append(f'class {cls["name"]}:')
        lines.append(f'    """{cls["description"]}"""')
        lines.append("")

        for t in cls.get("tests", []):
            if t["mock_factory"] not in all_factory_names:
                continue

            raw_expected = t.get("expected_subsequence", [])
            expected_list, const_name = _resolve_expected(raw_expected)

            lines.append(f'    def {t["method_name"]}(self, runner, eval_record):')
            lines.append(f'        """{t["docstring"]}"""')
            lines.append(f'        result = runner.run("{t["user_message"]}", {t["mock_factory"]}())')
            lines.append("")

            if const_name:
                lines.append(f"        expected = {const_name}")
            else:
                lines.append(f"        expected = {expected_list}")

            forbidden = t.get("forbidden_tools", [])
            min_calls = t.get("min_calls", 6)

            if forbidden:
                lines.append(f"        _record_planning_scores(")
                lines.append(f"            result, eval_record, \"{t['method_name'].replace('test_', '')}\",")
                lines.append(f"            expected_seq=expected, forbidden={forbidden}, min_calls={min_calls},")
                lines.append(f"        )")
            else:
                lines.append(f"        _record_planning_scores(")
                lines.append(f"            result, eval_record, \"{t['method_name'].replace('test_', '')}\",")
                lines.append(f"            expected_seq=expected, min_calls={min_calls},")
                lines.append(f"        )")

            lines.append("")
            lines.append("        _assert_any_expected_subsequence(result, expected)")

            if forbidden:
                lines.append(f"        _assert_no_forbidden_tools(result, {forbidden})")

            lines.append("")

    if test_meta.get("failure_tests"):
        lines.append("")
        lines.append('class TestFailureRecovery:')
        lines.append('    """Verify failure-recovery task planning."""')
        lines.append("")

        for t in test_meta.get("failure_tests", []):
            if t["mock_factory"] not in all_factory_names:
                continue
            lines.append(f'    def {t["method_name"]}(self, runner, eval_record):')
            lines.append(f'        """{t["docstring"]}"""')
            lines.append(f'        result = runner.run("{t["user_message"]}", {t["mock_factory"]}())')
            lines.append("")
            expected = t.get("expected_subsequence", [])
            min_calls = t.get("min_calls", 6)
            lines.append(f"        expected = {expected}")
            lines.append(f"        _record_planning_scores(")
            lines.append(f"            result, eval_record, \"{t['method_name'].replace('test_', '')}\",")
            lines.append(f"            expected_seq=expected, min_calls={min_calls},")
            lines.append(f"        )")
            lines.append("")
            lines.append("        _assert_any_expected_subsequence(result, expected)")

            for tool_name, min_count in t.get("assert_min_calls", {}).items():
                lines.append(f'        {tool_name}_calls = result.get_calls_for("{tool_name}")')
                lines.append(f"        assert len({tool_name}_calls) >= {min_count}, (")
                lines.append(f'            f"Expected at least {min_count} {tool_name} calls, got {{len({tool_name}_calls)}}"')
                lines.append(f"        )")

            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", output_path)


def _write_test_function_calling(
    test_meta: dict,
    commands_data: dict,
    mock_meta: dict,
    output_path: Path,
    *,
    param_case_limit: int = 18,
) -> None:
    all_factory_names = set()
    for f in mock_meta.get("mock_factories", []):
        all_factory_names.add(f["name"])

    memory_item_names = [mi["var_name"] for mi in mock_meta.get("memory_items", [])]

    first_factory = next(iter(all_factory_names), "lambda: MockResultProvider()")
    first_memory_item = memory_item_names[0] if memory_item_names else None
    memory_item_types = {
        mi["var_name"]: mi.get("data", {}).get("type", "")
        for mi in mock_meta.get("memory_items", [])
    }
    memory_item_display_names = {
        mi["var_name"]: mi.get("data", {}).get("name", "")
        for mi in mock_meta.get("memory_items", [])
    }
    factory_primary_items: dict[str, str] = {}
    for factory in mock_meta.get("mock_factories", []):
        candidates: list[str] = []
        for items in factory.get("memory_queries", {}).values():
            candidates.extend(items)
        object_candidates = [
            item for item in candidates
            if memory_item_types.get(item) == "object"
        ]
        if object_candidates:
            factory_primary_items[factory["name"]] = object_candidates[0]
        elif candidates:
            factory_primary_items[factory["name"]] = candidates[0]

    commands_by_test_name = {
        cmd.get("test_name", ""): cmd
        for cmd in commands_data.get("commands", [])
        if cmd.get("test_name")
    }

    candidate_cases: list[dict] = []
    seen_factories: set[str] = set()
    for cls in test_meta.get("test_classes", []):
        for t in cls.get("tests", []):
            factory = t.get("mock_factory")
            item = factory_primary_items.get(factory)
            if not factory or factory in seen_factories or not item:
                continue
            if factory not in all_factory_names:
                continue
            seen_factories.add(factory)
            test_name = t["method_name"].replace("test_", "")
            command_meta = commands_by_test_name.get(test_name, {})
            expected_seq = t.get("expected_subsequence")
            if not _expects_move_to_param_check(expected_seq, command_meta):
                continue
            candidate_cases.append({
                "method_name": t["method_name"],
                "name": test_name,
                "user_message": t["user_message"],
                "mock_factory": factory,
                "item": item,
                "category": command_meta.get("category", ""),
                "target_furniture": command_meta.get("target_furniture", ""),
                "target_object": command_meta.get("target_object", ""),
                "object_kind": memory_item_display_names.get(item, ""),
            })

    param_cases = _select_param_cases(candidate_cases, limit=param_case_limit)

    lines = [
        '"""Layer 1: Function calling parameter validation tests.',
        "",
        "Verify the model passes correct parameter VALUES to tool calls.",
        "",
        'Owns the **param_accuracy** scoring dimension.',
        '"""',
        "",
        "import pytest",
        "",
        "from conftest import assert_tool_contains",
        "from evaluation import score_param_accuracy, score_param_accuracy_multi, ScoreDetail",
        "from mock_tool_results import (",
    ]

    for name in sorted(memory_item_names):
        lines.append(f"    {name},")
    for name in sorted(all_factory_names):
        lines.append(f"    {name},")
    lines.extend([
        ")",
        "",
        "",
        "def _record_fc_scores(result, eval_record, test_name, checks, tolerance=0.01):",
        '    """Compute param_accuracy scores and record to eval report."""',
        "    pa = score_param_accuracy_multi(result, checks, tolerance)",
        '    eval_record(test_name, {"param_accuracy": pa.to_dict()}, result=result)',
        "",
        "",
        "def _bool_param_accuracy(passed: bool, detail: str) -> dict:",
        '    """Build a param_accuracy score dict from a boolean check result."""',
        "    return ScoreDetail(",
        '        "param_accuracy",',
        "        1.0 if passed else 0.0,",
        "        detail,",
        '        sub_scores={"matched": 1 if passed else 0, "total": 1},',
        "    ).to_dict()",
        "",
        "",
        "def _best_move_to_index(result, item):",
        '    """Pick the move_to call closest to the target object coordinates."""',
        '    move_calls = result.get_calls_for("move_to")',
        "    if not move_calls:",
        "        return 0",
        "    target = (",
        '        item.get("obj_x"), item.get("obj_y"),',
        '        item.get("coordinate_x"), item.get("coordinate_y"),',
        "    )",
        "    def _dist(call):",
        "        total = 0.0",
        "        count = 0",
        '        for key, expected in zip(("obj_x", "obj_y", "coordinate_x", "coordinate_y"), target):',
        "            actual = call.get(key)",
        "            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):",
        "                total += abs(actual - expected)",
        "                count += 1",
        "        return total if count else float('inf')",
        "    best_idx = min(range(len(move_calls)), key=lambda idx: _dist(move_calls[idx]))",
        "    return best_idx",
        "",
        "",
        "class TestParamAccuracy:",
        '    """Verify parameter values in tool calls."""',
        "",
    ])

    if first_memory_item and first_factory:
        first_cmd = "测试指令"
        for cls in test_meta.get("test_classes", []):
            for t in cls.get("tests", []):
                if t["mock_factory"] in all_factory_names:
                    first_cmd = t["user_message"]
                    first_factory = t["mock_factory"]
                    first_memory_item = factory_primary_items.get(first_factory, first_memory_item)
                    break
            else:
                continue
            break

        lines.extend([
            f'    def test_move_to_coords(self, runner, eval_record):',
            f'        """move_to should use coordinates from spatial_memory_query_vec."""',
            f'        result = runner.run("{first_cmd}", {first_factory}())',
            f'        item = {first_memory_item}',
            f'',
            f'        checks = [(',
            f'            "move_to",',
            f'            {{',
            f'                "obj_x": item["obj_x"],',
            f'                "obj_y": item["obj_y"],',
            f'                "coordinate_x": item["coordinate_x"],',
            f'                "coordinate_y": item["coordinate_y"],',
            f'                "coordinate_z": item["coordinate_z"],',
            f'                "coordinate_yaw": item["coordinate_yaw"],',
            f'            }},',
            f'            _best_move_to_index(result, item),',
            f'        )]',
            f'',
            f'        _record_fc_scores(result, eval_record, "fc_move_to_coords", checks)',
            f'        assert_tool_contains(result, "move_to")',
            f'        pa = score_param_accuracy_multi(result, checks)',
            f'        assert pa.score >= 0.5, f"Param accuracy too low: {{pa.details}}"',
            f'',
            f'    def test_grasp_english(self, runner, eval_record):',
            f'        """grasp_start should receive English item name."""',
            f'        result = runner.run("{first_cmd}", {first_factory}())',
            f'        grasp_calls = result.get_calls_for("grasp_start")',
            f'',
            f'        passed = bool(grasp_calls) and grasp_calls[0].get("item", "").isascii()',
            f'        item_arg = grasp_calls[0].get("item", "") if grasp_calls else "(未调用)"',
            f'        eval_record("fc_grasp_lang", {{',
            f'            "param_accuracy": _bool_param_accuracy(',
            f'                passed, f"grasp_start item=\'{{item_arg}}\', 英文={{\'是\' if passed else \'否\'}}",',
            f'            ),',
            f'        }}, result=result)',
            f'        if grasp_calls:',
            f'            assert grasp_calls[0].get("item", "").isascii()',
            f'',
            f'    def test_spatial_query_chinese(self, runner, eval_record):',
            f'        """spatial_memory_query_vec name param should be in Chinese."""',
            f'        result = runner.run("{first_cmd}", {first_factory}())',
            f'        query_calls = result.get_calls_for("spatial_memory_query_vec")',
            f'',
            f'        if query_calls:',
            f'            name = query_calls[0].get("name", "")',
            f'            has_chinese = any("\\u4e00" <= c <= "\\u9fff" for c in name)',
            f'            detail = f"query name=\'{{name}}\', 含中文={{\'是\' if has_chinese else \'否\'}}"',
            f'        else:',
            f'            has_chinese = False',
            f'            detail = "spatial_memory_query_vec 未调用"',
            f'',
            f'        eval_record("fc_spatial_query_chinese", {{',
            f'            "param_accuracy": _bool_param_accuracy(has_chinese, detail),',
            f'        }}, result=result)',
            f'        if query_calls:',
            f'            name = query_calls[0].get("name", "")',
            f'            assert any("\\u4e00" <= c <= "\\u9fff" for c in name)',
            f'',
        ])

    if param_cases:
        lines.extend([
            "    @pytest.mark.parametrize(",
            '        "case",',
            "        [",
        ])
        for case in param_cases:
            lines.append("            {")
            lines.append(f'                "name": "{case["name"]}",')
            lines.append(f'                "user_message": {json.dumps(case["user_message"], ensure_ascii=False)},')
            lines.append(f'                "mock_factory": {case["mock_factory"]},')
            lines.append(f'                "item": {case["item"]},')
            lines.append("            },")
        lines.extend([
            "        ],",
            "        ids=lambda case: case[\"name\"],",
            "    )",
            "    def test_move_to_coords_by_command(self, runner, eval_record, case):",
            '        """move_to should use the coordinates returned by the command-specific mock."""',
            "        result = runner.run(case[\"user_message\"], case[\"mock_factory\"]())",
            "        item = case[\"item\"]",
            "",
            "        checks = [(",
            "            \"move_to\",",
            "            {",
            "                \"obj_x\": item[\"obj_x\"],",
            "                \"obj_y\": item[\"obj_y\"],",
            "                \"coordinate_x\": item[\"coordinate_x\"],",
            "                \"coordinate_y\": item[\"coordinate_y\"],",
            "                \"coordinate_z\": item[\"coordinate_z\"],",
            "                \"coordinate_yaw\": item[\"coordinate_yaw\"],",
            "            },",
            "            _best_move_to_index(result, item),",
            "        )]",
            "",
            "        test_name = f\"fc_move_to_coords_{case['name']}\"",
            "        _record_fc_scores(result, eval_record, test_name, checks)",
            "        assert_tool_contains(result, \"move_to\")",
            "        pa = score_param_accuracy_multi(result, checks)",
            "        assert pa.score >= 0.5, f\"Param accuracy too low: {pa.details}\"",
            "",
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", output_path)


def _select_param_cases(candidates: list[dict], *, limit: int) -> list[dict]:
    """Select a representative parameter-check subset instead of filling by order.

    The generated commands heavily skew toward repeated move_to checks. For deep
    evaluation, prefer a compact but diverse subset spanning task category,
    furniture target, object kind, and action prefix.
    """
    if limit <= 0 or not candidates:
        return []

    remaining = sorted(
        candidates,
        key=lambda case: (
            case.get("category", ""),
            case.get("target_furniture", ""),
            case.get("object_kind", ""),
            case.get("name", ""),
        ),
    )
    selected: list[dict] = []
    seen_categories: set[str] = set()
    seen_furniture: set[str] = set()
    seen_object_kinds: set[str] = set()
    seen_action_prefixes: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    while remaining and len(selected) < limit:
        best_idx = 0
        best_score = None
        for idx, case in enumerate(remaining):
            action_prefix = _action_prefix(case.get("name", ""))
            category = case.get("category", "")
            furniture = case.get("target_furniture", "")
            object_kind = case.get("object_kind", "")
            pair = (category, furniture)
            score = (
                4 if category and category not in seen_categories else 0,
                3 if furniture and furniture not in seen_furniture else 0,
                2 if object_kind and object_kind not in seen_object_kinds else 0,
                2 if action_prefix and action_prefix not in seen_action_prefixes else 0,
                1 if category and furniture and pair not in seen_pairs else 0,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        category = chosen.get("category", "")
        furniture = chosen.get("target_furniture", "")
        object_kind = chosen.get("object_kind", "")
        action_prefix = _action_prefix(chosen.get("name", ""))
        if category:
            seen_categories.add(category)
        if furniture:
            seen_furniture.add(furniture)
        if object_kind:
            seen_object_kinds.add(object_kind)
        if action_prefix:
            seen_action_prefixes.add(action_prefix)
        if category and furniture:
            seen_pairs.add((category, furniture))

    return selected


def _action_prefix(test_name: str) -> str:
    if "_" not in test_name:
        return test_name
    return test_name.split("_", 1)[0]


def _expects_move_to_param_check(expected_seq, command_meta: dict) -> bool:
    category = command_meta.get("category", "")
    if category in {"查询检索", "整理折叠", "收纳归集", "复合任务"}:
        return False
    if _looks_like_push_adjustment(command_meta.get("text", "")):
        return False
    options = []
    if isinstance(expected_seq, list) and expected_seq and isinstance(expected_seq[0], str):
        options = [expected_seq]
    elif isinstance(expected_seq, list):
        options = expected_seq
    for option in options:
        if "move_to" in option:
            return True
    return False


def _write_test_response_quality(test_meta: dict, mock_meta: dict, output_path: Path) -> None:
    all_factory_names = set()
    for f in mock_meta.get("mock_factories", []):
        all_factory_names.add(f["name"])

    lines = [
        '"""Layer 1: Response text quality tests.',
        "",
        'Owns the **response_quality** scoring dimension.',
        '"""',
        "",
        "import pytest",
        "",
        "from evaluation import score_response_quality",
        "from mock_tool_results import (",
    ]

    for name in sorted(all_factory_names):
        lines.append(f"    {name},")
    lines.extend([
        ")",
        "",
        "",
        "def _record_quality_score(result, eval_record, test_name):",
        "    rq = score_response_quality(result)",
        '    eval_record(test_name, {"response_quality": rq.to_dict()}, result=result)',
        "    return rq",
        "",
        "",
        "class TestResponseQuality:",
        '    """Verify assistant response text quality across different instruction types."""',
        "",
    ])

    test_count = 0
    for cls in test_meta.get("test_classes", []):
        for t in cls.get("tests", []):
            if t["mock_factory"] not in all_factory_names:
                continue
            test_count += 1
            if test_count > 10:
                break
            method = f"test_rq_{t['method_name'].replace('test_', '')}"
            lines.extend([
                f'    def {method}(self, runner, eval_record):',
                f'        """Response quality for: {t["docstring"][:50]}"""',
                f'        result = runner.run("{t["user_message"]}", {t["mock_factory"]}())',
                f'        rq = _record_quality_score(result, eval_record, "rq_{t["method_name"].replace("test_", "")}")',
                f'        assert rq.score >= 0.4, f"Response quality too low: {{rq.details}}"',
                f'',
            ])
        if test_count > 10:
            break

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", output_path)
