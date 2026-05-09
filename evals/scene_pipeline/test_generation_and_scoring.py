from pathlib import Path
import json
import importlib.util
import sys
import types

import pytest

from evals.scene_pipeline.pipeline import _ARTIFACT_NAMES
from evals.scene_pipeline.pipeline import _atomic_write_text as _pipeline_atomic_write_text
from evals.scene_pipeline.pipeline import _cleanup_stage_and_downstream
from evals.scene_pipeline.pipeline import _ensure_generated_tests_nonempty
from evals.scene_pipeline.pipeline import _is_valid_tests_artifact
from evals.scene_pipeline.pipeline import _stage_or_downstream_outputs_exist
from evals.scene_pipeline.batch import resolve_single_scene_dir
from evals.scene_pipeline.paths import EVAL_GENERATED_DIR
from evals.scene_pipeline.steps.generate_commands import write_user_command_md
from evals.scene_pipeline.skill_registry import get_predefined_tool_names, get_scene_skills
from evals.scene_pipeline.steps.generate_server import generate_server
from evals.scene_pipeline.steps.generate_tests import (
    EVAL_PROTOCOL_VERSION,
    _augment_test_meta_with_semantic_variants,
    _collect_expected_for_text,
    _compound_expected_for_text,
    _expects_move_to_param_check,
    _has_closed_container_target,
    _place_expected_for_text,
    _stabilize_test_meta,
    _tidy_expected_for_text,
    _write_test_task_planning,
)


class _NoopLLMClient:
    def chat_json(self, *args, **kwargs):
        return {}

    def chat_text(self, *args, **kwargs):
        return ""


def _load_real_home_living_module(module_name: str, relative_path: str):
    root = Path(__file__).parent / "scenarios" / "real_home_living" / "tests"
    path = root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_generation_dir_points_to_eval_kit_data(tmp_path) -> None:
    assert EVAL_GENERATED_DIR == Path(
        "/home/houxianzhou/kaiwu_workspace/scene-pipeline-eval-kit/data"
    )
    image_path = tmp_path / "home_scene.png"
    image_path.write_bytes(b"image")
    scene_dir = resolve_single_scene_dir(None, image_path)
    assert scene_dir.parent == EVAL_GENERATED_DIR
    assert scene_dir.name.startswith("scene_home_scene_")


def test_write_test_task_planning_emits_single_failure_recovery_class(tmp_path) -> None:
    output_path = tmp_path / "test_task_planning.py"
    test_meta = {
        "test_classes": [
            {
                "name": "TestFailureRecovery",
                "description": "Verify failure recovery planning for grasp/move/memory fault scenarios.",
                "tests": [
                    {
                        "method_name": "test_grasp_fail_retry_camera",
                        "docstring": "grasp fail",
                        "user_message": "抓失败重试",
                        "mock_factory": "grasp_fail_retry_camera_mock",
                        "expected_subsequence": ["spatial_memory_query_vec", "grasp_start", "grasp_start"],
                        "min_calls": 3,
                    }
                ],
            }
        ],
        "failure_tests": [
            {
                "method_name": "test_grasp_fail_retry_camera",
                "docstring": "grasp fail",
                "user_message": "抓失败重试",
                "mock_factory": "grasp_fail_retry_camera_mock",
                "expected_subsequence": ["spatial_memory_query_vec", "grasp_start", "grasp_start"],
                "min_calls": 3,
                "assert_min_calls": {"grasp_start": 2},
            }
        ],
    }
    mock_meta = {
        "mock_factories": [],
        "failure_factories": [{"name": "grasp_fail_retry_camera_mock"}],
    }

    _write_test_task_planning(test_meta, mock_meta, output_path)
    content = output_path.read_text(encoding="utf-8")

    assert content.count("class TestFailureRecovery:") == 1
    assert content.count("def test_grasp_fail_retry_camera") == 1
    assert 'assert len(grasp_start_calls) >= 2' in content
    assert f'EVAL_PROTOCOL_VERSION = "{EVAL_PROTOCOL_VERSION}"' in content
    assert "score = score_tool_sequence(result, option)" in content
    assert "def _assert_no_forbidden_tools(result, forbidden):" in content


def test_generated_tests_artifact_requires_current_protocol_marker(tmp_path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    for name in ("test_task_planning.py", "test_function_calling.py", "test_response_quality.py"):
        (tests_dir / name).write_text(
            "class TestGenerated:\n"
            "    def test_example(self):\n"
            "        pass\n",
            encoding="utf-8",
        )

    assert not _is_valid_tests_artifact(tests_dir)

    (tests_dir / "test_task_planning.py").write_text(
        f'EVAL_PROTOCOL_VERSION = "{EVAL_PROTOCOL_VERSION}"\n'
        "class TestGenerated:\n"
        "    def test_example(self):\n"
        "        pass\n",
        encoding="utf-8",
    )

    assert _is_valid_tests_artifact(tests_dir)


def test_commands_resume_cleanup_removes_stage_and_downstream_outputs(tmp_path) -> None:
    scene_dir = tmp_path / "scene"
    tests_dir = scene_dir / "tests"
    tests_dir.mkdir(parents=True)
    files = [
        scene_dir / "user_command.md",
        scene_dir / _ARTIFACT_NAMES["commands_data"],
        scene_dir / "demo_scene.yaml",
        scene_dir / _ARTIFACT_NAMES["mock_meta"],
        tests_dir / "mock_tool_results.py",
        tests_dir / "mock_server.py",
        tests_dir / "test_task_planning.py",
        tests_dir / "test_function_calling.py",
        tests_dir / "test_response_quality.py",
    ]
    for path in files:
        path.write_text("stale", encoding="utf-8")

    assert _stage_or_downstream_outputs_exist(
        "commands", scene_dir, tests_dir, scene_name="demo_scene",
    )

    _cleanup_stage_and_downstream(
        "commands", scene_dir, tests_dir, scene_name="demo_scene",
    )

    assert all(not path.exists() for path in files)


def test_pipeline_atomic_write_replaces_json_without_tmp_file(tmp_path) -> None:
    output_path = tmp_path / "commands_data.json"

    _pipeline_atomic_write_text(output_path, json.dumps({"commands": [{"text": "旧"}]}, ensure_ascii=False))
    _pipeline_atomic_write_text(output_path, json.dumps({"commands": [{"text": "新"}]}, ensure_ascii=False))

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"commands": [{"text": "新"}]}
    assert not list(tmp_path.glob(".commands_data.json.*.tmp"))


def test_write_user_command_md_is_atomic_and_replaces_existing_file(tmp_path) -> None:
    output_path = tmp_path / "user_command.md"
    output_path.write_text("旧内容", encoding="utf-8")

    write_user_command_md(
        {
            "commands": [
                {"category": "拿取递送", "text": "把杯子递给我"},
            ],
            "failure_scenarios": [
                {"description": "抓取失败", "text": "如果没抓住杯子就重试"},
            ],
        },
        output_path,
    )

    content = output_path.read_text(encoding="utf-8")
    assert "旧内容" not in content
    assert '"把杯子递给我"' in content
    assert '抓取失败："如果没抓住杯子就重试"' in content
    assert not list(tmp_path.glob(".user_command.md.*.tmp"))


def test_predefined_scene_tools_are_deduped_before_server_generation(tmp_path) -> None:
    predefined_skills, category = get_scene_skills("kitchen")

    assert category == "home"
    predefined_tool_names = get_predefined_tool_names(predefined_skills)
    assert predefined_tool_names.count("insert") == 1
    assert predefined_tool_names.count("stack") == 1
    assert predefined_tool_names.count("unstack") == 1

    output_path = tmp_path / "server.py"
    all_tools = generate_server(
        _NoopLLMClient(),
        {
            "scene_display_name": "厨房",
            "extra_tools": ["stack", "stack"],
        },
        output_path,
        predefined_skills=predefined_skills,
    )

    assert all_tools.count("insert") == 1
    assert all_tools.count("stack") == 1
    assert all_tools.count("unstack") == 1

    server_code = output_path.read_text(encoding="utf-8")
    assert server_code.count("def insert(") == 1
    assert server_code.count("def stack(") == 1
    assert server_code.count("def unstack(") == 1


def test_semantic_variant_stabilization_uses_source_command() -> None:
    commands_data = {
        "commands": [
            {
                "test_name": "fetch_cup",
                "text": "把茶几上的杯子递给我。",
                "category": "拿取递送",
                "target_object": "cup",
                "target_furniture": "coffee_table",
            }
        ]
    }
    test_meta = {
        "test_classes": [
            {
                "name": "TestFetchAndDeliver",
                "description": "fetch tests",
                "tests": [
                    {
                        "method_name": "test_fetch_cup",
                        "docstring": "把茶几上的杯子递给我。",
                        "user_message": "把茶几上的杯子递给我。",
                        "mock_factory": "fetch_cup_mock",
                        "expected_subsequence": "FETCH_EXPECTED",
                        "forbidden_tools": ["place_down"],
                        "min_calls": 6,
                    }
                ],
            }
        ],
        "failure_tests": [],
    }

    augmented = _augment_test_meta_with_semantic_variants(test_meta, commands_data)
    stabilized = _stabilize_test_meta(augmented, commands_data)
    tests = stabilized["test_classes"][0]["tests"]
    variants = [test for test in tests if test["method_name"] != "test_fetch_cup"]

    assert variants
    assert all(
        test["expected_subsequence"] == [
            "spatial_memory_query_vec", "move_to", "grasp_start", "handover",
        ]
        for test in variants
    )


def test_push_adjustment_place_expected_and_param_filter() -> None:
    expected = _place_expected_for_text("把花瓶挪到斗柜左侧一点。")

    assert ["spatial_memory_query_vec", "move_to", "push_item"] in expected
    assert ["spatial_memory_query_vec", "move_to", "push"] in expected
    assert not _expects_move_to_param_check(
        expected,
        {"category": "放置移动", "text": "把花瓶挪到斗柜左侧一点。"},
    )


def test_collect_expected_distinguishes_box_from_surface_tidy() -> None:
    boxed = _collect_expected_for_text("把地毯上的白色运动鞋收进收纳箱")
    unboxed = _collect_expected_for_text("把开放式壁柜中上层的护肤品瓶罐都收在一起")

    assert ["fetch_box", "spatial_memory_query_vec", "move_to", "grasp_start", "put_box"] in boxed
    assert ["spatial_memory_query_vec", "move_to", "tidy_surface"] in unboxed
    assert ["spatial_memory_query_vec", "move_to", "sort"] in unboxed


def test_tidy_expected_accepts_surface_action_aliases() -> None:
    expected = _tidy_expected_for_text("把抽屉柜上的米白色和蓝色花瓶都整理成一排")

    assert ["spatial_memory_query_vec", "move_to", "tidy_surface"] in expected
    assert ["spatial_memory_query_vec", "move_to", "align"] in expected
    assert ["spatial_memory_query_vec", "move_to", "sort"] in expected
    assert ["spatial_memory_query_vec", "move_to", "clean"] in expected


def test_tidy_expected_accepts_push_and_fold_aliases_from_bad_cases() -> None:
    secure = _tidy_expected_for_text("把右侧墙边的蓝色冲浪板靠稳一点")
    folded = _tidy_expected_for_text("把灰绿色毯子收拢成一条直线")
    unblock = _tidy_expected_for_text("把黄色花束的位置整理一下，别挡着相框")

    assert ["spatial_memory_query_vec", "move_to", "push_item"] in secure
    assert ["spatial_memory_query_vec", "move_to", "push"] in secure
    assert ["spatial_memory_query_vec", "move_to", "push_item"] in unblock
    assert ["spatial_memory_query_vec", "move_to", "push"] in unblock
    assert ["spatial_memory_query_vec", "move_to", "fold_item"] in folded
    assert ["spatial_memory_query_vec", "move_to", "fold"] in folded


def test_tidy_expected_accepts_cleaning_aliases() -> None:
    expected = _tidy_expected_for_text("把洗手盆右侧台面整理干净一点")

    assert ["spatial_memory_query_vec", "move_to", "clean"] in expected
    assert ["spatial_memory_query_vec", "move_to", "wipe"] in expected


def test_storage_box_object_is_not_closed_container_target() -> None:
    assert not _has_closed_container_target("把抽屉柜顶部的花瓶和书本收纳盒分区摆齐")
    assert _has_closed_container_target("把抽屉柜顶部的花瓶收进收纳盒")


def test_stabilization_keeps_compound_tidy_then_confirm_as_tidy() -> None:
    commands_data = {
        "commands": [
            {
                "test_name": "align_storage_jars_then_confirm_black_teapot",
                "text": "把开放搁板上层的玻璃储物罐排整齐，再确认黑色茶壶在下层右侧",
                "category": "复合任务",
                "target_object": "玻璃储物罐",
                "target_furniture": "开放搁板",
            }
        ]
    }
    test_meta = {
        "test_classes": [
            {
                "name": "TestComplex",
                "description": "complex tests",
                "tests": [
                    {
                        "method_name": "test_align_storage_jars_then_confirm_black_teapot",
                        "docstring": "把开放搁板上层的玻璃储物罐排整齐，再确认黑色茶壶在下层右侧",
                        "user_message": "把开放搁板上层的玻璃储物罐排整齐，再确认黑色茶壶在下层右侧",
                        "mock_factory": "align_storage_jars_then_confirm_black_teapot_mock",
                        "expected_subsequence": ["spatial_memory_query_vec", "move_to", "sort", "spatial_memory_query_vec"],
                        "forbidden_tools": ["handover", "grasp_start"],
                        "min_calls": 4,
                    }
                ],
            }
        ],
        "failure_tests": [],
    }

    stabilized = _stabilize_test_meta(test_meta, commands_data)
    expected = stabilized["test_classes"][0]["tests"][0]["expected_subsequence"]

    assert ["spatial_memory_query_vec", "move_to", "tidy_surface"] in expected
    assert ["spatial_memory_query_vec", "move_to", "align"] in expected
    assert ["spatial_memory_query_vec", "move_to", "sort"] in expected
    assert all("handover" not in option for option in expected)
    assert all("grasp_start" not in option for option in expected)


def test_stabilization_does_not_turn_unboxed_storage_box_object_into_box_collect() -> None:
    commands_data = {
        "commands": [
            {
                "test_name": "group_vases_and_box_on_dresser",
                "text": "把抽屉柜顶部的花瓶和书本收纳盒分区摆齐",
                "category": "收纳归集",
                "target_object": "花瓶和书本收纳盒",
                "target_furniture": "抽屉柜",
            }
        ]
    }
    test_meta = {
        "test_classes": [
            {
                "name": "TestCollectAndOrganize",
                "description": "collect tests",
                "tests": [
                    {
                        "method_name": "test_group_vases_and_box_on_dresser",
                        "docstring": "把抽屉柜顶部的花瓶和书本收纳盒分区摆齐",
                        "user_message": "把抽屉柜顶部的花瓶和书本收纳盒分区摆齐",
                        "mock_factory": "group_vases_and_box_on_dresser_mock",
                        "expected_subsequence": ["spatial_memory_query_vec", "move_to", "sort"],
                        "forbidden_tools": ["handover"],
                        "min_calls": 4,
                    }
                ],
            }
        ],
        "failure_tests": [],
    }

    stabilized = _stabilize_test_meta(test_meta, commands_data)
    expected = stabilized["test_classes"][0]["tests"][0]["expected_subsequence"]

    assert ["spatial_memory_query_vec", "move_to", "tidy_surface"] in expected
    assert ["spatial_memory_query_vec", "move_to", "sort"] in expected
    assert not any(option and option[0] == "fetch_box" for option in expected)


def test_compound_expected_no_longer_requires_preface_query() -> None:
    expected = _compound_expected_for_text("把搭毯叠好再整理到梯架中部")

    assert expected
    assert ["spatial_memory_query_vec", "move_to", "fold_item", "tidy_surface"] in expected
    assert ["spatial_memory_query_vec", "move_to", "fold", "sort"] in expected


def test_compound_expected_allows_fetch_handover_and_move_then_tidy() -> None:
    fetch_expected = _compound_expected_for_text("把冰箱门上的白色便签取下来递给我，然后不要放到别处")
    mixed_expected = _compound_expected_for_text("把沙发上的黄色玩偶拿到单人椅上，然后把单人椅周围整理一下")

    assert fetch_expected == ["spatial_memory_query_vec", "move_to", "grasp_start", "handover"]
    assert ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down"] in mixed_expected
    assert ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down", "tidy_surface"] in mixed_expected


def test_score_tool_sequence_penalizes_extra_calls() -> None:
    fake_conftest = types.ModuleType("conftest")

    class ConversationResult:
        def __init__(self, tool_calls, messages=None, usage=None):
            self.tool_calls = tool_calls
            self.messages = messages or []
            self.usage = usage or {}

        @property
        def tool_names(self):
            return [name for name, _ in self.tool_calls]

        def get_calls_for(self, tool_name):
            return [args for name, args in self.tool_calls if name == tool_name]

        @property
        def turn_count(self):
            return 0

        @property
        def total_tool_calls_count(self):
            return len(self.tool_calls)

        @property
        def assistant_messages(self):
            return []

    fake_conftest.ConversationResult = ConversationResult
    original = sys.modules.get("conftest")
    sys.modules["conftest"] = fake_conftest
    try:
        module = _load_real_home_living_module(
            "scene_eval_test_module",
            "evaluation.py",
        )
    finally:
        if original is not None:
            sys.modules["conftest"] = original
        else:
            sys.modules.pop("conftest", None)

    result = ConversationResult(
        [
            ("spatial_memory_query_vec", {}),
            ("move_to", {}),
            ("scene_recognition", {}),
            ("grasp_start", {}),
            ("perception_custom", {}),
            ("handover", {}),
            ("back_station", {}),
        ]
    )
    score = module.score_tool_sequence(
        result,
        [
            "spatial_memory_query_vec",
            "move_to",
            "scene_recognition",
            "grasp_start",
            "perception_custom",
            "handover",
        ],
    )

    assert score.score < 1.0
    assert score.sub_scores["extra_total"] == 1
    assert score.sub_scores["extra_penalty"] > 0


def test_score_tool_sequence_treats_baseline_support_calls_as_neutral() -> None:
    fake_conftest = types.ModuleType("conftest")

    class ConversationResult:
        def __init__(self, tool_calls, messages=None, usage=None):
            self.tool_calls = tool_calls
            self.messages = messages or []
            self.usage = usage or {}

        @property
        def tool_names(self):
            return [name for name, _ in self.tool_calls]

        def get_calls_for(self, tool_name):
            return [args for name, args in self.tool_calls if name == tool_name]

        @property
        def turn_count(self):
            return 0

        @property
        def total_tool_calls_count(self):
            return len(self.tool_calls)

        @property
        def assistant_messages(self):
            return []

    fake_conftest.ConversationResult = ConversationResult
    original = sys.modules.get("conftest")
    sys.modules["conftest"] = fake_conftest
    try:
        module = _load_real_home_living_module(
            "scene_eval_neutral_support_module",
            "evaluation.py",
        )
    finally:
        if original is not None:
            sys.modules["conftest"] = original
        else:
            sys.modules.pop("conftest", None)

    result = ConversationResult(
        [
            ("fetch_box", {}),
            ("spatial_memory_query_vec", {}),
            ("move_to", {}),
            ("scene_recognition", {}),
            ("grasp_start", {}),
            ("delete_spatial_memory_by_id", {}),
            ("put_box", {}),
        ]
    )
    sequence = module.score_tool_sequence(
        result,
        ["fetch_box", "move_to", "grasp_start", "put_box"],
    )
    efficiency = module.score_efficiency(result, 4)

    assert sequence.score == 1.0
    assert sequence.sub_scores["extra_total"] == 0
    assert sequence.sub_scores["neutral_extra_total"] == 3
    assert efficiency.score == 1.0
    assert efficiency.sub_scores["billable_calls"] == 4


def test_score_tool_sequence_matches_semantic_tool_aliases() -> None:
    fake_conftest = types.ModuleType("conftest")

    class ConversationResult:
        def __init__(self, tool_calls, messages=None, usage=None):
            self.tool_calls = tool_calls
            self.messages = messages or []
            self.usage = usage or {}

        @property
        def tool_names(self):
            return [name for name, _ in self.tool_calls]

        @property
        def turn_count(self):
            return 0

        @property
        def total_tool_calls_count(self):
            return len(self.tool_calls)

        @property
        def assistant_messages(self):
            return []

    fake_conftest.ConversationResult = ConversationResult
    original = sys.modules.get("conftest")
    sys.modules["conftest"] = fake_conftest
    try:
        module = _load_real_home_living_module(
            "scene_eval_tool_alias_module",
            "evaluation.py",
        )
    finally:
        if original is not None:
            sys.modules["conftest"] = original
        else:
            sys.modules.pop("conftest", None)

    push_result = ConversationResult([
        ("spatial_memory_query_vec", {}),
        ("move_to", {}),
        ("push", {}),
    ])
    fold_result = ConversationResult([
        ("spatial_memory_query_vec", {}),
        ("move_to", {}),
        ("fold", {}),
    ])
    insert_result = ConversationResult([
        ("spatial_memory_query_vec", {}),
        ("move_to", {}),
        ("grasp_start", {}),
        ("insert", {}),
    ])

    assert module.score_tool_sequence(
        push_result,
        ["spatial_memory_query_vec", "move_to", "push_item"],
    ).score == 1.0
    assert module.score_tool_sequence(
        fold_result,
        ["spatial_memory_query_vec", "move_to", "fold_item"],
    ).score == 1.0
    assert module.score_tool_sequence(
        ConversationResult([
            ("spatial_memory_query_vec", {}),
            ("move_to", {}),
            ("wipe", {}),
        ]),
        ["spatial_memory_query_vec", "move_to", "clean"],
    ).score == 1.0
    assert module.score_tool_sequence(
        insert_result,
        ["spatial_memory_query_vec", "move_to", "grasp_start", "place_down"],
    ).score == 1.0


def test_score_efficiency_penalizes_excessive_support_calls() -> None:
    fake_conftest = types.ModuleType("conftest")

    class ConversationResult:
        def __init__(self, tool_calls, messages=None, usage=None):
            self.tool_calls = tool_calls
            self.messages = messages or []
            self.usage = usage or {}

        @property
        def tool_names(self):
            return [name for name, _ in self.tool_calls]

        @property
        def turn_count(self):
            return 0

        @property
        def total_tool_calls_count(self):
            return len(self.tool_calls)

        @property
        def assistant_messages(self):
            return []

    fake_conftest.ConversationResult = ConversationResult
    original = sys.modules.get("conftest")
    sys.modules["conftest"] = fake_conftest
    try:
        module = _load_real_home_living_module(
            "scene_eval_efficiency_allowance_module",
            "evaluation.py",
        )
    finally:
        if original is not None:
            sys.modules["conftest"] = original
        else:
            sys.modules.pop("conftest", None)

    result = ConversationResult(
        [
            ("spatial_memory_query_vec", {}),
            ("trigger_spatial_processing", {}),
            ("spatial_memory_query_vec", {}),
            ("trigger_spatial_processing", {}),
            ("spatial_memory_query_vec", {}),
            ("trigger_spatial_processing", {}),
            ("spatial_memory_query_vec", {}),
            ("move_to", {}),
            ("grasp_start", {}),
            ("handover", {}),
        ]
    )
    efficiency = module.score_efficiency(result, 3)

    assert efficiency.sub_scores["free_support_calls"] == 4
    assert efficiency.sub_scores["billable_calls"] == 6
    assert efficiency.score < 1.0


def test_ensure_generated_tests_nonempty_rejects_duplicate_class_names(tmp_path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_task_planning.py").write_text(
        """
class TestDup:
    def test_a(self):
        pass

class TestDup:
    def test_b(self):
        pass
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tests_dir / "test_function_calling.py").write_text(
        "def test_fc():\n    pass\n",
        encoding="utf-8",
    )
    (tests_dir / "test_response_quality.py").write_text(
        "def test_rq():\n    pass\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="重复测试类"):
        _ensure_generated_tests_nonempty(tests_dir)


def test_ensure_generated_tests_nonempty_rejects_duplicate_test_names(tmp_path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_task_planning.py").write_text(
        """
class TestDup:
    def test_a(self):
        pass

    def test_a(self):
        pass
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tests_dir / "test_function_calling.py").write_text(
        "def test_fc():\n    pass\n",
        encoding="utf-8",
    )
    (tests_dir / "test_response_quality.py").write_text(
        "def test_rq():\n    pass\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="重复测试方法"):
        _ensure_generated_tests_nonempty(tests_dir)
