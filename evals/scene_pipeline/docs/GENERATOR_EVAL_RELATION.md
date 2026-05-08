# Scene Pipeline 生成与评测边界说明

`evals/scene_pipeline` 在 `scene-pipeline-eval-kit` 中负责“评测链路生成”能力，生成产物直接写入同一仓库的 `data/`，供评测运行 UI/CLI 使用。

## 当前边界

- 生成侧：
  - 图片分析
  - 工具、配置、mock、server 生成
  - `tests/test_*.py` 等评测链路文件生成
  - 生成 CLI / 生成 UI
- 评测侧：
  - 运行评测 CLI
  - 评测 UI
  - 批量评测汇总逻辑
  - 报告汇总逻辑

## 当前入口

- CLI: `python -m evals.scene_pipeline --image ...`
- 生成 UI: `bash evals/scene_pipeline/run_generate_ui.sh`
- 生成 UI 等价入口: `bash evals/scene_pipeline/run_ui.sh`

## 生成产物

生成完成后，每个场景会产出一个 `scene_*` 目录，核心文件包括：

- `server.py`
- `commands_data.json`
- `scene_analysis.json`
- `*.yaml`
- `tests/conftest.py`
- `tests/evaluation.py`
- `tests/mock_tool_results.py`
- `tests/test_task_planning.py`
- `tests/test_function_calling.py`
- `tests/test_response_quality.py`

这些目录默认写入 `scene-pipeline-eval-kit/data/`，也是评测运行代码读取的数据集目录。

## 评测侧位置

评测运行入口位于 `scene-pipeline-eval-kit/run_eval_ui.sh` 和 `scene-pipeline-eval-kit/eval_runner.py`。
