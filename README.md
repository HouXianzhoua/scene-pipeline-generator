# Scene Pipeline Generator

Standalone generator for scene evaluation datasets.

## Scope

- Contains only the dataset generation pipeline and generation UI.
- Does not contain evaluation runtime or report UI.
- Writes generated `scene_*` datasets to the evaluation runtime repository by default.

## Default Output

Generated datasets are written to:

```text
/home/houxianzhou/kaiwu_workspace/scene-pipeline-eval-kit/data
```

Override it when needed:

```bash
OUTPUT_DIR=/path/to/data bash scripts/run_generate_ui.sh
```

## Quick Start

```bash
python3 -m pip install -r requirements.txt
bash scripts/run_generate_ui.sh
```

CLI generation:

```bash
python3 -m scene_pipeline --image /path/to/scene.jpg --output-dir /home/houxianzhou/kaiwu_workspace/scene-pipeline-eval-kit/data
```

Legacy entry points under `evals.scene_pipeline` remain as compatibility
wrappers, but new commands should use `scene_pipeline`.
