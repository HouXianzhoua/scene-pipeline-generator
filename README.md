# Scene Pipeline Generator

Standalone generator for scene evaluation datasets.

## Scope

- Contains only the dataset generation pipeline and generation UI.
- Does not contain evaluation runtime or report UI.
- Writes generated `scene_*` datasets to the evaluation runtime repository by default.

## Environment Setup

Use a dedicated conda environment so local user-site packages do not affect the
generator.

```bash
conda create -n scene-pipeline python=3.12 -y
conda activate scene-pipeline
python -m pip install -r requirements.txt
```

The UI launch scripts also expect a conda environment named `scene-pipeline`.
If you use a different name, pass it with `CONDA_ENV`:

```bash
CONDA_ENV=my-env bash scripts/run_generate_ui.sh
```

## Configuration

The launch scripts read these environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL` | `gpt-5.5` | Model used for scene analysis and code generation. |
| `BASE_URL` | `https://api.chatanywhere.tech/v1` | OpenAI-compatible API base URL used by the scripts. |
| `API_KEY` | empty | API key passed to the LLM client. Do not commit real keys. |
| `OUTPUT_DIR` | `/home/houxianzhou/kaiwu_workspace/scene-pipeline-eval-kit/data` | Directory where generated scene datasets are written. Override this on other machines. |
| `GRADIO_SERVER_PORT` | `7860` | Port for the Gradio UI. Use another port if it is occupied. |
| `CONDA_ENV` | `scene-pipeline` | Conda environment used by the launch scripts. |

The Python CLI defaults are defined in `scene_pipeline/config.py`. The scripts
override the base URL default for the UI startup command.

## Run The UI

Start the generation UI:

```bash
API_KEY=$YOUR_API_KEY \
OUTPUT_DIR=/path/to/scene-pipeline-eval-kit/data \
bash scripts/run_generate_ui.sh
```

If port `7860` is already in use:

```bash
GRADIO_SERVER_PORT=7861 \
API_KEY=$YOUR_API_KEY \
OUTPUT_DIR=/path/to/scene-pipeline-eval-kit/data \
bash scripts/run_generate_ui.sh
```

The script will re-enter the configured conda environment automatically and set
`PYTHONNOUSERSITE=1` so packages from `~/.local` are ignored.

## Run The CLI

Generate from one image:

```bash
conda activate scene-pipeline
python -m scene_pipeline \
  --image /path/to/scene.jpg \
  --output-dir /path/to/scene-pipeline-eval-kit/data \
  --model gpt-5.5 \
  --base-url https://api.chatanywhere.tech/v1 \
  --api-key $YOUR_API_KEY
```

Generate from 1-5 images as one batch:

```bash
conda activate scene-pipeline
python -m scene_pipeline \
  --image /path/to/home1.jpg,/path/to/home2.jpg \
  --output-dir /path/to/scene-pipeline-eval-kit/data \
  --model gpt-5.5 \
  --base-url https://api.chatanywhere.tech/v1 \
  --api-key $YOUR_API_KEY
```

Resume a previous output directory:

```bash
python -m scene_pipeline \
  --image /path/to/scene.jpg \
  --output-dir /path/to/existing-scene-dir \
  --resume \
  --api-key $YOUR_API_KEY
```

## Default Output

Generated datasets are written to this path unless callers override it:

```text
/home/houxianzhou/kaiwu_workspace/scene-pipeline-eval-kit/data
```

On another machine, override it:

```bash
OUTPUT_DIR=/path/to/data bash scripts/run_generate_ui.sh
```

The package-level fallback can also be overridden with
`SCENE_PIPELINE_GENERATED_DIR` when using lower-level code paths.

## Verify Installation

```bash
conda run -n scene-pipeline python -s -c "import scene_pipeline.web_ui; print('ui imports ok')"
conda run -n scene-pipeline python -s -m pip check
```

Expected result:

```text
ui imports ok
No broken requirements found.
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'huggingface_hub'`: install
  dependencies in the conda environment with `python -m pip install -r requirements.txt`.
- `Cannot find empty port in range: 7860-7860`: start with
  `GRADIO_SERVER_PORT=7861` or another free port.
- `conda: command not found`: install Miniforge/Miniconda, or run the CLI with
  another Python environment that has `requirements.txt` installed.
- Generated files go to the wrong place: set `OUTPUT_DIR` explicitly.

Legacy entry points under `evals.scene_pipeline` remain as compatibility
wrappers, but new commands should use `scene_pipeline`.
