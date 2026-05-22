#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-scene-pipeline}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" && "${SCENE_PIPELINE_CONDA_RUN:-}" != "1" ]]; then
    export SCENE_PIPELINE_CONDA_RUN=1
    export PYTHONNOUSERSITE=1
    exec conda run --no-capture-output -n "$CONDA_ENV" bash "$0" "$@"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

MODEL="${MODEL:-gpt-5.5}"
BASE_URL="${BASE_URL:-https://api.chatanywhere.tech/v1}"
API_KEY="${API_KEY:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/houxianzhou/kaiwu_workspace/scene-pipeline-eval-kit/data}"

cd "$REPO_ROOT"

echo "Starting Scene Pipeline UI on http://127.0.0.1:${GRADIO_SERVER_PORT:-7860}"

python -m scene_pipeline --generate-ui \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --output-dir "$OUTPUT_DIR" \
    --resume \
    "$@"
