#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${MODEL:-gpt-5.5}"
BASE_URL="${BASE_URL:-https://api.chatanywhere.tech/v1}"
API_KEY="${API_KEY:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/houxianzhou/kaiwu_workspace/scene-pipeline-eval-kit/data}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

cd "$REPO_ROOT"

python3 -m evals.scene_pipeline --generate-ui \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --output-dir "$OUTPUT_DIR" \
    --resume \
    "$@"
