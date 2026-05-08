"""Shared paths for the scene pipeline package.

The pipeline is intentionally rooted at this package directory, not at the
hosting SDK repository. This lets ``evals/scene_pipeline`` be copied into a
standalone repository without changing generated/evaluation paths.
"""

from __future__ import annotations

import os
from pathlib import Path


EVALS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVALS_ROOT.parent.parent if EVALS_ROOT.parent.name == "evals" else EVALS_ROOT.parent
EVAL_SCENARIOS_DIR = EVALS_ROOT / "scenarios"
EVAL_GENERATED_DIR = Path(
    os.environ.get(
        "SCENE_PIPELINE_GENERATED_DIR",
        "/home/houxianzhou/kaiwu_workspace/scene-pipeline-eval-kit/data",
    )
).expanduser()
EVAL_IMAGE_ARCHIVE_DIR = EVALS_ROOT / "source_images"
