"""Shared paths for the scene pipeline generator package.

The pipeline is intentionally rooted at this package directory. Generated
scene-eval datasets are written to the external evaluation kit data directory
unless callers override the output path.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
EVAL_SCENARIOS_DIR = PACKAGE_ROOT / "scenarios"
EVAL_GENERATED_DIR = Path(
    os.environ.get(
        "SCENE_PIPELINE_GENERATED_DIR",
        "/home/houxianzhou/kaiwu_workspace/scene-pipeline-eval-kit/data",
    )
).expanduser()
EVAL_IMAGE_ARCHIVE_DIR = PACKAGE_ROOT / "source_images"
