"""Legacy module entry point for ``python -m evals.scene_pipeline``."""

from __future__ import annotations

import sys

from scene_pipeline.__main__ import main


if __name__ == "__main__":
    sys.exit(main())
