"""Allow running as: python -m evals.scene_pipeline [--generate-ui ...]"""

import sys

ui_mode = None
if "--generate-ui" in sys.argv:
    sys.argv.remove("--generate-ui")
    ui_mode = "generate"

if ui_mode is not None:
    from .web_ui import launch, parse_ui_args

    args = parse_ui_args()
    launch(cli_args=args, mode=ui_mode)
else:
    from .cli import main

    sys.exit(main())
