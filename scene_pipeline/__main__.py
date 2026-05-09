"""Allow running as: python -m scene_pipeline [--generate-ui ...]."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    ui_mode = None
    if "--generate-ui" in argv:
        argv.remove("--generate-ui")
        ui_mode = "generate"

    if ui_mode is not None:
        from .web_ui import launch, parse_ui_args

        args = parse_ui_args(argv)
        launch(cli_args=args, mode=ui_mode)
        return 0

    from .cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
