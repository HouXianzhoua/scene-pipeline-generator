"""CLI entry point for the scene pipeline generator."""

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    DEFAULT_GEN_BASE_URL,
    DEFAULT_GEN_MODEL,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scene Pipeline: generate scene-eval assets from scene photos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Generate one scene-eval chain with default settings
  python -m scene_pipeline --image photo.jpg

  # Generate 1-5 photos as one batch
  python -m scene_pipeline \\
      --image home1.jpg,home2.jpg \\
      --output-dir ./data

  # Use GPT-4o for generation
  python -m scene_pipeline \\
      --image photo.jpg \\
      --model gpt-4o --base-url https://api.openai.com/v1 --api-key $KEY

  # Resume a failed run (skip steps that already have output)
  python -m scene_pipeline \\
      --image photo.jpg --resume \\
      --output-dir ./data/my_scene \\
      --model gpt-4o --api-key $KEY
""",
    )

    gen_group = parser.add_argument_group("Generation")
    gen_group.add_argument(
        "--image", type=str,
        help="Path to one scene photo, or comma-separated paths for 1-5 photos",
    )
    gen_group.add_argument(
        "--provider", type=str, default="openai",
        choices=["openai", "anthropic", "local"],
        help="LLM provider for generation (default: openai)",
    )
    gen_group.add_argument(
        "--model", type=str, default=DEFAULT_GEN_MODEL,
        help=f"Model for scene analysis and code generation (default: {DEFAULT_GEN_MODEL})",
    )
    gen_group.add_argument(
        "--base-url", type=str, default=DEFAULT_GEN_BASE_URL,
        help=f"LLM API base URL (default: {DEFAULT_GEN_BASE_URL})",
    )
    gen_group.add_argument(
        "--api-key", type=str, default="EMPTY",
        help="LLM API key",
    )
    gen_group.add_argument(
        "--vision-model", type=str, default=None,
        help="Separate model for vision tasks (default: same as --model)",
    )
    gen_group.add_argument(
        "--max-tokens", type=int, default=16384,
        help="Max output tokens per LLM call (default: 16384)",
    )
    gen_group.add_argument(
        "--scene-category", type=str, default=None,
        choices=["home", "office", "retail", "industrial"],
        help="Override auto-detected scene skill category. "
             "When omitted, the category is inferred from scene_type "
             "(e.g. living_room → home).",
    )

    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: auto-generated under the eval kit data/ directory)",
    )
    out_group.add_argument(
        "--resume", action="store_true",
        help="Resume a previous run: skip steps whose output already exists. "
             "Requires --output-dir.",
    )
    out_group.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    if not args.image:
        parser.error("--image is required")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger("scene_pipeline")
    logger.info("Scene Pipeline starting...")

    from .batch import parse_image_inputs

    try:
        image_paths = parse_image_inputs(args.image) if args.image else []
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    from .batch import resolve_single_scene_dir, run_batch_pipeline
    from .llm_client import LLMClient

    if len(image_paths) > 1 and not args.output_dir:
        logger.error("--output-dir is required when --image contains multiple photos")
        return 1
    for image_path in image_paths:
        if not image_path.exists():
            logger.error("Image file not found: %s", image_path)
            return 1

    client = LLMClient(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        vision_model=args.vision_model,
        max_tokens=args.max_tokens,
    )

    single_mode = len(image_paths) == 1
    generation_output_dir = (
        resolve_single_scene_dir(args.output_dir, image_paths[0])
        if single_mode
        else Path(args.output_dir)
    )
    payload = run_batch_pipeline(
        image_paths=image_paths,
        client=client,
        output_dir=Path(generation_output_dir),
        resume=args.resume,
        scene_category=args.scene_category,
    )

    scenes = payload.get("scenes", [])
    if not scenes:
        logger.error("Pipeline failed to produce output directory")
        return 1

    if single_mode:
        scene = scenes[0]
        scene_dir = Path(scene["scene_dir"])
        generation_errors = scene.get("generation_errors") or []
        if generation_errors:
            logger.error("Pipeline completed with errors:")
            for err in generation_errors:
                logger.error("  - %s", err)
        logger.info("Generation complete: %s", scene_dir)
        timings = scene.get("timings") or {}
        if timings:
            logger.info("Timings: %s", {k: f"{v:.1f}s" for k, v in timings.items()})
    else:
        scene_dir = Path(args.output_dir)
        logger.info("Batch JSON report: %s", payload.get("report_json_path"))
        logger.info("Batch Markdown report: %s", payload.get("report_md_path"))
        logger.info("Batch generation complete: %s", scene_dir)

    return 0
