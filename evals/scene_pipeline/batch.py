"""Batch helpers for generating scene_pipeline assets from 1-8 scene images."""

from __future__ import annotations

import json
import logging
import re
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
from typing import Any

from .llm_client import LLMClient
from .pipeline import run_pipeline
from .paths import EVAL_GENERATED_DIR

logger = logging.getLogger(__name__)


def has_runnable_tests(scene_dir: Path) -> bool:
    tests_dir = Path(scene_dir) / "tests"
    return tests_dir.is_dir() and any(tests_dir.glob("test_*.py"))


def scene_dir_name_for_image(image_path: str | Path, image_hash: str | None = None) -> str:
    """Return a stable scene directory name for one image."""
    image_path = Path(image_path)
    hash_prefix = (image_hash or compute_file_sha256(image_path))[:8]
    return f"scene_{_safe_name(image_path.stem)}_{hash_prefix}"


def scene_dir_for_image(output_dir: Path, image_path: str | Path, image_hash: str | None = None) -> Path:
    """Return the canonical generated scene directory for an image."""
    return Path(output_dir) / scene_dir_name_for_image(image_path, image_hash)


def resolve_single_scene_dir(output_dir: str | Path | None, image_path: str | Path) -> Path:
    """Resolve single-image generation to the same stable layout as batches.

    Existing concrete scene directories are still accepted for resume/backward
    compatibility, but the package generated root is always treated as a root.
    """
    root = Path(output_dir).expanduser() if output_dir else EVAL_GENERATED_DIR
    if _is_concrete_scene_dir(root):
        return root
    return scene_dir_for_image(root, image_path)


def _is_concrete_scene_dir(path: Path) -> bool:
    if path.name.startswith("scene_"):
        return True
    try:
        if path.resolve() == EVAL_GENERATED_DIR.resolve():
            return False
    except OSError:
        pass
    indicators = (
        "scene_analysis.json",
        "all_tools.json",
        "commands_data.json",
        "mock_meta.json",
    )
    if any((path / name).exists() for name in indicators):
        return True
    return (path / "tests" / "test_task_planning.py").exists()


def compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def materialize_scene_source_image(
    scene_dir: Path,
    image_path: Path,
    *,
    original_image_name: str | None = None,
) -> dict[str, str]:
    """Copy the input image into the scene dataset so it stays self-contained."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    source = image_path.expanduser().resolve()
    ext = source.suffix.lower() or ".img"
    preferred_name = (original_image_name or source.name or f"source_image{ext}").strip()
    target = scene_dir / preferred_name

    if target.exists():
        same_target = False
        try:
            same_target = target.resolve() == source
        except OSError:
            same_target = False
        if not same_target:
            source_hash = compute_file_sha256(source)
            try:
                target_hash = compute_file_sha256(target)
            except OSError:
                target_hash = None
            if target_hash != source_hash:
                safe_stem = _safe_name(Path(preferred_name).stem) or "source_image"
                target = scene_dir / f"{safe_stem}_{source_hash[:8]}{ext}"

    if not target.exists():
        shutil.copy2(source, target)

    return {
        "source_image_path": target.name,
        "thumbnail_path": target.name,
    }


def build_scene_batch_meta(
    *,
    scene_dir: Path,
    image_index: int,
    image_path: Path,
    image_hash: str,
) -> dict[str, Any]:
    """Build portable scene metadata that stores the image inside the dataset."""
    local_image_meta = materialize_scene_source_image(
        scene_dir,
        image_path,
        original_image_name=image_path.name,
    )
    local_image_path = local_image_meta["source_image_path"]
    return {
        "image_index": image_index,
        "input_image_path": str(image_path.resolve()),
        "image_path": local_image_path,
        "scene_dir": scene_dir.name,
        "image_hash": image_hash,
        "archived_image_path": local_image_path,
        "original_image_name": image_path.name,
        **local_image_meta,
    }


def write_scene_batch_meta(scene_dir: Path, scene_meta: dict[str, Any]) -> dict[str, Any]:
    meta_path = scene_dir / "batch_meta.json"
    existing: dict[str, Any] = {}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    merged = {**existing, **scene_meta}
    meta_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return merged


def find_existing_scene_by_hash(output_dir: Path, image_hash: str) -> Path | None:
    for meta_path in sorted(output_dir.rglob("batch_meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("image_hash") == image_hash:
            return meta_path.parent
    return None


def parse_image_inputs(value: str | None) -> list[Path]:
    """Parse one CLI image field into 1-8 image paths.

    Accepts a single path or comma-separated paths. Commas are intentionally
    simple here because scene image filenames should not contain commas.
    """
    if not value:
        return []
    paths = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if len(paths) > 8:
        raise ValueError("最多支持 8 张图片")
    if not paths:
        raise ValueError("没有解析到图片路径")
    return paths


def run_batch_pipeline(
    *,
    image_paths: list[Path],
    client: LLMClient | None,
    output_dir: Path,
    resume: bool = False,
    scene_category: str | None = None,
) -> dict[str, Any]:
    """Generate one scene-eval chain per image and write one aggregate report."""
    if not (1 <= len(image_paths) <= 8):
        raise ValueError("image_paths must contain 1-8 images")

    output_dir.mkdir(parents=True, exist_ok=True)
    scenes: list[dict[str, Any]] = []
    started = time.time()

    indexed_images = list(enumerate(image_paths, 1))

    def run_one(index: int, image_path: Path) -> tuple[int, dict[str, Any]]:
        image_hash = compute_file_sha256(image_path)
        scene_dir = scene_dir_for_image(output_dir, image_path, image_hash)
        existing_scene_dir = find_existing_scene_by_hash(output_dir, image_hash)
        if existing_scene_dir is not None and existing_scene_dir != scene_dir and has_runnable_tests(existing_scene_dir):
            scene_meta = build_scene_batch_meta(
                scene_dir=existing_scene_dir,
                image_index=index,
                image_path=image_path,
                image_hash=image_hash,
            )
            write_scene_batch_meta(existing_scene_dir, scene_meta)
            scene_info = {
                **scene_meta,
                "scene_dir": str(existing_scene_dir),
                "reused_existing_scene_dir": str(existing_scene_dir),
                "generation_skipped": True,
            }
            return index, scene_info
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_meta = build_scene_batch_meta(
            scene_dir=scene_dir,
            image_index=index,
            image_path=image_path,
            image_hash=image_hash,
        )
        write_scene_batch_meta(scene_dir, scene_meta)
        scene_info: dict[str, Any] = {
            **scene_meta,
            "scene_dir": str(scene_dir),
        }
        logger.info("Batch scene %d/%d: %s", index, len(image_paths), image_path)

        if client is not None:
            result = run_pipeline(
                image_path,
                client,
                output_dir=scene_dir,
                resume=resume,
                scene_category=scene_category,
            )
            scene_info["generation_errors"] = result.errors
            scene_info["generation_warnings"] = result.consistency_warnings
            scene_info["timings"] = result.timings
            if result.scene_dir is not None:
                scene_dir = result.scene_dir
                scene_meta = build_scene_batch_meta(
                    scene_dir=scene_dir,
                    image_index=index,
                    image_path=image_path,
                    image_hash=image_hash,
                )
                write_scene_batch_meta(scene_dir, scene_meta)
                scene_info.update(scene_meta)
                scene_info["scene_dir"] = str(scene_dir)

        return index, scene_info

    if client is not None and len(indexed_images) > 1:
        workers = min(len(indexed_images), 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(run_one, index, image_path): index
                for index, image_path in indexed_images
            }
            for future in as_completed(future_map):
                _index, scene_info = future.result()
                scenes.append(scene_info)
    else:
        for index, image_path in indexed_images:
            _index, scene_info = run_one(index, image_path)
            scenes.append(scene_info)

    scenes.sort(key=lambda item: item.get("image_index", 0))

    summary = {
        "image_count": len(image_paths),
        "elapsed_seconds": round(time.time() - started, 2),
    }

    payload = {
        "summary": summary,
        "scenes": scenes,
    }
    report_dir = output_dir / "batch_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "generation_report.json"
    md_path = report_dir / "generation_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_format_generation_markdown(payload), encoding="utf-8")
    payload["report_json_path"] = str(json_path)
    payload["report_md_path"] = str(md_path)
    return payload


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or "item"


def _format_generation_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    scenes = payload.get("scenes") or []
    lines = [
        "# Scene Pipeline Batch Generation Report",
        "",
        f"- 图片数: {summary.get('image_count', 0)}",
        f"- 总耗时: {summary.get('elapsed_seconds', 0)}s",
        "",
        "## 场景结果",
        "",
    ]
    for scene in scenes:
        image_name = Path(scene.get("image_path") or scene.get("scene_dir") or "").name or "-"
        lines.append(f"- {image_name} -> `{scene.get('scene_dir', '-')}`")
        errors = scene.get("generation_errors") or []
        if errors:
            lines.append(f"  errors: {'; '.join(str(err) for err in errors)}")
    return "\n".join(lines) + "\n"
