"""Gradio Web UI for scene-eval chain generation.

Launch with::

    python -m scene_pipeline --generate-ui
    python -m scene_pipeline --generate-ui --model gpt-5.5 --api-key $KEY
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from .config import DEFAULT_GEN_BASE_URL, DEFAULT_GEN_MODEL
from .pipeline import PipelineResult, StepEvent
from .paths import EVAL_GENERATED_DIR
from .batch import (
    build_scene_batch_meta,
    compute_file_sha256,
    find_existing_scene_by_hash,
    has_runnable_tests,
    resolve_single_scene_dir,
    scene_dir_for_image,
    write_scene_batch_meta,
)


@dataclass
class UIDefaults:
    """Default values for generate UI form fields."""

    image: str = ""
    provider: str = "openai"
    model: str = DEFAULT_GEN_MODEL
    base_url: str = DEFAULT_GEN_BASE_URL
    api_key: str = ""
    vision_model: str = ""
    max_tokens: int = 16384
    scene_category: str = ""
    output_dir: str = ""
    resume: bool = False


def parse_ui_args(argv: list[str] | None = None) -> UIDefaults:
    """Parse CLI flags into *UIDefaults* (no mandatory args, never errors)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default=DEFAULT_GEN_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_GEN_BASE_URL)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--vision-model", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--scene-category", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--generate-ui", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args, _unknown = parser.parse_known_args(argv)
    return UIDefaults(
        image=args.image or "",
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        vision_model=args.vision_model or "",
        max_tokens=args.max_tokens,
        scene_category=args.scene_category or "",
        output_dir=args.output_dir or "",
        resume=args.resume,
    )


logger = logging.getLogger(__name__)
_ACTIVE_GRADIO_APP: gr.Blocks | None = None
_AUTO_CLOSE_AFTER_GENERATE = False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


STEP_DEFS: list[dict[str, str]] = [
    {"id": "compress", "name": "图像压缩"},
    {"id": "analyze", "name": "场景分析"},
    {"id": "adaptive", "name": "自适应配置"},
    {"id": "server", "name": "Server 生成"},
    {"id": "commands", "name": "指令生成"},
    {"id": "config", "name": "配置生成"},
    {"id": "mocks", "name": "Mock 生成"},
    {"id": "tests", "name": "测试生成"},
    {"id": "validation", "name": "一致性校验"},
]

_STATUS_ICON = {
    "pending": ("○", "#9ca3af"),
    "running": ("◉", "#3b82f6"),
    "done": ("●", "#22c55e"),
    "skipped": ("◎", "#eab308"),
    "error": ("✕", "#ef4444"),
}

_STATUS_LABEL = {
    "pending": "待执行",
    "running": "运行中",
    "done": "完成",
    "skipped": "已复用",
    "error": "失败",
}


def _render_progress(
    statuses: dict[str, str],
    timings: dict[str, float | None],
    running_since: dict[str, float] | None = None,
    step_messages: dict[str, str] | None = None,
) -> str:
    """Return an HTML snippet showing the step progress bar."""
    parts: list[str] = []
    parts.append(
        '<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;padding:10px 0;">'
    )
    for index, step_def in enumerate(STEP_DEFS):
        step_id = step_def["id"]
        status = statuses.get(step_id, "pending")
        icon, color = _STATUS_ICON.get(status, _STATUS_ICON["pending"])
        label = _STATUS_LABEL.get(status, status)
        elapsed = timings.get(step_id)
        if status == "running" and running_since and step_id in running_since:
            elapsed = max(0.0, time.time() - running_since[step_id])
        timing_str = f" ({elapsed:.1f}s)" if elapsed is not None else ""
        anim = ' class="spin"' if status == "running" else ""
        step_msg = (step_messages or {}).get(step_id, "").strip()
        step_msg_html = (
            f'<div style="font-size:11px;color:#374151;margin-top:4px;max-width:120px;'
            f'text-align:center;line-height:1.35;word-break:break-word">{html.escape(step_msg)}</div>'
            if step_msg else ""
        )
        parts.append(
            f'<div style="display:flex;flex-direction:column;align-items:center;min-width:80px;'
            f'padding:4px 6px;border-radius:8px;'
            f'background:{"#eff6ff" if status == "running" else "transparent"}">'
            f'<span{anim} style="font-size:20px;color:{color};line-height:1">{icon}</span>'
            f'<span style="font-size:12px;color:{color};margin-top:2px;'
            f'font-weight:{"600" if status == "running" else "400"}">{step_def["name"]}</span>'
            f'<span style="font-size:11px;color:{color};margin-top:2px">{label}{timing_str}</span>'
            f"{step_msg_html}"
            "</div>"
        )
        if index < len(STEP_DEFS) - 1:
            line_color = "#22c55e" if status == "done" else "#d1d5db"
            parts.append(
                f'<div style="width:24px;height:2px;background:{line_color};margin-top:-10px"></div>'
            )
    parts.append("</div>")
    parts.append(
        "<style>.spin{animation:spin 1s linear infinite}"
        "@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}"
        "</style>"
    )
    return "".join(parts)


def _read_file_safe(path: Path | str, max_lines: int = 200) -> str:
    """Read a file, returning at most *max_lines* lines."""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n\n... ({len(lines) - max_lines} more lines)"
    return "\n".join(lines)


def _schedule_generate_app_close(delay: float = 1.0) -> None:
    """Release the terminal shortly after the generate-only UI finishes."""
    if not _AUTO_CLOSE_AFTER_GENERATE or _ACTIVE_GRADIO_APP is None:
        return

    def close_app() -> None:
        if _ACTIVE_GRADIO_APP is not None:
            _ACTIVE_GRADIO_APP.close()

    threading.Timer(delay, close_app).start()


def _render_batch_generation_progress(batch_items: list[dict[str, Any]], output_dir: str) -> str:
    title = "生成目录" if len(batch_items) == 1 else "批量生成目录"
    cards: list[str] = [
        '<div style="display:flex;flex-direction:column;gap:12px;padding:8px 0">',
        f'<div style="font-weight:600">{html.escape(title)}: {html.escape(output_dir)}</div>',
    ]
    for item in sorted(batch_items, key=lambda entry: entry.get("index", 0)):
        step_html = _render_progress(
            item.get("statuses", {}),
            item.get("timings", {}),
            item.get("running_since", {}),
            item.get("step_messages", {}),
        )
        status = item.get("final_status", "running")
        status_color = {
            "running": "#2563eb",
            "done": "#16a34a",
            "error": "#dc2626",
        }.get(status, "#6b7280")
        cards.append(
            '<div style="border:1px solid #d1d5db;border-radius:10px;padding:12px">'
            '<div style="display:flex;justify-content:space-between;gap:12px;align-items:center">'
            f'<div><div style="font-weight:600">[{item.get("index", "-")}] {html.escape(item.get("image_name", "-"))}</div>'
            f'<div style="font-size:12px;color:#6b7280">{html.escape(item.get("scene_dir", "-"))}</div></div>'
            f'<div style="font-size:12px;font-weight:600;color:{status_color}">{html.escape(item.get("message", ""))}</div>'
            "</div>"
            f"{step_html}"
            "</div>"
        )
    cards.append("</div>")
    return "".join(cards)


def _make_llm_event_handler(event_queue: queue.Queue, *, item_index: int | None = None) -> Any:
    def _handler(payload: dict[str, Any]) -> None:
        event_queue.put({"type": "llm_event", "item_index": item_index, **payload})

    return _handler


def run_pipeline_ui(
    image,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    vision_model: str,
    max_tokens: int,
    scene_category: str,
    output_dir: str,
):
    """Gradio generator that drives the pipeline and yields UI updates."""
    resume = True
    if isinstance(image, list):
        image_paths = [str(getattr(item, "name", item)) for item in image if item]
    elif isinstance(image, str):
        image_paths = [image]
    elif hasattr(image, "name"):
        image_paths = [image.name]
    else:
        image_paths = [str(image)] if image else []

    if not image_paths:
        yield (
            '<div style="color:#ef4444;padding:8px">请先上传场景照片</div>',
            None, None, None, None, None, None, None, None, "",
        )
        return

    if len(image_paths) > 8:
        yield (
            '<div style="color:#ef4444;padding:8px">最多支持 8 张图片</div>',
            None, None, None, None, None, None, None, None, "",
        )
        return

    if len(image_paths) > 1:
        seen_hashes: dict[str, str] = {}
        duplicate_pairs: list[str] = []
        for path_str in image_paths:
            path = Path(path_str)
            image_hash = compute_file_sha256(path)
            if image_hash in seen_hashes:
                duplicate_pairs.append(f"{Path(seen_hashes[image_hash]).name} = {path.name}")
            else:
                seen_hashes[image_hash] = path_str
        if duplicate_pairs:
            yield (
                '<div style="color:#ef4444;padding:8px">'
                '检测到重复图片内容，请移除重复上传后再生成：'
                + html.escape("；".join(duplicate_pairs))
                + "</div>",
                None, None, None, None, None, None, None, None, "",
            )
            return

    out_scene_analysis: Any = None
    out_adaptive: Any = None
    out_server: Any = None
    out_commands: str = ""
    out_config: str = ""
    out_mocks: Any = None
    out_tests: str = ""
    out_validation: Any = None
    scene_dir_path: str | None = None
    final_status_html = ""
    event_q: queue.Queue = queue.Queue()

    scene_category_value = scene_category.strip() if scene_category else None
    output_dir_value = output_dir.strip() if output_dir else None
    vision_model_value = vision_model.strip() if vision_model else None

    if len(image_paths) > 1 and not output_dir_value:
        yield (
            '<div style="color:#ef4444;padding:8px">多图片模式必须指定输出目录，作为 batch root</div>',
            out_scene_analysis, out_adaptive, out_server,
            out_commands, out_config, out_mocks, out_tests, out_validation,
            "",
        )
        return

    try:
        from .llm_client import LLMClient
        from .pipeline import run_pipeline

        batch_root = Path(output_dir_value) if output_dir_value else Path(resolve_single_scene_dir(output_dir_value, image_paths[0])).parent
        batch_root.mkdir(parents=True, exist_ok=True)
        batch_items: list[dict[str, Any]] = []
        batch_lock = threading.Lock()
        done_event_q: queue.Queue = queue.Queue()
        batch_item_by_index: dict[int, dict[str, Any]] = {}
        single_mode = len(image_paths) == 1

        for index, path in enumerate(image_paths, 1):
            image_hash = compute_file_sha256(Path(path))
            scene_dir = (
                Path(resolve_single_scene_dir(output_dir_value, path))
                if single_mode
                else Path(scene_dir_for_image(batch_root, path, image_hash))
            )
            item = {
                "index": index,
                "image_name": Path(path).name,
                "image_path": path,
                "scene_dir": str(scene_dir),
                "image_hash": image_hash,
                "statuses": {},
                "timings": {},
                "running_since": {},
                "step_messages": {},
                "final_status": "running",
                "message": "等待开始",
                "errors": [],
            }
            batch_items.append(item)
            batch_item_by_index[index] = item

        def _current_running_step_id(item: dict[str, Any]) -> str | None:
            for step_def in STEP_DEFS:
                if item["statuses"].get(step_def["id"]) == "running":
                    return step_def["id"]
            return None

        def update_step_message(message: str, *, item: dict[str, Any], step_id: str | None = None) -> None:
            target_step = step_id or _current_running_step_id(item) or STEP_DEFS[0]["id"]
            item["step_messages"][target_step] = message

        def sync_single_outputs(item: dict[str, Any], evt: StepEvent | None = None, result: PipelineResult | None = None) -> None:
            nonlocal out_scene_analysis, out_adaptive, out_server, out_commands, out_config, out_mocks, out_tests, out_validation
            if not single_mode:
                return
            if evt is not None:
                if evt.step_id == "analyze" and evt.status == "done" and evt.data:
                    out_scene_analysis = evt.data
                elif evt.step_id == "adaptive" and evt.status == "done" and evt.data:
                    out_adaptive = evt.data
                elif evt.step_id == "server" and evt.status == "done":
                    out_server = evt.data
                elif evt.step_id == "commands" and evt.status == "done" and evt.data:
                    commands = evt.data
                    lines = []
                    for command in commands.get("commands", []):
                        lines.append(f"- **{command.get('category', '')}**: {command.get('text', '')}")
                    if commands.get("failure_scenarios"):
                        lines.append("\n### 故障恢复场景")
                        for scenario in commands["failure_scenarios"]:
                            lines.append(f"- {scenario.get('text', '')}")
                    out_commands = "\n".join(lines) if lines else json.dumps(commands, ensure_ascii=False, indent=2)
                elif evt.step_id == "config" and evt.status == "done" and evt.data:
                    yaml_path = evt.data.get("yaml_path", "")
                    out_config = _read_file_safe(yaml_path, max_lines=300) if yaml_path else ""
                elif evt.step_id == "mocks" and evt.status == "done" and evt.data:
                    out_mocks = evt.data
                elif evt.step_id == "tests" and evt.status == "done" and evt.data:
                    tests_dir = evt.data.get("tests_dir", "")
                    if tests_dir:
                        parts = []
                        for file_name in (
                            "test_task_planning.py",
                            "test_function_calling.py",
                            "test_response_quality.py",
                        ):
                            file_path = Path(tests_dir) / file_name
                            if file_path.exists():
                                parts.append(f"### {file_name}\n```python\n{_read_file_safe(file_path, 60)}\n```")
                        out_tests = "\n\n".join(parts)
                elif evt.step_id == "validation" and evt.status == "done" and evt.data:
                    out_validation = evt.data
            if result is not None and result.scene_dir is not None:
                return

        def emit_progress() -> tuple[Any, ...]:
            summary_lines = []
            for item in sorted(batch_items, key=lambda entry: entry["index"]):
                summary_lines.append(f"- [{item['index']}] `{item['image_name']}`: {item['message']}")

            lead_item = batch_items[0]
            if single_mode:
                status_html = _render_batch_generation_progress(batch_items, lead_item.get("scene_dir", "")) + final_status_html
            else:
                status_html = _render_batch_generation_progress(batch_items, str(batch_root)) + final_status_html
            return (
                status_html,
                out_scene_analysis,
                out_adaptive,
                out_server,
                "\n".join(summary_lines) if not single_mode else out_commands,
                out_config,
                out_mocks,
                out_tests,
                out_validation,
                str(lead_item.get("scene_dir") or ""),
            )

        def worker(item: dict[str, Any]) -> None:
            def on_step(evt: StepEvent) -> None:
                with batch_lock:
                    item["statuses"][evt.step_id] = evt.status
                    if evt.status == "running":
                        item["running_since"][evt.step_id] = time.time()
                        item["step_messages"][evt.step_id] = ""
                    elif evt.status in {"done", "skipped", "error"}:
                        item["running_since"].pop(evt.step_id, None)
                        if evt.status in {"done", "skipped"}:
                            item["step_messages"].pop(evt.step_id, None)
                        elif evt.error:
                            item["step_messages"][evt.step_id] = evt.error
                    if evt.timing is not None:
                        item["timings"][evt.step_id] = evt.timing
                    item["message"] = f"{evt.step_name} - {_STATUS_LABEL.get(evt.status, evt.status)}"
                    if evt.status == "error" and evt.error:
                        item["errors"].append(evt.error)
                    if evt.step_id == "complete" and evt.data:
                        item["scene_dir"] = evt.data.get("scene_dir", item["scene_dir"])
                    sync_single_outputs(item, evt=evt)

            try:
                with batch_lock:
                    item["message"] = "开始生成"
                client = LLMClient(
                    provider=provider,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    vision_model=vision_model_value,
                    max_tokens=int(max_tokens),
                    event_callback=_make_llm_event_handler(event_q, item_index=int(item["index"])),
                )
                health = client.health_check()
                if not health["ok"]:
                    raise RuntimeError(f"生成侧 endpoint 健康检查失败: {health['reason']}")

                image_hash = str(item.get("image_hash") or "")
                existing_scene_dir = find_existing_scene_by_hash(batch_root, image_hash)
                if (
                    existing_scene_dir is not None
                    and str(existing_scene_dir) != item["scene_dir"]
                    and has_runnable_tests(existing_scene_dir)
                ):
                    scene_meta = build_scene_batch_meta(
                        scene_dir=existing_scene_dir,
                        image_index=int(item["index"]),
                        image_path=Path(item["image_path"]),
                        image_hash=image_hash,
                    )
                    write_scene_batch_meta(existing_scene_dir, scene_meta)
                    with batch_lock:
                        item["scene_dir"] = str(existing_scene_dir)
                        item["final_status"] = "done"
                        item["message"] = "检测到同图已生成，已跳过"
                        item["statuses"] = {step_def["id"]: "skipped" for step_def in STEP_DEFS}
                    return

                scene_dir = Path(item["scene_dir"])
                scene_dir.mkdir(parents=True, exist_ok=True)
                scene_meta = build_scene_batch_meta(
                    scene_dir=scene_dir,
                    image_index=int(item["index"]),
                    image_path=Path(item["image_path"]),
                    image_hash=image_hash,
                )
                write_scene_batch_meta(scene_dir, scene_meta)

                result = run_pipeline(
                    item["image_path"],
                    client,
                    output_dir=item["scene_dir"],
                    resume=resume,
                    scene_category=scene_category_value,
                    on_step=on_step,
                )
                with batch_lock:
                    item["scene_dir"] = str(result.scene_dir or item["scene_dir"])
                    item["final_status"] = "error" if result.errors else "done"
                    item["message"] = "生成完成" if not result.errors else f"生成完成但有错误: {'; '.join(result.errors)}"
                    item["errors"].extend(result.errors)
                    final_scene_dir = Path(item["scene_dir"])
                    final_scene_meta = build_scene_batch_meta(
                        scene_dir=final_scene_dir,
                        image_index=int(item["index"]),
                        image_path=Path(item["image_path"]),
                        image_hash=image_hash,
                    )
                    write_scene_batch_meta(final_scene_dir, final_scene_meta)
                    if single_mode:
                        nonlocal scene_dir_path, final_status_html
                        scene_dir_path = item["scene_dir"]
                        if result.errors:
                            final_status_html = (
                                '<div style="color:#ef4444;margin-top:8px">'
                                f'Pipeline 完成但存在错误：{"; ".join(result.errors)}'
                                "</div>"
                            )
                        else:
                            skipped = ", ".join(result.steps_skipped) if result.steps_skipped else "无"
                            total = result.timings.get("total")
                            timing = f"；总耗时 {total:.1f}s" if isinstance(total, (int, float)) else ""
                            final_status_html = (
                                '<div style="color:#22c55e;margin-top:8px">'
                                f"生成完成{timing}；输出目录：{scene_dir_path}；跳过步骤：{skipped}"
                                "</div>"
                            )
                        sync_single_outputs(item, result=result)
            except Exception as exc:
                with batch_lock:
                    item["final_status"] = "error"
                    item["message"] = f"生成异常: {exc}"
                    item["errors"].append(str(exc))
                    if single_mode:
                        final_status_html = f'<div style="color:#ef4444;margin-top:8px">Pipeline 异常: {html.escape(str(exc))}</div>'
            finally:
                done_event_q.put(item["index"])

        yield emit_progress()

        threads = [threading.Thread(target=worker, args=(item,), daemon=True) for item in batch_items]
        for thread in threads:
            thread.start()

        finished = 0
        total = len(threads)
        while finished < total:
            try:
                event = event_q.get(timeout=0.3)
            except queue.Empty:
                event = None

            if isinstance(event, dict) and event.get("type") == "llm_event":
                target_index = event.get("item_index")
                batch_item = batch_item_by_index.get(int(target_index)) if target_index is not None else None
                if batch_item is not None:
                    event_name = event.get("event")
                    if event_name == "llm_health_check_start":
                        update_step_message(
                            f"endpoint 预检中，timeout={event.get('timeout')}s",
                            item=batch_item,
                            step_id=STEP_DEFS[0]["id"],
                        )
                    elif event_name == "llm_health_check_result":
                        if event.get("ok"):
                            update_step_message("endpoint 预检通过", item=batch_item, step_id=STEP_DEFS[0]["id"])
                        else:
                            update_step_message(
                                f"endpoint 预检失败: {event.get('reason', '-')}",
                                item=batch_item,
                                step_id=STEP_DEFS[0]["id"],
                            )
                    elif event_name == "llm_request_error":
                        update_step_message(
                            f"请求异常，第 {event.get('attempt')}/{event.get('max_attempts')} 次: {event.get('error', '-')}",
                            item=batch_item,
                        )
                    elif event_name == "llm_retry_scheduled":
                        update_step_message(
                            f"{event.get('backoff_seconds')}s 后重试，第 {event.get('next_attempt')}/{event.get('max_attempts')} 次",
                            item=batch_item,
                        )
                yield emit_progress()
                continue

            try:
                done_event_q.get_nowait()
                finished += 1
            except queue.Empty:
                pass

            yield emit_progress()

        for thread in threads:
            thread.join(timeout=5)

        scene_summaries = [
            {
                "image_index": item["index"],
                "image_name": item["image_name"],
                "scene_dir": item["scene_dir"],
                "status": item["final_status"],
                "errors": item["errors"],
            }
            for item in sorted(batch_items, key=lambda entry: entry["index"])
        ]

        if not single_mode:
            yield (
                _render_batch_generation_progress(batch_items, str(batch_root))
                + '<div style="color:#22c55e;padding:8px">Batch 生成完成</div>',
                out_scene_analysis,
                out_adaptive,
                out_server,
                f"```json\n{json.dumps(scene_summaries, ensure_ascii=False, indent=2)}\n```",
                out_config,
                out_mocks,
                out_tests,
                out_validation,
                str(batch_root),
            )
        else:
            yield emit_progress()

        _schedule_generate_app_close()
    except Exception as exc:
        final_status_html = (
            f'<div style="color:#ef4444;padding:8px">'
            f'{"Batch 异常" if len(image_paths) > 1 else "Pipeline 异常"}: {html.escape(str(exc))}'
            "</div>"
        )
        yield (
            _render_batch_generation_progress(
                batch_items if "batch_items" in locals() else [],
                str(batch_root) if "batch_root" in locals() else (output_dir_value or ""),
            )
            + final_status_html,
            out_scene_analysis,
            out_adaptive,
            out_server,
            out_commands,
            out_config,
            out_mocks,
            out_tests,
            out_validation,
            scene_dir_path or (str(batch_root) if "batch_root" in locals() else ""),
        )
        _schedule_generate_app_close()


_CSS = """
.api-key-input textarea,
.api-key-input input {
  background: var(--input-background-fill, color-mix(in srgb, CanvasText 4%, Canvas)) !important;
  color: var(--body-text-color, CanvasText) !important;
}
.api-key-input textarea:-webkit-autofill,
.api-key-input input:-webkit-autofill,
.api-key-input textarea:-webkit-autofill:hover,
.api-key-input input:-webkit-autofill:hover,
.api-key-input textarea:-webkit-autofill:focus,
.api-key-input input:-webkit-autofill:focus,
.api-key-input textarea:-webkit-autofill:active,
.api-key-input input:-webkit-autofill:active {
  -webkit-text-fill-color: var(--body-text-color, CanvasText) !important;
  box-shadow: 0 0 0 1000px var(--input-background-fill, color-mix(in srgb, CanvasText 4%, Canvas)) inset !important;
  caret-color: var(--body-text-color, CanvasText) !important;
  transition: background-color 99999s ease-out 0s !important;
}
"""

_HEAD = """
<script>
(() => {
  const patchApiKeyInputs = () => {
    const nodes = document.querySelectorAll('.api-key-input input, .api-key-input textarea');
    nodes.forEach((node) => {
      node.setAttribute('autocomplete', 'off');
      node.setAttribute('autocorrect', 'off');
      node.setAttribute('autocapitalize', 'off');
      node.setAttribute('spellcheck', 'false');
      node.setAttribute('data-lpignore', 'true');
      node.setAttribute('data-1p-ignore', 'true');
      node.setAttribute('data-form-type', 'other');
    });
  };
  window.setInterval(patchApiKeyInputs, 500);
  window.addEventListener('load', patchApiKeyInputs);
  document.addEventListener('DOMContentLoaded', patchApiKeyInputs);
})();
</script>
"""


def build_generate_app(defaults: UIDefaults | None = None) -> gr.Blocks:
    """Build the generate-only Gradio app."""
    defaults = defaults or UIDefaults()
    default_images = [item.strip() for item in defaults.image.split(",") if item.strip()]

    with gr.Blocks(title="Scene Pipeline Generator") as app:
        gr.Markdown("# 评测链路生成 UI\n上传 1-8 张场景图片，并行生成对应评测链路。")

        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.File(
                    label="场景照片（支持 1-8 张）",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    value=default_images or None,
                )
            with gr.Column(scale=3):
                with gr.Row():
                    provider = gr.Dropdown(
                        choices=["openai", "anthropic", "local"],
                        value=defaults.provider,
                        label="Provider",
                    )
                    model = gr.Textbox(value=defaults.model, label="Model")
                    scene_category = gr.Dropdown(
                        choices=["", "home", "office", "retail", "industrial"],
                        value=defaults.scene_category,
                        label="场景类别（留空自动检测）",
                    )
                with gr.Row():
                    base_url = gr.Textbox(value=defaults.base_url, label="Base URL")
                    api_key = gr.Textbox(
                        value=defaults.api_key,
                        label="API Key",
                        type="password",
                        elem_classes="api-key-input",
                    )
                with gr.Row():
                    vision_model = gr.Textbox(value=defaults.vision_model, label="Vision Model（留空同 Model）")
                    max_tokens = gr.Number(value=defaults.max_tokens, label="Max Tokens", precision=0)
                    output_dir = gr.Textbox(value=defaults.output_dir, label="输出目录（多图 batch root）")

        with gr.Row():
            run_btn = gr.Button("开始生成评测链路", variant="primary", elem_id="run-btn")

        progress_html = gr.HTML(value=_render_progress({}, {}), label="生成进度")
        out_commands = gr.Markdown(label="生成结果 / Batch 摘要")
        output_dir_box = gr.Textbox(label="生成输出目录", interactive=False)

        out_scene = gr.JSON(label="scene_analysis.json", visible=False)
        out_adaptive = gr.JSON(label="自适应配置", visible=False)
        out_server = gr.JSON(label="all_tools", visible=False)
        out_config = gr.Code(label="scene.yaml", language="yaml", visible=False)
        out_mocks = gr.JSON(label="mock_meta", visible=False)
        out_tests = gr.Markdown(label="测试文件", visible=False)
        out_validation = gr.JSON(label="校验结果", visible=False)

        run_btn.click(
            fn=run_pipeline_ui,
            inputs=[
                image_input,
                provider,
                model,
                base_url,
                api_key,
                vision_model,
                max_tokens,
                scene_category,
                output_dir,
            ],
            outputs=[
                progress_html,
                out_scene,
                out_adaptive,
                out_server,
                out_commands,
                out_config,
                out_mocks,
                out_tests,
                out_validation,
                output_dir_box,
            ],
        )

    return app


def launch(*, cli_args: UIDefaults | None = None, mode: str = "generate", **kwargs):
    """Build and launch the generate-only Gradio app."""
    global _ACTIVE_GRADIO_APP, _AUTO_CLOSE_AFTER_GENERATE

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("gradio").setLevel(logging.WARNING)

    if mode != "generate":
        raise ValueError("Only generate mode is supported in the original repository")

    app = build_generate_app(defaults=cli_args)
    server_port = int(os.environ.get("GRADIO_SERVER_PORT") or kwargs.get("server_port", 7860))
    _ACTIVE_GRADIO_APP = app
    _AUTO_CLOSE_AFTER_GENERATE = _env_flag("SCENE_PIPELINE_AUTO_CLOSE", False)
    allowed_paths = {str(EVAL_GENERATED_DIR)}
    if cli_args and cli_args.output_dir:
        allowed_paths.add(str(Path(cli_args.output_dir).expanduser()))

    app.queue()
    app.launch(
        server_name=kwargs.get("server_name", "0.0.0.0"),
        server_port=server_port,
        share=kwargs.get("share", False),
        quiet=True,
        allowed_paths=sorted(allowed_paths),
        css=_CSS,
        head=_HEAD,
    )
