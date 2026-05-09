"""Multi-provider LLM client with unified chat/vision interface."""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from .config import LLM_REQUEST_TIMEOUT_SECONDS, MAX_LLM_RETRIES, Provider

logger = logging.getLogger(__name__)

_JSON_REPAIR_RETRIES = 2
LLMEventCallback = Callable[[dict[str, Any]], None]


class LLMClient:
    """Unified LLM client supporting OpenAI-compatible APIs and Anthropic."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "EMPTY",
        vision_model: str | None = None,
        max_tokens: int = 16384,
        event_callback: LLMEventCallback | None = None,
    ):
        self.provider = Provider(provider)
        self.model = model
        self.vision_model = vision_model or model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.request_timeout = LLM_REQUEST_TIMEOUT_SECONDS
        self._event_callback = event_callback

        if self.provider == Provider.ANTHROPIC:
            self._init_anthropic()
        else:
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=self.request_timeout,
            )

    def _init_anthropic(self):
        try:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package required for Anthropic provider. "
                "Install with: pip install anthropic"
            )

    def _emit_event(self, event: str, **payload: Any) -> None:
        if self._event_callback is None:
            return
        try:
            self._event_callback({
                "event": event,
                "provider": self.provider.value,
                "model": self.model,
                "base_url": self.base_url,
                **payload,
            })
        except Exception:
            logger.debug("LLM event callback raised; ignoring", exc_info=True)

    def health_check(
        self,
        *,
        timeout: float = 20.0,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        model = (model_override or self.model).strip()
        if not model:
            return {"ok": False, "reason": "模型名称为空"}
        if not self.base_url.strip():
            return {"ok": False, "reason": "Base URL 为空"}

        self._emit_event(
            "llm_health_check_start",
            check_model=model,
            timeout=timeout,
        )
        try:
            if self.provider == Provider.ANTHROPIC:
                response = self._anthropic_client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": "Reply with ok."}],
                    max_tokens=64,
                    temperature=0,
                )
                content = response.content[0].text.strip() if getattr(response, "content", None) else ""
                if not content:
                    result = {"ok": False, "reason": "模型返回了空回复"}
                else:
                    result = {"ok": True, "reason": "健康检查通过"}
            else:
                client = OpenAI(
                    base_url=self.base_url.strip(),
                    api_key=self.api_key.strip() or "EMPTY",
                    timeout=timeout,
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a health check endpoint."},
                        {"role": "user", "content": "Reply with ok."},
                    ],
                    temperature=0,
                    max_tokens=64,
                )
                choices = getattr(response, "choices", None) or []
                if not choices:
                    result = {"ok": False, "reason": "模型接口返回空 choices"}
                else:
                    message = getattr(choices[0], "message", None)
                    content = (getattr(message, "content", None) or "").strip() if message else ""
                    tool_calls = getattr(message, "tool_calls", None) if message else None
                    if not content and not tool_calls:
                        result = {"ok": False, "reason": "模型返回了空回复"}
                    else:
                        result = {"ok": True, "reason": "健康检查通过"}
        except Exception as exc:
            result = {"ok": False, "reason": _format_health_error(exc)}

        self._emit_event(
            "llm_health_check_result",
            check_model=model,
            timeout=timeout,
            ok=result["ok"],
            reason=result["reason"],
        )
        return result

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0,
        response_format: dict | None = None,
    ) -> str:
        """Text-only chat completion with retry logic."""
        for attempt in range(MAX_LLM_RETRIES + 1):
            try:
                return self._do_chat(messages, temperature, response_format)
            except Exception as e:
                self._emit_event(
                    "llm_request_error",
                    request_kind="chat",
                    attempt=attempt + 1,
                    max_attempts=MAX_LLM_RETRIES + 1,
                    error=str(e),
                    retrying=attempt < MAX_LLM_RETRIES,
                )
                if attempt == MAX_LLM_RETRIES:
                    raise
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
                self._emit_event(
                    "llm_retry_scheduled",
                    request_kind="chat",
                    next_attempt=attempt + 2,
                    max_attempts=MAX_LLM_RETRIES + 1,
                    backoff_seconds=2 ** attempt,
                )
                time.sleep(2 ** attempt)
        return ""

    def vision(
        self,
        prompt: str,
        image_path: str | Path,
        temperature: float = 0,
        response_format: dict | None = None,
    ) -> str:
        """Vision chat: analyze an image with a text prompt."""
        image_data = _encode_image(image_path)
        ext = Path(image_path).suffix.lower().lstrip(".")
        media_type = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"

        if self.provider == Provider.ANTHROPIC:
            return self._anthropic_vision(prompt, image_data, media_type, temperature)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}",
                        },
                    },
                ],
            }
        ]
        for attempt in range(MAX_LLM_RETRIES + 1):
            try:
                return self._do_chat(
                    messages, temperature, response_format,
                    model_override=self.vision_model,
                )
            except Exception as e:
                self._emit_event(
                    "llm_request_error",
                    request_kind="vision",
                    attempt=attempt + 1,
                    max_attempts=MAX_LLM_RETRIES + 1,
                    error=str(e),
                    retrying=attempt < MAX_LLM_RETRIES,
                )
                if attempt == MAX_LLM_RETRIES:
                    raise
                logger.warning("Vision call failed (attempt %d): %s", attempt + 1, e)
                self._emit_event(
                    "llm_retry_scheduled",
                    request_kind="vision",
                    next_attempt=attempt + 2,
                    max_attempts=MAX_LLM_RETRIES + 1,
                    backoff_seconds=2 ** attempt,
                )
                time.sleep(2 ** attempt)
        return ""

    def _do_chat(
        self,
        messages: list[dict],
        temperature: float,
        response_format: dict | None,
        model_override: str | None = None,
    ) -> str:
        model = model_override or self.model

        if self.provider == Provider.ANTHROPIC:
            return self._anthropic_chat(messages, temperature, model)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.request_timeout,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        if finish_reason == "length":
            usage = getattr(response, "usage", None)
            comp_tokens = getattr(usage, "completion_tokens", "?") if usage else "?"
            logger.warning(
                "LLM output TRUNCATED (finish_reason=length, "
                "completion_tokens=%s, max_tokens=%d). "
                "Consider increasing --max-tokens.",
                comp_tokens, self.max_tokens,
            )
        return response.choices[0].message.content or ""

    def _anthropic_chat(
        self, messages: list[dict], temperature: float, model: str
    ) -> str:
        system = None
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": filtered,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        response = self._anthropic_client.messages.create(**kwargs)
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "max_tokens":
            usage = getattr(response, "usage", None)
            out_tokens = getattr(usage, "output_tokens", "?") if usage else "?"
            logger.warning(
                "Anthropic output TRUNCATED (stop_reason=max_tokens, "
                "output_tokens=%s, max_tokens=%d). "
                "Consider increasing --max-tokens.",
                out_tokens, self.max_tokens,
            )
        return response.content[0].text

    def _anthropic_vision(
        self, prompt: str, image_data: str, media_type: str, temperature: float
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._anthropic_chat(messages, temperature, self.vision_model)

    # ------------------------------------------------------------------
    # JSON output methods with extraction, repair, and retry
    # ------------------------------------------------------------------

    def chat_json(
        self,
        messages: list[dict],
        temperature: float = 0,
    ) -> dict:
        """Chat expecting JSON output. Parses with extraction + repair fallback."""
        if self.provider != Provider.ANTHROPIC:
            raw = self.chat(
                messages, temperature,
                response_format={"type": "json_object"},
            )
        else:
            raw = self.chat(messages, temperature)

        return self._parse_json_robust(raw, messages, temperature)

    def vision_json(
        self,
        prompt: str,
        image_path: str | Path,
        temperature: float = 0,
    ) -> dict:
        """Vision chat expecting JSON output."""
        if self.provider != Provider.ANTHROPIC:
            raw = self.vision(
                prompt, image_path, temperature,
                response_format={"type": "json_object"},
            )
        else:
            raw = self.vision(prompt, image_path, temperature)

        return self._parse_json_robust(raw, [{"role": "user", "content": prompt}], temperature)

    def _parse_json_robust(
        self, raw: str, original_messages: list[dict], temperature: float
    ) -> dict:
        """Try multiple strategies to parse JSON from the LLM response.

        Strategy chain:
            1. Direct parse (after stripping fences)
            2. Extract JSON object via brace matching
            3. Ask LLM to fix the malformed JSON (retry)
        """
        cleaned = _strip_markdown_fences(raw)
        parse_error: json.JSONDecodeError | None = None

        # Strategy 1: direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            parse_error = exc
            logger.warning("JSON parse failed: %s", exc)

        # Strategy 2: extract the outermost { ... } via brace matching
        extracted = _extract_json_object(raw)
        if extracted and extracted != cleaned:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                logger.warning("Extracted JSON also invalid")

        # Strategy 3: close truncated JSON — DISABLED
        # Silently closing truncated JSON hides incomplete LLM output, making
        # it hard to diagnose missing data.  Keep the code for reference but
        # skip it so that truncation surfaces as a parse error.
        # closed = _try_close_truncated_json(cleaned)
        # if closed is not None:
        #     try:
        #         result = json.loads(closed)
        #         logger.info("Truncated JSON repair succeeded (closed unclosed brackets)")
        #         return result
        #     except json.JSONDecodeError:
        #         logger.warning("Truncated JSON closure did not produce valid JSON")

        # Strategy 4: ask LLM to repair the JSON
        for attempt in range(_JSON_REPAIR_RETRIES):
            logger.info(
                "Requesting LLM JSON repair (attempt %d/%d)",
                attempt + 1, _JSON_REPAIR_RETRIES,
            )
            try:
                repaired = self._repair_json(cleaned, parse_error)
                return json.loads(repaired)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("JSON repair attempt %d failed: %s", attempt + 1, e)

        raise parse_error  # type: ignore[misc]

    def _repair_json(self, malformed: str, error: json.JSONDecodeError) -> str:
        """Ask the LLM to fix malformed JSON."""
        truncated = malformed
        if len(truncated) > 12000:
            start = max(0, error.pos - 500)
            end = min(len(truncated), error.pos + 500)
            context = truncated[start:end]
            repair_prompt = (
                "以下 JSON 文本在解析时出错。\n\n"
                f"## 错误\n{error}\n\n"
                f"## 错误附近的上下文（字符 {start}-{end}）\n```\n{context}\n```\n\n"
                f"## 完整 JSON 的前 2000 字符\n```\n{truncated[:2000]}\n```\n\n"
                f"## 完整 JSON 的后 2000 字符\n```\n{truncated[-2000:]}\n```\n\n"
                "请修复这段 JSON 并返回完整的、合法的 JSON 对象。只返回 JSON，不要加任何其他文字。"
            )
        else:
            repair_prompt = (
                "以下 JSON 文本在解析时出错。\n\n"
                f"## 错误\n{error}\n\n"
                f"## 原始文本\n```\n{truncated}\n```\n\n"
                "请修复这段 JSON 并返回完整的、合法的 JSON 对象。只返回 JSON，不要加任何其他文字。"
            )

        raw = self.chat(
            [{"role": "user", "content": repair_prompt}],
            temperature=0,
        )
        return _strip_markdown_fences(raw)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON content."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _extract_json_object(text: str) -> str | None:
    """Extract the outermost {...} from text using brace-depth tracking.

    Handles thinking-model output where the JSON may be preceded
    by reasoning/thinking tokens.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    end = start

    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                return text[start:end + 1]

    return None


def _try_close_truncated_json(text: str) -> str | None:
    """Attempt to fix truncated JSON by closing unclosed brackets and braces.

    When a model hits its output token limit, the JSON is often cut off
    mid-way, leaving unclosed ``{``, ``[``, or ``"``.  This function
    tracks the nesting depth and appends the missing closers.

    Returns the repaired string, or *None* if the JSON is already balanced.
    """
    in_string = False
    escape = False
    stack: list[str] = []

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    if not stack and not in_string:
        return None

    result = text.rstrip()

    if in_string:
        result += '"'

    # Strip trailing incomplete key-value pair fragments (e.g. trailing comma,
    # colon, or the start of a value that was never finished).
    result = result.rstrip(" \t\n\r,:")

    # Close all remaining open brackets / braces in LIFO order.
    result += "".join(reversed(stack))

    return result


def _format_health_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).strip()
    if status_code:
        return f"HTTP {status_code}: {message}"
    return message or exc.__class__.__name__


def _encode_image(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
