"""Configured OpenAI-compatible LLM calls with disk caching and JSONL logs."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAX_ATTEMPTS = 3
_CACHE_FORMAT_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _optional_int(value: Any) -> int | None:
    """Convert provider usage fields when present; ignore empty/malformed values."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _request_key(*, model: str, base_url: str, messages: Sequence[Mapping[str, str]],
                 temperature: float, json_mode: bool,
                 extra_body: Mapping[str, Any] | None = None,
                 prompt_sha256: str | None = None,
                 max_tokens: int | None = None) -> str:
    payload = {
        "model": model,
        "base_url": base_url,
        "messages": list(messages),
        "temperature": temperature,
        "json_mode": json_mode,
        "extra_body": dict(extra_body or {}),
        "prompt_sha256": prompt_sha256,
        "cache_format_version": _CACHE_FORMAT_VERSION,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _load_role(role: str, config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if role not in {"teacher", "generator", "judge", "judge_primary", "judge_secondary"}:
        raise ValueError(f"Unknown LLM role: {role!r}")
    settings = config.get(role)
    required = ("model", "base_url", "api_key_env")
    if not isinstance(settings, dict) or any(not settings.get(name) for name in required):
        raise ValueError(
            f"Fill model, base_url, and api_key_env for role {role!r} in {config_path}"
        )
    loaded: dict[str, Any] = {name: str(settings[name]) for name in required}
    loaded["thinking_mode"] = str(settings.get("thinking_mode", "unspecified"))
    loaded["thinking_parameters"] = dict(settings.get("thinking_parameters") or {})
    return loaded


def _is_response_format_unsupported(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()
    mentions_format = "response_format" in message or "json_object" in message
    indicates_unsupported = any(
        phrase in message
        for phrase in ("not supported", "unsupported", "unknown parameter", "unrecognized")
    )
    return status in {400, 422} and mentions_format and indicates_unsupported


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (APITimeoutError, RateLimitError, APIConnectionError, TimeoutError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def _backoff(attempt: int) -> None:
    # Deterministic exponential backoff: 1s, then 2s.
    time.sleep(2 ** (attempt - 1))


def _append_log(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical_json(event) + "\n")


def _read_cache(path: Path, json_mode: bool) -> dict[str, Any] | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if cached.get("cache_format_version") != _CACHE_FORMAT_VERSION:
        return None
    if json_mode:
        try:
            parsed = json.loads(cached["text"])
        except (KeyError, json.JSONDecodeError, TypeError):
            return None
        cached["json"] = parsed
    else:
        cached["json"] = None
    return cached


def _write_cache(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(
        _canonical_json({
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "text": result["text"],
            "usage": result["usage"],
            "model": result["model"],
            "api_model": result.get("api_model", result["model"]),
            "request_parameters": result.get("request_parameters", {}),
        }),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def call_llm(
    role: str,
    messages: Sequence[Mapping[str, str]],
    *,
    json_mode: bool = False,
    temperature: float = 0.0,
    extra_body: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    client: Any | None = None,
    prompt_sha256: str | None = None,
    log_messages: bool = False,
    max_tokens: int | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Call a configured model and return text, parsed JSON, and usage metadata.

    JSON parse failures trigger one corrective retry. Timeout, rate-limit,
    connection, and server errors retry up to three total API attempts.
    Failed calls and invalid JSON are never cached.
    """
    config_file = Path(config_path) if config_path else _PROJECT_ROOT / "configs" / "models.yaml"
    env_file = _PROJECT_ROOT / ".env"
    load_dotenv(env_file, override=False)
    settings = _load_role(role, config_file)
    api_key = os.getenv(settings["api_key_env"])
    if not api_key:
        raise ValueError(f"Environment variable {settings['api_key_env']!r} is not set")

    base_cache = Path(cache_dir) if cache_dir else _PROJECT_ROOT / "outputs" / "cache" / "llm"
    base_logs = Path(log_dir) if log_dir else _PROJECT_ROOT / "outputs" / "logs"
    base_cache.mkdir(parents=True, exist_ok=True)
    effective_extra_body = {**settings.get("thinking_parameters", {}), **dict(extra_body or {})}
    default_request_parameters = {
        "model": settings["model"],
        "temperature": temperature,
        "response_format": {"type": "json_object"} if json_mode else None,
        "max_tokens": max_tokens,
        "extra_body": effective_extra_body,
        "thinking_mode": settings["thinking_mode"],
    }
    cache_key = _request_key(
        model=settings["model"], base_url=settings["base_url"], messages=messages,
        temperature=temperature, json_mode=json_mode, extra_body=effective_extra_body,
        prompt_sha256=prompt_sha256, max_tokens=max_tokens,
    )
    cache_path = base_cache / f"{cache_key}.json"
    log_path = base_logs / f"{role}.jsonl"
    started = time.perf_counter()

    cached = _read_cache(cache_path, json_mode)
    if cached is not None:
        cached.setdefault("api_model", cached.get("model", settings["model"]))
        cached.setdefault("request_parameters", default_request_parameters)
        elapsed = time.perf_counter() - started
        _append_log(log_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "model": settings["model"],
            "api_model": cached["api_model"],
            "base_url": settings["base_url"],
            "request_hash": cache_key,
            "cache_hit": True,
            "attempts": 0,
            "temperature": temperature,
            "json_mode": json_mode,
            "extra_body": effective_extra_body,
            "thinking_mode": settings["thinking_mode"],
            "request_parameters": cached.get("request_parameters", {}),
            "token_usage": cached.get("usage"),
            "elapsed_seconds": elapsed,
            "status": "success",
            "prompt_sha256": prompt_sha256,
            "api_request_sent": False,
            **({"messages": list(messages)} if log_messages else {}),
        })
        return {**cached, "cache_hit": True, "elapsed_seconds": elapsed}

    if client is None:
        client = OpenAI(api_key=api_key, base_url=settings["base_url"], timeout=float(timeout), max_retries=0)

    call_messages = [dict(message) for message in messages]
    format_supported = json_mode
    parse_retry_used = False
    attempts = 0
    usage: dict[str, int] | None = None
    last_error: Exception | None = None

    while attempts < _MAX_ATTEMPTS:
        attempts += 1
        request: dict[str, Any] = {
            "model": settings["model"],
            "messages": call_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request["max_tokens"] = int(max_tokens)
        if format_supported:
            request["response_format"] = {"type": "json_object"}
        if effective_extra_body:
            request["extra_body"] = effective_extra_body
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            if format_supported and _is_response_format_unsupported(exc):
                format_supported = False
                last_error = exc
                if attempts < _MAX_ATTEMPTS:
                    continue
                break
            elif _is_retryable(exc) and attempts < _MAX_ATTEMPTS:
                last_error = exc
                _backoff(attempts)
                continue
            else:
                last_error = exc
                break

        choice = response.choices[0]
        text = choice.message.content or ""
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage_details = raw_usage.model_dump() if hasattr(raw_usage, "model_dump") else {}
            usage = {
                "prompt_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(raw_usage, "total_tokens", 0) or 0),
            }
            details = usage_details.get("completion_tokens_details", {}) if isinstance(usage_details, dict) else {}
            if not details:
                details = getattr(raw_usage, "completion_tokens_details", None)
                details = details.model_dump() if hasattr(details, "model_dump") else details
            reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
            if reasoning_tokens is None and isinstance(usage_details, dict):
                reasoning_tokens = usage_details.get("reasoning_tokens")
            parsed_reasoning_tokens = _optional_int(reasoning_tokens)
            if parsed_reasoning_tokens is not None:
                usage["reasoning_tokens"] = parsed_reasoning_tokens

        parsed: Any = None
        if json_mode:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                if not parse_retry_used and attempts < _MAX_ATTEMPTS:
                    parse_retry_used = True
                    call_messages = [
                        *call_messages,
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": "The previous response was invalid JSON. Return only one valid JSON object.",
                        },
                    ]
                    last_error = exc
                    continue
                last_error = exc
                break

        result = {
            "text": text,
            "json": parsed,
            "usage": usage,
            "model": settings["model"],
            "api_model": str(getattr(response, "model", None) or settings["model"]),
            "request_parameters": {
                "model": settings["model"],
                "temperature": temperature,
                "response_format": {"type": "json_object"} if format_supported else None,
                "max_tokens": max_tokens,
                "extra_body": effective_extra_body,
                "thinking_mode": settings["thinking_mode"],
            },
            "cache_hit": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_cache(cache_path, result)
        _append_log(log_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "model": settings["model"],
            "api_model": result["api_model"],
            "base_url": settings["base_url"],
            "request_hash": cache_key,
            "cache_hit": False,
            "attempts": attempts,
            "temperature": temperature,
            "json_mode": json_mode,
            "extra_body": effective_extra_body,
            "thinking_mode": settings["thinking_mode"],
            "request_parameters": result["request_parameters"],
            "token_usage": usage,
            "elapsed_seconds": result["elapsed_seconds"],
            "status": "success",
            "prompt_sha256": prompt_sha256,
            "api_request_sent": True,
            **({"messages": call_messages} if log_messages else {}),
        })
        return result

    elapsed = time.perf_counter() - started
    _append_log(log_path, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "model": settings["model"],
        "base_url": settings["base_url"],
        "request_hash": cache_key,
        "cache_hit": False,
        "attempts": attempts,
        "temperature": temperature,
        "json_mode": json_mode,
        "extra_body": effective_extra_body,
        "thinking_mode": settings["thinking_mode"],
        "token_usage": usage,
        "elapsed_seconds": elapsed,
        "status": "error",
        "error_type": type(last_error).__name__ if last_error else "UnknownError",
        "error": str(last_error) if last_error else "LLM call failed",
        "prompt_sha256": prompt_sha256,
        "api_request_sent": True,
        **({"messages": call_messages} if log_messages else {}),
    })
    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM call failed without a response")
