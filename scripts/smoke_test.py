"""Run small live API and metric checks for the configured RQ4 models.

Usage:
    .venv/bin/python scripts/smoke_test.py

This makes three small JSON requests and two judge-based metric checks. Results
are cached under outputs/cache/ and request metadata is appended to outputs/logs/.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv

from rq4.llm import _load_role, call_llm

QUESTION = "How long are adult blue whales, and which sex is generally larger?"
PASSAGE = (
    "Adult blue whales typically measure 24 to 30 meters long. "
    "Females are generally larger than males."
)
REFERENCE_ANSWER = (
    "Adult blue whales typically measure 24 to 30 meters long. "
    "Females are generally larger than males."
)
TOY_RESPONSE = (
    "Adult blue whales are usually 24 to 30 meters long, and females are "
    "generally larger than males."
)
# RAGChecker 0.1.9 accepts reviewed keypoints directly as strings when joint
# checking is disabled.
KEYPOINTS = [
    "Adult blue whales typically measure 24 to 30 meters long",
    "Female blue whales are generally larger than male blue whales",
]


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _request_hash(role: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{role}\n{raw}".encode("utf-8")).hexdigest()


def _api_key(settings: dict[str, str]) -> str:
    value = os.getenv(settings["api_key_env"])
    if not value:
        raise RuntimeError(f"Missing environment variable {settings['api_key_env']}")
    return value


def smoke_role(role: str, settings: dict[str, str]) -> dict[str, Any]:
    message = {
        "role": "user",
        "content": (
            f'Return exactly one JSON object with keys "ok" and "role". '
            f'Use true for ok and "{role}" for role. Do not add other text.'
        ),
    }
    # GPT-6 reasoning models require temperature=1.0. Other role pings are greedy.
    temperature = 1.0 if role == "teacher" else 0.0
    extra_body = {"enable_thinking": False} if role == "judge" else None
    result = call_llm(
        role,
        [message],
        json_mode=True,
        temperature=temperature,
        extra_body=extra_body,
    )
    parsed = result["json"]
    if not isinstance(parsed, dict) or parsed.get("ok") is not True or parsed.get("role") != role:
        raise ValueError(f"{role} returned parseable JSON with an unexpected shape")
    return result


def _make_ragas_http_client(log_path: Path, role: str, model: str, base_url: str):
    import httpx

    async def record_start(request: httpx.Request) -> None:
        request.extensions["rq4_started_at"] = time.perf_counter()

    async def record_response(response: httpx.Response) -> None:
        try:
            await response.aread()
            payload = response.json()
        except Exception:
            payload = {}
        usage = payload.get("usage") if isinstance(payload, dict) else None
        started = response.request.extensions.get("rq4_started_at", time.perf_counter())
        _append_jsonl(log_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "llm_http_response",
            "role": role,
            "model": model,
            "base_url": base_url,
            "http_status": response.status_code,
            "token_usage": usage,
            "elapsed_seconds": time.perf_counter() - started,
        })

    return httpx.AsyncClient(
        timeout=httpx.Timeout(90.0, connect=20.0),
        event_hooks={"request": [record_start], "response": [record_response]},
    )


async def run_faithfulness(settings: dict[str, str], key: str) -> float:
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    cache = DiskCacheBackend(str(PROJECT_ROOT / "outputs" / "cache" / "ragas_faithfulness"))
    http_client = _make_ragas_http_client(
        PROJECT_ROOT / "outputs" / "logs" / "judge.jsonl",
        "judge",
        settings["model"],
        settings["base_url"],
    )
    started = time.perf_counter()
    metric_payload = {
        "user_input": QUESTION,
        "response": TOY_RESPONSE,
        "retrieved_contexts": [PASSAGE],
    }
    request_hash = _request_hash("ragas_faithfulness", metric_payload)
    try:
        async with AsyncOpenAI(
            api_key=key,
            base_url=settings["base_url"],
            timeout=90.0,
            max_retries=0,
            http_client=http_client,
        ) as client:
            llm = llm_factory(
                settings["model"],
                client=client,
                temperature=0.0,
                max_tokens=1024,
                extra_body={"enable_thinking": False},
                cache=cache,
            )
            score = await Faithfulness(llm=llm).ascore(**metric_payload)
        value = float(score.value)
        _append_jsonl(PROJECT_ROOT / "outputs" / "logs" / "judge.jsonl", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "metric_result",
            "metric": "ragas_faithfulness",
            "role": "judge",
            "model": settings["model"],
            "request_hash": request_hash,
            "score": value,
            "elapsed_seconds": time.perf_counter() - started,
            "status": "success",
        })
        return value
    except Exception as exc:
        _append_jsonl(PROJECT_ROOT / "outputs" / "logs" / "judge.jsonl", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "metric_result",
            "metric": "ragas_faithfulness",
            "role": "judge",
            "model": settings["model"],
            "request_hash": request_hash,
            "elapsed_seconds": time.perf_counter() - started,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise


def run_ragchecker(settings: dict[str, str], key: str) -> list[str]:
    # Avoid RAGChecker/LiteLLM's optional online cost-map fetch during import.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from ragchecker import RAGChecker
    from ragchecker.container import RAGResult, RAGResults, RetrievedDoc

    result = RAGResult(
        query_id="toy-blue-whale-001",
        query=QUESTION,
        gt_answer=REFERENCE_ANSWER,
        response=TOY_RESPONSE,
        retrieved_context=[RetrievedDoc(doc_id="toy-passage-001", text=PASSAGE)],
        gt_answer_claims=KEYPOINTS,
    )
    results = RAGResults(results=[result])
    def cached_judge_calls(prompts: list[str]) -> list[str]:
        answers = []
        for prompt in prompts:
            response = call_llm(
                "judge",
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                extra_body={"enable_thinking": False},
                max_tokens=10,
            )
            answers.append(response["text"])
        return answers

    checker = RAGChecker(
        extractor_name=f"openai/{settings['model']}",
        checker_name=f"openai/{settings['model']}",
        extractor_api_base=settings["base_url"],
        checker_api_base=settings["base_url"],
        openai_api_key=key,
        joint_check=False,
        custom_llm_api_func=cached_judge_calls,
    )

    checker.check_claims(results, check_type="response2answer")
    labels = result.response2answer
    if not isinstance(labels, list) or len(labels) != len(KEYPOINTS):
        raise ValueError(f"Unexpected RAGChecker response2answer output: {labels!r}")
    return [str(label) for label in labels]


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    settings = {role: _load_role(role, config_path) for role in ("teacher", "generator", "judge")}
    failures: list[str] = []

    print("Model JSON smoke checks:")
    for role, role_settings in settings.items():
        try:
            result = smoke_role(role, role_settings)
            usage = result.get("usage") or {}
            print(
                f"  PASS {role}: {role_settings['model']} returned valid JSON "
                f"(tokens={usage.get('total_tokens', 'n/a')}, "
                f"cache_hit={result['cache_hit']})"
            )
        except Exception as exc:
            failures.append(f"{role} JSON: {type(exc).__name__}: {exc}")
            print(f"  FAIL {role}: {type(exc).__name__}: {exc}")

    judge_settings = settings["judge"]
    try:
        faithfulness = asyncio.run(run_faithfulness(judge_settings, _api_key(judge_settings)))
        print(f"Ragas Faithfulness: {faithfulness:.3f}")
    except Exception as exc:
        failures.append(f"Ragas Faithfulness: {type(exc).__name__}: {exc}")
        print(f"Ragas Faithfulness: FAIL ({type(exc).__name__}: {exc})")

    try:
        labels = run_ragchecker(judge_settings, _api_key(judge_settings))
        print(f"RAGChecker response2answer: {json.dumps(labels, ensure_ascii=False)}")
    except Exception as exc:
        failures.append(f"RAGChecker response2answer: {type(exc).__name__}: {exc}")
        print(f"RAGChecker response2answer: FAIL ({type(exc).__name__}: {exc})")

    if failures:
        print("\nSmoke test completed with failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nAll configured model and metric smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
