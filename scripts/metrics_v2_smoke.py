"""Run Step 1 smoke checks for the metrics_v2 scoring pipeline."""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
from rq4.llm import _load_role, call_llm

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "metrics_v2"
CACHE_DIR = OUTPUT_ROOT / "cache"
LOG_DIR = OUTPUT_ROOT / "logs"
LABELS = {"Entailment", "Neutral", "Contradiction"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _protected_pairs() -> set[tuple[str, str]]:
    protected: set[tuple[str, str]] = set()
    for rel in ("human_check_key.csv", "human_validation/answers_key.csv"):
        path = PROJECT_ROOT / "outputs" / "final" / rel
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                question_id = str(row.get("question_id", "")).strip()
                condition = str(row.get("condition", "")).strip()
                if question_id and condition:
                    protected.add((question_id, condition))
    return protected


def _select_smoke_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    answers = _read_jsonl(PROJECT_ROOT / "outputs" / "final" / "answers_final.jsonl")
    questions = {row["id"]: row for row in _read_jsonl(PROJECT_ROOT / "outputs" / "final" / "questions_final.jsonl")}
    prior = {
        (str(row.get("id")), str(row.get("condition"))): row
        for row in _read_jsonl(PROJECT_ROOT / "outputs" / "final" / "scores_final_revised.jsonl")
    }
    protected = _protected_pairs()
    eligible = [
        row for row in answers
        if (str(row.get("id")), str(row.get("condition"))) not in protected
        and row.get("status") == "generated"
        and prior.get((str(row.get("id")), str(row.get("condition"))), {}).get("abstention", {}).get("label") != "abstains"
    ]
    random.Random(42).shuffle(eligible)
    c4 = next((row for row in eligible if row.get("condition") == "C4"), None)
    if c4 is None:
        raise RuntimeError("Could not select a non-blind C4 smoke item")
    selected = [c4]
    selected.extend(row for row in eligible if row is not c4 and row.get("id") != c4.get("id"))
    selected = selected[:3]
    if len(selected) != 3:
        raise RuntimeError("Could not select three non-blind C4 smoke items")
    return selected, questions


def _ragchecker(settings: dict[str, Any], role: str, answer: dict[str, Any], question: dict[str, Any], check_type: str) -> dict[str, Any]:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from ragchecker import RAGChecker
    from ragchecker.container import RAGResult, RAGResults, RetrievedDoc

    keypoints = [str(point["text"]) for point in question.get("keypoints", [])]
    result = RAGResult(
        query_id=f"{answer['id']}-{answer['condition']}-{role}-{check_type}",
        query=str(question["question"]),
        gt_answer=str(question.get("reference_answer", "")),
        response=str(answer.get("answer", "")),
        retrieved_context=[
            RetrievedDoc(doc_id=str(p["passage_id"]), text=str(p["text"]))
            for p in answer.get("passages", [])
        ],
        gt_answer_claims=keypoints,
    )
    results = RAGResults(results=[result])
    trace: list[dict[str, Any]] = []

    def judge_calls(prompts: list[str]) -> list[str]:
        values: list[str] = []
        for prompt in prompts:
            response = call_llm(
                role,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                extra_body=settings.get("thinking_parameters", {}),
                max_tokens=10,
                config_path=PROJECT_ROOT / "configs" / "models.yaml",
                cache_dir=CACHE_DIR / "llm",
                log_dir=LOG_DIR,
                prompt_sha256="ragchecker-0.1.9-one-word-v1",
                log_messages=True,
            )
            label = str(response.get("text", "")).strip()
            trace.append({
                "label": label,
                "usage": response.get("usage"),
                "elapsed_seconds": response.get("elapsed_seconds"),
                "cache_hit": response.get("cache_hit"),
            })
            if label not in LABELS:
                raise RuntimeError(f"RAGChecker returned a non-label response (length={len(label)})")
            values.append(label)
        return values

    checker = RAGChecker(
        extractor_name=f"openai/{settings['model']}",
        checker_name=f"openai/{settings['model']}",
        extractor_api_base=settings["base_url"],
        checker_api_base=settings["base_url"],
        openai_api_key=os.getenv(settings["api_key_env"], "unused-custom-callback"),
        joint_check=False,
        custom_llm_api_func=judge_calls,
    )
    checker.check_claims(results, check_type=check_type)
    if check_type == "retrieved2answer":
        computed = checker.evaluate(results, metrics=["claim_recall", "context_precision"])
        return {"values": result.retrieved2answer, "metrics": computed, "calls": trace}
    labels = getattr(result, check_type)
    return {"values": labels, "calls": trace}


def _json_call(role: str, prompt_name: str, instruction: str, payload: dict[str, Any], temperature: float = 0.0) -> dict[str, Any]:
    result = call_llm(
        role,
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        json_mode=True,
        temperature=temperature,
        config_path=PROJECT_ROOT / "configs" / "models.yaml",
        cache_dir=CACHE_DIR / "llm",
        log_dir=LOG_DIR,
        prompt_sha256=f"metrics-v2-smoke-{prompt_name}-v1",
        log_messages=True,
        max_tokens=1024,
    )
    if not isinstance(result.get("json"), dict):
        raise RuntimeError(f"{prompt_name} did not return a JSON object")
    return result


async def _ragas_faithfulness(role: str, settings: dict[str, Any], answer: dict[str, Any], question: dict[str, Any]) -> tuple[float, int]:
    import httpx
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    started = time.perf_counter()
    calls: list[dict[str, Any]] = []

    async def request_start(request: httpx.Request) -> None:
        request.extensions["started"] = time.perf_counter()

    async def response_done(response: httpx.Response) -> None:
        try:
            await response.aread()
            body = response.json()
        except Exception:
            body = {}
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        entry = {
            "status": response.status_code,
            "api_model": body.get("model") if isinstance(body, dict) else None,
            "token_usage": usage,
            "elapsed_seconds": time.perf_counter() - response.request.extensions.get("started", time.perf_counter()),
        }
        calls.append(entry)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / f"{role}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "ragas_faithfulness", "role": role, **entry}, sort_keys=True) + "\n")

    client = AsyncOpenAI(
        api_key=os.environ[settings["api_key_env"]],
        base_url=settings["base_url"],
        timeout=120,
        max_retries=0,
        http_client=httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=20),
            event_hooks={"request": [request_start], "response": [response_done]},
        ),
    )
    llm = llm_factory(
        settings["model"], client=client, temperature=0.0, max_tokens=1024,
        extra_body=settings.get("thinking_parameters", {}),
        cache=DiskCacheBackend(str(CACHE_DIR / "ragas" / role)),
    )
    try:
        contexts = [str(p["text"]) for p in answer.get("passages", [])]
        score = await Faithfulness(llm=llm).ascore(
            user_input=str(question["question"]),
            response=str(answer["answer"]),
            retrieved_contexts=contexts,
        )
        if score.value is None:
            raise RuntimeError("Ragas Faithfulness returned no score")
        return float(score.value), sum(int((entry.get("token_usage") or {}).get("total_tokens", 0) or 0) for entry in calls)
    finally:
        await client.close()


def main() -> int:
    logging.getLogger().setLevel(logging.ERROR)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = PROJECT_ROOT / "configs" / "models.yaml"
    primary = _load_role("judge_primary", config)
    secondary = _load_role("judge_secondary", config)
    for settings in (primary, secondary):
        if not os.getenv(settings["api_key_env"]):
            raise RuntimeError(f"Missing environment variable {settings['api_key_env']}")

    rows, questions = _select_smoke_rows()
    smoke_usage: dict[str, Any] = {"selected_count": len(rows), "tests": {}, "token_totals": {}}
    for role, settings in (("judge_primary", primary), ("judge_secondary", secondary)):
        role_status: dict[str, bool] = {
            "ragchecker_keypoints": False,
            "ragchecker_retriever": False,
            "response_relevance": False,
        }
        tokens = 0
        for index, answer in enumerate(rows):
            question = questions[str(answer["id"])]
            keypoint_result = _ragchecker(settings, role, answer, question, "response2answer")
            labels = keypoint_result["values"]
            if not isinstance(labels, list) or len(labels) != len(question.get("keypoints", [])) or any(str(x) not in LABELS for x in labels):
                raise RuntimeError("RAGChecker keypoint smoke check failed")
            role_status["ragchecker_keypoints"] = True
            for call in keypoint_result["calls"]:
                tokens += int((call.get("usage") or {}).get("total_tokens", 0))
            if answer.get("condition") == "C4":
                retrieval_result = _ragchecker(settings, role, answer, question, "retrieved2answer")
                if retrieval_result.get("metrics") is None:
                    raise RuntimeError("RAGChecker retriever smoke check failed")
                role_status["ragchecker_retriever"] = True
                for call in retrieval_result["calls"]:
                    tokens += int((call.get("usage") or {}).get("total_tokens", 0))
            if index == 0:
                relevance = _json_call(
                    role,
                    "response_relevance",
                    "Rate whether the answer addresses the question. Return JSON with integer score 1-5 and a one-sentence reason.",
                    {"question": question["question"], "answer": answer["answer"]},
                )
                if type(relevance["json"].get("score")) is not int or relevance["json"]["score"] not in range(1, 6) or not relevance["json"].get("reason"):
                    raise RuntimeError("Response relevance smoke check failed JSON validation")
                role_status["response_relevance"] = True
                tokens += int((relevance.get("usage") or {}).get("total_tokens", 0))
        smoke_usage["tests"][role] = role_status
        smoke_usage["token_totals"][role] = tokens

    faithfulness_scores: list[float] = []
    faithfulness_tokens = 0
    for answer in rows:
        question = questions[str(answer["id"])]
        faithfulness, item_tokens = asyncio.run(_ragas_faithfulness("judge_primary", primary, answer, question))
        if not 0.0 <= faithfulness <= 1.0:
            raise RuntimeError("Ragas Faithfulness smoke check returned an invalid score")
        faithfulness_scores.append(faithfulness)
        faithfulness_tokens += item_tokens
    answer, question = rows[0], questions[str(rows[0]["id"])]
    decomposition = _json_call(
        "judge_primary",
        "statement_decomposition",
        "Split the response into sentences or list items and atomic claims. Return a JSON object with a statements list; each item must have statement, cited_passage_ids, and claims.",
        {"answer": answer["answer"]},
    )
    if not isinstance(decomposition["json"].get("statements"), list):
        raise RuntimeError("Statement decomposition smoke check failed JSON validation")
    smoke_usage["tests"]["judge_primary"]["ragas_faithfulness"] = True
    smoke_usage["tests"]["judge_primary"]["statement_decomposition"] = True
    smoke_usage["faithfulness_smoke_count"] = len(faithfulness_scores)
    smoke_usage["faithfulness_smoke_mean"] = sum(faithfulness_scores) / len(faithfulness_scores)
    smoke_usage["token_totals"]["judge_primary"] += int((decomposition.get("usage") or {}).get("total_tokens", 0)) + faithfulness_tokens
    smoke_usage["ragas_faithfulness_tokens"] = faithfulness_tokens
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "step1_smoke.json").write_text(json.dumps(smoke_usage, indent=2), encoding="utf-8")
    print("Step 1 smoke tests: PASS")
    print("  Items tested: 3 (selected inside the script; none from the blind samples)")
    for role in ("judge_primary", "judge_secondary"):
        print(f"  {role}: {', '.join(name for name, passed in smoke_usage['tests'][role].items() if passed)}; tokens={smoke_usage['token_totals'][role]}")
    print(f"  Primary Ragas Faithfulness: PASS (n={len(faithfulness_scores)})")
    print("  Primary statement decomposition: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
