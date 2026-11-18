"""Score pilot answers with RAGChecker, adapted FLASK, abstention, and Ragas."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
from rq4.llm import _load_role, call_llm
from scripts.build_index import _read_jsonl, _write_jsonl
from scripts.check_questions import (
    MODELS_CONFIG,
    check_decisive_labels,
    ragchecker_response2answer,
    _write_xlsx_minimal,
)

QUESTIONS_PATH = PROJECT_ROOT / "outputs" / "pilot" / "questions_reviewed.jsonl"
CONDITIONS_PATH = PROJECT_ROOT / "outputs" / "pilot" / "conditions_pilot.jsonl"
ANSWERS_PATH = PROJECT_ROOT / "outputs" / "pilot" / "answers_pilot.jsonl"
SCORES_PATH = PROJECT_ROOT / "outputs" / "pilot" / "scores_pilot.jsonl"
REVIEW_PATH = PROJECT_ROOT / "outputs" / "pilot" / "scores_review.xlsx"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "pilot" / "score_summary.csv"
PAIRED_DROPS_PATH = PROJECT_ROOT / "outputs" / "pilot" / "paired_drops.csv"
FLASK_PROMPT = PROJECT_ROOT / "prompts" / "judge_correctness_flask.md"
ABSTENTION_PROMPT = PROJECT_ROOT / "prompts" / "judge_abstention.md"


def _load_system(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index("## System") + len("## System")
    end = text.index("## User template", start)
    return text[start:end].strip()


def _keypoints_text(points: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {'[D]' if point.get('decisive') else '[ ]'} {point.get('text', '')}"
        for point in points
    )


def _judge_json(
    system_prompt: str, user_prompt: str, *, llm_call: Callable[..., dict[str, Any]] = call_llm,
    prompt_sha256: str | None = None,
) -> dict[str, Any]:
    return llm_call(
        "judge",
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        json_mode=True,
        temperature=0.0,
        prompt_sha256=prompt_sha256 or hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        log_messages=True,
    )


def _flask_user(question: str, reference: str, keypoints: list[dict[str, Any]], answer: str) -> str:
    return (
        f"Question: {question}\n\nReference answer: {reference}\n\n"
        f"Keypoints (D = decisive):\n{_keypoints_text(keypoints)}\n\nAnswer to grade:\n{answer}"
    )


def _abstention_user(question: str, answer: str) -> str:
    return f"Question: {question}\n\nAnswer:\n{answer}"


def keypoint_metrics(keypoints: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    decisive = [point for point in keypoints if point.get("decisive")]
    normalized = [str(label).strip() for label in labels]
    correctness = bool(decisive) and all(
        normalized[index].casefold() == "entailment"
        for index, point in enumerate(keypoints) if point.get("decisive")
    ) and not any(label.casefold() == "contradiction" for label in normalized)
    completeness = (
        sum(label.casefold() == "entailment" for label in normalized) / len(keypoints)
        if keypoints else None
    )
    return {"correctness": correctness, "completeness": completeness, "keypoint_labels": normalized}


def needs_manual_check(correctness: bool, flask_score: int) -> bool:
    return (correctness and flask_score <= 3) or (not correctness and flask_score >= 4)


def score_answer_record(
    answer_row: dict[str, Any],
    question: dict[str, Any],
    condition: dict[str, Any],
    *,
    flask_system: str,
    abstention_system: str,
    judge_json: Callable[[str, str], dict[str, Any]],
    ragchecker_fn: Callable[[dict[str, Any], dict[str, Any]], tuple[list[str], list[dict[str, Any]]]],
    ragas_fn: Callable[[str, dict[str, Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    answer = str(answer_row.get("answer", ""))
    qtext = str(question["question"])
    keypoints = question.get("keypoints", [])
    condition_name = str(answer_row["condition"])
    record: dict[str, Any] = {
        **{key: answer_row.get(key) for key in ("id", "type", "question", "reference_answer", "condition", "gold_position", "answer", "passage_map")},
        "status": "scored",
        "raw_outputs": {},
        "ragchecker": None,
        "flask": None,
        "abstention": None,
        "faithfulness": None,
        "context_recall": None,
    }
    if condition_name in {"C1", "C2", "C4"}:
        try:
            labels, trace = ragchecker_fn(answer_row, question)
            record["ragchecker"] = keypoint_metrics(keypoints, labels)
            record["raw_outputs"]["ragchecker"] = trace
        except Exception as exc:
            record["ragchecker"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if condition_name in {"C1", "C2", "C4"}:
        user_prompt = _flask_user(qtext, str(question.get("reference_answer", "")), keypoints, answer)
        try:
            response = judge_json(flask_system, user_prompt)
            data = response.get("json")
            if not isinstance(data, dict) or type(data.get("score")) is not int or data["score"] not in range(1, 6):
                raise ValueError("FLASK output must contain an integer score from 1 to 5")
            record["flask"] = {"score": data["score"], "reason": str(data.get("reason", ""))}
            record["raw_outputs"]["flask"] = {"text": response.get("text"), "json": data}
        except Exception as exc:
            record["flask"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    try:
        response = judge_json(abstention_system, _abstention_user(qtext, answer))
        data = response.get("json")
        allowed = {"answers", "abstains", "partially_answers"}
        if not isinstance(data, dict) or data.get("label") not in allowed:
            raise ValueError(f"Abstention output label must be one of {sorted(allowed)}")
        record["abstention"] = {"label": data["label"], "reason": str(data.get("reason", ""))}
        record["raw_outputs"]["abstention"] = {"text": response.get("text"), "json": data}
    except Exception as exc:
        record["abstention"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    no_claim_reason = None
    if not answer.strip():
        no_claim_reason = "The answer is empty; no response claims can be extracted."
    elif record.get("abstention", {}).get("label") == "abstains":
        no_claim_reason = "The answer abstains without making a substantive claim."
    if no_claim_reason:
        record["faithfulness"] = {"value": None, "status": "NaN", "reason": no_claim_reason}
    else:
        try:
            value, raw = ragas_fn("faithfulness", answer_row, question)
            record["faithfulness"] = value
            record["raw_outputs"]["ragas_faithfulness"] = raw
        except Exception as exc:
            record["faithfulness"] = {"value": None, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    if condition_name == "C4":
        try:
            value, raw = ragas_fn("context_recall", answer_row, question)
            record["context_recall"] = value
            record["raw_outputs"]["ragas_context_recall"] = raw
        except Exception as exc:
            record["context_recall"] = {"value": None, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    rag = record.get("ragchecker")
    flask = record.get("flask")
    if isinstance(rag, dict) and isinstance(rag.get("correctness"), bool) and isinstance(flask, dict) and isinstance(flask.get("score"), int):
        record["needs_manual_check"] = needs_manual_check(rag["correctness"], flask["score"])
    else:
        record["needs_manual_check"] = None
    return record


def _score_system_call(prompt_path: Path, q: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    system = _load_system(prompt_path)
    if prompt_path == FLASK_PROMPT:
        user = _flask_user(q["question"], q["reference_answer"], q["keypoints"], answer["answer"])
    else:
        user = _abstention_user(q["question"], answer["answer"])
    return _judge_json(system, user)


def _ragchecker_call(answer: dict[str, Any], question: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    context = [(str(p["passage_id"]), str(p["text"])) for p in answer.get("passages", [])]
    trace: list[dict[str, Any]] = []
    labels = ragchecker_response2answer(
        question_id=f"{answer['id']}-{answer['condition']}-score",
        question=str(question["question"]),
        reference_answer=str(question["reference_answer"]),
        answer=str(answer.get("answer", "")),
        keypoints=question.get("keypoints", []),
        context=context,
        trace=trace,
    )
    return labels, trace


def _http_response_raw(payload: dict[str, Any]) -> Any:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        return choices[0].get("message", {}).get("content") or choices[0].get("text")
    return payload.get("output") or payload.get("content")


def _reasoning_tokens(usage: dict[str, Any]) -> int | None:
    value = usage.get("reasoning_tokens")
    if value is None:
        details = usage.get("completion_tokens_details") or {}
        if isinstance(details, dict):
            value = details.get("reasoning_tokens")
    return int(value) if value is not None else None


def _make_ragas_client(cache_name: str, captured_calls: list[dict[str, Any]], *, max_tokens: int = 1024):
    import httpx
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = _load_role("judge", MODELS_CONFIG)
    api_key = os.getenv(settings["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing environment variable {settings['api_key_env']}")
    log_path = PROJECT_ROOT / "outputs" / "logs" / "judge.jsonl"

    async def request_start(request: httpx.Request) -> None:
        request.extensions["rq4_started_at"] = time.perf_counter()

    async def response_done(response: httpx.Response) -> None:
        try:
            await response.aread()
            payload = response.json()
        except Exception:
            payload = {}
        request_body = {}
        try:
            request_body = json.loads(response.request.content.decode("utf-8"))
        except Exception:
            pass
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        call = {
            "api_model": payload.get("model", settings["model"]) if isinstance(payload, dict) else settings["model"],
            "raw_response": _http_response_raw(payload) if isinstance(payload, dict) else None,
            "token_usage": usage,
            "reasoning_tokens": _reasoning_tokens(usage) if isinstance(usage, dict) else None,
            "request_parameters": {
                key: request_body.get(key)
                for key in ("model", "temperature", "max_tokens", "max_completion_tokens", "response_format", "extra_body", "enable_thinking", "reasoning", "reasoning_effort")
                if key in request_body
            },
            "elapsed_seconds": time.perf_counter() - response.request.extensions.get("rq4_started_at", time.perf_counter()),
            "http_status": response.status_code,
        }
        captured_calls.append(call)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": "ragas_metric_llm", "role": "judge", **call}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    async_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=20.0),
        event_hooks={"request": [request_start], "response": [response_done]},
    )
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=settings["base_url"],
        timeout=120.0,
        max_retries=0,
        http_client=async_client,
    )
    from ragas.cache import DiskCacheBackend
    cache = DiskCacheBackend(str(PROJECT_ROOT / "outputs" / "cache" / f"ragas_{cache_name}"))
    llm = llm_factory(
        settings["model"],
        client=client,
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body=settings["thinking_parameters"],
        cache=cache,
    )
    return client, llm


def _run_ragas(metric: str, answer_row: dict[str, Any], question: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from ragas.metrics.collections import ContextRecall, Faithfulness

    captured: list[dict[str, Any]] = []
    client, llm = _make_ragas_client(metric, captured)
    contexts = [str(row["text"]) for row in answer_row.get("passages", [])]
    payload: dict[str, Any]
    if metric == "faithfulness":
        ragas_metric = Faithfulness(llm=llm)
        payload = {
            "user_input": str(question["question"]),
            "response": str(answer_row.get("answer", "")),
            "retrieved_contexts": contexts,
        }
    else:
        ragas_metric = ContextRecall(llm=llm)
        payload = {
            "user_input": str(question["question"]),
            "retrieved_contexts": contexts,
            "reference": str(question["reference_answer"]),
        }

    async def evaluate() -> Any:
        try:
            return await ragas_metric.ascore(**payload)
        finally:
            await client.close()

    started = time.perf_counter()
    result = asyncio.run(evaluate())
    value = float(result.value)
    if metric == "faithfulness" and not np.isfinite(value):
        return (
            {"value": None, "status": "NaN", "reason": "Ragas Faithfulness extracted no statements to check."},
            {"metric": metric, "payload": payload, "score": None, "status": "NaN",
             "reason": "Ragas Faithfulness extracted no statements to check.",
             "elapsed_seconds": time.perf_counter() - started, "raw_calls": captured},
        )
    return (
        {"value": value, "status": "scored"},
        {"metric": metric, "payload": payload, "score": value, "elapsed_seconds": time.perf_counter() - started, "raw_calls": captured},
    )


def _preflight_ragchecker(answer: dict[str, Any], question: dict[str, Any]) -> None:
    labels, trace = _ragchecker_call(answer, question)
    if not trace:
        raise RuntimeError("RAGChecker preflight made no judge calls")
    allowed = {"Entailment", "Neutral", "Contradiction"}
    if any(row.get("raw_response") not in allowed for row in trace):
        raise ValueError("RAGChecker preflight returned an invalid label")
    print("RAGChecker preflight — raw judge prompt:")
    print(trace[0]["prompt"])
    print("RAGChecker preflight — raw judge response:")
    print(trace[0]["raw_response"])
    print("RAGChecker preflight labels:", labels)
    print("All preflight responses are exact Entailment / Neutral / Contradiction labels.")


def score_all(
    questions: list[dict[str, Any]], conditions: list[dict[str, Any]], answers: list[dict[str, Any]],
    *, judge_json: Callable[[str, str], dict[str, Any]] | None = None,
    ragchecker_fn: Callable[[dict[str, Any], dict[str, Any]], tuple[list[str], list[dict[str, Any]]]] | None = None,
    ragas_fn: Callable[[str, dict[str, Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    flask_system = _load_system(FLASK_PROMPT)
    abstention_system = _load_system(ABSTENTION_PROMPT)
    question_by_id = {str(row["id"]): row for row in questions}
    condition_by_id = {str(row["id"]): row for row in conditions}
    if judge_json is None:
        judge_json = lambda system, user: _judge_json(system, user)
    if ragchecker_fn is None:
        ragchecker_fn = _ragchecker_call
    if ragas_fn is None:
        ragas_fn = _run_ragas
    preflight_answer = next(
        (row for row in answers if row.get("condition") == "C1" and row.get("status") == "generated"), None
    )
    if preflight_answer is None:
        raise ValueError("No successful C1 answer is available for the RAGChecker preflight")
    _preflight_ragchecker(preflight_answer, question_by_id[str(preflight_answer["id"])])
    output = []
    for position, answer in enumerate(answers, start=1):
        qid = str(answer["id"])
        print(f"Scoring {position}/{len(answers)}: {qid} {answer['condition']}", flush=True)
        output.append(score_answer_record(
            answer,
            question_by_id[qid],
            condition_by_id[qid],
            flask_system=flask_system,
            abstention_system=abstention_system,
            judge_json=judge_json,
            ragchecker_fn=ragchecker_fn,
            ragas_fn=ragas_fn,
        ))
    return output


def _score_number(score: dict[str, Any] | None) -> float | None:
    if not isinstance(score, dict):
        return None
    value = score.get("value")
    if value is None or score.get("status") in {"error", "NaN"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def summary_frames(scores: list[dict[str, Any]], conditions: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_question: dict[str, dict[str, dict[str, Any]]] = {}
    for row in scores:
        by_question.setdefault(str(row["id"]), {})[str(row["condition"])] = row
    condition_rows = []
    for condition_name in ("C1", "C2", "C3", "C4"):
        rows = [r for r in scores if r["condition"] == condition_name]
        correct = [r["ragchecker"].get("correctness") for r in rows if isinstance(r.get("ragchecker"), dict) and "correctness" in r["ragchecker"]]
        completeness = [r["ragchecker"].get("completeness") for r in rows if isinstance(r.get("ragchecker"), dict) and r["ragchecker"].get("completeness") is not None]
        flask = [r["flask"].get("score") for r in rows if isinstance(r.get("flask"), dict) and isinstance(r["flask"].get("score"), int)]
        abstention_rate = [r["abstention"].get("label") == "abstains" for r in rows if isinstance(r.get("abstention"), dict) and r["abstention"].get("label")]
        faithfulness = [n for r in rows if (n := _score_number(r.get("faithfulness"))) is not None]
        row: dict[str, Any] = {
            "condition": condition_name,
            "correctness_rate": float(np.mean(correct)) if correct else None,
            "average_completeness": float(np.mean(completeness)) if completeness else None,
            "average_flask": float(np.mean(flask)) if flask else None,
            "abstention_rate": float(np.mean(abstention_rate)) if abstention_rate else None,
            "average_faithfulness": float(np.mean(faithfulness)) if faithfulness else None,
            "faithfulness_n": len(faithfulness),
        }
        flagged = [r for r in rows if r.get("needs_manual_check") is True]
        row["needs_manual_check_count"] = len(flagged)
        row["needs_manual_check_ids"] = "; ".join(str(r["id"]) for r in flagged)
        if condition_name == "C4":
            cond_by_id = {str(q["id"]): q for q in conditions}
            chunk_hits = [cond_by_id[str(r["id"])].get("c4_gold_hit_rank") is not None for r in rows]
            gold_page_hits = [cond_by_id[str(r["id"])].get("c4_gold_page_hit_rank", {}).get("gold") is not None for r in rows]
            partner_page_hits = [cond_by_id[str(r["id"])].get("c4_gold_page_hit_rank", {}).get("partner") is not None for r in rows if cond_by_id[str(r["id"])].get("c4_gold_page_hit_rank", {}).get("partner") is not None or r.get("type") == "comparison"]
            recalls = [n for r in rows if (n := _score_number(r.get("context_recall"))) is not None]
            row.update({
                "gold_chunk_hit_at_10": float(np.mean(chunk_hits)) if chunk_hits else None,
                "gold_page_hit_at_10": float(np.mean(gold_page_hits)) if gold_page_hits else None,
                "partner_page_hit_at_10": float(np.mean(partner_page_hits)) if partner_page_hits else None,
                "average_context_recall": float(np.mean(recalls)) if recalls else None,
            })
        condition_rows.append(row)
    paired_rows = []
    for qid, condition_map in by_question.items():
        for pair_name, left_name, right_name in (("C1-C2", "C1", "C2"), ("C1-C4", "C1", "C4")):
            left, right = condition_map.get(left_name), condition_map.get(right_name)
            if not left or not right:
                continue
            row = {"id": qid, "comparison": pair_name}
            for metric in ("correctness", "completeness"):
                left_metric = left.get("ragchecker") or {}
                right_metric = right.get("ragchecker") or {}
                lv, rv = left_metric.get(metric), right_metric.get(metric)
                row[f"{metric}_drop"] = (float(lv) - float(rv)) if lv is not None and rv is not None else None
            for metric in ("flask", "faithfulness"):
                lv = left.get(metric, {}).get("score") if metric == "flask" and isinstance(left.get(metric), dict) else _score_number(left.get(metric))
                rv = right.get(metric, {}).get("score") if metric == "flask" and isinstance(right.get(metric), dict) else _score_number(right.get(metric))
                row[f"{metric}_drop"] = (float(lv) - float(rv)) if lv is not None and rv is not None else None
            paired_rows.append(row)
    summary = pd.DataFrame(condition_rows)
    paired = pd.DataFrame(paired_rows)
    wide_rows = []
    condition_by_id = {str(row["id"]): row for row in conditions}
    for qid, condition_map in by_question.items():
        first = condition_map.get("C1", next(iter(condition_map.values())))
        row: dict[str, Any] = {"id": qid, "type": first["type"], "question": first["question"]}
        for cname in ("C1", "C2", "C3", "C4"):
            scored = condition_map.get(cname, {})
            rag = scored.get("ragchecker") or {}
            row[f"{cname}_correctness"] = rag.get("correctness")
            row[f"{cname}_completeness"] = rag.get("completeness")
            row[f"{cname}_FLASK"] = (scored.get("flask") or {}).get("score")
            row[f"{cname}_abstention"] = (scored.get("abstention") or {}).get("label")
            faith = scored.get("faithfulness") or {}
            row[f"{cname}_faithfulness"] = faith.get("value") if faith.get("value") is not None else "NaN"
        row["needs_manual_check"] = any(
            condition_row.get("needs_manual_check") is True
            for condition_row in condition_map.values()
        )
        row["C4_gold_chunk_rank"] = condition_by_id[qid].get("c4_gold_hit_rank")
        page_rank = condition_by_id[qid].get("c4_gold_page_hit_rank", {})
        row["C4_gold_page_rank"] = page_rank.get("gold")
        row["C4_partner_page_rank"] = page_rank.get("partner")
        row["C4_ContextRecall"] = (condition_map.get("C4", {}).get("context_recall") or {}).get("value")
        wide_rows.append(row)
    return summary, paired, pd.DataFrame(wide_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--conditions", type=Path, default=CONDITIONS_PATH)
    parser.add_argument("--answers", type=Path, default=ANSWERS_PATH)
    parser.add_argument("--out", type=Path, default=SCORES_PATH)
    parser.add_argument("--review-xlsx", type=Path, default=REVIEW_PATH)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--paired-csv", type=Path, default=PAIRED_DROPS_PATH)
    parser.add_argument("--preflight-only", action="store_true", help="Print and validate one raw RAGChecker judge call, then stop.")
    args = parser.parse_args()
    questions = _read_jsonl(args.questions)
    conditions = _read_jsonl(args.conditions)
    answers = _read_jsonl(args.answers)
    if len(answers) != len(questions) * 4:
        raise ValueError(f"Expected {len(questions) * 4} answer records; found {len(answers)}")
    if args.preflight_only:
        question_by_id = {str(row["id"]): row for row in questions}
        preflight_answer = next(
            (row for row in answers if row.get("condition") == "C1" and row.get("status") == "generated"), None
        )
        if preflight_answer is None:
            raise ValueError("No successful C1 answer is available for the RAGChecker preflight")
        _preflight_ragchecker(preflight_answer, question_by_id[str(preflight_answer["id"])])
        return
    scores = score_all(questions, conditions, answers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out, scores)
    summary, paired, wide = summary_frames(scores, conditions)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    paired.to_csv(args.paired_csv, index=False)
    try:
        with pd.ExcelWriter(args.review_xlsx, engine="openpyxl") as writer:
            wide.to_excel(writer, index=False, sheet_name="Scores review")
            summary.to_excel(writer, index=False, sheet_name="Summary")
            paired.to_excel(writer, index=False, sheet_name="Paired drops")
            for sheet in writer.sheets.values():
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        _write_xlsx_minimal(wide, args.review_xlsx)
    print("Summary by condition:")
    print(summary.to_string(index=False))
    print("Paired drops (C1−C2 and C1−C4):")
    print(paired.to_string(index=False))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.review_xlsx}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.paired_csv}")


if __name__ == "__main__":
    main()
