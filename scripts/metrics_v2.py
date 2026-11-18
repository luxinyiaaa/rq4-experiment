"""Definition-anchored RQ4 metric scoring for the frozen S1 answer set."""
from __future__ import annotations

import argparse
import asyncio
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
from rq4.llm import _load_role, call_llm

FINAL_DIR = PROJECT_ROOT / "outputs" / "final"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics_v2"
PROMPT_DIR = PROJECT_ROOT / "prompts" / "judge_metrics"
CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"
LABELS = {"Entailment", "Neutral", "Contradiction"}
ABSTENTION_LABELS = {"answers", "abstains", "partly_answers"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prompt_text(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _call_json(
    role: str,
    prompt_name: str,
    payload: dict[str, Any],
    *,
    mock_llm: Callable[..., dict[str, Any]] | None = None,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    system = _prompt_text(prompt_name)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    llm_fn = mock_llm or call_llm
    result: dict[str, Any] | None = None
    for attempt in range(2):
        result = llm_fn(
            role,
            messages,
            json_mode=True,
            temperature=0.0,
            config_path=CONFIG_PATH,
            cache_dir=METRICS_DIR / "cache" / "llm",
            log_dir=METRICS_DIR / "logs",
            prompt_sha256=_prompt_hash(system),
            log_messages=True,
            max_tokens=max_tokens,
        )
        if isinstance(result.get("json"), dict):
            result["_metrics_parse_retry_used"] = attempt == 1
            return result
        if attempt == 0:
            messages = [*messages, {
                "role": "user",
                "content": f"The previous response was not a JSON object. Return exactly one JSON object for {prompt_name}; do not return an array or plain text.",
            }]
    raise ValueError(f"{prompt_name} response was not a JSON object after one retry")


def _protected_pairs() -> set[tuple[str, str]]:
    """Read blind-answer keys internally; callers must not print these values."""
    pairs: set[tuple[str, str]] = set()
    for rel in ("human_check_key.csv", "human_validation/answers_key.csv"):
        path = FINAL_DIR / rel
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                question_id = str(row.get("question_id", "")).strip()
                condition = str(row.get("condition", "")).strip()
                if question_id and condition:
                    pairs.add((question_id, condition))
    return pairs


def select_pilot(
    answers: list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    prior_scores: dict[tuple[str, str], dict[str, Any]],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    protected = _protected_pairs()
    rng = random.Random(seed)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answers:
        key = (str(row.get("id", "")), str(row.get("condition", "")))
        if key not in protected and row.get("status") == "generated":
            by_condition[key[1]].append(row)
    for rows in by_condition.values():
        rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    counts = Counter()
    types_seen: set[str] = set()

    def take(row: dict[str, Any]) -> None:
        selected.append(row)
        counts[str(row["condition"])] += 1
        types_seen.add(str(questions[str(row["id"])].get("type", "")))
        by_condition[str(row["condition"])].remove(row)

    # Reserve one C4 answer that the saved RAGChecker result labelled incorrect.
    c4_wrong = [
        row for row in by_condition["C4"]
        if prior_scores.get((str(row["id"]), "C4"), {}).get("ragchecker", {}).get("correctness") is False
    ]
    if not c4_wrong:
        raise RuntimeError("No non-blind incorrect C4 answer is available for the pilot")
    anchor = c4_wrong[0]
    take(anchor)
    # Keep a matched C1/C2 counterpart for the required robustness pilot where available.
    for condition in ("C1", "C2"):
        counterpart = next((row for row in by_condition[condition] if row["id"] == anchor["id"]), None)
        if counterpart is not None:
            take(counterpart)

    # Include a known abstention if one is available and does not violate quotas.
    abstainers = [
        row for condition in ("C3", "C1", "C2", "C4")
        for row in by_condition[condition]
        if prior_scores.get((str(row["id"]), condition), {}).get("abstention", {}).get("label") == "abstains"
        and counts[condition] < 3
    ]
    if not abstainers:
        raise RuntimeError("No non-blind abstaining answer is available for the pilot")
    take(abstainers[0])

    question_types = {str(q.get("type", "")) for q in questions.values()}
    for question_type in sorted(question_types - types_seen):
        candidates = [
            row for condition in ("C1", "C2", "C3", "C4")
            for row in by_condition[condition]
            if counts[condition] < 3 and str(questions[str(row["id"])].get("type", "")) == question_type
        ]
        if not candidates:
            raise RuntimeError("Could not cover all question types with non-blind pilot answers")
        candidates.sort(key=lambda row: (counts[str(row["condition"])], rng.random()))
        take(candidates[0])

    for condition in ("C1", "C2", "C3", "C4"):
        while counts[condition] < 3:
            if not by_condition[condition]:
                raise RuntimeError(f"Insufficient non-blind answers for {condition}")
            take(by_condition[condition][0])
    if len(selected) != 12 or any(counts[c] != 3 for c in ("C1", "C2", "C3", "C4")):
        raise RuntimeError("Pilot selection did not meet the 12-answer stratified design")
    return sorted(selected, key=lambda row: (("C1", "C2", "C3", "C4").index(row["condition"]), str(row["id"])))


def _ragchecker_run(
    role: str,
    answer_row: dict[str, Any],
    question: dict[str, Any],
    check_type: str,
    *,
    settings: dict[str, Any] | None = None,
    mock_llm: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from ragchecker import RAGChecker
    from ragchecker.container import RAGResult, RAGResults, RetrievedDoc

    settings = settings or _load_role(role, CONFIG_PATH)
    trace: list[dict[str, Any]] = []
    result = RAGResult(
        query_id=f"{answer_row['id']}-{answer_row['condition']}-{check_type}",
        query=str(question["question"]),
        gt_answer=str(question.get("reference_answer", "")),
        response=str(answer_row.get("answer", "")),
        retrieved_context=[
            RetrievedDoc(doc_id=str(p["passage_id"]), text=str(p["text"]))
            for p in answer_row.get("passages", [])
        ],
        gt_answer_claims=[str(point["text"]) for point in question.get("keypoints", [])],
    )
    results = RAGResults(results=[result])

    def request_labels(prompts: list[str]) -> list[str]:
        def request_one(prompt: str) -> tuple[str, dict[str, Any]]:
            llm_fn = mock_llm or call_llm
            response = llm_fn(
                role,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                extra_body=settings.get("thinking_parameters", {}),
                max_tokens=10,
                config_path=CONFIG_PATH,
                cache_dir=METRICS_DIR / "cache" / "llm",
                log_dir=METRICS_DIR / "logs",
                prompt_sha256=f"ragchecker-0.1.9-{check_type}-nonjoint-v1",
                log_messages=True,
            )
            label = str(response.get("text", "")).strip()
            if label not in LABELS:
                raise ValueError("RAGChecker must return exactly Entailment, Neutral, or Contradiction")
            trace_row = {
                "label": label,
                "reason": f"RAGChecker labelled this comparison {label}.",
                "model_id": response.get("api_model", response.get("model")),
                "token_usage": response.get("usage"),
                "request_parameters": response.get("request_parameters"),
                "cache_hit": response.get("cache_hit"),
            }
            return label, trace_row

        # Keypoint/chunk comparisons are independent. A small bounded pool avoids
        # serial latency while preserving prompt order in the returned labels.
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(request_one, prompts))
        trace.extend(row for _, row in results)
        return [label for label, _ in results]

    checker = RAGChecker(
        extractor_name=f"openai/{settings['model']}",
        checker_name=f"openai/{settings['model']}",
        extractor_api_base=settings["base_url"],
        checker_api_base=settings["base_url"],
        openai_api_key=os.getenv(settings["api_key_env"], "custom-callback"),
        joint_check=False,
        custom_llm_api_func=request_labels,
    )
    checker.check_claims(results, check_type=check_type)
    if check_type == "retrieved2answer":
        checker.evaluate(results, metrics=["context_precision", "claim_recall"])
        metrics = {k: float(v) for k, v in result.metrics.items() if k in {"context_precision", "claim_recall"}}
        return {"values": result.retrieved2answer, "metrics": metrics, "trace": trace}
    values = getattr(result, check_type)
    return {"values": values, "trace": trace}


def keypoint_summary(points: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    if len(points) != len(labels):
        raise ValueError("RAGChecker returned a different number of labels than keypoints")
    normalized = [str(label).strip() for label in labels]
    decisive = [i for i, point in enumerate(points) if bool(point.get("decisive"))]
    correct = bool(decisive) and all(normalized[i] == "Entailment" for i in decisive) and "Contradiction" not in normalized
    complete = sum(label == "Entailment" for label in normalized) / len(normalized) if normalized else None
    return {"correct": correct, "completeness": complete, "labels": normalized}


def _split_answer(answer: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer)
    return [part.strip() for part in parts if part.strip()]


def _page_texts() -> dict[str, str]:
    import pandas as pd
    frame = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "chunks_clean.csv", dtype=str, keep_default_na=False)
    frame = frame.sort_values(["url", "chunk_number"], kind="stable")
    return {url: "\n".join(group["content"].tolist()) for url, group in frame.groupby("url", sort=False)}


def _gold_page_text(question: dict[str, Any], pages: dict[str, str]) -> str:
    urls = [str(question.get("url", ""))]
    partner = question.get("partner")
    if isinstance(partner, dict) and partner.get("url"):
        urls.append(str(partner["url"]))
    return "\n\n".join(pages.get(url, "") for url in urls)


def _passage_map(answer: dict[str, Any]) -> dict[str, str]:
    return {str(p.get("passage_id")): str(p.get("text", "")) for p in answer.get("passages", [])}


def _cited_ids(statement: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[(P\d+)\]", statement)))


def citation_validity(answer: str, allowed_ids: set[str]) -> dict[str, Any]:
    """Programmatic R1 check: require a citation and reject unknown passage IDs."""
    found = set(re.findall(r"\[(P\d+)\]", answer))
    return {"valid": bool(found) and found.issubset(allowed_ids), "found_ids": sorted(found),
            "invalid_ids": sorted(found - allowed_ids)}


def metric_applicability(metric: str, condition: str, abstention: str | None,
                         has_citations: bool = True) -> tuple[bool, str | None]:
    """Return the fixed per-answer applicability decision and an exclusion reason."""
    if metric in {"answer_correctness", "completeness"} and condition == "C3":
        return False, "C3 passages contain no answer."
    if metric in {"retrieved_evidence_relevance", "retrieved_evidence_coverage"} and condition != "C4":
        return False, "Applicable to C4 retrieval only."
    if metric == "evidence_faithfulness" and abstention == "abstains":
        return False, "Answer is labelled as an abstention."
    if metric == "factual_accuracy" and abstention == "abstains":
        return False, "Answer is labelled as an abstention."
    if metric == "citation_support" and (abstention == "abstains" or not has_citations):
        return False, "No cited answer statement is available."
    if metric == "citation_completeness" and abstention == "abstains":
        return False, "Answer is labelled as an abstention."
    return True, None


def terminal_pilot_summary(primary_calls: int, primary_tokens: int,
                           secondary_calls: int, secondary_tokens: int) -> list[str]:
    """Aggregate-only terminal output; deliberately accepts no item-level data."""
    return [
        "Metrics v2 pilot complete: 12 answers; 3 per condition; all four question types represented.",
        "Protected blind-sample items excluded during in-code selection.",
        f"Primary judge: {primary_calls} logged calls; {primary_tokens} logged tokens.",
        f"Secondary judge: {secondary_calls} logged calls; {secondary_tokens} logged tokens.",
        f"Scaled estimate for 176 answers: primary {round(primary_calls * 176 / 12):,} calls / {round(primary_tokens * 176 / 12):,} tokens; secondary {round(secondary_calls * 176 / 12):,} calls / {round(secondary_tokens * 176 / 12):,} tokens. No prices are configured.",
    ]


def _make_ragas_client(role: str, settings: dict[str, Any], name: str, token_limit: int = 1024):
    import httpx
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory

    usage_calls: list[dict[str, Any]] = []
    log_path = METRICS_DIR / "logs" / f"{role}.jsonl"

    async def start(request: httpx.Request) -> None:
        request.extensions["metrics_started"] = time.perf_counter()

    async def done(response: httpx.Response) -> None:
        try:
            await response.aread()
            raw = response.json()
        except Exception:
            raw = {}
        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "ragas_metric_call", "role": role,
            "api_model": raw.get("model") if isinstance(raw, dict) else None,
            "token_usage": usage,
            "elapsed_seconds": time.perf_counter() - response.request.extensions.get("metrics_started", time.perf_counter()),
            "http_status": response.status_code, "metric": name,
        }
        usage_calls.append(event)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    client = AsyncOpenAI(
        api_key=os.environ[settings["api_key_env"]], base_url=settings["base_url"], timeout=120,
        max_retries=0,
        http_client=httpx.AsyncClient(timeout=httpx.Timeout(120, connect=20), event_hooks={"request": [start], "response": [done]}),
    )
    llm = llm_factory(
        settings["model"], client=client, temperature=0.0, max_tokens=token_limit,
        extra_body=settings.get("thinking_parameters", {}),
        cache=DiskCacheBackend(str(METRICS_DIR / "cache" / f"ragas_{role}_{name}")),
    )
    return client, llm, usage_calls


async def _ragas_score(role: str, settings: dict[str, Any], metric: str, answer: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    from ragas.metrics.collections import Faithfulness
    contexts = [str(row.get("text", "")) for row in answer.get("passages", [])]
    last_error: Exception | None = None
    for attempt, token_limit in enumerate((1024, 2048)):
        client, llm, calls = _make_ragas_client(role, settings, metric, token_limit=token_limit)
        try:
            score = await Faithfulness(llm=llm).ascore(
                user_input=str(question["question"]), response=str(answer.get("answer", "")), retrieved_contexts=contexts,
            )
            value = float(score.value) if score.value is not None else None
            return {"value": value, "status": "scored" if value is not None else "NaN", "calls": calls,
                    "retry_used": attempt == 1,
                    "reason": "Ragas Faithfulness measures supported extracted claims divided by all extracted claims." if value is not None else "Ragas extracted no answer claims to assess."}
        except Exception as exc:
            last_error = exc
            if attempt == 1:
                raise
        finally:
            await client.close()
    raise RuntimeError("Faithfulness failed after the single larger-token retry") from last_error


def _empty_exclusion(reason: str) -> dict[str, Any]:
    return {"value": None, "status": "excluded", "reason": reason}


def _factual_claim_output(data: dict[str, Any]) -> dict[str, Any]:
    """Accept the judge's per-claim result wrapper as well as a direct label object."""
    if data.get("label") in {"supported", "contradicted", "cannot_be_determined"}:
        return data
    for key in ("claims", "results"):
        rows = data.get(key)
        if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
            return rows[0]
    return data


def score_one(
    answer: dict[str, Any],
    question: dict[str, Any],
    *,
    settings_by_role: dict[str, dict[str, Any]],
    pages: dict[str, str],
    prior_abstention: str | None = None,
    mock_llm: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    condition = str(answer["condition"])
    answer_text = str(answer.get("answer", ""))
    keypoints = question.get("keypoints", [])
    passage_texts = _passage_map(answer)
    record: dict[str, Any] = {
        "id": answer["id"], "type": question.get("type"), "condition": condition,
        "answer": answer_text, "metrics": {}, "judgements": [], "status": "scored",
    }

    # The primary abstention decision controls the applicability of metrics 1–10
    # for both judges. The secondary judge's own label is retained for metric 11.
    primary_abstention_result = _call_json(
        "judge_primary", "appropriate_abstention",
        {"question": question["question"], "answer": answer_text}, mock_llm=mock_llm,
    )
    primary_abstention_data = primary_abstention_result["json"]
    if primary_abstention_data.get("label") not in ABSTENTION_LABELS or not primary_abstention_data.get("reason"):
        raise ValueError("Primary abstention output is invalid")
    record["primary_abstention_label"] = primary_abstention_data["label"]
    record["judgements"].append({
        "metric": "appropriate_abstention", "judge": "primary", "output": primary_abstention_data,
        "token_usage": primary_abstention_result.get("usage"), "model_id": primary_abstention_result.get("api_model"),
    })

    for role in ("judge_primary", "judge_secondary"):
        prefix = "primary" if role == "judge_primary" else "secondary"
        role_settings = settings_by_role[role]
        labels_info = None
        retrieval_info = None
        decisive_retrieved: bool | None = None
        if condition in {"C1", "C2", "C4"}:
            response_result = _ragchecker_run(role, answer, question, "response2answer", settings=role_settings, mock_llm=mock_llm)
            labels = [str(x) for x in response_result["values"]]
            labels_info = keypoint_summary(keypoints, labels)
            record["metrics"].setdefault("answer_correctness", {})[prefix] = labels_info["correct"]
            record["metrics"].setdefault("completeness", {})[prefix] = labels_info["completeness"]
            record["judgements"].append({"metric": "answer_correctness_completeness", "judge": prefix,
                                          "labels": [{"label": lab, "reason": f"RAGChecker labelled keypoint {i+1} {lab}."} for i, lab in enumerate(labels)],
                                          "trace": response_result["trace"]})
            if condition == "C4":
                retr = _ragchecker_run(role, answer, question, "retrieved2answer", settings=role_settings, mock_llm=mock_llm)
                retrieval_info = retr.get("metrics", {})
                record["metrics"].setdefault("retrieved_evidence_relevance", {})[prefix] = retrieval_info.get("context_precision")
                record["metrics"].setdefault("retrieved_evidence_coverage", {})[prefix] = retrieval_info.get("claim_recall")
                matrix = retr.get("values") or []
                decisive_indexes = [i for i, point in enumerate(keypoints) if point.get("decisive")]
                decisive_retrieved = bool(decisive_indexes) and all(
                    i < len(matrix) and any(str(value).strip() == "Entailment" for value in (matrix[i] if isinstance(matrix[i], list) else [matrix[i]]))
                    for i in decisive_indexes
                )
                chunk_labels = []
                for chunk_index, passage in enumerate(answer.get("passages", [])):
                    labels_for_chunk = matrix[chunk_index] if chunk_index < len(matrix) else []
                    if not isinstance(labels_for_chunk, list):
                        labels_for_chunk = [labels_for_chunk]
                    chunk_labels.append({"passage_id": passage.get("passage_id"), "labels": labels_for_chunk})
                record["judgements"].append({"metric": "retrieval", "judge": prefix,
                                              "metrics": retrieval_info, "decisive_keypoints_all_retrieved": decisive_retrieved,
                                              "keypoint_chunk_labels": matrix, "per_chunk_labels": chunk_labels,
                                              "trace": retr.get("trace", [])})
        else:
            record["metrics"].setdefault("answer_correctness", {})[prefix] = _empty_exclusion("C3 evidence does not contain the answer")
            record["metrics"].setdefault("completeness", {})[prefix] = _empty_exclusion("C3 evidence does not contain the answer")

        relevance = _call_json(role, "response_relevance", {"question": question["question"], "answer": answer_text}, mock_llm=mock_llm)
        relevance_data = relevance["json"]
        if type(relevance_data.get("score")) is not int or relevance_data["score"] not in range(1, 6) or not relevance_data.get("reason"):
            raise ValueError("Response relevance output is invalid")
        record["metrics"].setdefault("response_relevance", {})[prefix] = {"score": relevance_data["score"], "reason": relevance_data["reason"]}
        record["judgements"].append({"metric": "response_relevance", "judge": prefix, "output": relevance_data,
                                      "token_usage": relevance.get("usage"), "model_id": relevance.get("api_model")})

        if role == "judge_primary":
            abstention_data = primary_abstention_data
            abstention = primary_abstention_result
        else:
            abstention = _call_json(role, "appropriate_abstention", {"question": question["question"], "answer": answer_text}, mock_llm=mock_llm)
            abstention_data = abstention["json"]
            if abstention_data.get("label") not in ABSTENTION_LABELS or not abstention_data.get("reason"):
                raise ValueError("Abstention output is invalid")
        record["metrics"].setdefault("appropriate_abstention", {})[prefix] = {"label": abstention_data["label"], "reason": abstention_data["reason"]}
        if condition == "C3":
            record["metrics"]["appropriate_abstention"][prefix]["measure"] = {
                "name": "abstention_rate", "value": abstention_data["label"] == "abstains"
            }
        elif condition in {"C1", "C2"}:
            record["metrics"]["appropriate_abstention"][prefix]["measure"] = {
                "name": "false_abstention_rate", "value": abstention_data["label"] == "abstains"
            }
        elif condition == "C4":
            if decisive_retrieved:
                record["metrics"]["appropriate_abstention"][prefix]["measure"] = {
                    "name": "false_abstention_rate", "value": abstention_data["label"] == "abstains"
                }
            else:
                record["metrics"]["appropriate_abstention"][prefix]["measure"] = {
                    "name": "false_abstention_rate", "status": "excluded",
                    "reason": "C4 does not retrieve every decisive keypoint."
                }
        if prefix == "secondary":
            record["judgements"].append({"metric": "appropriate_abstention", "judge": prefix, "output": abstention_data,
                                          "token_usage": abstention.get("usage"), "model_id": abstention.get("api_model")})
        label = str(primary_abstention_data["label"])
        non_abstaining = label != "abstains"

        if non_abstaining:
            faithfulness = asyncio.run(_ragas_score(role, role_settings, "faithfulness", answer, question))
            record["metrics"].setdefault("evidence_faithfulness", {})[prefix] = faithfulness
            record["judgements"].append({"metric": "evidence_faithfulness", "judge": prefix,
                                          "output": {"score": faithfulness.get("value"), "reason": faithfulness.get("reason", "")},
                                          "calls": faithfulness.get("calls", [])})
        else:
            faithfulness = _empty_exclusion("Answer is labelled as an abstention; no claims are scored for evidence faithfulness")
            record["metrics"].setdefault("evidence_faithfulness", {})[prefix] = faithfulness

        if non_abstaining and prefix == "primary":
            decomp = _call_json(role, "statement_decomposition", {"answer": answer_text}, mock_llm=mock_llm)
            statements = decomp["json"].get("statements")
            if not isinstance(statements, list):
                raise ValueError("Statement decomposition must contain a statements list")
            record["statement_decomposition"] = statements
            record["judgements"].append({"metric": "statement_decomposition", "judge": prefix, "output": decomp["json"],
                                          "token_usage": decomp.get("usage"), "model_id": decomp.get("api_model")})
        elif non_abstaining and prefix == "secondary" and "statement_decomposition" not in record:
            decomp = _call_json("judge_primary", "statement_decomposition", {"answer": answer_text}, mock_llm=mock_llm)
            statements = decomp["json"].get("statements")
            if not isinstance(statements, list):
                raise ValueError("Statement decomposition must contain a statements list")
            record["statement_decomposition"] = statements
            record["judgements"].append({"metric": "statement_decomposition", "judge": "primary", "output": decomp["json"],
                                          "token_usage": decomp.get("usage"), "model_id": decomp.get("api_model")})
        elif non_abstaining and "statement_decomposition" not in record:
            statements = []
        else:
            statements = record.get("statement_decomposition", [])

        if non_abstaining:
            full_gold = _gold_page_text(question, pages)
            claim_rows: list[dict[str, Any]] = []
            claim_inputs = [
                (statement_index, claim_index, str(claim))
                for statement_index, statement_obj in enumerate(statements)
                if isinstance(statement_obj, dict)
                for claim_index, claim in enumerate(statement_obj.get("claims", []))
            ]
            with ThreadPoolExecutor(max_workers=4) as pool:
                claim_results = list(pool.map(
                    lambda item: _call_json(role, "factual_accuracy", {"claim": item[2], "gold_page_text": full_gold}, mock_llm=mock_llm),
                    claim_inputs,
                ))
            for (statement_index, claim_index, _), result in zip(claim_inputs, claim_results):
                data = _factual_claim_output(result["json"])
                if data.get("label") not in {"supported", "contradicted", "cannot_be_determined"} or not data.get("reason"):
                    raise ValueError("Factual accuracy output is invalid")
                claim_rows.append({"statement_index": statement_index, "claim_index": claim_index, **data})
                record["judgements"].append({"metric": "factual_accuracy", "judge": prefix,
                                              "statement_index": statement_index, "claim_index": claim_index,
                                              "output": data, "token_usage": result.get("usage"), "model_id": result.get("api_model")})
            decidable = [row for row in claim_rows if row["label"] in {"supported", "contradicted"}]
            record["metrics"].setdefault("factual_accuracy", {})[prefix] = {
                "supported_share": sum(row["label"] == "supported" for row in claim_rows) / len(decidable) if decidable else None,
                "cannot_be_determined_share": sum(row["label"] == "cannot_be_determined" for row in claim_rows) / len(claim_rows) if claim_rows else None,
                "supported_count": sum(row["label"] == "supported" for row in claim_rows),
                "contradicted_count": sum(row["label"] == "contradicted" for row in claim_rows),
                "cannot_be_determined_count": sum(row["label"] == "cannot_be_determined" for row in claim_rows),
                "all_claim_count": len(claim_rows),
                "claims": claim_rows,
            }

            citation_rows: list[dict[str, Any]] = []
            citation_inputs: list[tuple[int, list[str], str]] = []
            for statement_index, statement_obj in enumerate(statements):
                if not isinstance(statement_obj, dict):
                    continue
                statement = str(statement_obj.get("statement", ""))
                cited = list(dict.fromkeys([str(x) for x in statement_obj.get("cited_passage_ids", [])] + _cited_ids(statement)))
                if not cited:
                    continue
                invalid = [pid for pid in cited if pid not in passage_texts]
                if invalid:
                    citation_rows.append({"statement_index": statement_index, "entailed": False,
                                          "reason": "The cited passage ID is not among the passages supplied to the answer.", "invalid_ids": invalid})
                    record["judgements"].append({"metric": "citation_support", "judge": prefix,
                                                  "statement_index": statement_index, "output": citation_rows[-1]})
                    continue
                citation_inputs.append((statement_index, cited, statement))
            with ThreadPoolExecutor(max_workers=4) as pool:
                citation_results = list(pool.map(
                    lambda item: _call_json(role, "citation_support", {
                        "statement": item[2],
                        "cited_passages": [{"passage_id": pid, "text": passage_texts[pid]} for pid in item[1]],
                    }, mock_llm=mock_llm),
                    citation_inputs,
                ))
            for (statement_index, _, _), result in zip(citation_inputs, citation_results):
                data = result["json"]
                if type(data.get("entailed")) is not bool or not data.get("reason"):
                    raise ValueError("Citation support output is invalid")
                citation_rows.append({"statement_index": statement_index, "entailed": data["entailed"], "reason": data["reason"]})
                record["judgements"].append({"metric": "citation_support", "judge": prefix,
                                              "statement_index": statement_index, "output": data,
                                              "token_usage": result.get("usage"), "model_id": result.get("api_model")})
            record["metrics"].setdefault("citation_support", {})[prefix] = {
                "score": sum(row["entailed"] for row in citation_rows) / len(citation_rows) if citation_rows else None,
                "cited_statements": citation_rows,
            }

            attribution_payload = {
                "statements": [{"statement_id": index, "statement": x.get("statement", ""), "cited_passage_ids": x.get("cited_passage_ids", [])}
                               for index, x in enumerate(statements) if isinstance(x, dict)],
                "passages": [{"passage_id": pid, "text": text} for pid, text in passage_texts.items()],
            }
            attribution = _call_json(role, "citation_completeness", attribution_payload, mock_llm=mock_llm)
            attribution_data = attribution["json"].get("citation_completeness", attribution["json"])
            attribution_rows = attribution_data.get("statements", attribution_data.get("results")) if isinstance(attribution_data, dict) else None
            if not isinstance(attribution_rows, list) or len(attribution_rows) != len(statements):
                attribution_payload["correction"] = (
                    f"The prior response returned {len(attribution_rows) if isinstance(attribution_rows, list) else 0} rows; "
                    f"return exactly {len(statements)} rows, one for each statement_id, with no aggregate."
                )
                attribution = _call_json(role, "citation_completeness", attribution_payload, mock_llm=mock_llm)
                attribution_data = attribution["json"].get("citation_completeness", attribution["json"])
                attribution_rows = attribution_data.get("statements", attribution_data.get("results")) if isinstance(attribution_data, dict) else None
            if not isinstance(attribution_rows, list) or len(attribution_rows) != len(statements):
                raise ValueError("Citation completeness must label every statement")
            record["judgements"].append({"metric": "citation_completeness", "judge": prefix,
                                          "output": attribution["json"], "token_usage": attribution.get("usage"),
                                          "model_id": attribution.get("api_model")})
            completeness_rows = []
            for index, (statement_obj, attr) in enumerate(zip(statements, attribution_rows)):
                if "statement_id" in attr and attr["statement_id"] != index:
                    raise ValueError("Citation completeness returned statement IDs out of order")
                needs = attr.get("needs_attribution")
                if type(needs) is not bool or not attr.get("reason"):
                    raise ValueError("Citation completeness statement result is invalid")
                cited = list(dict.fromkeys([str(x) for x in statement_obj.get("cited_passage_ids", [])] + _cited_ids(str(statement_obj.get("statement", "")))))
                valid = any(pid in passage_texts for pid in cited)
                completeness_rows.append({"statement_index": index, "needs_attribution": needs, "has_valid_citation": valid, "reason": attr["reason"]})
            required = [row for row in completeness_rows if row["needs_attribution"]]
            record["metrics"].setdefault("citation_completeness", {})[prefix] = {
                "score": sum(row["has_valid_citation"] for row in required) / len(required) if required else None,
                "statements": completeness_rows,
            }

            if condition == "C3" or label in {"abstains", "partly_answers"}:
                passages = [str(x.get("text", "")) for x in answer.get("passages", [])]
                r2 = _call_json(role, "instruction_fulfilment", {
                    "generator_instruction": "Answer only from the passages; say when the passages do not provide enough evidence.",
                    "passages": passages,
                    "answer": answer_text,
                }, mock_llm=mock_llm)
                r2data = r2["json"]
                if type(r2data.get("met")) is not bool or not r2data.get("reason"):
                    raise ValueError("Instruction fulfilment R2 output is invalid")
                r2_value = r2data
                record["judgements"].append({"metric": "instruction_fulfilment_R2", "judge": prefix,
                                              "output": r2data, "token_usage": r2.get("usage"), "model_id": r2.get("api_model")})
            else:
                r2_value = _empty_exclusion("R2 is not applicable to this answer and condition")
            given_ids = set(passage_texts)
            r1 = citation_validity(answer_text, given_ids)["valid"]
            r3 = faithfulness.get("value") == 1.0
            record["metrics"].setdefault("instruction_fulfilment", {})[prefix] = {
                "R1_citation_format_and_valid_ids": r1,
                "R2_says_when_evidence_is_insufficient": r2_value,
                "R3_faithfulness_equals_1": r3,
                "DRFR": None,
            }
            r2_met = r2_value.get("met") if isinstance(r2_value, dict) else None
            requirements = [r1, r3] + ([r2_met] if r2_met is not None else [])
            record["metrics"]["instruction_fulfilment"][prefix]["DRFR"] = sum(bool(x) for x in requirements) / len(requirements) if requirements else None
        else:
            for name in ("factual_accuracy", "citation_support", "citation_completeness"):
                record["metrics"].setdefault(name, {})[prefix] = _empty_exclusion("Answer is an abstention")
            if condition == "C3" or label in {"abstains", "partly_answers"}:
                r2 = _call_json(role, "instruction_fulfilment", {
                    "generator_instruction": "Answer only from the passages; say when the passages do not provide enough evidence.",
                    "passages": [str(x.get("text", "")) for x in answer.get("passages", [])],
                    "answer": answer_text,
                }, mock_llm=mock_llm)
                r2_value = r2["json"]
                if type(r2_value.get("met")) is not bool or not r2_value.get("reason"):
                    raise ValueError("Instruction fulfilment R2 output is invalid")
                record["judgements"].append({"metric": "instruction_fulfilment_R2", "judge": prefix,
                                              "output": r2_value, "token_usage": r2.get("usage"), "model_id": r2.get("api_model")})
                record["metrics"].setdefault("instruction_fulfilment", {})[prefix] = {
                    "R1_citation_format_and_valid_ids": None,
                    "R2_says_when_evidence_is_insufficient": r2_value,
                    "R3_faithfulness_equals_1": None,
                    "DRFR": 1.0 if r2_value["met"] else 0.0,
                }
            else:
                record["metrics"].setdefault("instruction_fulfilment", {})[prefix] = _empty_exclusion("R2 is not applicable; R1 and R3 are not scored for a refusal")

    record["metrics"].setdefault("retrieved_evidence_relevance", {}).setdefault("primary", _empty_exclusion("Only applicable to C4"))
    record["metrics"].setdefault("retrieved_evidence_relevance", {}).setdefault("secondary", _empty_exclusion("Only applicable to C4"))
    record["metrics"].setdefault("retrieved_evidence_coverage", {}).setdefault("primary", _empty_exclusion("Only applicable to C4"))
    record["metrics"].setdefault("retrieved_evidence_coverage", {}).setdefault("secondary", _empty_exclusion("Only applicable to C4"))
    record["metrics"].setdefault("controlled_robustness", {}).setdefault("primary", _empty_exclusion("No matched comparison in this pilot sample"))
    record["metrics"].setdefault("controlled_robustness", {}).setdefault("secondary", _empty_exclusion("No matched comparison in this pilot sample"))
    return record


def summarize_pilot(records: list[dict[str, Any]], calls_by_role: dict[str, int], tokens_by_role: dict[str, int], parse_failures: int) -> str:
    label_disagreements: Counter[str] = Counter()
    score_disagreements: Counter[str] = Counter()
    continuous_disagreements: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    for record in records:
        for metric, values in record.get("metrics", {}).items():
            primary = values.get("primary")
            secondary = values.get("secondary")
            if isinstance(primary, dict) and primary.get("status") == "excluded": exclusions[f"{metric}: {primary.get('reason')}"] += 1
            if isinstance(secondary, dict) and secondary.get("status") == "excluded": exclusions[f"{metric}: {secondary.get('reason')}"] += 1
            if primary is not None and secondary is not None:
                plabel = _comparable_value(primary)
                slabel = _comparable_value(secondary)
                if metric in {"answer_correctness", "appropriate_abstention"} and plabel is not None and slabel is not None and plabel != slabel:
                    label_disagreements[metric] += 1
                if metric == "response_relevance":
                    pscore = primary.get("score") if isinstance(primary, dict) else primary
                    sscore = secondary.get("score") if isinstance(secondary, dict) else secondary
                    if isinstance(pscore, (float, int)) and isinstance(sscore, (float, int)) and abs(pscore - sscore) >= 2:
                        score_disagreements[metric] += 1
                pcontinuous = _continuous_values(primary)
                scontinuous = _continuous_values(secondary)
                if pcontinuous is not None and scontinuous is not None and abs(pcontinuous - scontinuous) >= 0.3:
                    continuous_disagreements[metric] += 1
    lines = [
        "# Metrics v2 pilot summary",
        "",
        "Pilot scoring uses the existing S1 answers only. No answer was regenerated. The blind human-validation sample was read inside the selection code and excluded; no protected key values are reproduced here.",
        "",
        "## Workload and usage",
        "",
        f"- Pilot answer count: {len(records)} (three per condition).",
        f"- Primary judge logical calls: {calls_by_role.get('judge_primary', 0)}; logged tokens: {tokens_by_role.get('judge_primary', 0)}.",
        f"- Secondary judge logical calls: {calls_by_role.get('judge_secondary', 0)}; logged tokens: {tokens_by_role.get('judge_secondary', 0)}.",
        f"- Full-run estimate (176 answers, scaled from pilot): about {round(calls_by_role.get('judge_primary', 0) * 176 / 12):,} primary API calls and {round(tokens_by_role.get('judge_primary', 0) * 176 / 12):,} tokens; about {round(calls_by_role.get('judge_secondary', 0) * 176 / 12):,} secondary API calls and {round(tokens_by_role.get('judge_secondary', 0) * 176 / 12):,} tokens. Ragas claim-extraction/check calls vary with answer length. No model prices are configured, so cost is not estimated.",
        f"- JSON/label parse failures after the single retry: {parse_failures}.",
        "- Secondary-judge results reused: none; old score prompts or inputs differ from this protocol.",
        "",
        "## Applicability exclusions",
        "",
    ]
    if exclusions:
        lines.extend(f"- {name}: {count}" for name, count in sorted(exclusions.items()))
    else:
        lines.append("- None recorded in the pilot.")
    lines.extend(["", "## Judge disagreements", ""])
    exceeded_labels = {metric: count for metric, count in label_disagreements.items() if count > 3}
    if exceeded_labels:
        lines.extend(f"- Label disagreements above 3/12 for {metric}: {count} of 12." for metric, count in sorted(exceeded_labels.items()))
    else:
        lines.append("- No categorical metric exceeded 3 disagreements out of 12.")
    exceeded_scores = {metric: count for metric, count in score_disagreements.items() if count > 3}
    if exceeded_scores:
        lines.extend(f"- Score differences of at least 2 points above 3/12 for {metric}: {count} of 12." for metric, count in sorted(exceeded_scores.items()))
    else:
        lines.append("- No 1–5 score metric had differences of 2 or more points on more than 3 answers.")
    exceeded_continuous = {metric: count for metric, count in continuous_disagreements.items() if count > 3}
    if exceeded_continuous:
        lines.extend(f"- Continuous differences of at least 0.3 above 3/12 for {metric}: {count} of 12." for metric, count in sorted(exceeded_continuous.items()))
    else:
        lines.append("- No continuous metric differed by at least 0.3 on more than 3 answers.")
    return "\n".join(lines) + "\n"


def _comparable_value(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("status") == "excluded":
            return None
        for key in ("label", "score", "value", "correct", "supported_share", "DRFR"):
            if key in value and value[key] is not None:
                return value[key]
    return value


def _continuous_values(value: Any) -> float | None:
    if not isinstance(value, dict) or value.get("status") == "excluded":
        return None
    for key in ("value", "score", "supported_share", "cannot_be_determined_share", "completeness", "DRFR"):
        candidate = value.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    return None


def add_pilot_robustness(records: list[dict[str, Any]]) -> None:
    by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_question[str(record["id"])][str(record["condition"])] = record
    metric_names = [
        "answer_correctness", "evidence_faithfulness", "factual_accuracy", "response_relevance",
        "instruction_fulfilment", "completeness", "citation_support", "citation_completeness",
    ]
    for conditions in by_question.values():
        base = conditions.get("C1")
        if base is None:
            continue
        for target_name in ("C2", "C4"):
            target = conditions.get(target_name)
            if target is None:
                continue
            for metric in metric_names:
                for judge in ("primary", "secondary"):
                    base_value = _comparable_value(base.get("metrics", {}).get(metric, {}).get(judge))
                    target_value = _comparable_value(target.get("metrics", {}).get(metric, {}).get(judge))
                    if isinstance(base_value, (int, float, bool)) and isinstance(target_value, (int, float, bool)):
                        difference = float(target_value) - float(base_value)
                        for item in (base, target):
                            item.setdefault("metrics", {}).setdefault("controlled_robustness", {}).setdefault(judge, {})[
                                f"C1_to_{target_name}_{metric}"
                            ] = {"difference": difference, "reason": f"Paired {target_name} minus C1 for the same question."}


def _pilot_usage(pilot_started_at: str, roles: list[str]) -> tuple[Counter[str], Counter[str]]:
    calls: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for role in roles:
        path = METRICS_DIR / "logs" / f"{role}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            timestamp = event.get("timestamp")
            # Older Ragas response hooks did not timestamp their events; those events
            # are unique to the pilot (the Step 1 hook uses a different event name).
            in_pilot = timestamp is None or timestamp >= pilot_started_at
            if not in_pilot:
                continue
            is_api_call = event.get("event") == "ragas_metric_call" or (
                event.get("cache_hit") is False and event.get("status") == "success"
            )
            if is_api_call:
                calls[role] += 1
                tokens[role] += int((event.get("token_usage") or {}).get("total_tokens", 0) or 0)
    return calls, tokens


def run_pilot(*, mock_llm: Callable[..., dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    answers = read_jsonl(FINAL_DIR / "answers_final.jsonl")
    questions_list = read_jsonl(FINAL_DIR / "questions_final.jsonl")
    questions = {str(row["id"]): row for row in questions_list}
    prior_rows = read_jsonl(FINAL_DIR / "scores_final_revised.jsonl")
    prior = {(str(row["id"]), str(row["condition"])): row for row in prior_rows}
    selected = select_pilot(answers, questions, prior)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = {role: _load_role(role, CONFIG_PATH) for role in ("judge_primary", "judge_secondary")}
    pages = _page_texts()
    scored: list[dict[str, Any]] = []
    state_path = METRICS_DIR / "items" / "pilot_v2_run_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        pilot_started_at = str(json.loads(state_path.read_text(encoding="utf-8"))["started_at"])
    else:
        pilot_started_at = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps({"started_at": pilot_started_at}, indent=2) + "\n", encoding="utf-8")
    parse_failures = 0
    for answer in selected:
        question = questions[str(answer["id"])]
        row = score_one(answer, question, settings_by_role=settings, pages=pages, prior_abstention=prior.get((answer["id"], answer["condition"]), {}).get("abstention", {}).get("label"), mock_llm=mock_llm)
        scored.append(row)
    add_pilot_robustness(scored)
    call_counts, token_counts = _pilot_usage(pilot_started_at, list(settings))
    out_dir = METRICS_DIR / "pilot_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    item_dir = METRICS_DIR / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    (item_dir / "pilot_v2_judgements.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8")
    answers_by_key = {(str(row["id"]), str(row["condition"])): row for row in answers}
    review = ["# Metrics v2 pilot review", "", "The following 12 S1 answers are outside the protected human-validation samples. Item-level detail is confined to this review file and `outputs/metrics_v2/items/`.", ""]
    for record in scored:
        source = answers_by_key[(str(record["id"]), str(record["condition"]))]
        question = questions[str(record["id"])]
        review.extend([
            f"## {record['id']} — {record['condition']} — {record['type']}",
            "",
            f"**Question:** {question['question']}",
            f"**Passage IDs:** {', '.join(str(p.get('passage_id')) for p in source.get('passages', []))}",
            "",
            "**Answer:**",
            "",
            source.get("answer", ""),
            "",
            "**Statement decomposition:**",
            "",
            "```json",
            json.dumps(record.get("statement_decomposition", []), ensure_ascii=False, indent=2),
            "```",
            "",
            "| Metric | Primary value | Primary reason | Secondary value | Secondary reason |",
            "|---|---|---|---|---|",
        ])
        for metric, values in record.get("metrics", {}).items():
            primary = values.get("primary")
            secondary = values.get("secondary")
            pvalue, preason = _review_value_reason(primary, metric, record, "primary")
            svalue, sreason = _review_value_reason(secondary, metric, record, "secondary")
            review.append(f"| {metric} | {pvalue} | {preason} | {svalue} | {sreason} |")
        review.append("")
    (out_dir / "pilot_review.md").write_text("\n".join(review), encoding="utf-8")
    (out_dir / "pilot_summary.md").write_text(summarize_pilot(scored, call_counts, token_counts, parse_failures), encoding="utf-8")
    # Aggregate-only terminal output: do not print protected sample contents or individual pilot records.
    for line in terminal_pilot_summary(call_counts["judge_primary"], token_counts["judge_primary"],
                                       call_counts["judge_secondary"], token_counts["judge_secondary"]):
        print(line)
    print(f"Pilot review: {out_dir / 'pilot_review.md'}")
    print(f"Pilot summary: {out_dir / 'pilot_summary.md'}")
    return scored


def run_full() -> list[dict[str, Any]]:
    """Score every frozen S1 answer and append completed items for safe resume."""
    checks_path = METRICS_DIR / "pilot_v2" / "checks.md"
    if not checks_path.exists() or "| PASS |" not in checks_path.read_text(encoding="utf-8"):
        raise RuntimeError("Pilot v2 checks must pass before the full run")
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    answers = read_jsonl(FINAL_DIR / "answers_final.jsonl")
    questions = {str(row["id"]): row for row in read_jsonl(FINAL_DIR / "questions_final.jsonl")}
    prior_rows = read_jsonl(FINAL_DIR / "scores_final_revised.jsonl")
    prior = {(str(row["id"]), str(row["condition"])): row for row in prior_rows}
    settings = {role: _load_role(role, CONFIG_PATH) for role in ("judge_primary", "judge_secondary")}
    pages = _page_texts()
    item_dir = METRICS_DIR / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    state_path = item_dir / "full_run_state.json"
    if not state_path.exists():
        state_path.write_text(json.dumps({"started_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    output_path = item_dir / "judgements.jsonl"
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if output_path.exists():
        for row in read_jsonl(output_path):
            completed[(str(row["id"]), str(row["condition"]))] = row
    with output_path.open("a", encoding="utf-8") as stream:
        for answer in answers:
            key = (str(answer["id"]), str(answer["condition"]))
            if answer.get("status") != "generated" or key in completed:
                continue
            question = questions[key[0]]
            scored = score_one(
                answer, question, settings_by_role=settings, pages=pages,
                prior_abstention=prior.get(key, {}).get("abstention", {}).get("label"),
            )
            stream.write(json.dumps(scored, ensure_ascii=False) + "\n")
            stream.flush()
            completed[key] = scored
    expected = {(str(a["id"]), str(a["condition"])) for a in answers if a.get("status") == "generated"}
    if set(completed) != expected:
        raise RuntimeError(f"Full scoring incomplete: {len(completed)} records for {len(expected)} generated answers")
    records = [completed[(str(a["id"]), str(a["condition"]))] for a in answers if (str(a["id"]), str(a["condition"])) in expected]
    print(f"Full scoring complete: {len(records)} S1 answers; no S2 answer set was available.")
    print(f"Item-level judgements saved under {item_dir}.")
    return records


def _review_value_reason(value: Any, metric: str | None = None,
                         record: dict[str, Any] | None = None,
                         judge: str | None = None) -> tuple[str, str]:
    if value is None:
        return "—", "Not applicable or not computed."
    if isinstance(value, dict):
        if "status" in value and value.get("status") in {"excluded", "NaN"}:
            return str(value.get("value", "N/A")), str(value.get("reason", "Not applicable."))
        if "label" in value:
            measure = value.get("measure")
            suffix = f"; {measure.get('name')}={measure.get('value', measure.get('status'))}" if isinstance(measure, dict) else ""
            return f"{value['label']}{suffix}", str(value.get("reason", ""))
        if "supported_share" in value:
            claims = value.get("claims", [])
            return (
                f"supported/(supported+contradicted)={value['supported_share']}; cannot_be_determined/all={value.get('cannot_be_determined_share')}",
                json.dumps(claims, ensure_ascii=False),
            )
        if metric in {"answer_correctness", "completeness"} and record:
            item = next((j for j in record.get("judgements", [])
                         if j.get("metric") == "answer_correctness_completeness" and j.get("judge") == judge), None)
            if item:
                labels = item.get("labels", [])
                return str(value), json.dumps(labels, ensure_ascii=False)
        if metric in {"retrieved_evidence_relevance", "retrieved_evidence_coverage"} and record:
            item = next((j for j in record.get("judgements", [])
                         if j.get("metric") == "retrieval" and j.get("judge") == judge), None)
            if item:
                return str(value), json.dumps(item.get("per_chunk_labels", []), ensure_ascii=False)
        if metric in {"citation_support", "citation_completeness"}:
            return str(value.get("score", value)), json.dumps(value.get("statements", value.get("cited_statements", [])), ensure_ascii=False)
        if "score" in value:
            return str(value["score"]), str(value.get("reason") or f"Calculated from {len(value.get('cited_statements', value.get('statements', [])))} judged statements.")
        if "DRFR" in value:
            r2 = value.get("R2_says_when_evidence_is_insufficient")
            reason = r2.get("reason", "") if isinstance(r2, dict) else "R2 was not applicable."
            return str(value["DRFR"]), reason or "Calculated from applicable instruction requirements."
        if "value" in value:
            return str(value.get("value")), str(value.get("reason", ""))
        return json.dumps(value, ensure_ascii=False), "See structured judgement detail."
    return str(value), "Derived from the corresponding judged labels or program check."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Score the 12-answer pilot")
    parser.add_argument("--pilot-v2", action="store_true", help="Rerun the fixed 12-answer pilot without overwriting the first run")
    parser.add_argument("--full", action="store_true", help="Score every frozen answer after pilot checks pass")
    args = parser.parse_args()
    if args.full:
        run_full()
    elif args.pilot or args.pilot_v2:
        run_pilot()
    else:
        parser.error("Select --pilot, --pilot-v2, or --full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
