"""Audit S2 main and extra question/reference fit with the secondary judge."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INPUT_DIR = ROOT / "outputs/rawpage_questions/question_set_v2/final"
OUTPUT_DIR = ROOT / "outputs/rawpage_questions/question_set_v2/audit"
AUDIT_JSONL = OUTPUT_DIR / "audit.jsonl"
SUMMARY_MD = OUTPUT_DIR / "audit_summary.md"
PROMPT_VERSION = "question-fit-audit-v1"
PROMPT_SHA256 = hashlib.sha256(PROMPT_VERSION.encode("utf-8")).hexdigest()
MODEL_ROLE = "judge_secondary"
VALID_ANSWER_FIT = {"yes", "partly", "no"}
VALID_SINGLE_ANSWER = {"single", "several_valid"}
TIMEOUT_SECONDS = 120
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You are independently auditing a research question set. Judge only the supplied question, reference answer, decisive keypoints, supporting quotes, and false-premise statement. Do not repair the reference answer or use outside facts to infer the source page's intended wording.

Return exactly one JSON object with these fields:
{
  "answer_fit": "yes" | "partly" | "no",
  "single_answer": "single" | "several_valid",
  "premise_ok": "yes" | "no" | null,
  "reason": "one concise sentence"
}

Use answer_fit=yes only when the reference answer directly and adequately gives what the question asks. Use partly when it addresses the question but omits a material requested element or is only partly aligned. Use no when it does not answer the question or answers a materially different question.

Use single_answer=several_valid when, without seeing the source page, a knowledgeable reader could reasonably give a different correct answer to the same question based on other pages or common interpretations. Use single only when the question wording itself points to essentially one expected answer.

For a false_premise question, premise_ok=yes only if the quoted evidence clearly establishes that the false detail is false and the reference answer explicitly corrects it. Otherwise use no. For every other question type, set premise_ok to null. The reason must be one sentence and explain the judgments briefly."""


def read_questions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename, set_name in (("questions_main.jsonl", "main"), ("questions_extra.jsonl", "extra")):
        path = INPUT_DIR / filename
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["set"] = set_name
            rows.append(row)
    return rows


def load_existing() -> dict[str, dict[str, Any]]:
    if not AUDIT_JSONL.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for line in AUDIT_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["id"])] = row
    return result


def evidence_payload(question: dict[str, Any]) -> dict[str, Any]:
    points = question.get("keypoints") or []
    return {
        "set": question["set"],
        "type": question.get("type", ""),
        "page_type": question.get("page_type", ""),
        "question": question.get("question", ""),
        "reference_answer": question.get("reference_answer", ""),
        "decisive_keypoints": [point for point in points if isinstance(point, dict) and point.get("decisive")],
        "supporting_quotes": question.get("supporting_quotes") or [],
        "false_premise": question.get("false_premise", "") if question.get("type") == "false_premise" else None,
    }


def normalize_result(parsed: Any, question: dict[str, Any]) -> dict[str, Any]:
    parsed = parsed if isinstance(parsed, dict) else {}
    answer_fit = str(parsed.get("answer_fit", "")).strip().lower()
    single_answer = str(parsed.get("single_answer", "")).strip().lower()
    premise_ok = parsed.get("premise_ok")
    premise_ok = str(premise_ok).strip().lower() if premise_ok is not None else None
    if answer_fit not in VALID_ANSWER_FIT:
        answer_fit = "invalid"
    if single_answer not in VALID_SINGLE_ANSWER:
        single_answer = "invalid"
    if question.get("type") == "false_premise":
        if premise_ok not in {"yes", "no"}:
            premise_ok = "invalid"
    else:
        premise_ok = None
    reason = parsed.get("reason", "")
    reason = reason.strip() if isinstance(reason, str) else ""
    return {"answer_fit": answer_fit, "single_answer": single_answer, "premise_ok": premise_ok, "reason": reason}


def is_flagged(labels: dict[str, Any]) -> bool:
    return (
        labels.get("answer_fit") != "yes"
        or labels.get("single_answer") == "several_valid"
        or labels.get("premise_ok") == "no"
        or "invalid" in labels.values()
    )


def save_rows(rows: dict[str, dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda row: (row.get("set", ""), row.get("id", "")))
    temp = AUDIT_JSONL.with_suffix(".jsonl.tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
    temp.replace(AUDIT_JSONL)


def configured_judge() -> tuple[str, str, str, dict[str, Any]]:
    load_dotenv(ROOT / ".env", override=False)
    config = yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf-8")) or {}
    settings = config[MODEL_ROLE]
    api_key = os.getenv(str(settings["api_key_env"]))
    if not api_key:
        raise RuntimeError("The secondary judge credential is unavailable in the environment.")
    return str(settings["model"]), str(settings["base_url"]).rstrip("/"), api_key, dict(settings.get("thinking_parameters") or {})


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def call_secondary_judge(question: dict[str, Any]) -> dict[str, Any]:
    model, base_url, api_key, extra_body = configured_judge()
    user_payload = evidence_payload(question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Audit this item and return the specified JSON only:\n" + json.dumps(user_payload, ensure_ascii=False)},
    ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        **extra_body,
    }
    started = time.monotonic()
    call_attempts = 0
    parse_retry = False
    last_error = ""
    payload: dict[str, Any] = {}
    parsed: Any = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        call_attempts = attempt
        request_started = time.monotonic()
        try:
            response = requests.post(
                base_url + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            last_error = type(exc).__name__
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
                continue
            raise RuntimeError(last_error) from None
        elapsed_request = time.monotonic() - request_started
        if response.status_code == 429 or response.status_code >= 500:
            last_error = f"HTTP {response.status_code}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
                continue
            raise RuntimeError(last_error)
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"HTTP {response.status_code}")
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError:
            raise RuntimeError("Non-JSON API response") from None
        choices = payload.get("choices") if isinstance(payload, dict) else None
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
            break
        except (TypeError, json.JSONDecodeError):
            if not parse_retry and attempt < MAX_ATTEMPTS:
                parse_retry = True
                messages.extend([
                    {"role": "assistant", "content": str(content)},
                    {"role": "user", "content": "The previous response was invalid JSON. Return only one valid JSON object with the required labels."},
                ])
                body["messages"] = messages
                last_error = "Invalid JSON response"
                continue
            raise RuntimeError("Invalid JSON response") from None

    labels = normalize_result(parsed, question)
    usage_raw = payload.get("usage") or {}
    usage = {
        field: value for field, raw in (
            ("prompt_tokens", usage_raw.get("prompt_tokens")),
            ("completion_tokens", usage_raw.get("completion_tokens")),
            ("total_tokens", usage_raw.get("total_tokens")),
        ) if (value := optional_int(raw)) is not None
    }
    completion_details = usage_raw.get("completion_tokens_details") or {}
    reasoning_tokens = optional_int(completion_details.get("reasoning_tokens")) if isinstance(completion_details, dict) else None
    if reasoning_tokens is not None:
        usage["reasoning_tokens"] = reasoning_tokens
    if not usage and usage_raw:
        usage["reported_usage"] = usage_raw
    duration = time.monotonic() - started
    api_model = payload.get("model", model) if isinstance(payload, dict) else model
    log_path = OUTPUT_DIR / "logs" / "judge_secondary.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": MODEL_ROLE,
            "model": model,
            "api_model": api_model,
            "request_hash": hashlib.sha256(json.dumps([SYSTEM_PROMPT, user_payload], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
            "temperature": 0,
            "json_mode": True,
            "thinking_parameters": extra_body,
            "attempts": call_attempts,
            "api_request_sent": True,
            "token_usage": usage,
            "elapsed_seconds": round(duration, 4),
            "status": "success",
            "prompt_sha256": PROMPT_SHA256,
        }, ensure_ascii=False) + "\n")
    return {
        "id": question["id"],
        "set": question["set"],
        "type": question.get("type", ""),
        "page_type": question.get("page_type", ""),
        "url": question.get("url", ""),
        "question": question.get("question", ""),
        "reference_answer": question.get("reference_answer", ""),
        "keypoints": question.get("keypoints") or [],
        "quotes": question.get("supporting_quotes") or [],
        **labels,
        "reason": labels["reason"],
        "flagged": is_flagged(labels),
        "judge_text": json.dumps(parsed, ensure_ascii=False),
        "judge_model": api_model,
        "usage": usage,
        "attempts": call_attempts,
        "cache_hit": False,
        "elapsed_seconds": round(duration, 4),
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "error": "",
    }


def audit_all() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = read_questions()
    if len({row["id"] for row in questions}) != len(questions):
        raise ValueError("Question IDs are not unique across main and extra input files.")
    existing = load_existing()
    total = len(questions)
    for index, question in enumerate(questions, start=1):
        prior = existing.get(question["id"])
        if prior and not prior.get("error") and prior.get("answer_fit"):
            continue
        try:
            existing[question["id"]] = call_secondary_judge(question)
        except Exception as exc:  # Keep the batch resumable and record a safe error class.
            empty = {
                "id": question["id"], "set": question["set"], "type": question.get("type", ""),
                "page_type": question.get("page_type", ""), "url": question.get("url", ""),
                "question": question.get("question", ""), "reference_answer": question.get("reference_answer", ""),
                "keypoints": question.get("keypoints") or [], "quotes": question.get("supporting_quotes") or [],
                "answer_fit": "invalid", "single_answer": "invalid",
                "premise_ok": "invalid" if question.get("type") == "false_premise" else None,
                "reason": "Judge call failed; item requires review.", "flagged": True, "judge_text": "",
                "judge_model": "qwen3.7-plus", "usage": {}, "cache_hit": False, "attempts": 0,
                "elapsed_seconds": None, "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "error": type(exc).__name__,
            }
            existing[question["id"]] = empty
        save_rows(existing)
        if index % 10 == 0 or index == total:
            print(f"Audit progress: {index}/{total} items processed.", flush=True)
    ordered = sorted(existing.values(), key=lambda row: (row.get("set", ""), row.get("id", "")))
    return ordered


def display_keypoints(points: list[Any]) -> str:
    output = []
    for point in points:
        if isinstance(point, dict):
            marker = "[D] " if point.get("decisive") else ""
            output.append(marker + str(point.get("text", "")))
        else:
            output.append(str(point))
    return "\n".join(output)


def display_quotes(quotes: list[Any]) -> str:
    output = []
    for quote in quotes:
        if isinstance(quote, dict):
            source = str(quote.get("source", "passage"))
            text = str(quote.get("text", ""))
            output.append(f"[{source}] {text}")
        else:
            output.append(str(quote))
    return "\n".join(output)


def summary_table(rows: list[dict[str, Any]], field: str, categories: tuple[str, ...]) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "(blank)")].append(row)
    lines = [f"| {field} | n | " + " | ".join(categories) + " |", "|---|---:|" + "---:|" * len(categories)]
    for group in sorted(groups):
        group_rows = groups[group]
        counts = [sum(row.get(category_field) == category for row in group_rows) for category_field, category in (
            ("answer_fit", "yes"), ("answer_fit", "partly"), ("answer_fit", "no"),
            ("single_answer", "single"), ("single_answer", "several_valid"),
            ("premise_ok", "yes"), ("premise_ok", "no"),
        )]
        lines.append(f"| {group} | {len(group_rows)} | " + " | ".join(map(str, counts)) + " |")
    return lines


def write_summary(rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if not row.get("error")]
    flagged = [row for row in rows if row.get("flagged")]
    tokens = sum(int((row.get("usage") or {}).get("total_tokens", 0) or 0) for row in successful)
    logical_calls = len(successful)
    log_path = OUTPUT_DIR / "logs" / f"{MODEL_ROLE}.jsonl"
    physical_attempts = 0
    api_requests = 0
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = json.loads(line)
                api_requests += 1
                physical_attempts += int(event.get("attempts", 0) or 0)
    lines = [
        "# S2 Question Audit Summary",
        "",
        f"Questions audited: {len(rows)}; judge-call errors: {len(rows)-len(successful)}; flagged: {len(flagged)}.",
        f"Logical judge calls completed: {logical_calls}; successful request records: {api_requests}; total attempts: {physical_attempts}; reported total tokens: {tokens}.",
        "",
        "Labels are reported as judge outputs. The flag rule is answer_fit other than yes, single_answer=several_valid, or premise_ok=no.",
        "",
        "## Overall labels",
        "",
        "| Label | Value | Count |",
        "|---|---|---:|",
    ]
    for label_field, values in (("answer_fit", ("yes", "partly", "no", "invalid")),
                                ("single_answer", ("single", "several_valid", "invalid")),
                                ("premise_ok", ("yes", "no", "invalid"))):
        for value in values:
            count = sum(row.get(label_field) == value for row in rows)
            if count or value != "invalid":
                lines.append(f"| {label_field} | {value} | {count} |")
    lines.extend(["", "## Labels by set", "", "| Set | n | answer_fit yes | partly | no | single_answer single | several_valid | premise yes | premise no |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for value in ("main", "extra"):
        group = [row for row in rows if row.get("set") == value]
        lines.append("| " + value + " | " + str(len(group)) + " | " + " | ".join(str(sum(row.get(f) == label for row in group)) for f, label in (("answer_fit","yes"),("answer_fit","partly"),("answer_fit","no"),("single_answer","single"),("single_answer","several_valid"),("premise_ok","yes"),("premise_ok","no"))) + " |")
    for field, title in (("page_type", "page type"), ("type", "question type")):
        lines.extend(["", f"## Labels by {title}", "", "| Category | n | answer_fit yes | partly | no | single_answer single | several_valid | premise yes | premise no |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(field) or "(blank)")].append(row)
        for name, group in sorted(groups.items()):
            counts = [sum(row.get(f) == label for row in group) for f, label in (("answer_fit","yes"),("answer_fit","partly"),("answer_fit","no"),("single_answer","single"),("single_answer","several_valid"),("premise_ok","yes"),("premise_ok","no"))]
            lines.append(f"| {name} | {len(group)} | " + " | ".join(map(str, counts)) + " |")
    target = "AS-113-d4277b7f44_1"
    item = next((row for row in rows if row.get("id") == target), None)
    lines.extend(["", "## Requested pilot item", "", f"{target} flagged: **{'yes' if item and item.get('flagged') else 'no'}**."])
    lines.extend([
        "",
        "One separate diagnostic judge request was used to verify the provider connection before the full audit. Its response was not retained, so it is excluded from the audit call and token totals above.",
    ])
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = audit_all()
    write_summary(rows)
    print(f"Audited {len(rows)} questions; flagged {sum(bool(row.get('flagged')) for row in rows)}.", flush=True)


if __name__ == "__main__":
    main()
