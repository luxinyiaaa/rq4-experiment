"""Answer a seed-7 sample of five S2 questions with the four Expert API models."""

from __future__ import annotations

import csv
import json
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "outputs/rawpage_questions/question_set_v2/final/questions_main.jsonl"
OUTPUT_DIR = ROOT / "outputs/api_answers/pilot"
ANSWERS_PATH = OUTPUT_DIR / "answers.jsonl"
CONFIG_PATH = OUTPUT_DIR / "pilot_config.json"
SUMMARY_PATH = OUTPUT_DIR / "pilot_summary.md"
WORKBOOK_PATH = OUTPUT_DIR / "pilot_answers.xlsx"
KB_PATHS = (ROOT / "data/raw/site_pages_rows.csv", ROOT / "site_pages_rows.csv")
MODEL_NAMES = ("openai", "openai_gpt56_sol", "openai_gpt6_astra", "egi")
QUESTION_TYPES = ("list_all", "paraphrased", "false_premise")
TIMEOUT_SECONDS = 300
INTER_REQUEST_SECONDS = 5
RETRY_WAIT_SECONDS = 30
MAX_RETRIES = 1
SUFFIX = "Please cite the URL of the source for each statement in your answer."
MARKDOWN_URL_RE = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)", re.I)
BARE_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.I)
TRAILING_URL_CHARS = ".,;:!?)]}】"


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name.strip()] = value
    return values


def load_main_questions() -> list[dict[str, Any]]:
    return [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def draw_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use seed 7, enforcing all three types and at least one dataset record."""
    rng = random.Random(7)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_first(predicate) -> None:
        for row in shuffled:
            if row["id"] not in selected_ids and predicate(row):
                selected.append(row)
                selected_ids.add(row["id"])
                return
        raise ValueError("The main question set does not satisfy the requested sampling constraints.")

    add_first(lambda row: row.get("page_type") == "dataset_record")
    for question_type in QUESTION_TYPES:
        add_first(lambda row, question_type=question_type: row.get("type") == question_type)
    for row in shuffled:
        if len(selected) == 5:
            break
        if row["id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["id"])
    return selected


def normalize_url(value: str) -> str:
    parts = urlsplit(value.strip())
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = parts.port
    netloc = host if port is None else f"{host}:{port}"
    path = parts.path.rstrip("/")
    return urlunsplit(("", netloc, path, "", "")).lower()


def load_kb_urls() -> set[str]:
    path = next((candidate for candidate in KB_PATHS if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("The knowledge-base CSV was not found at the expected data/raw or workspace-root path.")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "url" not in (reader.fieldnames or []):
            raise ValueError("The knowledge-base CSV has no url column.")
        return {normalize_url(row["url"]) for row in reader if row.get("url", "").strip()}


def extract_urls(answer: str) -> list[str]:
    # The broad URL scan also catches URLs inside 【...】 and markdown links.
    # Citation-only markers such as 【0†source】 contain no URL and are ignored.
    matches = [(match.start(), match.group(1)) for match in MARKDOWN_URL_RE.finditer(answer)]
    matches.extend((match.start(), match.group(0)) for match in BARE_URL_RE.finditer(answer))
    urls = [url for _, url in sorted(matches, key=lambda item: item[0])]
    result: list[str] = []
    seen: set[str] = set()
    for url in urls:
        clean = url.rstrip(TRAILING_URL_CHARS + "\"'`")
        if clean and clean not in seen:
            result.append(clean)
            seen.add(clean)
    return result


def call_once(endpoint: str, query: str, model_name: str) -> tuple[Any, int | None, float, str | None]:
    started = time.monotonic()
    try:
        response = requests.post(endpoint, json={"query": query, "model_name": model_name}, timeout=TIMEOUT_SECONDS)
        elapsed = time.monotonic() - started
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError:
            payload = None
            return payload, response.status_code, elapsed, "non-JSON response"
        if not response.ok:
            return payload, response.status_code, elapsed, f"HTTP {response.status_code}"
        if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
            return payload, response.status_code, elapsed, "response JSON missing string answer field"
        return payload, response.status_code, elapsed, None
    except requests.RequestException as exc:
        elapsed = time.monotonic() - started
        # Never persist exception text because it may contain the endpoint URL.
        error_kind = "timeout" if isinstance(exc, requests.Timeout) else type(exc).__name__
        return None, None, elapsed, error_kind


def call_with_retry(endpoint: str, query: str, model_name: str) -> dict[str, Any]:
    total_latency = 0.0
    retries = 0
    status: int | None = None
    payload: Any = None
    error: str | None = None
    timestamp = datetime.now(timezone.utc).isoformat()
    for attempt in range(MAX_RETRIES + 1):
        if attempt:
            retries += 1
            time.sleep(RETRY_WAIT_SECONDS)
        payload, status, elapsed, error = call_once(endpoint, query, model_name)
        total_latency += elapsed
        should_retry = error == "timeout" or (status is not None and status >= 500)
        if not should_retry or attempt == MAX_RETRIES:
            break
    answer = payload.get("answer", "") if isinstance(payload, dict) else ""
    if not isinstance(answer, str):
        answer = ""
    return {
        "http_status": status,
        "latency_seconds": round(total_latency, 3),
        "retries": retries,
        "answer": answer,
        "model_name_returned": payload.get("model_name", payload.get("model")) if isinstance(payload, dict) else None,
        "provider_returned": payload.get("provider") if isinstance(payload, dict) else None,
        "error": error,
    }


def write_jsonl(rows: list[dict[str, Any]]) -> None:
    ANSWERS_PATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def summarize(
    rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    kb_urls: set[str],
    summary_path: Path = SUMMARY_PATH,
) -> None:
    by_id = {row["id"]: row for row in questions}
    lines = [
        "# API Pilot Summary",
        "",
        "No scoring was performed.",
        "",
        "| Model | Succeeded | Failed | Mean latency (s) | Maximum latency (s) | Mean answer words | Answers with URL | Mean distinct URLs | Cited URLs in KB | Answers citing source URL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        group = [row for row in rows if row["model_name"] == model]
        successful = [row for row in group if row["http_status"] is not None and 200 <= row["http_status"] < 300 and not row["error"]]
        latencies = [row["latency_seconds"] for row in successful]
        word_counts = [len(row["answer"].split()) for row in successful]
        all_urls = [(url, row) for row in successful for url in extract_urls(row["answer"])]
        answers_with_urls = sum(bool(extract_urls(row["answer"])) for row in successful)
        source_cites = sum(
            any(normalize_url(url) == normalize_url(str(by_id[row["id"]].get("url", ""))) for url in extract_urls(row["answer"]))
            for row in successful
        )
        kb_hits = sum(normalize_url(url) in kb_urls for url, _ in all_urls)
        mean_latency = f"{statistics.mean(latencies):.2f}" if latencies else "NA"
        maximum_latency = f"{max(latencies):.2f}" if latencies else "NA"
        mean_words = f"{statistics.mean(word_counts):.1f}" if word_counts else "NA"
        mean_distinct_urls = f"{statistics.mean([len(extract_urls(row['answer'])) for row in successful]):.2f}" if successful else "NA"
        lines.append(
            f"| {model} | {len(successful)} | {len(group)-len(successful)} | "
            f"{mean_latency} | {maximum_latency} | {mean_words} | "
            f"{answers_with_urls}/{len(successful)} | "
            f"{mean_distinct_urls} | "
            f"{kb_hits}/{len(all_urls)} | {source_cites}/{len(successful)} |"
        )
    mean_request_latency = statistics.mean([row["latency_seconds"] for row in rows if row["http_status"] is not None]) if any(row["http_status"] is not None for row in rows) else 0.0
    estimated_seconds = 1320 * mean_request_latency + 1319 * INTER_REQUEST_SECONDS
    lines.extend([
        "",
        f"Estimated serial time for 330 questions × 4 models: {estimated_seconds / 3600:.2f} hours ({estimated_seconds / 60:.1f} minutes), using this pilot's mean request latency and 5 seconds between requests; retries are excluded.",
        "",
        "URL and source-page matches are normalized for scheme, `www.`, trailing slash, and case. URL shares use successful answers; knowledge-base URL share uses all distinct cited URL occurrences across successful answers.",
    ])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    env_values = load_env_file(ROOT / ".env")
    endpoint = env_values.get("ENVRI_EXPERT_URL", "").strip()
    if not endpoint:
        raise SystemExit("ENVRI_EXPERT_URL is missing from .env; no API calls were made.")
    questions = draw_questions(load_main_questions())
    if len(questions) != 5 or set(row.get("type") for row in questions) != set(QUESTION_TYPES):
        raise ValueError("The deterministic sample failed its size or question-type constraints.")
    if not any(row.get("page_type") == "dataset_record" for row in questions):
        raise ValueError("The deterministic sample contains no dataset_record question.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({
        "query_suffix": SUFFIX,
        "query_format": "{question}\n\n{query_suffix}",
        "model_names": list(MODEL_NAMES),
        "timeout_seconds": TIMEOUT_SECONDS,
        "inter_request_seconds": INTER_REQUEST_SECONDS,
        "retry_on": ["timeout", "HTTP 5xx"],
        "max_retries": MAX_RETRIES,
        "retry_wait_seconds": RETRY_WAIT_SECONDS,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    kb_urls = load_kb_urls()
    records: list[dict[str, Any]] = []
    total = len(questions) * len(MODEL_NAMES)
    call_index = 0
    for question in questions:
        for model_name in MODEL_NAMES:
            if call_index:
                time.sleep(INTER_REQUEST_SECONDS)
            call_index += 1
            query = f"{question['question']}\n\n{SUFFIX}"
            started_at = datetime.now(timezone.utc).isoformat()
            result = call_with_retry(endpoint, query, model_name)
            records.append({
                "id": question["id"],
                "type": question.get("type", ""),
                "page_type": question.get("page_type", ""),
                "question": question["question"],
                "reference_answer": question.get("reference_answer", ""),
                "source_url": question.get("url", ""),
                "model_name": model_name,
                "query_sent": query,
                "request_time_utc": started_at,
                **result,
            })
            write_jsonl(records)
            success_so_far = sum(row["http_status"] is not None and 200 <= row["http_status"] < 300 and not row["error"] for row in records)
            print(f"Call {call_index}/{total} recorded; successful so far: {success_so_far}.")

    summarize(records, questions, kb_urls)


if __name__ == "__main__":
    main()
