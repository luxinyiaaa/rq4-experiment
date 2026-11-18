"""Run the final S1 questions against the Environmental Expert API."""

from __future__ import annotations

import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "outputs" / "final" / "questions_final.jsonl"
PARAPHRASE_PATH = ROOT / "outputs" / "final" / "paraphrase_control.xlsx"
PAGES_PATH = ROOT / "data" / "processed" / "pages.csv"
OUTPUT_DIR = ROOT / "outputs" / "expert_api" / "answers"
RAW_DIR = OUTPUT_DIR / "raw"
ANSWERS_PATH = OUTPUT_DIR / "answers_S1.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "run_summary.md"
MODEL_NAMES = ("openai", "openai_gpt56_sol", "openai_gpt6_astra", "egi")
TIMEOUT_SECONDS = 300
MAX_RETRIES = 2
RETRY_WAIT_SECONDS = 30

MARKDOWN_URL_RE = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)", re.I)
BARE_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.I)


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
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


def load_question_records() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_wording_questions() -> list[dict[str, Any]]:
    workbook = load_workbook(PARAPHRASE_PATH, read_only=True, data_only=True)
    try:
        sheet = workbook["Paraphrase control"]
        rows = sheet.iter_rows(values_only=True)
        headers = next(rows)
        records = [dict(zip(headers, row)) for row in rows if row and row[0]]
    finally:
        workbook.close()

    source_questions: list[dict[str, Any]] = []
    for row in records:
        if str(row.get("human_decision", "")).strip().lower() == "reject":
            continue
        question = str(row.get("source_worded_question") or "").strip()
        if row.get("id") == "W009":
            note = str(row.get("human_note") or "")
            marker = "Replace source_worded_question with:"
            if marker in note:
                question = note.split(marker, 1)[1].strip()
        if not question:
            raise ValueError(f"Missing source-wording question for {row.get('id')}")
        source_questions.append({
            "id": str(row["id"]),
            "type": str(row.get("type") or ""),
            "question": question,
            "variant": "source_wording",
        })
    if len(source_questions) != 14:
        raise ValueError(f"Expected 14 source-wording questions; found {len(source_questions)}")
    return source_questions


def load_page_url_keys(path: Path) -> set[str]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "url" not in (reader.fieldnames or []):
            raise ValueError("data/processed/pages.csv has no url column")
        return {normalize_url(row.get("url", "")) for row in reader if row.get("url", "").strip()}


def normalize_url(url: str) -> str:
    value = url.strip()
    parts = urlsplit(value)
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = parts.port
    netloc = host if port is None else f"{host}:{port}"
    path = parts.path.rstrip("/")
    return urlunsplit(("", netloc, path, parts.query, parts.fragment)).lower()


def extract_urls(answer: str) -> list[str]:
    found = MARKDOWN_URL_RE.findall(answer) + BARE_URL_RE.findall(answer)
    unique: list[str] = []
    seen: set[str] = set()
    for raw_url in found:
        url = raw_url.rstrip(".,;:!?\"'`)")
        if url and url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def read_existing_answers() -> dict[tuple[str, str, str], dict[str, Any]]:
    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not ANSWERS_PATH.exists():
        return existing
    for line in ANSWERS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["id"], row["variant"], row["model_name_requested"])
        existing[key] = row
    return existing


def raw_path(item_id: str, variant: str, model_name: str) -> Path:
    return RAW_DIR / f"{item_id}_{variant}_{model_name}.json"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_answer_row(row: dict[str, Any], existing: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    key = (row["id"], row["variant"], row["model_name_requested"])
    existing[key] = row
    temp_path = ANSWERS_PATH.with_suffix(".jsonl.tmp")
    ordered = sorted(existing.values(), key=lambda item: (item["id"], item["variant"], item["model_name_requested"]))
    temp_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered), encoding="utf-8")
    temp_path.replace(ANSWERS_PATH)


def response_record(
    item: dict[str, Any],
    variant: str,
    model_name: str,
    payload: Any,
    http_status: int | None,
    latency: float,
    timestamp: str,
    error: str | None,
    page_urls: set[str],
) -> dict[str, Any]:
    answer = payload.get("answer", "") if isinstance(payload, dict) else ""
    if not isinstance(answer, str):
        answer = ""
    urls = extract_urls(answer)
    valid_response = isinstance(payload, dict) and isinstance(payload.get("answer"), str)
    success = http_status is not None and 200 <= http_status < 300 and valid_response and error is None
    return {
        "id": item["id"],
        "type": item.get("type", ""),
        "variant": variant,
        "model_name_requested": model_name,
        "model_name_returned": payload.get("model_name", payload.get("model")) if isinstance(payload, dict) else None,
        "provider_returned": payload.get("provider") if isinstance(payload, dict) else None,
        "answer": answer,
        "status": http_status,
        "success": success,
        "error": error,
        "latency": round(latency, 3),
        "timestamp": timestamp,
        "cited_urls": urls,
        "cited_url_kb_membership": [
            {"url": url, "in_knowledge_base": normalize_url(url) in page_urls}
            for url in urls
        ],
    }


def call_once(endpoint: str, query: str, model_name: str) -> tuple[Any, int | None, float, str | None]:
    started = time.monotonic()
    try:
        response = requests.post(
            endpoint,
            json={"query": query, "model_name": model_name},
            timeout=TIMEOUT_SECONDS,
        )
        latency = time.monotonic() - started
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError:
            payload = {"raw_response_text": response.text}
            return payload, response.status_code, latency, "non-JSON response"
        if not response.ok:
            return payload, response.status_code, latency, f"HTTP {response.status_code}"
        if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
            return payload, response.status_code, latency, "response JSON missing string answer field"
        return payload, response.status_code, latency, None
    except requests.RequestException as exc:
        latency = time.monotonic() - started
        # Do not persist or display exception text; it can include the endpoint URL.
        return {"error": type(exc).__name__}, None, latency, type(exc).__name__


def run_calls(endpoint: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    originals = load_question_records()
    if len(originals) != 44:
        raise ValueError(f"Expected 44 final S1 questions; found {len(originals)}")
    paraphrases = source_wording_questions()
    page_urls = load_page_url_keys(PAGES_PATH)
    existing = read_existing_answers()

    work: list[tuple[dict[str, Any], str, str]] = []
    for question in originals:
        work.append(({
            "id": question["id"],
            "type": question["type"],
            "question": question["question"],
        }, "original", question["question"]))
    for question in paraphrases:
        work.append((question, "source_wording", question["question"]))

    for item, variant, query in work:
        for model_name in MODEL_NAMES:
            key = (item["id"], variant, model_name)
            target = raw_path(*key)
            prior = existing.get(key)
            if prior and prior.get("success") and target.exists():
                continue

            final_payload: Any = {"error": "no attempt made"}
            final_status: int | None = None
            final_latency = 0.0
            final_error: str | None = None
            final_timestamp = datetime.now(timezone.utc).isoformat()
            for attempt in range(MAX_RETRIES + 1):
                payload, status, latency, error = call_once(endpoint, query, model_name)
                final_payload = payload
                final_status = status
                final_latency = latency
                final_error = error
                final_timestamp = datetime.now(timezone.utc).isoformat()
                if error is None:
                    break
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_SECONDS)

            row = response_record(
                item=item,
                variant=variant,
                model_name=model_name,
                payload=final_payload,
                http_status=final_status,
                latency=final_latency,
                timestamp=final_timestamp,
                error=final_error,
                page_urls=page_urls,
            )
            write_json(target, final_payload)
            save_answer_row(row, existing)

    return existing


def write_summary(rows_by_key: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    rows = list(rows_by_key.values())
    variant_order = ("original", "source_wording")
    successful = sum(bool(row.get("success")) for row in rows)
    failed = len(rows) - successful
    lines = [
        "# Environmental Expert API Run Summary",
        "",
        f"Failed calls: {failed}",
        f"Successful calls: {successful}",
        "",
        "## Call counts by model and variant",
        "",
        "| Model | Variant | Successful | Failed |",
        "|---|---|---:|---:|",
    ]
    for model in MODEL_NAMES:
        for variant in variant_order:
            group = [row for row in rows if row["model_name_requested"] == model and row["variant"] == variant]
            good = sum(bool(row.get("success")) for row in group)
            lines.append(f"| {model} | {variant} | {good} | {len(group) - good} |")

    lines.extend([
        "",
        "## Aggregate metrics by model",
        "",
        "| Model | Median latency (s) | Median answer length (words) | Answers with cited URL | Cited URLs in knowledge base |",
        "|---|---:|---:|---:|---:|",
    ])
    for model in MODEL_NAMES:
        group = [row for row in rows if row["model_name_requested"] == model and row.get("success")]
        latencies = [float(row["latency"]) for row in group]
        lengths = [len(row["answer"].split()) for row in group]
        cited_answer_count = sum(bool(row["cited_urls"]) for row in group)
        cited_urls = [membership for row in group for membership in row["cited_url_kb_membership"]]
        kb_count = sum(bool(item["in_knowledge_base"]) for item in cited_urls)
        median_latency = f"{statistics.median(latencies):.2f}" if latencies else "NA"
        median_words = f"{statistics.median(lengths):.1f}" if lengths else "NA"
        cited_share = f"{cited_answer_count}/{len(group)}" if group else "0/0"
        kb_share = f"{kb_count}/{len(cited_urls)}" if cited_urls else "0/0"
        lines.append(f"| {model} | {median_latency} | {median_words} | {cited_share} | {kb_share} |")

    lines.extend([
        "",
        "Shares are shown as numerator/denominator counts. Latency and answer-length medians use successful calls only.",
        "",
    ])
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    env = load_env_file(ROOT / ".env")
    endpoint = env.get("ENVRI_EXPERT_URL", "").strip()
    if not endpoint:
        raise SystemExit("ENVRI_EXPERT_URL is missing from .env; no API calls were made.")
    rows = run_calls(endpoint)
    write_summary(rows)
    successful = sum(bool(row.get("success")) for row in rows.values())
    failed = len(rows) - successful
    print(f"Successful calls: {successful}; failed calls: {failed}.")
    print(f"Summary saved to {SUMMARY_PATH.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
