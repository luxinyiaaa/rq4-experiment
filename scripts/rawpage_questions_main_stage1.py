"""Generate stage-1 raw-web questions for eligible non-pilot KB pages.

Network and teacher calls are performed only when run without --prepare-only.
No API credentials or API base URL are written to outputs or logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import trafilatura
import yaml
from dotenv import load_dotenv
from rapidfuzz.fuzz import partial_ratio_alignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rawpage_pilot_v2 import build_prompt as build_v2_prompt  # noqa: E402

PAGES_CSV = ROOT / "data/processed/pages.csv"
CLEAN_KB_CSV = ROOT / "data/processed/chunks_clean.csv"
RAW_KB_CSV = ROOT / "data/raw/site_pages_rows.csv"
PILOT_PAGES = ROOT / "outputs/rawpage_questions/pilot/pilot_pages.jsonl"
OUT_DIR = ROOT / "outputs/rawpage_questions/main"
RAW_WEB_DIR = ROOT / "data/raw_web/main"
ORDER_CSV = OUT_DIR / "sampling_order.csv"
QUESTIONS_JSONL = OUT_DIR / "questions.jsonl"
PAGES_OUT_CSV = OUT_DIR / "pages.csv"
SUMMARY_MD = OUT_DIR / "stage1_summary.md"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)
QUESTION_TARGET = 40
SITE_QUESTION_CAP = QUESTION_TARGET // 4
QUESTION_TYPES = ("list_all", "paraphrased", "false_premise")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temp, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def site_for(row: dict[str, Any]) -> str:
    return (str(row.get("domain") or urlsplit(str(row["url"])).netloc)).strip().lower().removeprefix("www.")


def make_page_id(page_type: str, rank: int, url: str) -> str:
    prefix = "DR" if page_type == "dataset_record" else "AS"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{rank:03d}-{digest}"


def prepare_sampling_order() -> list[dict[str, Any]]:
    """Create the immutable seed-42 type-specific order, excluding pilot pages."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if ORDER_CSV.exists():
        with ORDER_CSV.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    pilot_urls = {row["url"] for row in read_jsonl(PILOT_PAGES)}
    with PAGES_CSV.open(newline="", encoding="utf-8") as handle:
        eligible = [
            row for row in csv.DictReader(handle)
            if str(row.get("eligible", "")).strip().lower() in {"true", "1", "yes"}
            and row.get("page_type") in {"dataset_record", "about_science"}
            and row.get("url") not in pilot_urls
        ]
    by_type: dict[str, list[dict[str, str]]] = {
        "dataset_record": [],
        "about_science": [],
    }
    for row in eligible:
        by_type[row["page_type"]].append(row)
    import random
    rng = random.Random(42)
    order: list[dict[str, Any]] = []
    global_rank = 0
    for page_type in ("dataset_record", "about_science"):
        rng.shuffle(by_type[page_type])
        for type_rank, page in enumerate(by_type[page_type], start=1):
            global_rank += 1
            order.append({
                "sequence": global_rank,
                "type_order": type_rank,
                "id": make_page_id(page_type, type_rank, page["url"]),
                "url": page["url"],
                "site": site_for(page),
                "page_type": page_type,
            })
    write_csv_atomic(ORDER_CSV, order, ["sequence", "type_order", "id", "url", "site", "page_type"])
    return order


def markdown_text(text: str) -> str:
    return re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text or "")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", markdown_text(text).lower())


def token_ngrams(text: str, n: int = 8) -> list[tuple[str, ...]]:
    words = tokens(text)
    return [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]


def ngram_coverage(source_text: str, target_text: str, n: int = 8) -> float | None:
    source = token_ngrams(source_text, n)
    if not source:
        return None
    target = set(token_ngrams(target_text, n))
    return sum(gram in target for gram in source) / len(source)


def similarity_label(status: int | None, live_in_kb: float | None) -> str:
    if status != 200:
        return "unavailable"
    value = 0.0 if live_in_kb is None else live_in_kb
    if value >= 0.85:
        return "unchanged"
    if value >= 0.5:
        return "changed"
    return "very_different"


def chunk_sort(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except (TypeError, ValueError):
        return 0, str(value)


def load_chunk_texts(path: Path) -> dict[str, str]:
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row.get("url", "")].append(row)
    return {
        url: "\n".join(row.get("content", "") or "" for row in sorted(items, key=lambda r: chunk_sort(r.get("chunk_number", ""))))
        for url, items in rows.items()
    }


class RawKBIndex:
    """In-memory token corpora for exact URL and all-URL quote checks."""

    def __init__(self, path: Path):
        chunks_by_url: dict[str, list[dict[str, str]]] = defaultdict(list)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                chunks_by_url[row.get("url", "")].append(row)
        self.chunks: dict[str, list[str]] = {}
        self.token_text_by_url: dict[str, str] = {}
        self.no_separator_token_text_by_url: dict[str, str] = {}
        for url, rows in chunks_by_url.items():
            contents = [row.get("content", "") or "" for row in sorted(rows, key=lambda r: chunk_sort(r.get("chunk_number", "")))]
            self.chunks[url] = contents
            self.token_text_by_url[url] = " " + " ".join(tokens("\n".join(contents))) + " "
            self.no_separator_token_text_by_url[url] = " " + " ".join(tokens("".join(contents))) + " "

    def matches_url(self, quote_tokens: list[str], url: str) -> bool:
        needle = " " + " ".join(quote_tokens) + " "
        return bool(quote_tokens) and (needle in self.token_text_by_url.get(url, "") or needle in self.no_separator_token_text_by_url.get(url, ""))

    def other_urls(self, quote_tokens: list[str], own_url: str, limit: int = 3) -> list[str]:
        needle = " " + " ".join(quote_tokens) + " "
        found = []
        for url, token_text in self.token_text_by_url.items():
            if url != own_url and quote_tokens and (needle in token_text or needle in self.no_separator_token_text_by_url.get(url, "")):
                found.append(url)
                if len(found) >= limit:
                    break
        return found


def contains_tokens(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    first = needle[0]
    for i, token in enumerate(haystack):
        if token == first and haystack[i:i + width] == needle:
            return True
    return False


def fuzzy_score_and_context(quote: str, chunks: list[str], radius: int = 40) -> tuple[float, str]:
    source = "\n".join(chunks)
    transformed = markdown_text(source).lower()
    spans = list(re.finditer(r"[a-z0-9]+", transformed))
    query_parts = tokens(quote)
    if not query_parts or not spans:
        return 0.0, ""
    target_parts: list[str] = []
    target_spans: list[tuple[int, int]] = []
    cursor = 0
    for span in spans:
        if target_parts:
            cursor += 1
        part = span.group(0)
        target_parts.append(part)
        target_spans.append((cursor, cursor + len(part)))
        cursor += len(part)
    aligned = partial_ratio_alignment(" ".join(query_parts), " ".join(target_parts))
    if aligned is None:
        return 0.0, ""
    first = next((i for i, (_, end) in enumerate(target_spans) if end > aligned.dest_start), 0)
    last = next((i for i, (start, _) in enumerate(target_spans) if start >= aligned.dest_end), len(spans))
    start_at = spans[first].start() if first < len(spans) else 0
    end_at = spans[max(first, last - 1)].end()
    context = transformed[max(0, start_at - radius):min(len(transformed), end_at + radius)]
    return float(aligned.score), context.strip()


def label_matches(values: list[bool]) -> str:
    count = sum(values)
    if not values or count == 0:
        return "none"
    return "all" if count == len(values) else "some"


def validate_page_response(value: Any) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(value, dict):
        return [], "response_not_object"
    if value.get("skip_page") is True:
        reason = value.get("skip_reason")
        if not isinstance(reason, str) or not reason.strip():
            return [], "skip_page_missing_reason"
        return [], "teacher_skip:" + reason.strip()
    questions = value.get("questions")
    if not isinstance(questions, list) or len(questions) != 2:
        return [], "expected_two_questions"
    issues: list[str] = []
    for i, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            issues.append(f"question_{i}_not_object")
            continue
        if question.get("skip") is True:
            issues.append(f"question_{i}_skip_not_supported_for_pair")
            continue
        if question.get("type") not in QUESTION_TYPES:
            issues.append(f"question_{i}_invalid_type")
        for field in ("question", "reference_answer"):
            if not isinstance(question.get(field), str) or not question[field].strip():
                issues.append(f"question_{i}_missing_{field}")
        keypoints = question.get("keypoints")
        if not isinstance(keypoints, list) or not keypoints:
            issues.append(f"question_{i}_missing_keypoints")
        elif any(not isinstance(k, dict) or not isinstance(k.get("text"), str) or not k["text"].strip() or not isinstance(k.get("decisive"), bool) for k in keypoints):
            issues.append(f"question_{i}_invalid_keypoints")
        quotes = question.get("supporting_quotes")
        if not isinstance(quotes, list) or not quotes:
            issues.append(f"question_{i}_missing_quotes")
        elif any(not isinstance(q, dict) or q.get("source") != "passage" or not isinstance(q.get("text"), str) or not q["text"] for q in quotes):
            issues.append(f"question_{i}_invalid_quotes")
        if question.get("type") == "false_premise" and not isinstance(question.get("false_premise"), str):
            issues.append(f"question_{i}_missing_false_premise")
    if questions[0].get("type") == questions[1].get("type"):
        issues.append("question_types_not_different")
    if not issues:
        decisive0 = {" ".join(tokens(k["text"])) for k in questions[0]["keypoints"] if k.get("decisive")}
        decisive1 = {" ".join(tokens(k["text"])) for k in questions[1]["keypoints"] if k.get("decisive")}
        quotes0 = {" ".join(tokens(q["text"])) for q in questions[0]["supporting_quotes"]}
        quotes1 = {" ".join(tokens(q["text"])) for q in questions[1]["supporting_quotes"]}
        if decisive0 & decisive1 or quotes0 & quotes1:
            issues.append("questions_reuse_decisive_fact_or_quote")
    if issues:
        return [], ";".join(issues)
    return questions, None


def contains_sensitive_contact(text: str) -> bool:
    if re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, flags=re.I):
        return True
    for match in re.finditer(r"(?<!\w)(?:\+?\d[\d ()./-]{7,}\d)(?!\w)", text):
        if sum(ch.isdigit() for ch in match.group(0)) >= 10:
            return True
    if re.search(r"\b\d{1,6}\s+[A-Za-z0-9.' -]{1,50}\b(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|close|crescent|court|place|postcode|postal code)\b", text, flags=re.I):
        return True
    return False


def fetch_page(page: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    attempt_log: list[dict[str, Any]] = []
    last_response = None
    for attempt in range(1, 4):
        try:
            response = requests.get(
                page["url"],
                headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"},
                timeout=60,
                allow_redirects=True,
            )
            last_response = response
            attempt_log.append({"attempt": attempt, "http_status": response.status_code})
            if response.status_code == 200:
                break
        except requests.RequestException as exc:
            last_response = None
            attempt_log.append({"attempt": attempt, "error": type(exc).__name__})
        if attempt < 3:
            time.sleep(60)
    body = last_response.content if last_response is not None else b""
    status = last_response.status_code if last_response is not None else None
    final_url = last_response.url if last_response is not None else ""
    content_type = last_response.headers.get("Content-Type", "") if last_response is not None else ""
    record = {
        "id": page["id"],
        "original_url": page["url"],
        "final_url": final_url,
        "http_status": status,
        "content_type": content_type,
        "fetch_time_utc": datetime.now(timezone.utc).isoformat(),
        "attempts": attempt_log,
        "retries": max(0, len(attempt_log) - 1),
    }
    return record, body


def teacher_settings() -> tuple[dict[str, Any], str]:
    load_dotenv(ROOT / ".env", override=False)
    with (ROOT / "configs/models.yaml").open(encoding="utf-8") as handle:
        settings = (yaml.safe_load(handle) or {}).get("judge_primary") or {}
    required = ("model", "base_url", "api_key_env")
    if any(not settings.get(key) for key in required):
        raise ValueError("judge_primary configuration is incomplete")
    api_key = os.getenv(str(settings["api_key_env"]))
    if not api_key:
        raise ValueError("judge_primary credential is missing from the local environment")
    return settings, api_key


def teacher_call(endpoint: str, api_key: str, model: str, extra_body: dict[str, Any], prompt: str, page: dict[str, Any], text: str, prompt_hash: str) -> dict[str, Any]:
    truncated = len(text) > 15000
    supplied = text[:15000]
    user_message = (
        "Generate two questions from this current live page. Use only the supplied text.\n"
        f"PAGE URL: {page['url']}\n"
        f"PAGE TYPE: {page['page_type']}\n"
        f"INPUT NOTE: {'Only the first 15,000 characters are supplied; the text is truncated.' if truncated else 'The complete extracted page text is supplied.'}\n"
        f"PAGE TEXT:\n<<<\n{supplied}\n>>>\nReturn the required JSON only."
    )
    system = prompt + "\n\n## Additional stage 1 rules\n- The page's two questions must rely on different decisive facts; do not ask both questions about the same fact.\n- Do not include e-mail addresses, phone numbers, postal addresses, or private individuals in questions or answers."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_message}]
    all_usage = Counter()
    calls = 0
    actual_model = model
    json_supported = True
    last_error = ""
    for correction_try in range(2):
        call_messages = list(messages)
        if correction_try:
            call_messages.append({"role": "user", "content": f"The previous response was invalid ({last_error}). Correct the reported issue and return exactly one valid JSON object with the required fields, or skip the page with a concise reason."})
        response_payload = None
        for attempt in range(3):
            calls += 1
            try:
                request: dict[str, Any] = {"model": model, "messages": call_messages, "temperature": 0}
                if json_supported:
                    request["response_format"] = {"type": "json_object"}
                if extra_body:
                    request.update(extra_body)
                response = requests.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=request,
                    timeout=180,
                )
                if json_supported and response.status_code in {400, 422}:
                    body_lower = response.text.lower()
                    if "response_format" in body_lower and any(word in body_lower for word in ("unsupported", "not supported", "unknown parameter", "unrecognized")):
                        json_supported = False
                        continue
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    return {"json": None, "error": "teacher_api_http_error:" + str(response.status_code), "calls": calls, "usage": dict(all_usage), "model": actual_model, "truncated": truncated, "prompt_sha256": prompt_hash}
                if not response.ok:
                    return {"json": None, "error": "teacher_api_http_error:" + str(response.status_code), "calls": calls, "usage": dict(all_usage), "model": actual_model, "truncated": truncated, "prompt_sha256": prompt_hash}
                response_payload = response.json()
                break
            except Exception as exc:
                last_error = type(exc).__name__
                retryable = isinstance(exc, (requests.Timeout, requests.ConnectionError))
                if retryable and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return {"json": None, "error": "teacher_api_error:" + last_error, "calls": calls, "usage": dict(all_usage), "model": actual_model, "truncated": truncated, "prompt_sha256": prompt_hash}
        if response_payload is None:
            break
        actual_model = str(response_payload.get("model") or model)
        usage = response_payload.get("usage")
        if usage is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                all_usage[key] += int(usage.get(key, 0) or 0)
        try:
            content = response_payload["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            last_error = "missing_response_content"
            continue
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            last_error = "invalid_json"
            continue
        questions, error = validate_page_response(value)
        if error:
            if error.startswith("teacher_skip:"):
                return {"json": value, "questions": [], "skip_reason": error.removeprefix("teacher_skip:"), "calls": calls, "usage": dict(all_usage), "model": actual_model, "truncated": truncated, "prompt_sha256": prompt_hash}
            last_error = error
            continue
        sensitive = any(contains_sensitive_contact(q.get("question", "") + " " + q.get("reference_answer", "")) for q in questions)
        if sensitive:
            last_error = "sensitive_contact_or_address_detected"
            continue
        return {"json": value, "questions": questions, "error": "", "calls": calls, "usage": dict(all_usage), "model": actual_model, "truncated": truncated, "prompt_sha256": prompt_hash}
    return {"json": None, "questions": [], "error": "generation_failed:" + (last_error or "invalid_response"), "calls": calls, "usage": dict(all_usage), "model": actual_model, "truncated": truncated, "prompt_sha256": prompt_hash}


def question_record(page: dict[str, Any], question: dict[str, Any], slot: int, live_text: str, raw_index: RawKBIndex, similarity: str, live_in_kb: float | None, teacher: dict[str, Any]) -> dict[str, Any]:
    quote_checks = []
    other_urls: list[str] = []
    for item in question.get("supporting_quotes", []):
        quote = str(item.get("text", ""))
        quote_tokens = tokens(quote)
        in_url = raw_index.matches_url(quote_tokens, page["url"])
        other = raw_index.other_urls(quote_tokens, page["url"], limit=3) if quote_tokens and not in_url else []
        fuzzy, context = fuzzy_score_and_context(quote, raw_index.chunks.get(page["url"], []))
        quote_checks.append({
            "source": item.get("source", "passage"),
            "quote": quote,
            "quote_in_page": bool(quote) and quote in live_text,
            "in_kb": in_url,
            "quote_kb_fuzzy": round(fuzzy, 2),
            "kb_best_match": "" if in_url else context,
            "in_kb_any": in_url or bool(other),
            "kb_other_urls": other,
        })
        other_urls.extend(other)
    in_kb = label_matches([q["in_kb"] for q in quote_checks])
    in_kb_any = label_matches([q["in_kb_any"] for q in quote_checks])
    keypoint_lines = []
    for point in question.get("keypoints", []):
        keypoint_lines.append(("[decisive] " if point.get("decisive") else "[non-decisive] ") + point["text"])
    quote_lines = [
        f"[source: {item.get('source', 'passage')}] {item.get('text', '')}"
        for item in question.get("supporting_quotes", [])
    ]
    best_lines = [
        f"{item['quote']} [fuzzy={item['quote_kb_fuzzy']:.2f}] {item['kb_best_match']}"
        for item in quote_checks if not item["in_kb"]
    ]
    return {
        "id": f"{page['id']}_{slot}",
        "page_id": page["id"],
        "url": page["url"],
        "site": page["site"],
        "page_type": page["page_type"],
        "live_in_kb": live_in_kb,
        "similarity_label": similarity,
        "type": question["type"],
        "type_reason": question.get("type_reason", ""),
        "question": question["question"],
        "reference_answer": question["reference_answer"],
        "keypoints": question["keypoints"],
        "supporting_quotes": question["supporting_quotes"],
        "false_premise": question.get("false_premise", ""),
        "quote_checks": quote_checks,
        "quote_in_page": all(q["quote_in_page"] for q in quote_checks) if quote_checks else False,
        "in_kb": in_kb,
        "in_kb_any": in_kb_any,
        "kb_other_urls": list(dict.fromkeys(other_urls))[:3],
        "kb_best_match": "\n".join(best_lines),
        "keypoints_with_quotes": "\n".join(keypoint_lines + quote_lines),
        "truncated": teacher["truncated"],
        "teacher_model": teacher["model"],
        "teacher_prompt_sha256": teacher["prompt_sha256"],
    }


PAGE_FIELDS = [
    "id", "url", "final_url", "site", "page_type", "http_status", "retries",
    "extracted_characters", "truncated", "kb_in_live", "live_in_kb", "similarity_label",
    "questions_generated", "questions_skipped", "skip_reason", "teacher_calls",
    "prompt_tokens", "completion_tokens", "total_tokens",
]


def load_pages_out() -> list[dict[str, Any]]:
    if not PAGES_OUT_CSV.exists():
        return []
    with PAGES_OUT_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_progress(page_rows: list[dict[str, Any]], questions: list[dict[str, Any]]) -> None:
    write_csv_atomic(PAGES_OUT_CSV, page_rows, PAGE_FIELDS)
    write_jsonl_atomic(QUESTIONS_JSONL, questions)


def run(network: bool) -> None:
    started = time.monotonic()
    order = prepare_sampling_order()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages_done = load_pages_out()
    page_by_id = {str(row["id"]): row for row in pages_done}
    questions = read_jsonl(QUESTIONS_JSONL)
    completed = set(page_by_id)
    questions_by_type = Counter(row["page_type"] for row in questions)
    site_counts = Counter((row["page_type"], row["site"]) for row in questions)
    clean_texts = load_chunk_texts(CLEAN_KB_CSV)
    raw_index = RawKBIndex(RAW_KB_CSV)
    prompt = build_stage_prompt()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    endpoint = ""
    api_key = ""
    model = ""
    extra_body: dict[str, Any] = {}
    if network:
        config, api_key = teacher_settings()
        model = str(config["model"])
        endpoint = str(config["base_url"]).rstrip("/") + "/chat/completions"
        extra_body = dict(config.get("thinking_parameters") or {})

    for order_row in order:
        page_type = str(order_row["page_type"])
        if questions_by_type[page_type] >= QUESTION_TARGET:
            continue
        page_id = str(order_row["id"])
        if page_id in completed:
            continue
        page = {**order_row, "url": str(order_row["url"]), "site": str(order_row["site"]), "page_type": page_type}
        site_key = (page_type, page["site"])
        if site_counts[site_key] >= SITE_QUESTION_CAP:
            page_report = {
                "id": page_id, "url": page["url"], "final_url": "", "site": page["site"],
                "page_type": page_type, "http_status": "not_fetched_site_cap", "retries": 0,
                "extracted_characters": 0, "truncated": "no", "kb_in_live": "", "live_in_kb": "",
                "similarity_label": "", "questions_generated": 0, "questions_skipped": 2,
                "skip_reason": "site_question_cap_reached", "teacher_calls": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            }
            pages_done.append(page_report)
            completed.add(page_id)
            page_by_id[page_id] = page_report
            write_progress(pages_done, questions)
            continue

        fetch_record, html_bytes = fetch_page(page)
        prefix = re.sub(r"[^A-Za-z0-9_-]", "_", page_id)
        RAW_WEB_DIR.mkdir(parents=True, exist_ok=True)
        html_path = RAW_WEB_DIR / f"{prefix}.html"
        tmp_html = html_path.with_suffix(".html.tmp")
        tmp_html.write_bytes(html_bytes)
        os.replace(tmp_html, html_path)
        fetch_json = RAW_WEB_DIR / f"{prefix}.json"
        fetch_json.write_text(json.dumps(fetch_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        status = fetch_record["http_status"]
        live_text = trafilatura.extract(html_bytes, output_format="markdown", include_tables=True, include_comments=False, include_links=True) or "" if status == 200 else ""
        extracted = len(live_text)
        kb_text = clean_texts.get(page["url"], "")
        kb_live = ngram_coverage(kb_text, live_text) if status == 200 else None
        live_kb = ngram_coverage(live_text, kb_text) if status == 200 else None
        label = similarity_label(status, live_kb)
        truncated = len(live_text) > 15000
        teacher = {"calls": 0, "usage": {}, "model": "", "truncated": truncated, "prompt_sha256": prompt_hash}
        generated: list[dict[str, Any]] = []
        reason = ""
        if status != 200:
            reason = "fetch_unavailable"
        elif extracted < 600:
            reason = "extracted_text_under_600_characters"
        elif not network:
            reason = "prepared_only_no_network_calls"
        else:
            teacher = teacher_call(endpoint, api_key, model, extra_body, prompt, page, live_text, prompt_hash)
            teacher["truncated"] = truncated
            if teacher.get("error"):
                reason = str(teacher["error"])
            elif teacher.get("skip_reason"):
                reason = "teacher_skip:" + str(teacher["skip_reason"])
            else:
                generated = [
                    question_record(page, q, i, live_text, raw_index, label, live_kb, teacher)
                    for i, q in enumerate(teacher["questions"], start=1)
                ]
        if generated:
            room = QUESTION_TARGET - questions_by_type[page_type]
            generated = generated[:room]
            questions.extend(generated)
            questions_by_type[page_type] += len(generated)
            site_counts[site_key] += len(generated)
            if questions_by_type[page_type] >= QUESTION_TARGET and len(generated) == 1:
                reason = "target_reached_after_final_question"
        elif reason.startswith("teacher_skip:"):
            questions_by_type[page_type] += 0
        usage = teacher.get("usage") or {}
        page_report = {
            "id": page_id, "url": page["url"], "final_url": fetch_record["final_url"],
            "site": page["site"], "page_type": page_type, "http_status": status if status is not None else "request_error",
            "retries": fetch_record["retries"], "extracted_characters": extracted,
            "truncated": "yes" if truncated else "no", "kb_in_live": kb_live if kb_live is not None else "",
            "live_in_kb": live_kb if live_kb is not None else "", "similarity_label": label,
            "questions_generated": len(generated), "questions_skipped": max(0, 2 - len(generated)),
            "skip_reason": reason, "teacher_calls": teacher.get("calls", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        pages_done.append(page_report)
        completed.add(page_id)
        page_by_id[page_id] = page_report
        write_progress(pages_done, questions)
        time.sleep(1)
        if all(questions_by_type[t] >= QUESTION_TARGET for t in ("dataset_record", "about_science")):
            break

    write_progress(pages_done, questions)
    write_summary(pages_done, questions, time.monotonic() - started)


def build_stage_prompt() -> str:
    base = build_v2_prompt()
    additions = [
        "## Raw web stage 1 rules",
        "- Generate exactly two questions per substantive page, and use two different types from list_all, paraphrased, and false_premise. If a type is unsuitable, use another allowed type and explain why.",
        "- Ask what a researcher looking for data or information would ask: content, measured variables, time and spatial coverage, methods, access, or responsible organisation.",
        "- Do not ask for an answer that consists only of a URL, identifier, file production/modification timestamp, or collection name.",
        "- If the page has no substantive content beyond metadata, return skip_page=true with a concise skip_reason.",
        "- If an answer comes from a table, state the row and column used in reference_answer and include any supplied unit.",
        "- The two questions must rely on different decisive facts.",
        "- Do not include e-mail addresses, phone numbers, postal addresses, or private individuals in questions or answers.",
        "- Do not use outside knowledge. Quotes must be verbatim substrings of the supplied live page text.",
        "Return JSON: {\"skip_page\": false, \"skip_reason\": \"\", \"questions\": [{\"type\": \"list_all|paraphrased|false_premise\", \"type_reason\": \"...\", \"question\": \"...\", \"reference_answer\": \"...\", \"keypoints\": [{\"text\": \"...\", \"decisive\": true}], \"supporting_quotes\": [{\"source\": \"passage\", \"text\": \"...\"}], \"false_premise\": \"... or empty\"}, {same fields}]}.",
    ]
    return base + "\n\n" + "\n".join(additions)


def write_summary(pages: list[dict[str, Any]], questions: list[dict[str, Any]], elapsed: float) -> None:
    status_counts = Counter(str(row["http_status"]) for row in pages)
    page_type_counts = Counter(row["page_type"] for row in pages)
    question_type_counts = Counter(row["type"] for row in questions)
    questions_by_page_type = Counter(row["page_type"] for row in questions)
    questions_by_site = Counter(row["site"] for row in questions)
    similarity_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pages:
        if row.get("similarity_label"):
            similarity_by_type[row["page_type"]][row["similarity_label"]] += 1
    skips: Counter[str] = Counter()
    for row in pages:
        reason = str(row.get("skip_reason", ""))
        if reason:
            category = reason if reason.startswith("teacher_skip:") else reason.split(":", 1)[0]
            skips[category] += 1
    in_kb = Counter((row["similarity_label"], row["in_kb"]) for row in questions)
    in_kb_any = Counter((row["similarity_label"], row["in_kb_any"]) for row in questions)
    quote_failures = sum(not item.get("quote_in_page", False) for row in questions for item in row.get("quote_checks", []))
    teacher_calls = sum(int(row.get("teacher_calls") or 0) for row in pages)
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in pages)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in pages)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in pages)
    lines = [
        "# Raw-Web Question Set — Stage 1 Summary", "",
        "## Pages processed by HTTP status and page type", "",
        "| HTTP status | dataset_record | about_science | total |", "|---|---:|---:|---:|",
    ]
    for status in sorted(status_counts):
        cells = [sum(1 for row in pages if str(row["http_status"]) == status and row["page_type"] == ptype) for ptype in ("dataset_record", "about_science")]
        lines.append(f"| {status} | {cells[0]} | {cells[1]} | {status_counts[status]} |")
    lines.append(f"| Total pages | {page_type_counts.get('dataset_record', 0)} | {page_type_counts.get('about_science', 0)} | {len(pages)} |")
    lines += ["", f"Usable pages (HTTP 200 and at least 600 extracted characters): {sum(1 for row in pages if str(row['http_status']) == '200' and int(row['extracted_characters']) >= 600)}", "", "## Skips by reason", ""]
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(skips.items()))
    if not skips:
        lines.append("- none: 0")
    lines += ["", "## Questions by page type", "", f"- dataset_record: {questions_by_page_type.get('dataset_record', 0)}", f"- about_science: {questions_by_page_type.get('about_science', 0)}", "", "## Questions by question type", ""]
    lines.extend(f"- {qtype}: {question_type_counts.get(qtype, 0)}" for qtype in QUESTION_TYPES)
    lines += ["", "## Questions by site", ""]
    lines.extend(f"- {site}: {count}" for site, count in sorted(questions_by_site.items()))
    lines += ["", "## Question-level KB matches overall", "", "| Match label | in_kb | in_kb_any |", "|---|---:|---:|"]
    for label in ("all", "some", "none"):
        lines.append(f"| {label} | {sum(1 for row in questions if row['in_kb'] == label)} | {sum(1 for row in questions if row['in_kb_any'] == label)} |")
    lines += ["", "## Question-level KB matches by similarity label", "", "| Similarity | Check | all | some | none |", "|---|---|---:|---:|---:|"]
    for sim in ("unchanged", "changed", "very_different", "unavailable"):
        for check, counter in (("in_kb", in_kb), ("in_kb_any", in_kb_any)):
            lines.append(f"| {sim} | {check} | {counter[(sim, 'all')]} | {counter[(sim, 'some')]} | {counter[(sim, 'none')]} |")
    lines += ["", "## Similarity labels by page type", "", "| Page type | unchanged | changed | very_different | unavailable |", "|---|---:|---:|---:|---:|"]
    for page_type in ("dataset_record", "about_science"):
        c = similarity_by_type[page_type]
        lines.append(f"| {page_type} | {c['unchanged']} | {c['changed']} | {c['very_different']} | {c['unavailable']} |")
    lines += ["", f"Quote-in-page failures: {quote_failures}", "", "## Teacher usage", "", f"- Calls: {teacher_calls}", f"- Prompt tokens: {prompt_tokens}", f"- Completion tokens: {completion_tokens}", f"- Total tokens: {total_tokens}", f"- Run time seconds: {elapsed:.1f}", ""]
    pilot_urls = {row["url"] for row in read_jsonl(PILOT_PAGES)}
    with PAGES_CSV.open(newline="", encoding="utf-8") as handle:
        eligible_dataset = [
            row for row in csv.DictReader(handle)
            if str(row.get("eligible", "")).strip().lower() in {"true", "1", "yes"}
            and row.get("page_type") == "dataset_record"
            and row.get("url") not in pilot_urls
        ]
    eligible_by_site = Counter(site_for(row) for row in eligible_dataset)
    fetched_stage1 = {row["url"] for row in pages if str(row.get("http_status")) != "not_fetched_site_cap"}
    remaining_by_site = Counter(site_for(row) for row in eligible_dataset if row["url"] not in fetched_stage1)
    dataset_question_count = sum(1 for row in questions if row["page_type"] == "dataset_record")
    dataset_fetched_count = sum(1 for row in pages if row["page_type"] == "dataset_record" and str(row.get("http_status")) != "not_fetched_site_cap")
    stage1_yield = dataset_question_count / dataset_fetched_count if dataset_fetched_count else 0.0
    cap25 = int(150 * 0.25)
    cap30 = int(150 * 0.30)
    estimate25_by_site = {site: min(count * stage1_yield, cap25) for site, count in remaining_by_site.items()}
    estimate30_by_site = {site: min(count * stage1_yield, cap30) for site, count in remaining_by_site.items()}
    lines += [
        "## Dataset-record site availability and stage 2 estimate", "",
        f"Stage 1 yield: {stage1_yield:.2f} questions per fetched dataset_record page ({dataset_question_count} questions / {dataset_fetched_count} fetched pages). Stage 2 estimates use eligible pages not fetched in stage 1, multiply by that yield, and cap each site's contribution at 37 questions under 25% or 45 under 30% of a 150-question target.",
        "",
        "| Site | Eligible pages | Remaining after stage 1 | Estimated stage 2 questions at 25% cap | Estimated stage 2 questions at 30% cap |",
        "|---|---:|---:|---:|---:|",
    ]
    for site in sorted(eligible_by_site):
        lines.append(f"| {site} | {eligible_by_site[site]} | {remaining_by_site.get(site, 0)} | {estimate25_by_site.get(site, 0.0):.1f} | {estimate30_by_site.get(site, 0.0):.1f} |")
    lines.append(f"| Estimated total | {sum(eligible_by_site.values())} | {sum(remaining_by_site.values())} | {sum(estimate25_by_site.values()):.1f} | {sum(estimate30_by_site.values()):.1f} |")
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true", help="Write the deterministic sampling order without network access or teacher calls")
    parser.add_argument("--summary-only", action="store_true", help="Rebuild the summary from saved stage-1 outputs without network access")
    args = parser.parse_args()
    if args.prepare_only:
        order = prepare_sampling_order()
        counts = Counter(row["page_type"] for row in order)
        print(json.dumps({"dataset_record_ordered": counts["dataset_record"], "about_science_ordered": counts["about_science"], "sampling_order": str(ORDER_CSV.relative_to(ROOT))}))
        return
    if args.summary_only:
        page_rows = load_pages_out()
        question_rows = read_jsonl(QUESTIONS_JSONL)
        elapsed = 0.0
        if SUMMARY_MD.exists():
            match = re.search(r"Run time seconds: ([0-9.]+)", SUMMARY_MD.read_text(encoding="utf-8"))
            if match:
                elapsed = float(match.group(1))
        write_summary(page_rows, question_rows, elapsed)
        print(json.dumps({"pages": len(page_rows), "questions": len(question_rows), "summary": str(SUMMARY_MD.relative_to(ROOT))}))
        return
    run(network=True)


if __name__ == "__main__":
    main()
