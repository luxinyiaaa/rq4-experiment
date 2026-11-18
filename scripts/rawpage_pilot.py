"""Fetch a seeded sample of live ENVRI pages and generate pilot questions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rq4.llm import call_llm

PAGES_CSV = PROJECT_ROOT / "data" / "processed" / "pages.csv"
RAW_KB_CSV = PROJECT_ROOT / "data" / "raw" / "site_pages_rows.csv"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "teacher_generate.md"
RAW_WEB_DIR = PROJECT_ROOT / "data" / "raw_web" / "pilot"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rawpage_questions" / "pilot"
QUESTIONS_JSONL = OUTPUT_DIR / "pilot_questions.jsonl"
SUMMARY_MD = OUTPUT_DIR / "pilot_summary.md"
TYPE_QUOTAS = {"dataset_record": 4, "about_science": 4, "vocabulary": 2}
QUESTION_TYPES = ("list_all", "paraphrased", "false_premise")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _site_key(page: dict[str, str]) -> str:
    return (page.get("domain") or urlsplit(page["url"]).netloc).strip().lower().removeprefix("www.")


def sample_pages(seed: int, policy: str) -> list[dict[str, str]]:
    with PAGES_CSV.open(newline="", encoding="utf-8") as handle:
        pages = [
            row for row in csv.DictReader(handle)
            if _is_true(row.get("eligible")) and row.get("page_type") in TYPE_QUOTAS
        ]
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for page in pages:
        by_type[page["page_type"]].append(page)

    quotas = dict(TYPE_QUOTAS)
    vocab_site_duplicate: str | None = None
    if policy == "allow-two-vocabulary-from-one-site":
        vocab_sites = sorted({_site_key(page) for page in by_type["vocabulary"]})
        if len(vocab_sites) != 1:
            raise ValueError("The vocabulary-site exception requires exactly one eligible vocabulary site")
        vocab_site_duplicate = vocab_sites[0]
    elif policy in {"reallocate-extra-to-dataset_record", "reallocate-extra-to-about_science"}:
        quotas["vocabulary"] = 1
        destination = policy.removeprefix("reallocate-extra-to-")
        quotas[destination] += 1

    rng = random.Random(seed)
    for group in by_type.values():
        rng.shuffle(group)

    # Process the sparsest categories first, while retaining the fixed seeded
    # order within each category. Backtracking finds a valid cross-category draw.
    type_order = sorted(quotas, key=lambda t: (len({_site_key(p) for p in by_type[t]}), t))
    slots = [kind for kind in type_order for _ in range(quotas[kind])]
    selected: list[dict[str, str]] = []
    site_counts: Counter[str] = Counter()

    def allowed(page: dict[str, str]) -> bool:
        site = _site_key(page)
        if not site_counts[site]:
            return True
        return (
            page["page_type"] == "vocabulary"
            and site == vocab_site_duplicate
            and site_counts[site] < 2
            and all(p["page_type"] == "vocabulary" for p in selected if _site_key(p) == site)
        )

    def search(slot: int) -> bool:
        if slot == len(slots):
            return True
        kind = slots[slot]
        used_urls = {p["url"] for p in selected}
        for page in by_type[kind]:
            if page["url"] in used_urls or not allowed(page):
                continue
            selected.append(page)
            site_counts[_site_key(page)] += 1
            if search(slot + 1):
                return True
            site_counts[_site_key(page)] -= 1
            selected.pop()
        return False

    if not search(0):
        raise ValueError("Could not satisfy the requested quotas and site cap")
    return selected


def load_raw_kb(selected_urls: set[str]) -> dict[str, list[dict[str, str]]]:
    by_url: dict[str, list[dict[str, str]]] = defaultdict(list)
    with RAW_KB_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"url", "chunk_number", "content"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Raw knowledge-base CSV lacks url, chunk_number, or content")
        for row in reader:
            if row.get("url") in selected_urls:
                by_url[row["url"]].append(row)
    for rows in by_url.values():
        rows.sort(key=lambda row: _chunk_sort_key(row.get("chunk_number", "")))
    return by_url


def _chunk_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except (TypeError, ValueError):
        return 0, str(value)


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def kb_similarity(kb_rows: list[dict[str, str]], live_text: str) -> tuple[float, int, int]:
    kb_chunks = [normalise_text(row.get("content", "")) for row in kb_rows]
    live = normalise_text(live_text)
    comparable = [chunk for chunk in kb_chunks if chunk]
    matched = sum(chunk in live for chunk in comparable)
    share = matched / len(comparable) if comparable else 0.0
    return share, matched, len(comparable)


def similarity_label(status: int | None, share: float) -> str:
    if status != 200:
        return "unavailable"
    if share >= 0.9:
        return "unchanged"
    if share >= 0.5:
        return "changed"
    return "very_different"


def safe_stem(index: int, page: dict[str, str]) -> str:
    digest = hashlib.sha256(page["url"].encode("utf-8")).hexdigest()[:10]
    return f"{page['page_type']}_{index:02d}_{digest}"


def adapted_teacher_prompt() -> str:
    source = PROMPT_PATH.read_text(encoding="utf-8")
    system_intro = source.split("## System", 1)[1].split("## User template", 1)[0].strip()
    general_rules = source.split("## Rules for every type", 1)[1].split("## Rules per type", 1)[0].strip()
    type_rules_block = source.split("## Rules per type", 1)[1].split("## Output format", 1)[0]
    allowed_rules = []
    headings = list(re.finditer(r"(?m)^- \*\*([^*]+)\*\*:", type_rules_block))
    for i, match in enumerate(headings):
        name = match.group(1).strip()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(type_rules_block)
        if name in QUESTION_TYPES:
            allowed_rules.append(type_rules_block[match.start():end].strip())

    example_blocks = []
    examples = source.split("## Examples", 1)[1]
    for heading, next_heading in (
        ("### Example 1: list_all", "### Example 2: comparison"),
        ("### Example 3: paraphrased", "### Example 4: false_premise"),
        ("### Example 4: false_premise", None),
    ):
        if heading not in examples:
            continue
        block = examples.split(heading, 1)[1]
        if next_heading and next_heading in block:
            block = block.split(next_heading, 1)[0]
        example_blocks.append(f"{heading}\n{block.strip()}")

    return "\n\n".join([
        "# Live-page question generation pilot",
        system_intro,
        "## Rules for every type\n" + general_rules,
        "## Allowed type rules\n" + "\n\n".join(allowed_rules),
        "## Output format\nReturn JSON only in this format. Generate exactly two question objects for the page; each object must use one of the three allowed types. If one question cannot be made responsibly, use skip=true and a reason for that object.\n"
        "```json\n{\n  \"questions\": [\n    {\n      \"skip\": false,\n      \"type\": \"list_all | paraphrased | false_premise\",\n      \"question\": \"...\",\n      \"reference_answer\": \"...\",\n      \"keypoints\": [{\"text\": \"...\", \"decisive\": true}],\n      \"supporting_quotes\": [{\"source\": \"passage\", \"text\": \"exact quote from supplied page text\"}],\n      \"false_premise\": \"... or empty string\"\n    },\n    {\n      \"skip\": false,\n      \"type\": \"list_all | paraphrased | false_premise\",\n      \"question\": \"...\",\n      \"reference_answer\": \"...\",\n      \"keypoints\": [{\"text\": \"...\", \"decisive\": true}],\n      \"supporting_quotes\": [{\"source\": \"passage\", \"text\": \"exact quote from supplied page text\"}],\n      \"false_premise\": \"... or empty string\"\n    }\n  ]\n}\n```",
        "## Examples\n" + "\n\n".join(example_blocks),
    ])


def user_message(page: dict[str, str], live_text: str) -> tuple[str, bool]:
    truncated = len(live_text) > 15000
    page_text = live_text[:15000]
    message = (
        "Generate exactly two independent questions from this live web page. "
        f"Allowed types: {', '.join(QUESTION_TYPES)}.\n"
        f"PAGE URL: {page['url']}\n"
        f"INPUT NOTE: The supplied page text is {'the first 15,000 characters and is truncated' if truncated else 'the complete extracted page text'}; "
        "use only the supplied text.\n"
        "PAGE TEXT:\n<<<\n" + page_text + "\n>>>\n"
        "Return JSON only using the required two-question format."
    )
    return message, truncated


def validate_generated(result: Any) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(result, dict) or not isinstance(result.get("questions"), list):
        return [], "response must contain a questions list"
    questions = result["questions"]
    if len(questions) != 2:
        return [], f"expected 2 question objects, received {len(questions)}"
    issues: list[str] = []
    for i, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            issues.append(f"question {i} is not an object")
            continue
        if question.get("skip") is True:
            if not question.get("reason"):
                issues.append(f"question {i} skip response lacks a reason")
            continue
        if question.get("type") not in QUESTION_TYPES:
            issues.append(f"question {i} has a disallowed type")
        for field in ("question", "reference_answer"):
            if not isinstance(question.get(field), str) or not question[field].strip():
                issues.append(f"question {i} lacks {field}")
        if not isinstance(question.get("keypoints"), list) or not question["keypoints"]:
            issues.append(f"question {i} lacks keypoints")
        if not isinstance(question.get("supporting_quotes"), list) or not question["supporting_quotes"]:
            issues.append(f"question {i} lacks supporting_quotes")
    return (questions, None) if not issues else ([], "; ".join(issues))


def check_quotes(question: dict[str, Any], live_text: str, kb_text: str) -> dict[str, Any]:
    live_norm = normalise_text(live_text)
    kb_norm = normalise_text(kb_text)
    per_quote = []
    for quote in question.get("supporting_quotes", []):
        text = str(quote.get("text", "")) if isinstance(quote, dict) else ""
        exact_in_page = bool(text) and text in live_text
        normalised_quote = normalise_text(text)
        in_kb = bool(normalised_quote) and normalised_quote in kb_norm
        per_quote.append({
            "quote": text,
            "exact_in_page": exact_in_page,
            "in_kb": in_kb,
        })
    total = len(per_quote)
    kb_matches = sum(bool(row["in_kb"]) for row in per_quote)
    if total == 0 or kb_matches == 0:
        in_kb = "none"
    elif kb_matches == total:
        in_kb = "all"
    else:
        in_kb = "some"
    return {
        "quote_checks": per_quote,
        "quote_in_page": all(row["exact_in_page"] for row in per_quote) if total else False,
        "in_kb": in_kb,
    }


def keypoints_with_quotes(question: dict[str, Any], quote_checks: list[dict[str, Any]]) -> str:
    points = question.get("keypoints", [])
    quote_lines = []
    for quote in quote_checks:
        marker = "page exact" if quote["exact_in_page"] else "page mismatch"
        kb_marker = "KB match" if quote["in_kb"] else "not in KB"
        quote_lines.append(f"quote: {quote['quote']} [{marker}; {kb_marker}]")
    if not points:
        return "\n".join(quote_lines)
    lines: list[str] = []
    for point in points:
        if isinstance(point, dict):
            text = str(point.get("text", ""))
            decisive = bool(point.get("decisive"))
        else:
            text, decisive = str(point), False
        lines.append(("[decisive] " if decisive else "") + text)
    lines.extend(quote_lines)
    return "\n".join(lines)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def write_summary(page_records: list[dict[str, Any]], question_records: list[dict[str, Any]]) -> None:
    status_counts = Counter(str(row.get("http_status")) for row in page_records)
    similarity_counts = Counter(row.get("kb_similarity_label", "unavailable") for row in page_records)
    needs_js = sum(row.get("page_status") == "needs_js_rendering" for row in page_records)
    skipped_questions = sum(len(row.get("skipped_questions", [])) for row in page_records)
    quote_failures = sum(
        not quote.get("exact_in_page", False)
        for row in question_records
        for quote in row.get("quote_checks", [])
    )
    in_kb_counts = Counter(row.get("in_kb", "none") for row in question_records)
    lines = [
        "# Raw Web Page Question Pilot Summary",
        "",
        f"Pages fetched: {len(page_records)}",
        "",
        "## HTTP status codes",
        "",
    ]
    lines.extend(f"- {status}: {count}" for status, count in sorted(status_counts.items()))
    lines.extend([
        "",
        f"Pages needing JS rendering: {needs_js}",
        "",
        "## Knowledge-base similarity labels",
        "",
    ])
    lines.extend(f"- {label}: {similarity_counts.get(label, 0)}" for label in ("unchanged", "changed", "very_different", "unavailable"))
    lines.extend([
        "",
        f"Questions generated: {len(question_records)}",
        f"Questions skipped by teacher: {skipped_questions}",
        f"Quote-check failures: {quote_failures}",
        "",
        "## Question-level knowledge-base quote match",
        "",
    ])
    lines.extend(f"- {label}: {in_kb_counts.get(label, 0)}" for label in ("all", "some", "none"))
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(policy: str) -> None:
    RAW_WEB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = sample_pages(seed=42, policy=policy)
    selected_urls = {page["url"] for page in pages}
    kb_rows_by_url = load_raw_kb(selected_urls)
    teacher_prompt = adapted_teacher_prompt()
    prompt_hash = hashlib.sha256(teacher_prompt.encode("utf-8")).hexdigest()
    page_records: list[dict[str, Any]] = []
    question_records: list[dict[str, Any]] = []

    for index, page in enumerate(pages, start=1):
        stem = safe_stem(index, page)
        html_path = RAW_WEB_DIR / f"{stem}.html"
        metadata_path = RAW_WEB_DIR / f"{stem}.json"
        response = None
        error = None
        try:
            response = requests.get(
                page["url"],
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                timeout=60,
                allow_redirects=True,
            )
            fetched_at = datetime.now(timezone.utc).isoformat()
            html_path.write_bytes(response.content)
            metadata = {
                "original_url": page["url"],
                "final_url": response.url,
                "http_status": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "fetch_time_utc": fetched_at,
                "page_type": page["page_type"],
            }
            html = response.text
        except requests.RequestException as exc:
            fetched_at = datetime.now(timezone.utc).isoformat()
            error = type(exc).__name__
            metadata = {
                "original_url": page["url"],
                "final_url": None,
                "http_status": None,
                "content_type": None,
                "fetch_time_utc": fetched_at,
                "page_type": page["page_type"],
                "fetch_error": error,
            }
            html = ""
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        status = metadata["http_status"]
        live_text = ""
        if response is not None and status == 200:
            live_text = trafilatura.extract(
                html,
                output_format="markdown",
                include_tables=True,
                include_comments=False,
                include_links=True,
            ) or ""
        kb_rows = kb_rows_by_url.get(page["url"], [])
        kb_text = "".join(row.get("content", "") or "" for row in kb_rows)
        similarity, matched_chunks, chunk_count = kb_similarity(kb_rows, live_text)
        label = similarity_label(status, similarity)
        if status != 200:
            page_status = "unavailable"
        else:
            page_status = "needs_js_rendering" if len(live_text) < 600 else "usable"
        page_record = {
            "page_index": index,
            "url": page["url"],
            "page_type": page["page_type"],
            "site": _site_key(page),
            "raw_html_file": html_path.name if response is not None else None,
            "metadata_file": metadata_path.name,
            "http_status": status,
            "fetch_error": error,
            "kb_similarity": similarity,
            "kb_similarity_label": label,
            "kb_chunks_matching": matched_chunks,
            "kb_chunks_compared": chunk_count,
            "extracted_text_chars": len(live_text),
            "page_status": page_status,
        }
        page_records.append(page_record)

        if page_status == "usable" and status == 200:
            message, truncated = user_message(page, live_text)
            response_llm: dict[str, Any] = {}
            generated: list[dict[str, Any]] = []
            validation_error: str | None = None
            try:
                response_llm = call_llm(
                    "judge_primary",
                    [
                        {"role": "system", "content": teacher_prompt},
                        {"role": "user", "content": message},
                    ],
                    json_mode=True,
                    temperature=0.0,
                    prompt_sha256=prompt_hash,
                    timeout=180.0,
                    log_messages=False,
                )
                generated, validation_error = validate_generated(response_llm.get("json"))
                if validation_error:
                    retry_message = message + "\n\nYour previous response did not meet the format: " + validation_error + ". Return exactly two valid question objects."
                    response_llm = call_llm(
                        "judge_primary",
                        [
                            {"role": "system", "content": teacher_prompt},
                            {"role": "user", "content": retry_message},
                        ],
                        json_mode=True,
                        temperature=0.0,
                        prompt_sha256=prompt_hash,
                        timeout=180.0,
                        log_messages=False,
                    )
                    generated, validation_error = validate_generated(response_llm.get("json"))
            except Exception as exc:
                # Do not expose provider details, credentials, or endpoint data in outputs.
                validation_error = f"teacher_call_failed:{type(exc).__name__}"
            page_record["truncated_for_questions"] = truncated
            page_record["generation_error"] = validation_error
            page_record["teacher_model"] = response_llm.get("model")
            page_record["teacher_prompt_sha256"] = prompt_hash

            for slot, question in enumerate(generated, start=1):
                if question.get("skip") is True:
                    page_record.setdefault("skipped_questions", []).append(str(question.get("reason", "")))
                    continue
                checks = check_quotes(question, live_text, kb_text)
                question_records.append({
                    "id": f"RP{index:02d}_{slot}",
                    "url": page["url"],
                    "page_type": page["page_type"],
                    "kb_similarity_label": label,
                    "type": question["type"],
                    "question": question["question"],
                    "reference_answer": question["reference_answer"],
                    "keypoints": question.get("keypoints", []),
                    "supporting_quotes": question.get("supporting_quotes", []),
                    "keypoints_with_quotes": keypoints_with_quotes(question, checks["quote_checks"]),
                    "quote_checks": checks["quote_checks"],
                    "quote_in_page": checks["quote_in_page"],
                    "in_kb": checks["in_kb"],
                    "truncated_for_questions": truncated,
                })

        # Wait between fetches only; no API endpoint calls are made here.
        if index < len(pages):
            time.sleep(1)

    write_jsonl(QUESTIONS_JSONL, question_records)
    write_jsonl(OUTPUT_DIR / "pilot_pages.jsonl", page_records)
    write_summary(page_records, question_records)
    print(json.dumps({
        "pages_fetched": len(page_records),
        "questions_generated": len(question_records),
        "summary": str(SUMMARY_MD.relative_to(PROJECT_ROOT)),
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocabulary-site-policy",
        choices=(
            "strict",
            "allow-two-vocabulary-from-one-site",
            "reallocate-extra-to-dataset_record",
            "reallocate-extra-to-about_science",
        ),
        default="strict",
    )
    args = parser.parse_args()
    run(args.vocabulary_site_policy)


if __name__ == "__main__":
    main()
