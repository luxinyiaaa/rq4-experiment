"""Re-fetch the pilot page sample and produce v2 questions and comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import trafilatura
from rapidfuzz.fuzz import partial_ratio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from rq4.llm import call_llm  # noqa: E402
from rawpage_pilot import adapted_teacher_prompt  # noqa: E402

PILOT_DIR = PROJECT_ROOT / "outputs" / "rawpage_questions" / "pilot"
RAW_WEB_DIR = PROJECT_ROOT / "data" / "raw_web" / "pilot_v2"
V1_PAGES = PILOT_DIR / "pilot_pages.jsonl"
V1_QUESTIONS = PILOT_DIR / "pilot_questions.jsonl"
CLEAN_KB = PROJECT_ROOT / "data" / "processed" / "chunks_clean.csv"
RAW_KB = PROJECT_ROOT / "data" / "raw" / "site_pages_rows.csv"
PROMPT_FILE = PROJECT_ROOT / "prompts" / "teacher_generate.md"
PAGE_CSV = PILOT_DIR / "pilot_pages.csv"
QUESTIONS_JSONL = PILOT_DIR / "pilot_v2_questions.jsonl"
V1_RECHECKS = PILOT_DIR / "pilot_v2_v1_rechecks.jsonl"
QUESTIONS_XLSX = PILOT_DIR / "pilot_v2_questions.xlsx"
SUMMARY_MD = PILOT_DIR / "pilot_v2_summary.md"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)
ALLOWED_TYPES = {"list_all", "paraphrased", "false_premise"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalise_text(text: str) -> str:
    """Convert markdown to comparable plain text and normalise whitespace/case."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    lines: list[str] = []
    for line in text.splitlines():
        if re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*", line):
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", line)
        line = line.replace("|", " ")
        line = re.sub(r"```+|`", "", line)
        line = re.sub(r"\*\*|__|~~|\*|_", "", line)
        lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines).lower()).strip()


def word_ngrams(text: str, n: int = 8) -> list[tuple[str, ...]]:
    words = re.findall(r"[^\W_]+(?:['’\-][^\W_]+)*|\d+", normalise_text(text), flags=re.UNICODE)
    return [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]


def ngram_coverage(source_text: str, target_text: str, n: int = 8) -> float | None:
    source = word_ngrams(source_text, n)
    if not source:
        return None
    target = set(word_ngrams(target_text, n))
    return sum(gram in target for gram in source) / len(source)


def chunk_sort(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except (TypeError, ValueError):
        return 0, str(value)


def load_kb_rows(path: Path, urls: set[str]) -> dict[str, list[dict[str, str]]]:
    by_url: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("url") in urls:
                by_url[row["url"]].append(row)
    for rows in by_url.values():
        rows.sort(key=lambda row: chunk_sort(row.get("chunk_number", "")))
    return by_url


def compose(rows: list[dict[str, str]]) -> str:
    return "\n".join(row.get("content", "") or "" for row in rows)


def fetch_page(page: dict[str, Any], index: int) -> tuple[dict[str, Any], str]:
    page_id = f"RP{int(page['page_index']):02d}"
    final_response = None
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, 4):
        try:
            response = requests.get(
                page["url"],
                headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"},
                timeout=60,
                allow_redirects=True,
            )
            final_response = response
            attempts.append({"attempt": attempt, "http_status": response.status_code})
            if response.status_code != 500 or attempt == 3:
                break
        except requests.RequestException as exc:
            attempts.append({"attempt": attempt, "error": type(exc).__name__})
            final_response = None
            break
        time.sleep(60)

    fetched_at = datetime.now(timezone.utc).isoformat()
    html = final_response.text if final_response is not None else ""
    if final_response is not None:
        RAW_WEB_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_WEB_DIR / f"{page_id}.html").write_bytes(final_response.content)
    status = final_response.status_code if final_response is not None else None
    metadata = {
        "page_id": page_id,
        "original_url": page["url"],
        "final_url": final_response.url if final_response is not None else None,
        "http_status": status,
        "content_type": final_response.headers.get("Content-Type") if final_response is not None else None,
        "fetch_time_utc": fetched_at,
        "attempts": attempts,
    }
    RAW_WEB_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_WEB_DIR / f"{page_id}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if index < 10:
        time.sleep(1)
    return metadata, html


def build_prompt() -> str:
    return adapted_teacher_prompt() + """

## Additional raw-page pilot v2 rules
- Produce exactly two questions per substantive page, and use two different types from list_all, paraphrased, and false_premise. If a type is unsuitable, use another allowed type and explain the choice in `type_reason`.
- Ask what an environmental researcher seeking data or information would ask: content, measured variables, time or spatial coverage, methods, access, or responsible organisation.
- Do not ask questions whose answer is only a URL, identifier, file production/modification timestamp, or the collection to which a vocabulary record belongs.
- If the page has no substantive content beyond such metadata, return `skip_page: true` with a concise `skip_reason`; do not invent questions.
- If an answer is read from a table, state in `reference_answer` which row and column were used and include the unit when the table supplies one.
- Do not use outside knowledge. Quotes must be verbatim substrings of the supplied extracted text.

Return exactly this JSON shape (unless skipping the page):
{"skip_page": false, "skip_reason": "", "questions": [
 {"type": "list_all|paraphrased|false_premise", "type_reason": "...", "question": "...", "reference_answer": "...", "keypoints": [{"text": "...", "decisive": true}], "supporting_quotes": [{"source": "passage", "text": "verbatim quote"}], "false_premise": "... or empty"},
 {"type": "list_all|paraphrased|false_premise", "type_reason": "...", "question": "...", "reference_answer": "...", "keypoints": [{"text": "...", "decisive": true}], "supporting_quotes": [{"source": "passage", "text": "verbatim quote"}], "false_premise": "... or empty"}
]}
"""


def validate_response(value: Any) -> tuple[list[dict[str, Any]], str | None, str | None]:
    if not isinstance(value, dict):
        return [], None, "response is not a JSON object"
    if value.get("skip_page") is True:
        reason = value.get("skip_reason")
        return [], str(reason or "missing skip reason"), None if reason else "skip_page lacks skip_reason"
    questions = value.get("questions")
    if not isinstance(questions, list) or len(questions) != 2:
        return [], None, "response must contain exactly two questions"
    issues: list[str] = []
    types: list[str] = []
    for index, question in enumerate(questions, 1):
        if not isinstance(question, dict):
            issues.append(f"question {index} is not an object")
            continue
        qtype = question.get("type")
        types.append(str(qtype))
        if qtype not in ALLOWED_TYPES:
            issues.append(f"question {index} has a disallowed type")
        for field in ("type_reason", "question", "reference_answer"):
            if not isinstance(question.get(field), str) or not question[field].strip():
                issues.append(f"question {index} lacks {field}")
        if not isinstance(question.get("keypoints"), list) or not question["keypoints"]:
            issues.append(f"question {index} lacks keypoints")
        if not isinstance(question.get("supporting_quotes"), list) or not question["supporting_quotes"]:
            issues.append(f"question {index} lacks supporting_quotes")
        if question.get("type") == "false_premise" and not isinstance(question.get("false_premise"), str):
            issues.append(f"question {index} lacks false_premise")
    if len(set(types)) != 2:
        issues.append("the two question types must be different")
    if issues:
        return [], None, "; ".join(issues)
    return questions, None, None


def check_question(question: dict[str, Any], live_text: str, raw_kb_text: str) -> dict[str, Any]:
    norm_kb = normalise_text(raw_kb_text)
    details: list[dict[str, Any]] = []
    for item in question.get("supporting_quotes", []):
        quote = str(item.get("text", "")) if isinstance(item, dict) else ""
        norm_quote = normalise_text(quote)
        exact = bool(quote) and quote in live_text
        in_kb = bool(norm_quote) and norm_quote in norm_kb
        fuzzy = float(partial_ratio(norm_quote, norm_kb)) if norm_quote and norm_kb else 0.0
        details.append({"quote": quote, "exact_in_page": exact, "in_kb": in_kb, "quote_kb_fuzzy": fuzzy})
    found = sum(bool(row["in_kb"]) for row in details)
    label = "none" if not details or found == 0 else "all" if found == len(details) else "some"
    return {
        "quote_checks": details,
        "quote_in_page": all(row["exact_in_page"] for row in details) if details else False,
        "in_kb": label,
    }


def questions_prompt(page: dict[str, Any], text: str) -> tuple[str, bool]:
    truncated = len(text) > 15000
    content = text[:15000]
    message = (
        "Generate questions from this extracted live page. Use only the supplied text.\n"
        f"PAGE URL: {page['url']}\n"
        f"INPUT NOTE: {'Only the first 15,000 characters are supplied; the text is truncated.' if truncated else 'The complete extracted text is supplied.'}\n"
        f"PAGE TYPE: {page['page_type']}\n"
        f"PAGE TEXT:\n<<<\n{content}\n>>>\nReturn the required JSON only."
    )
    return message, truncated


def generate_for_page(page: dict[str, Any], text: str, prompt: str, prompt_hash: str) -> tuple[list[dict[str, Any]], str | None, str | None, bool]:
    message, truncated = questions_prompt(page, text)
    msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": message}]
    response = call_llm("judge_primary", msgs, json_mode=True, temperature=0.0, prompt_sha256=prompt_hash, timeout=180, log_messages=False)
    questions, skip_reason, error = validate_response(response.get("json"))
    if error:
        retry = message + "\n\nThe previous response was invalid: " + error + ". Correct it and return exactly two valid questions of different types, or a page-level skip with a reason."
        response = call_llm("judge_primary", [{"role": "system", "content": prompt}, {"role": "user", "content": retry}], json_mode=True, temperature=0.0, prompt_sha256=prompt_hash, timeout=180, log_messages=False)
        questions, skip_reason, error = validate_response(response.get("json"))
    return questions, skip_reason, error, truncated


def quote_lines(question: dict[str, Any], checks: list[dict[str, Any]]) -> str:
    points = []
    for point in question.get("keypoints", []):
        text = str(point.get("text", ""))
        points.append(("[decisive] " if point.get("decisive") else "") + text)
    points.extend(
        f"quote: {q['quote']} [page exact: {'yes' if q['exact_in_page'] else 'no'}; KB: {'yes' if q['in_kb'] else 'no'}; fuzzy: {q['quote_kb_fuzzy']:.1f}]"
        for q in checks
    )
    return "\n".join(points)


def main() -> None:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    sample = read_jsonl(V1_PAGES)
    v1_questions = read_jsonl(V1_QUESTIONS)
    urls = {row["url"] for row in sample}
    clean_rows = load_kb_rows(CLEAN_KB, urls)
    raw_rows = load_kb_rows(RAW_KB, urls)
    old_by_url = {row["url"]: row for row in sample}
    prompt = build_prompt()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    generated: list[dict[str, Any]] = []
    page_reports: list[dict[str, Any]] = []
    v1_rechecks: list[dict[str, Any]] = []

    for index, page in enumerate(sample):
        meta, html = fetch_page(page, index)
        status = meta["http_status"]
        live_text = ""
        if status == 200:
            live_text = trafilatura.extract(html, output_format="markdown", include_tables=True, include_comments=False, include_links=True) or ""
        old = old_by_url[page["url"]]
        ctext = compose(clean_rows.get(page["url"], []))
        rtext = compose(raw_rows.get(page["url"], []))
        old_share = old.get("kb_similarity")
        kb_live = ngram_coverage(ctext, live_text) if status == 200 else None
        live_kb = ngram_coverage(live_text, ctext) if status == 200 else None
        truncated = len(live_text) > 15000
        page_id = f"RP{int(page['page_index']):02d}"
        reason: str | None = None
        questions: list[dict[str, Any]] = []
        generation_error: str | None = None
        if status != 200:
            reason = f"unavailable_http_{status}"
        elif len(live_text) < 600:
            reason = "needs_js_rendering: extracted text is under 600 characters"
        else:
            questions, reason, generation_error, truncated = generate_for_page(page, live_text, prompt, prompt_hash)
            if generation_error:
                reason = "generation_failed: " + generation_error
        page_generated = 0
        for qindex, question in enumerate(questions, start=1):
            checks = check_question(question, live_text, rtext)
            record = {
                "id": f"{page_id}_{qindex}",
                "url": page["url"],
                "page_type": page["page_type"],
                "kb_similarity_label": "",
                "type": question["type"],
                "type_reason": question.get("type_reason", ""),
                "question": question["question"],
                "reference_answer": question["reference_answer"],
                "keypoints": question["keypoints"],
                "supporting_quotes": question["supporting_quotes"],
                "keypoints_with_quotes": quote_lines(question, checks["quote_checks"]),
                "quote_checks": checks["quote_checks"],
                "quote_in_page": checks["quote_in_page"],
                "in_kb": checks["in_kb"],
                "quote_kb_fuzzy": [item["quote_kb_fuzzy"] for item in checks["quote_checks"]],
                "truncated_for_questions": truncated,
                "teacher_prompt_sha256": prompt_hash,
            }
            generated.append(record)
            page_generated += 1

        page_reports.append({
            "page_id": page_id,
            "url": page["url"],
            "final_url": meta["final_url"],
            "page_type": page["page_type"],
            "http_status": status,
            "fetch_attempts": len(meta["attempts"]),
            "retry_count": max(0, len(meta["attempts"]) - 1),
            "extracted_characters": len(live_text),
            "truncated": "yes" if truncated else "no",
            "questions_generated": page_generated,
            "questions_skipped": 2 - page_generated,
            "teacher_skip_reason": reason or "",
            "generation_error": generation_error or "",
            "old_chunk_share": old_share,
            "kb_in_live": kb_live,
            "live_in_kb": live_kb,
        })

        for v1 in (q for q in v1_questions if q["url"] == page["url"]):
            checks = check_question(v1, live_text, rtext)
            v1_rechecks.append({
                "id": v1["id"],
                "url": v1["url"],
                "old_in_kb": v1.get("in_kb"),
                "new_in_kb": checks["in_kb"],
                "label_changed": v1.get("in_kb") != checks["in_kb"],
                "quote_checks": checks["quote_checks"],
            })

    with PAGE_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(page_reports[0]))
        writer.writeheader()
        writer.writerows(page_reports)
    QUESTIONS_JSONL.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated), encoding="utf-8")
    V1_RECHECKS.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in v1_rechecks), encoding="utf-8")
    write_summary(page_reports, generated, v1_rechecks)
    print(json.dumps({"pages": len(page_reports), "questions_generated": len(generated), "v1_in_kb_labels_changed": sum(r["label_changed"] for r in v1_rechecks)}))


def write_summary(pages: list[dict[str, Any]], questions: list[dict[str, Any]], v1_rechecks: list[dict[str, Any]]) -> None:
    status_counts = Counter(str(row["http_status"]) for row in pages)
    type_counts = Counter(row["type"] for row in questions)
    kb_counts = Counter(row["in_kb"] for row in questions)
    quote_failures = sum(not item["exact_in_page"] for row in questions for item in row["quote_checks"])
    lines = ["# Raw Web Page Question Pilot v2 Summary", "", "## Pages by HTTP status", ""]
    lines.extend(f"- HTTP {code}: {count}" for code, count in sorted(status_counts.items()))
    lines += ["", "## Page similarity (8-gram coverage)", "", "| Page ID | HTTP status | old_chunk_share | kb_in_live | live_in_kb |", "|---|---:|---:|---:|---:|"]
    for row in pages:
        fmt = lambda value: "NA" if value is None else f"{float(value):.4f}"
        lines.append(f"| {row['page_id']} | {row['http_status']} | {fmt(row['old_chunk_share'])} | {fmt(row['kb_in_live'])} | {fmt(row['live_in_kb'])} |")
    lines += ["", f"Questions generated: {len(questions)}", f"Questions skipped: {sum(int(row['questions_skipped']) for row in pages)}", "", "## Per-page skip reasons", ""]
    lines.extend(f"- {row['page_id']}: {row['teacher_skip_reason'] or 'none'}" for row in pages if row["questions_skipped"])
    lines += ["", "## Questions by type", ""]
    lines.extend(f"- {kind}: {type_counts.get(kind, 0)}" for kind in ("list_all", "paraphrased", "false_premise"))
    lines += ["", f"Quote-check failures (quote items): {quote_failures}", "", "## Question-level in_kb", ""]
    lines.extend(f"- {label}: {kb_counts.get(label, 0)}" for label in ("all", "some", "none"))
    lines += ["", f"V1 in_kb labels changed by the v2 check: {sum(row['label_changed'] for row in v1_rechecks)}", ""]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
