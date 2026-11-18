"""Analyse the browser-rendered v2b pilot pages and conditionally generate questions."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import trafilatura

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from rawpage_pilot_v2 import (  # noqa: E402
    QUESTIONS_JSONL as _unused_questions_path,
    build_prompt,
    check_question,
    compose,
    generate_for_page,
    load_kb_rows,
    ngram_coverage,
    read_jsonl,
)

PILOT = ROOT / "outputs/rawpage_questions/pilot"
BROWSER_DIR = ROOT / "data/raw_web/pilot_v2"
PAGES = ROOT / "data/processed/pages.csv"
CLEAN_KB = ROOT / "data/processed/chunks_clean.csv"
RAW_KB = ROOT / "data/raw/site_pages_rows.csv"
V2_PAGES_CSV = PILOT / "pilot_pages.csv"
V2_QUESTIONS = PILOT / "pilot_v2_questions.jsonl"
V2_QUESTIONS_OUT = PILOT / "pilot_v2b_questions.jsonl"
SUMMARY = PILOT / "pilot_v2b_summary.md"
TARGETS = ("RP04", "RP05", "RP07", "RP08")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def collection_name(url: str) -> str:
    path = urlsplit(url).path
    if "/collection/" not in path:
        return "(no collection segment)"
    remainder = path.split("/collection/", 1)[1]
    segment = remainder.split("/", 1)[0]
    return segment or "(no collection segment)"


def definition_length(text: str) -> int | None:
    """Return normalized character length for a Definition field, if present."""
    patterns = (
        r"(?ims)^\s*\|\s*definition\s*\|\s*([^|\n]+)",
        r"(?im)^\s*\*\*definition\*\*\s*\|\s*(.+?)\s*$",
        r"(?im)^\s*definition\s*\|\s*(.+?)\s*$",
        r"(?im)^\s*\*\*definition:\*\*\s*(.+?)\s*$",
        r"(?im)^\s*definition:\s*(.+?)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip()
            return len(value)
    return None


def quote_lines(question: dict[str, Any], checks: list[dict[str, Any]]) -> str:
    lines = []
    for point in question.get("keypoints", []):
        lines.append(("[decisive] " if point.get("decisive") else "") + str(point.get("text", "")))
    lines.extend(
        f"quote: {item['quote']} [page exact: {'yes' if item['exact_in_page'] else 'no'}; KB: {'yes' if item['in_kb'] else 'no'}; fuzzy: {item['quote_kb_fuzzy']:.1f}]"
        for item in checks
    )
    return "\n".join(lines)


def main() -> None:
    v2_pages = {row["page_id"]: row for row in load_csv(V2_PAGES_CSV)}
    v2_metrics = {row["page_id"]: row for row in v2_pages.values()}
    clean_rows = load_kb_rows(CLEAN_KB, {row["url"] for row in v2_pages.values()})
    raw_rows = load_kb_rows(RAW_KB, {row["url"] for row in v2_pages.values()})
    question_prompt = build_prompt()
    prompt_hash = hashlib.sha256(question_prompt.encode("utf-8")).hexdigest()
    browser_table: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []

    for page_id in TARGETS:
        previous = v2_pages[page_id]
        metadata = json.loads((BROWSER_DIR / f"{page_id}_browser.json").read_text(encoding="utf-8"))
        html = (BROWSER_DIR / f"{page_id}_browser.html").read_text(encoding="utf-8")
        live_text = trafilatura.extract(html, output_format="markdown", include_tables=True, include_comments=False, include_links=True) or ""
        status = metadata.get("http_status")
        clean_text = compose(clean_rows.get(previous["url"], []))
        raw_text = compose(raw_rows.get(previous["url"], []))
        browser_kb_in_live = ngram_coverage(clean_text, live_text) if status == 200 else None
        browser_live_in_kb = ngram_coverage(live_text, clean_text) if status == 200 else None
        v2_live_in_kb = float(previous["live_in_kb"]) if previous.get("live_in_kb") not in ("", "NA", None) else None
        v2_kb_in_live = float(previous["kb_in_live"]) if previous.get("kb_in_live") not in ("", "NA", None) else None
        improved = (
            status == 200
            and v2_live_in_kb is not None
            and browser_live_in_kb is not None
            and browser_live_in_kb - v2_live_in_kb >= 0.2
        )
        newly_available = status == 200 and str(previous.get("http_status")) != "200"
        generate = bool(newly_available or improved)
        generated_for_page: list[dict[str, Any]] = []
        skip_reason = ""

        if generate:
            source_page = {"url": previous["url"], "page_type": previous["page_type"]}
            if len(live_text) < 600:
                skip_reason = "Browser text is under 600 characters; the v2 workflow marks it as needing JS rendering."
            else:
                generated_for_page, skip_reason, generation_error, truncated = generate_for_page(source_page, live_text, question_prompt, prompt_hash)
                if generation_error:
                    skip_reason = "generation_failed: " + generation_error
                if generated_for_page:
                    for index, question in enumerate(generated_for_page, start=1):
                        checks = check_question(question, live_text, raw_text)
                        questions.append({
                            "id": f"{page_id}_B{index}",
                            "url": previous["url"],
                            "page_type": previous["page_type"],
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
                        })
        else:
            if status != 200:
                skip_reason = "No browser page became available; browser fetch returned a non-200 status."
            else:
                skip_reason = "The browser live_in_kb improvement was below 0.2 and the page was already available in v2."

        browser_table.append({
            "page_id": page_id,
            "v2_status": previous.get("http_status"),
            "browser_status": status,
            "browser_extracted_characters": len(live_text),
            "v2_kb_in_live": v2_kb_in_live,
            "browser_kb_in_live": browser_kb_in_live,
            "v2_live_in_kb": v2_live_in_kb,
            "browser_live_in_kb": browser_live_in_kb,
            "questions_generated": len(generated_for_page),
            "skip_reason": skip_reason,
        })

    V2_QUESTIONS_OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions), encoding="utf-8")
    v2_question_count = len(read_jsonl(V2_QUESTIONS))
    eligible = [row for row in load_csv(PAGES) if row.get("eligible", "").strip().lower() in {"true", "1", "yes"}]
    eligible_type_counts = Counter(row["page_type"] for row in eligible)
    vocab = [row for row in eligible if row["page_type"] == "vocabulary"]
    collection_counts = Counter(collection_name(row["url"]) for row in vocab)
    non_exv = [row for row in vocab if collection_name(row["url"]) not in {"EXV", "(no collection segment)"}]
    sampled = random.Random(42).sample(non_exv, min(5, len(non_exv)))
    all_clean = load_kb_rows(CLEAN_KB, {row["url"] for row in sampled})
    def_rows = [
        {"url": row["url"], "collection": collection_name(row["url"]), "definition_characters": definition_length(compose(all_clean.get(row["url"], [])))}
        for row in sampled
    ]
    summary = render_summary(browser_table, eligible_type_counts, collection_counts, def_rows, questions, v2_question_count)
    SUMMARY.write_text(summary, encoding="utf-8")
    print(json.dumps({"eligible_pages": len(eligible), "browser_pages": len(browser_table), "questions_generated": len(questions), "eligible_vocabulary_pages_sampled": len(def_rows)}))


def render_summary(browser_table: list[dict[str, Any]], type_counts: Counter[str], collection_counts: Counter[str], definition_rows: list[dict[str, Any]], questions: list[dict[str, Any]], v2_question_count: int) -> str:
    fmt = lambda x: "NA" if x is None else f"{float(x):.4f}"
    lines = [
        "# Raw Web Page Question Pilot v2b Summary", "",
        "## Browser fetch and text comparison", "",
        "| Page | V2 status | Browser status | Browser text chars | V2 kb_in_live | Browser kb_in_live | V2 live_in_kb | Browser live_in_kb |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in browser_table:
        lines.append(f"| {row['page_id']} | {row['v2_status']} | {row['browser_status']} | {row['browser_extracted_characters']} | {fmt(row['v2_kb_in_live'])} | {fmt(row['browser_kb_in_live'])} | {fmt(row['v2_live_in_kb'])} | {fmt(row['browser_live_in_kb'])} |")
    lines += ["", "## Eligible pages by page type", ""]
    lines.extend(f"- {kind}: {count}" for kind, count in sorted(type_counts.items()))
    lines += ["", "## Eligible vocabulary pages by collection", ""]
    lines.extend(f"- {collection}: {count}" for collection, count in sorted(collection_counts.items()))
    lines += ["", "## Non-EXV vocabulary Definition lengths", "", f"Eligible pages in named collections other than EXV: {sum(collection != 'EXV' and collection != '(no collection segment)' for collection in collection_counts.elements())}", f"Pages sampled (seed 42, maximum 5): {len(definition_rows)}", ""]
    if definition_rows:
        lines += ["| URL | Collection | Definition characters |", "|---|---|---:|"]
        lines.extend(f"| {row['url']} | {row['collection']} | {row['definition_characters'] if row['definition_characters'] is not None else 'not found'} |" for row in definition_rows)
    else:
        lines.append("No eligible vocabulary pages had a named collection other than EXV; no Definition-length sample was available.")
    lines += ["", "## Questions generated under the browser-page trigger", "", f"V2 questions: {v2_question_count}", f"V2b questions generated: {len(questions)}", ""]
    if questions:
        for question in questions:
            lines += [f"### {question['id']} ({question['type']})", "", f"Question: {question['question']}", "", f"Reference answer: {question['reference_answer']}", "", f"Keypoints and quotes:\n\n{question['keypoints_with_quotes']}", ""]
    else:
        lines.append("No page met the availability or +0.2 live_in_kb trigger; no teacher calls were made.")
    lines += ["", "## Pages without generated questions", ""]
    for row in browser_table:
        if row["questions_generated"] == 0:
            lines.append(f"- {row['page_id']}: {row['skip_reason']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
