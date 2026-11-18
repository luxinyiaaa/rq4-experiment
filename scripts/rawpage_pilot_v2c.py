"""Recompute raw-page pilot quote checks with formatting-independent tokens."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import partial_ratio_alignment

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/rawpage_questions/pilot"
RAW_KB = ROOT / "data/raw/site_pages_rows.csv"
CLEAN_KB = ROOT / "data/processed/chunks_clean.csv"
V1_QUESTIONS = PILOT / "pilot_questions.jsonl"
V2_QUESTIONS = PILOT / "pilot_v2_questions.jsonl"
V2_PAGES = PILOT / "pilot_pages.csv"
V2_SUMMARY = PILOT / "pilot_v2_summary.md"
OUT_JSON = Path("/private/tmp/rawpage_pilot/v2c_workbook_data.json")
OUT_SUMMARY = PILOT / "pilot_v2c_summary.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def markdown_to_text(text: str) -> str:
    return re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text or "")


def markdown_to_text_with_source_map(text: str) -> tuple[str, list[int]]:
    """Replace links/images while retaining offsets back to the original text."""
    source = text or ""
    output: list[str] = []
    offsets: list[int] = []
    cursor = 0
    pattern = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
    for match in pattern.finditer(source):
        output.append(source[cursor:match.start()])
        offsets.extend(range(cursor, match.start()))
        label = match.group(1)
        output.append(label)
        label_start = match.start(1)
        offsets.extend(range(label_start, label_start + len(label)))
        cursor = match.end()
    output.append(source[cursor:])
    offsets.extend(range(cursor, len(source)))
    return "".join(output), offsets


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", markdown_to_text(text).lower())


def old_v1_normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def old_v2_normalise(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text or "")
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


def exact_v1(quote: str, kb: str) -> bool:
    q = old_v1_normalise(quote)
    return bool(q) and q in old_v1_normalise(kb)


def exact_v2(quote: str, kb: str) -> bool:
    q = old_v2_normalise(quote)
    return bool(q) and q in old_v2_normalise(kb)


def exact_v2c(quote: str, chunks: list[str]) -> bool:
    q = tokens(quote)
    if not q:
        return False
    joined = tokens("\n".join(chunks))
    # The second explicit join mode is retained as required, though tokenisation
    # makes a line break and no separator equivalent at the boundary.
    no_separator = tokens("".join(chunks))
    return contains_sequence(joined, q) or contains_sequence(no_separator, q)


def contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    width = len(needle)
    return any(haystack[i:i + width] == needle for i in range(len(haystack) - width + 1))


def fuzzy_context(quote: str, chunks: list[str], radius: int = 40) -> tuple[float, str]:
    source = "\n".join(chunks)
    display_text, source_map = markdown_to_text_with_source_map(source)
    source_tokens = list(re.finditer(r"[a-z0-9]+", display_text.lower()))
    query = " ".join(tokens(quote))
    target_parts: list[str] = []
    target_spans: list[tuple[int, int]] = []
    cursor = 0
    for match in source_tokens:
        if target_parts:
            cursor += 1  # separator in the token-normalised fuzzy target
        part = match.group(0)
        target_parts.append(part)
        target_spans.append((cursor, cursor + len(part)))
        cursor += len(part)
    target = " ".join(target_parts)
    if not query or not target:
        return 0.0, ""
    alignment = partial_ratio_alignment(query, target)
    if alignment is None:
        return 0.0, ""
    first = next((i for i, (_, end) in enumerate(target_spans) if end > alignment.dest_start), 0)
    last = next((i for i, (start, _) in enumerate(target_spans) if start >= alignment.dest_end), len(source_tokens))
    transformed_start = source_tokens[first].start() if first < len(source_tokens) else 0
    transformed_end = source_tokens[max(first, last - 1)].end() if source_tokens else 0
    source_start = source_map[transformed_start] if source_map and transformed_start < len(source_map) else 0
    source_end = source_map[transformed_end - 1] + 1 if source_map and transformed_end > 0 else source_start
    char_start = max(0, source_start - radius)
    char_end = min(len(source), source_end + radius)
    context = source[char_start:char_end].strip()
    return float(alignment.score), context


def word_ngrams(text: str, n: int = 8) -> list[tuple[str, ...]]:
    words = tokens(text)
    return [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]


def coverage(source: str, target: str, n: int = 8) -> float | None:
    source_grams = word_ngrams(source, n)
    if not source_grams:
        return None
    target_grams = set(word_ngrams(target, n))
    return sum(gram in target_grams for gram in source_grams) / len(source_grams)


def sort_chunk(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except (ValueError, TypeError):
        return 0, str(value)


def load_raw_kb(urls: set[str]) -> dict[str, list[str]]:
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with RAW_KB.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            url = row.get("url", "")
            if url in urls:
                rows[url].append(row)
    return {
        url: [row.get("content", "") or "" for row in sorted(items, key=lambda r: sort_chunk(r.get("chunk_number", "")))]
        for url, items in rows.items()
    }


def load_clean_kb(urls: set[str]) -> dict[str, str]:
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with CLEAN_KB.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("url", "") in urls:
                rows[row["url"]].append(row)
    return {
        url: "\n".join(row.get("content", "") or "" for row in sorted(items, key=lambda r: sort_chunk(r.get("chunk_number", ""))))
        for url, items in rows.items()
    }


def question_label(values: list[bool]) -> str:
    found = sum(values)
    if not values or found == 0:
        return "none"
    return "all" if found == len(values) else "some"


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    v1 = read_jsonl(V1_QUESTIONS)
    v2 = read_jsonl(V2_QUESTIONS)
    questions = [("v1", row) for row in v1] + [("v2", row) for row in v2]
    sampled_pages = read_jsonl(PILOT / "pilot_pages.jsonl")
    urls = {row["url"] for _, row in questions} | {row["url"] for row in sampled_pages}
    raw_by_url = load_raw_kb(urls)
    clean_by_url = load_clean_kb(urls)

    quote_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    for version, row in questions:
        chunks = raw_by_url.get(row["url"], [])
        raw_joined = "\n".join(chunks)
        quote_checks: list[dict[str, Any]] = []
        for item in row.get("supporting_quotes", []):
            quote = str(item.get("text", "")) if isinstance(item, dict) else ""
            # The original v1 workflow concatenated adjacent chunks with no
            # separator; v2 changed this to a newline between chunks.
            hit1 = exact_v1(quote, "".join(chunks))
            hit2 = exact_v2(quote, raw_joined)
            hit3 = exact_v2c(quote, chunks)
            fuzzy, context = fuzzy_context(quote, chunks)
            detail = {
                "id": row["id"],
                "question_version": version,
                "quote": quote,
                "in_kb_v1": hit1,
                "in_kb_v2": hit2,
                "in_kb_v2c": hit3,
                "quote_kb_fuzzy": round(fuzzy, 2),
                "kb_best_match": context,
            }
            quote_rows.append(detail)
            quote_checks.append(detail)
        question_rows.append({
            "id": row["id"],
            "question_version": version,
            "page_type": row.get("page_type", ""),
            "type": row.get("type", ""),
            "question": row.get("question", ""),
            "quote_count": len(quote_checks),
            "in_kb_v1": question_label([q["in_kb_v1"] for q in quote_checks]),
            "in_kb_v2": question_label([q["in_kb_v2"] for q in quote_checks]),
            "in_kb_v2c": question_label([q["in_kb_v2c"] for q in quote_checks]),
        })

    with V2_PAGES.open(newline="", encoding="utf-8") as handle:
        page_rows = list(csv.DictReader(handle))
    old_by_url = {r["url"]: r for r in sampled_pages}
    sim_rows: list[dict[str, Any]] = []
    for page in page_rows:
        url = page["url"]
        old = old_by_url[url]
        status = str(page.get("http_status", ""))
        live_path = ROOT / "data/raw_web/pilot_v2" / f"{page['page_id']}.html"
        live_text = ""
        if status == "200" and live_path.exists():
            import trafilatura
            live_text = trafilatura.extract(live_path.read_bytes(), output_format="markdown", include_tables=True, include_comments=False, include_links=True) or ""
        clean_text = clean_by_url.get(url, "")
        kb_live = coverage(clean_text, live_text) if status == "200" else None
        live_kb = coverage(live_text, clean_text) if status == "200" else None
        sim_rows.append({
            "page_id": page["page_id"],
            "http_status": status,
            "old_chunk_share": old.get("kb_similarity"),
            "v2_kb_in_live": page.get("kb_in_live"),
            "v2_live_in_kb": page.get("live_in_kb"),
            "v2c_kb_in_live": kb_live,
            "v2c_live_in_kb": live_kb,
        })
    return quote_rows, question_rows, sim_rows


def write_summary(quote_rows: list[dict[str, Any]], question_rows: list[dict[str, Any]], sim_rows: list[dict[str, Any]]) -> None:
    counts = {method: Counter(row[f"in_kb_{method}"] for row in question_rows) for method in ("v1", "v2", "v2c")}
    lines = [
        "# Raw Web Page Pilot v2c Summary",
        "",
        "Question-level KB quote-match labels across all 24 questions (14 v1 + 10 v2):",
        "",
        "| Check | all | some | none |",
        "|---|---:|---:|---:|",
    ]
    for method in ("v1", "v2", "v2c"):
        lines.append(f"| {method} | {counts[method].get('all', 0)} | {counts[method].get('some', 0)} | {counts[method].get('none', 0)} |")
    lines += ["", "## Quotes not found by v2c", ""]
    misses = [row for row in quote_rows if not row["in_kb_v2c"]]
    if misses:
        lines += ["| Question | Version | Quote | Fuzzy score | Best KB match (context) |", "|---|---|---|---:|---|"]
        for row in misses:
            fields = [row["id"], row["question_version"], row["quote"], f"{row['quote_kb_fuzzy']:.2f}", row["kb_best_match"]]
            lines.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in fields) + " |")
    else:
        lines.append("No quote was unmatched.")
    lines += ["", "## Page similarity (8-token n-gram coverage)", "", "| Page | HTTP | old_chunk_share | v2 kb_in_live | v2c kb_in_live | v2 live_in_kb | v2c live_in_kb |", "|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "NA" if x is None or x == "" else f"{float(x):.4f}"
    for row in sim_rows:
        lines.append("| " + " | ".join([row["page_id"], str(row["http_status"]), fmt(row["old_chunk_share"]), fmt(row["v2_kb_in_live"]), fmt(row["v2c_kb_in_live"]), fmt(row["v2_live_in_kb"]), fmt(row["v2c_live_in_kb"])]) + " |")
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    quote_rows, question_rows, sim_rows = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"quotes": quote_rows, "questions": question_rows, "similarity": sim_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(quote_rows, question_rows, sim_rows)
    print(json.dumps({"questions": len(question_rows), "quotes": len(quote_rows), "unmatched_v2c": sum(not r["in_kb_v2c"] for r in quote_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
