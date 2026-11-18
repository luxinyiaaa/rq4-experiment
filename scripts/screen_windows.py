"""Judge evidence-window quality and replace unsuitable windows.

Example:
    python scripts/screen_windows.py \
      --windows data/processed/windows.jsonl \
      --pages data/processed/pages.csv \
      --chunks data/processed/chunks_clean.csv \
      --out-windows data/processed/windows_screened.jsonl \
      --review-csv data/processed/window_replacements.csv

Each judge request uses the configured ``judge`` role through ``rq4.llm``;
requests are cached and logged by that shared client. Replacements stay in the
same type where possible; dataset-record shortfalls may be filled with
about-science windows while preserving all per-site caps.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rq4.llm import call_llm
from scripts.sample_windows import QUOTAS, has_concrete_fact, page_windows

FINAL_COUNTS = {"dataset_record": 23, "about_science": 22, "vocabulary": 5}

REVIEW_FIELDS = [
    "replaced_window_id", "replaced_url", "page_type", "replacement_reason",
    "candidate_page_type", "candidate_window_id", "candidate_url",
    "candidate_decision", "candidate_reason",
    "selected_as_replacement", "candidate_length_chars", "candidate_length_rule",
    "replaced_text", "candidate_text",
]

JUDGE_INSTRUCTIONS = """You are screening a short evidence window for use in a scientific knowledge-base question-answering experiment.
Return JSON with exactly two keys: decision and reason.
decision must be either suitable or unsuitable.
Mark unsuitable when the window is mainly coordinates, IDs, a list of links, code, a fragment of a table, raw JSON or API output, a search result or navigation list, or image-caption text; also mark it unsuitable when it does not communicate at least one complete, readable fact. The workflow separately checks whether a candidate shares a dataset title with a selected window or has normalized text similarity greater than 0.7; that programmatic check may reject a candidate before it is sent to you.
Mark suitable when a reader can understand at least one complete factual statement or a coherent, usable piece of dataset metadata from the text.
The reason must be one concise sentence explaining the decision. Judge only the supplied text; do not infer missing context."""


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _dataset_title(window: dict[str, Any]) -> str:
    """Extract an explicit title/name for conservative dataset duplicate checks."""
    text = str(window.get("text", ""))
    patterns = (
        r"(?im)^\s*(?:dataset\s+)?(?:title|name)\s*[:|]\s*([^\n|]+)",
        r"(?im)^\s*#\s+(.+?)\s*$",
        r'"(?:title|name)"\s*:\s*"([^"\n]+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            title = _normalized_text(match.group(1)).strip(" .#*_-`")
            if len(title) >= 8:
                return title
    return ""


def _duplicate_reason(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> str | None:
    candidate_title = _dataset_title(candidate) if candidate.get("page_type") == "dataset_record" else ""
    for item in selected:
        if candidate_title and item.get("page_type") == "dataset_record":
            other_title = _dataset_title(item)
            if other_title and candidate_title == other_title:
                return f"Same dataset title as selected window {item['window_id']}: {candidate_title}."
        candidate_text = _normalized_text(str(candidate.get("text", "")))
        selected_text = _normalized_text(str(item.get("text", "")))
        # Real windows are at least 600 characters; avoid treating tiny test
        # strings or boilerplate labels as meaningful near-duplicates.
        if min(len(candidate_text), len(selected_text)) >= 600:
            ratio = difflib.SequenceMatcher(
                None, candidate_text, selected_text, autojunk=False,
            ).ratio()
            if ratio > 0.7:
                return f"Text similarity {ratio:.3f} exceeds 0.7 against selected window {item['window_id']}."
    return None


def judge_window(window: dict[str, Any], llm_call: Callable[..., dict[str, Any]] | None = None) -> dict[str, str]:
    """Ask the configured judge to classify one window and validate its JSON."""
    call = llm_call or call_llm
    response = call(
        "judge",
        [
            {"role": "system", "content": JUDGE_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Page type: {window['page_type']}\n"
                    f"URL: {window['url']}\n\n"
                    f"Window text:\n{window['text']}"
                ),
            },
        ],
        json_mode=True,
        temperature=0.0,
        extra_body={"enable_thinking": False},
    )
    result = response.get("json")
    if not isinstance(result, dict):
        raise ValueError("Judge response must be a JSON object")
    decision = str(result.get("decision", "")).strip().lower()
    reason = str(result.get("reason", "")).strip()
    if decision not in {"suitable", "unsuitable"} or not reason:
        raise ValueError(f"Judge returned invalid decision/reason: {result!r}")
    return {"decision": decision, "reason": reason}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _short_window(frame: pd.DataFrame) -> dict[str, Any] | None:
    """Last-resort approximate-length window, bounded by the 600-char eligibility minimum."""
    rows = frame.sort_values("chunk_number", kind="stable")[["id", "content"]].to_dict("records")
    choices = []
    for start in range(len(rows)):
        for count in range(3, 6):
            part = rows[start : start + count]
            if len(part) != count:
                continue
            text = "".join(str(row["content"]) for row in part if str(row["content"]).strip())
            if 600 <= len(text) < 900 and has_concrete_fact(text):
                choices.append({
                    "chunk_ids": [str(row["id"]) for row in part],
                    "text": text,
                    "length": len(text),
                })
    if not choices:
        return None
    return min(choices, key=lambda item: (900 - item["length"], item["chunk_ids"][0]))


def build_candidates(
    pages_path: Path, chunks_path: Path, used_urls: set[str], seed: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    pages = pd.read_csv(pages_path, dtype={"url": str, "domain": str, "page_type": str})
    chunks = pd.read_csv(chunks_path, dtype={"url": str, "id": str, "content": str}, keep_default_na=False)
    eligible = pages[
        (pages["eligible"].astype(str).str.lower() == "true")
        & pages["page_type"].isin(QUOTAS)
    ]
    chunks_by_url = {url: frame for url, frame in chunks.groupby("url", sort=False)}
    candidates = {kind: [] for kind in QUOTAS}
    short_candidates = {kind: [] for kind in QUOTAS}
    for row in eligible.itertuples(index=False):
        if row.url in used_urls:
            continue
        frame = chunks_by_url.get(row.url)
        if frame is None:
            continue
        options = page_windows(frame)
        if options:
            # Keep alternate consecutive windows from the same page. A page's
            # first window can be navigation or a machine-readable fragment.
            for window in options:
                candidates[row.page_type].append({
                    "url": row.url, "domain": row.domain, "page_type": row.page_type,
                    "chunk_ids": window["chunk_ids"], "text": window["text"],
                    "length": window["length"], "length_rule": "900–1,500 chars",
                })
        short_window = _short_window(frame)
        if short_window is not None:
            short_candidates[row.page_type].append({
                "url": row.url, "domain": row.domain, "page_type": row.page_type,
                "chunk_ids": short_window["chunk_ids"], "text": short_window["text"],
                "length": short_window["length"],
                "length_rule": "600–899 chars (short fallback; manual review)",
            })
    rng = random.Random(seed)
    for pool in candidates.values():
        # Randomize page order while keeping each page's alternative windows
        # adjacent and ordered by the sampler's best-length ranking.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for candidate in pool:
            grouped.setdefault(candidate["url"], []).append(candidate)
        urls = list(grouped)
        rng.shuffle(urls)
        pool[:] = [candidate for url in urls for candidate in grouped[url]]
    for pool in short_candidates.values():
        rng.shuffle(pool)
    return candidates, short_candidates


def _site_cap(page_type: str) -> int:
    return 8 if page_type == "dataset_record" else 5


def _next_candidate(
    pool: list[dict[str, Any]], selected: list[dict[str, Any]], page_type: str,
    used_urls: set[str],
) -> dict[str, Any] | None:
    site_counts = Counter(
        (item["page_type"], item["domain"]) for item in selected
    )
    for index, item in enumerate(pool):
        if (item["url"] not in used_urls
                and site_counts[(page_type, item["domain"])] < _site_cap(page_type)):
            return pool.pop(index)
    return None


def screen_windows(
    windows: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    short_candidates: dict[str, list[dict[str, Any]]] | None = None,
    llm_call: Callable[..., dict[str, Any]] | None = None,
    final_counts: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Screen initial windows and replace every unsuitable one in its category."""
    if len(windows) != sum(QUOTAS.values()):
        raise ValueError(f"Expected {sum(QUOTAS.values())} input windows; got {len(windows)}")
    next_number = max(
        (int(str(item["window_id"]).lstrip("W")) for item in windows),
        default=0,
    ) + 1
    selected: list[dict[str, Any]] = []
    rejected: list[tuple[dict[str, Any], str]] = []
    for index, item in enumerate(windows, start=1):
        print(f"Judging initial window {index}/{len(windows)}: {item['window_id']}", flush=True)
        duplicate = _duplicate_reason(item, selected)
        judgement = (
            {"decision": "unsuitable", "reason": duplicate}
            if duplicate else judge_window(item, llm_call)
        )
        if judgement["decision"] == "suitable":
            selected.append({**item, "screening_decision": "suitable", "screening_reason": judgement["reason"]})
        else:
            rejected.append((item, judgement["reason"]))

    replacement_rows: list[dict[str, Any]] = []
    unfilled: list[dict[str, Any]] = []
    used_urls = {item["url"] for item in windows}
    short_candidates = short_candidates or {kind: [] for kind in QUOTAS}

    def replace_one(original: dict[str, Any], reason: str, target_type: str) -> bool:
        nonlocal next_number
        while True:
            candidate = _next_candidate(candidates[target_type], selected, target_type, used_urls)
            if candidate is None:
                candidate = _next_candidate(short_candidates[target_type], selected, target_type, used_urls)
            if candidate is None:
                return False
            candidate_id = f"W{next_number:03d}"
            next_number += 1
            candidate = {"window_id": candidate_id, **candidate}
            print(f"Judging replacement candidate {candidate_id} for {original['window_id']} ({target_type})", flush=True)
            duplicate = _duplicate_reason(candidate, selected)
            judgement = (
                {"decision": "unsuitable", "reason": duplicate}
                if duplicate else judge_window(candidate, llm_call)
            )
            keep = judgement["decision"] == "suitable"
            replacement_rows.append({
                "replaced_window_id": original["window_id"],
                "replaced_url": original["url"],
                "page_type": original["page_type"],
                "replacement_reason": reason,
                "candidate_page_type": target_type,
                "candidate_window_id": candidate_id,
                "candidate_url": candidate["url"],
                "candidate_decision": judgement["decision"],
                "candidate_reason": judgement["reason"],
                "selected_as_replacement": keep,
                "candidate_length_chars": len(candidate["text"]),
                "candidate_length_rule": candidate.get("length_rule", "unspecified"),
                "replaced_text": original["text"],
                "candidate_text": candidate["text"],
            })
            if keep:
                selected.append({
                    **candidate,
                    "screening_decision": "suitable",
                    "screening_reason": judgement["reason"],
                })
                used_urls.add(candidate["url"])
                return True

    for original, original_reason in rejected:
        if not replace_one(original, original_reason, original["page_type"]):
            unfilled.append(original)

    final_counts = final_counts or FINAL_COUNTS
    current_counts = Counter(item["page_type"] for item in selected)
    about_science_needed = max(0, final_counts.get("about_science", 0) - current_counts["about_science"])
    remaining_unfilled = []
    rejected_reasons = {item["window_id"]: reason for item, reason in rejected}
    for original in unfilled:
        if original["page_type"] == "dataset_record" and about_science_needed > 0:
            if replace_one(original, rejected_reasons[original["window_id"]], "about_science"):
                about_science_needed -= 1
                continue
        remaining_unfilled.append(original)
    unfilled = remaining_unfilled

    for original in unfilled:
        replacement_rows.append({
            "replaced_window_id": original["window_id"],
            "replaced_url": original["url"],
            "page_type": original["page_type"],
            "replacement_reason": rejected_reasons[original["window_id"]],
            "candidate_page_type": "",
            "candidate_window_id": "",
            "candidate_url": "",
            "candidate_decision": "no_eligible_candidate",
            "candidate_reason": "No unused page had a suitable window within its page-type site cap.",
            "selected_as_replacement": False,
            "candidate_length_chars": "",
            "candidate_length_rule": "",
            "replaced_text": original["text"],
            "candidate_text": "",
        })

    return selected, replacement_rows, unfilled


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def append_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Append a targeted replacement audit without rewriting prior decisions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerows(records)


def replace_specific_windows(
    windows: list[dict[str, Any]],
    target_ids: list[str],
    candidates: dict[str, list[dict[str, Any]]],
    short_candidates: dict[str, list[dict[str, Any]]],
    llm_call: Callable[..., dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace only named windows, preserving all other records and their order."""
    by_id = {item["window_id"]: item for item in windows}
    missing = [window_id for window_id in target_ids if window_id not in by_id]
    if missing:
        raise ValueError(f"Target windows not found: {missing}")
    selected = [dict(item) for item in windows if item["window_id"] not in target_ids]
    updated = {item["window_id"]: dict(item) for item in windows}
    used_urls = {item["url"] for item in windows}
    next_number = max(int(str(item["window_id"]).lstrip("W")) for item in windows) + 1
    rows: list[dict[str, Any]] = []

    for target_id in target_ids:
        original = by_id[target_id]
        kind = original["page_type"]
        while True:
            candidate = _next_candidate(candidates[kind], selected, kind, used_urls)
            pool_name = "900–1,500 chars"
            if candidate is None:
                candidate = _next_candidate(short_candidates[kind], selected, kind, used_urls)
                pool_name = "600–899 chars (short fallback; manual review)"
            if candidate is None:
                raise RuntimeError(f"No same-category candidate remains for {target_id}")
            candidate_id = f"W{next_number:03d}"
            next_number += 1
            candidate = {"window_id": candidate_id, **candidate}
            duplicate = _duplicate_reason(candidate, selected)
            if duplicate:
                judgement = {"decision": "unsuitable", "reason": duplicate}
            else:
                print(f"Judging replacement candidate {candidate_id} for {target_id} ({kind})", flush=True)
                judgement = judge_window(candidate, llm_call)
            keep = judgement["decision"] == "suitable"
            rows.append({
                "replaced_window_id": target_id,
                "replaced_url": original["url"],
                "page_type": kind,
                "replacement_reason": "User-identified missed screening issue; replace under expanded suitability and duplicate rules.",
                "candidate_page_type": kind,
                "candidate_window_id": candidate_id,
                "candidate_url": candidate["url"],
                "candidate_decision": judgement["decision"],
                "candidate_reason": judgement["reason"],
                "selected_as_replacement": keep,
                "candidate_length_chars": len(candidate["text"]),
                "candidate_length_rule": candidate.get("length_rule", pool_name),
                "replaced_text": original["text"],
                "candidate_text": candidate["text"],
            })
            if not keep:
                continue
            # Retain the stable target ID used by question/window references.
            replacement = {
                **candidate,
                "window_id": target_id,
                "screening_decision": "suitable",
                "screening_reason": judgement["reason"],
            }
            selected.append(replacement)
            used_urls.add(candidate["url"])
            updated[target_id] = replacement
            break

    return [updated[item["window_id"]] for item in windows], rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", required=True, type=Path, help="input windows.jsonl")
    parser.add_argument("--pages", required=True, type=Path, help="cleaned pages.csv")
    parser.add_argument("--chunks", required=True, type=Path, help="cleaned chunks_clean.csv")
    parser.add_argument("--out-windows", required=True, type=Path, help="output suitable windows JSONL")
    parser.add_argument("--review-csv", required=True, type=Path, help="replacement audit CSV for manual review")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace-ids", nargs="*", help="replace only these existing window IDs; leave all others unchanged")
    args = parser.parse_args()

    windows = read_jsonl(args.windows)
    candidates, short_candidates = build_candidates(
        args.pages,
        args.chunks,
        {item["url"] for item in windows},
        args.seed,
    )
    if args.replace_ids:
        screened, replacements = replace_specific_windows(
            windows, args.replace_ids, candidates, short_candidates
        )
        append_review_csv(args.review_csv, replacements)
        write_jsonl(args.out_windows, screened)
        print(f"Targeted replacements: {len(args.replace_ids)}; other windows preserved: {len(windows)-len(args.replace_ids)}")
        print(f"Appended replacement attempts: {len(replacements)} to {args.review_csv}")
        return

    screened, replacements, unfilled = screen_windows(windows, candidates, short_candidates)
    complete = not unfilled and dict(Counter(item["page_type"] for item in screened)) == FINAL_COUNTS
    if complete:
        write_jsonl(args.out_windows, screened)
        written_windows = args.out_windows
    else:
        partial_path = args.out_windows.with_name(
            f"{args.out_windows.stem}.partial{args.out_windows.suffix}"
        )
        write_jsonl(partial_path, screened)
        written_windows = partial_path
    write_review_csv(args.review_csv, replacements)
    print(f"Suitable windows: {len(screened)} / {sum(FINAL_COUNTS.values())}")
    print("Page counts by type:")
    for page_type, count in Counter(item["page_type"] for item in screened).items():
        print(f"  {page_type}: {count}")
    print(f"Replacement attempts: {len(replacements)}")
    print(f"Replaced original windows: {len({row['replaced_window_id'] for row in replacements})}")
    print(f"Unfilled replacements: {len(unfilled)}")
    print(f"Wrote screened windows: {written_windows}")
    print(f"Wrote replacement review table: {args.review_csv}")


if __name__ == "__main__":
    main()
