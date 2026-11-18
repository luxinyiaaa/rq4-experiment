"""Apply round-two review and replace rejected gold pages using the seeded page sampler."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.assign_types import _read_jsonl, write_jsonl
from scripts.build_index import DEFAULT_OUTDIR, _same_dataset, _text_similarity, _url_template, load_index
from scripts.check_questions import (
    CONFLICT_PROMPT, GENERATOR_PROMPT, MODIFIED_ANSWER, MODIFIED_KEYPOINTS,
    MODIFIED_QUESTION, MODIFIED_QUOTES, REVIEW_DECISION, _edited_version,
    _parse_modified_keypoints, _read_review_sheet, _review_rows,
    find_duplicate_titles, load_conflict_prompt, load_generator_systems,
    mark_near_duplicates, page_text_catalog, page_title_catalog,
    page_usage_violations, partner_gold_title_violations,
    run_checks, run_edited_question_checks, run_new_checks,
    run_template_answer_checks, template_answer_check,
)
from scripts.generate_questions import CHUNKS_PATH, PROMPT_PATH, generate_one, load_chunk_lookup, load_teacher_prompt
from scripts.generate_questions import duplicate_title_note
from scripts.prepare_main_round2 import apply_edited_quotes
from scripts.sample_windows import MAX_WINDOW_CHARS, MIN_WINDOW_CHARS, QUOTAS, page_windows

QUESTIONS_INPUT = PROJECT_ROOT / "outputs" / "main" / "questions_main_round2.jsonl"
REVIEW_INPUT = PROJECT_ROOT / "outputs" / "main" / "review_sheet_main_round2.xlsx"
PILOT_INPUT = PROJECT_ROOT / "outputs" / "pilot" / "questions_reviewed.jsonl"
QUESTIONS_OUTPUT = PROJECT_ROOT / "outputs" / "main" / "questions_main_round3.jsonl"
REVIEW_OUTPUT = PROJECT_ROOT / "outputs" / "main" / "review_sheet_main_round3.xlsx"
REPLACEMENTS_CSV = PROJECT_ROOT / "outputs" / "main" / "window_replacements_main.csv"
PAGES_PATH = PROJECT_ROOT / "data" / "processed" / "pages.csv"
CHUNKS_CSV = PROJECT_ROOT / "data" / "processed" / "chunks_clean.csv"
SAMPLED_WINDOWS = PROJECT_ROOT / "data" / "processed" / "windows.jsonl"
ROUND2_IDS = {"W010", "W020", "W026", "W034", "W047", "W069", "W080"}
ROUND2_REVIEW_IDS = {
    "W009", "W010", "W015", "W016", "W020", "W022", "W023", "W026",
    "W034", "W035", "W037", "W041", "W047", "W048", "W059", "W061",
    "W062", "W063", "W064", "W068", "W069", "W073", "W080",
}
DECISION_COUNTS = Counter({"accept": 13, "edit": 3, "reject": 7})
SITE_CAP = {"dataset_record": 8, "about_science": 5, "vocabulary": 5}
REPLACEMENT_FIELDS = [
    "question_id", "old_window_id", "old_url", "new_window_id", "new_url",
    "attempt_number", "candidate_source", "reason",
]


def _normalized(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _write_replacement_log(path: Path, rows: list[dict[str, str]], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prior: list[dict[str, str]] = []
    if append and path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                old_reason = row.get("reason", "")
                old_candidate = row.get("candidate_source", "") or "round2_distractor_candidate_pool"
                if old_candidate.startswith("seed42_eligible_sample_order_") or old_candidate == "workflow_summary":
                    continue
                try:
                    attempt = row.get("attempt_number", "") or re.search(r"(?:SKIP )?attempt (\d+)", old_reason, re.I).group(1)
                except (AttributeError, TypeError):
                    attempt = ""
                prior.append({
                    "question_id": row.get("question_id", ""),
                    "old_window_id": row.get("old_window_id", ""),
                    "old_url": row.get("old_url", ""),
                    "new_window_id": row.get("new_window_id", ""),
                    "new_url": row.get("new_url", ""),
                    "attempt_number": attempt,
                    "candidate_source": old_candidate,
                    "reason": old_reason,
                })
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPLACEMENT_FIELDS)
        writer.writeheader()
        writer.writerows([*prior, *rows])


def _sample_candidate_pages() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the recorded seed-42 replacement sequence beginning with W051.

    The replacement ledger was produced by the original eligible-page sampler
    after the first 50 pages. Rebuild chunk mappings from the cleaned chunks so
    the candidate source remains ordinary sampled windows, never CAND records.
    """
    pages = pd.read_csv(PAGES_PATH, dtype={"url": str, "domain": str, "page_type": str})
    chunks = pd.read_csv(CHUNKS_CSV, dtype={"url": str, "id": str, "content": str}, keep_default_na=False)
    page_catalog = {str(row.url): row for row in pages.itertuples(index=False)}
    chunks_by_url = {url: frame for url, frame in chunks.groupby("url", sort=False)}
    text_to_chunks: dict[tuple[str, str], list[str]] = {}
    for url, frame in chunks_by_url.items():
        for option in page_windows(frame):
            text_to_chunks[(str(url), str(option["text"]))] = list(option["chunk_ids"])
    ledger_path = PROJECT_ROOT / "data" / "processed" / "window_replacements.csv"
    with ledger_path.open(encoding="utf-8-sig", newline="") as stream:
        ledger = list(csv.DictReader(stream))
    ledger.sort(key=lambda row: int(str(row["candidate_window_id"]).lstrip("W")))
    candidate_numbers = [int(str(row["candidate_window_id"]).lstrip("W")) for row in ledger]
    if candidate_numbers != list(range(51, 51 + len(candidate_numbers))):
        raise RuntimeError("The saved replacement ledger does not preserve the contiguous W051+ sampling order")
    candidates: list[dict[str, Any]] = []
    for row in ledger:
        window_id = str(row.get("candidate_window_id", ""))
        if not window_id.startswith("W") or int(window_id[1:]) < 51:
            continue
        url = str(row.get("candidate_url", ""))
        page = page_catalog.get(url)
        text = str(row.get("candidate_text", ""))
        try:
            length = int(row.get("candidate_length_chars", 0))
        except (TypeError, ValueError):
            length = len(text)
        chunks_for_window = text_to_chunks.get((url, text))
        candidates.append({
            "window_id": window_id,
            "sample_order_rank": int(window_id[1:]),
            "url": url,
            "domain": str(getattr(page, "domain", "") or urlparse(url).netloc),
            "page_type": str(row.get("candidate_page_type", "")),
            "current_page_type": str(getattr(page, "page_type", "") or ""),
            "eligible": bool(page is not None and str(getattr(page, "eligible", "")).casefold() == "true"),
            "text": text,
            "chunk_ids": chunks_for_window or [],
            "length": length,
            "length_rule": str(row.get("candidate_length_rule", "")),
            "previous_screen_decision": str(row.get("candidate_decision", "")),
            "previous_screen_reason": str(row.get("candidate_reason", "")),
        })
    first_sample = _read_jsonl(SAMPLED_WINDOWS)
    if len(first_sample) != sum(QUOTAS.values()):
        raise RuntimeError(f"Expected {sum(QUOTAS.values())} original sampled windows, found {len(first_sample)}")
    if Counter(row.get("page_type") for row in first_sample) != Counter(QUOTAS):
        raise RuntimeError("The original 50 windows do not match the seed-42 sample quotas")
    return first_sample, candidates


def _read_chunk_lookup(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as stream:
        return {str(row["id"]): str(row["content"]) for row in csv.DictReader(stream)}


def _selection_review(
    questions: list[dict[str, Any]], reviews: dict[str, dict[str, str]], chunk_lookup: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, str]]]:
    decisions = Counter(str(row.get(REVIEW_DECISION, "")).strip().casefold() for row in reviews.values())
    if decisions != DECISION_COUNTS:
        raise ValueError(f"Round-two review counts must be {dict(DECISION_COUNTS)}, found {dict(decisions)}")
    by_id = {str(row["id"]): row for row in questions}
    review_ids = set(reviews)
    if review_ids != ROUND2_REVIEW_IDS:
        raise ValueError(f"Round-two review IDs differ from the expected set: {sorted(review_ids ^ ROUND2_REVIEW_IDS)}")
    active = []
    rejected = []
    for source in questions:
        row = dict(source)
        qid = str(row["id"])
        if qid not in reviews:
            active.append(row)
            continue
        review = reviews[qid]
        decision = str(review[REVIEW_DECISION]).strip().casefold()
        row["human_review"] = review
        row["review_decision"] = decision
        if decision == "reject":
            rejected.append(row)
        elif decision == "edit":
            effective = _edited_version(row)
            if effective is None:
                raise ValueError(f"Edit fields are missing for {qid}")
            row.update({key: effective[key] for key in ("question", "reference_answer", "keypoints")})
            apply_edited_quotes(row, review.get(MODIFIED_QUOTES, ""), chunk_lookup)
            row["review_applied"] = True
            active.append(row)
        else:
            active.append(row)
    return active, rejected, by_id


def _site_key(record: dict[str, Any]) -> str:
    return str(record.get("domain") or urlparse(str(record.get("url", ""))).netloc).casefold()


def _existing_page_state(questions: list[dict[str, Any]]) -> tuple[set[str], Counter[tuple[str, str]]]:
    used = set()
    counts: Counter[tuple[str, str]] = Counter()
    for row in questions:
        if row.get("url"):
            used.add(str(row["url"]))
            counts[(str(row.get("page_type", "")), _site_key(row))] += 1
        partner = row.get("partner") or {}
        if partner.get("url"):
            used.add(str(partner["url"]))
    return used, counts


def _gold_titles(questions: list[dict[str, Any]], titles: dict[str, str]) -> set[str]:
    return {titles[url] for row in questions if (url := str(row.get("url", ""))) in titles}


def _partner_candidates(
    gold: dict[str, Any], index: Any, titles: dict[str, str], gold_titles: set[str],
    blocked_urls: set[str],
) -> list[dict[str, Any]]:
    query = index._query_embedding(str(gold["text"]))
    scores = np.asarray(index.window_embeddings, dtype=np.float32) @ query
    template = str(gold.get("url_template") or _url_template(str(gold["url"])))
    same_template = []
    for i, candidate in enumerate(index.window_records):
        url = str(candidate.get("url", ""))
        if candidate.get("page_type") != "dataset_record" or url == str(gold["url"]):
            continue
        if str(candidate.get("url_template") or _url_template(url)) != template:
            continue
        if url in blocked_urls or url not in titles or titles[url] in gold_titles:
            continue
        if _same_dataset(gold, candidate) or _text_similarity(str(gold["text"]), str(candidate["text"])) > 0.7:
            continue
        candidate_title = titles[url]
        if _normalized(candidate_title).casefold() not in _normalized(candidate.get("text", "")).casefold():
            continue
        same_template.append((float(scores[i]), candidate))
    same_template.sort(key=lambda pair: (-pair[0], str(pair[1].get("url", ""))))
    return [
        {key: candidate.get(key) for key in (
            "window_id", "url", "domain", "page_type", "url_template", "text", "chunk_ids"
        )} | {"similarity": score}
        for score, candidate in same_template
    ]


def _log_attempt(
    log: list[dict[str, str]], question: dict[str, Any], candidate: dict[str, Any],
    attempt: int, reason: str,
) -> None:
    entry = {
        "question_id": str(question.get("id", "")),
        "old_window_id": str(question.get("window_id", "")),
        "old_url": str(question.get("url", "")),
        "new_window_id": str(candidate.get("window_id", "")),
        "new_url": str(candidate.get("url", "")),
        "attempt_number": str(attempt),
        "candidate_source": f"seed42_eligible_sample_order_{candidate.get('sample_order_rank', '')}",
        "reason": reason,
    }
    log.append(entry)
    print(
        f"  {question.get('id')} attempt {attempt} [{entry['candidate_source']}]: {reason}",
        flush=True,
    )


def _generate_replacement(
    question: dict[str, Any], candidates: list[dict[str, Any]], index: Any,
    titles: dict[str, str], all_gold_titles: set[str], page_texts: dict[str, str],
    used_urls: set[str], site_counts: Counter[tuple[str, str]],
    gold_embedding_matrix: np.ndarray, *, prompt: str, chunk_lookup: dict[str, str],
    llm_call: Any, start_index: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, str]], int]:
    log: list[dict[str, str]] = []
    attempts = 0
    position = start_index
    while position < len(candidates) and attempts < 10:
        candidate = candidates[position]
        position += 1
        if candidate["page_type"] != question.get("page_type"):
            continue
        attempts += 1
        url = str(candidate["url"])
        source_label = f"data/processed/window_replacements.csv:{candidate['window_id']}"
        if not candidate.get("eligible") or candidate.get("current_page_type") != question.get("page_type"):
            _log_attempt(log, question, candidate, attempts, f"Skipped: page is not currently eligible as {question.get('page_type')}.")
            continue
        if candidate.get("length_rule") != "900–1,500 chars" or not MIN_WINDOW_CHARS <= candidate.get("length", 0) <= MAX_WINDOW_CHARS:
            _log_attempt(log, question, candidate, attempts, "Skipped: candidate does not meet the 900–1,500-character window rule.")
            continue
        if not candidate.get("chunk_ids"):
            _log_attempt(log, question, candidate, attempts, "Skipped: sampled text could not be mapped to its consecutive cleaned chunks.")
            continue
        if not re.search(r"\d|[^|\n]+\|[^|\n]+", str(candidate.get("text", ""))):
            _log_attempt(log, question, candidate, attempts, "Skipped: candidate has no digit/year or field/value fact signal.")
            continue
        if url in used_urls:
            _log_attempt(log, question, candidate, attempts, "Skipped: page is already used as a gold or partner page.")
            continue
        title = titles.get(url)
        if title and title in all_gold_titles:
            _log_attempt(log, question, candidate, attempts, f"Skipped: page title duplicates an existing gold title: {title!r}.")
            continue
        cap = SITE_CAP.get(str(candidate["page_type"]), 0)
        if site_counts[(str(candidate["page_type"]), _site_key(candidate))] >= cap:
            _log_attempt(log, question, candidate, attempts, f"Skipped: site cap {cap} already reached for {_site_key(candidate)}.")
            continue
        embedding = index.encoder.encode(
            [str(candidate["text"])], convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )[0].astype(np.float32)
        maximum_similarity = float(np.max(gold_embedding_matrix @ embedding)) if gold_embedding_matrix.size else 0.0
        if maximum_similarity > 0.85:
            _log_attempt(log, question, candidate, attempts, f"Skipped: maximum cosine similarity to an existing gold window is {maximum_similarity:.4f} (> 0.85).")
            continue

        item = dict(question)
        for field in ("window_id", "url", "domain", "page_type", "text", "chunk_ids"):
            item[field] = candidate.get(field)
        item["url_template"] = _url_template(url)
        item["partner"] = None
        if item.get("type") == "comparison":
            blocked_partners = set(used_urls) | {url}
            titles_for_partners = set(all_gold_titles)
            candidate_title = titles.get(url)
            if candidate_title:
                titles_for_partners.add(candidate_title)
            partner_options = _partner_candidates(item, index, titles, titles_for_partners, blocked_partners)
            if not partner_options:
                _log_attempt(log, question, candidate, attempts, "Skipped: no unused same-template partner window contains its record title or dataset name.")
                continue
            item["partner"] = partner_options[0]

        by_title: dict[str, list[str]] = {}
        for page_url, page_title in titles.items():
            by_title.setdefault(page_title, []).append(page_url)
        duplicate_note = duplicate_title_note(item, titles, by_title)
        generated = generate_one(
            item, chunk_lookup, system_prompt=prompt, llm_call=llm_call,
            duplicate_note=duplicate_note,
        )
        if generated.get("status") != "generated":
            reason = generated.get("skip_reason") or generated.get("generation_error") or generated.get("status", "unknown generation result")
            _log_attempt(log, question, candidate, attempts, f"Skipped by teacher/generation: {reason}")
            continue
        generated["review_decision"] = "replaced_round2_reject"
        generated["replacement_reason"] = "Replaced from the seed-42 eligible-page sample sequence after W050."
        template_result = template_answer_check(generated, page_texts)
        generated["template_answer_check"] = template_result
        if template_result["flagged"]:
            details = "; ".join(
                f"quote {entry['quote']!r} repeats on {entry['other_page_count']} other pages"
                for entry in template_result["matching_quotes"]
            )
            _log_attempt(log, question, candidate, attempts, f"Skipped by template-answer check: {details}.")
            continue
        _log_attempt(log, question, candidate, attempts, f"Selected successfully from {source_label}.")
        return generated, item, log, position
    return None, None, log, position


def _export_review(rows: list[dict[str, Any]], output: Path) -> None:
    frame = _review_rows(rows) if rows else _review_rows([{}]).iloc[0:0]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="Main round 3")
            sheet = writer.sheets["Main round 3"]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        from scripts.check_questions import _write_xlsx_minimal
        _write_xlsx_minimal(frame, output)


def prepare_round3(
    *, questions_path: Path = QUESTIONS_INPUT, review_path: Path = REVIEW_INPUT,
    pilot_path: Path = PILOT_INPUT, index_dir: Path = DEFAULT_OUTDIR,
    llm_call: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], dict[str, list[dict[str, Any]]]]:
    from rq4.llm import call_llm
    llm_call = llm_call or call_llm
    source_questions = _read_jsonl(questions_path)
    pilot = _read_jsonl(pilot_path)
    reviews = _read_review_sheet(review_path)
    if len(source_questions) != 50 or len(pilot) != 10:
        raise ValueError(f"Expected 50 current questions and 10 pilot questions, found {len(source_questions)} and {len(pilot)}")
    index = load_index(index_dir)
    chunk_lookup = load_chunk_lookup(CHUNKS_PATH)
    active, rejected, _ = _selection_review(source_questions, reviews, chunk_lookup)
    if {str(row["id"]) for row in rejected} != ROUND2_IDS:
        raise ValueError(f"Expected rejected question IDs {sorted(ROUND2_IDS)}, found {[row['id'] for row in rejected]}")

    titles, by_title = page_title_catalog(index.chunk_records)
    page_texts = page_text_catalog(index.chunk_records)
    round2_edit_ids = {
        qid for qid, review in reviews.items()
        if str(review.get(REVIEW_DECISION, "")).strip().casefold() == "edit"
    }
    edited = [row for row in active if str(row["id"]) in round2_edit_ids]
    roundtrip_system, _closed_system = load_generator_systems(GENERATOR_PROMPT)
    if edited:
        print(f"Rechecking {len(edited)} edited questions: round trip and knowledge-base conflict.", flush=True)
        run_edited_question_checks(edited, index, load_conflict_prompt(CONFLICT_PROMPT), roundtrip_system, llm_call=llm_call)
        edited_by_id = {str(row["id"]): row for row in edited}
        active = [edited_by_id.get(str(row["id"]), row) for row in active]
    baseline = [*active, *rejected]
    baseline = run_template_answer_checks(baseline, page_texts)
    baseline_template_flags = [row for row in baseline if row["template_answer_check"]["flagged"]]

    first_sample, candidates = _sample_candidate_pages()
    if len(first_sample) != 50:
        raise RuntimeError(f"Seeded sampler reproduced {len(first_sample)} original pages, not 50")
    used_urls, site_counts = _existing_page_state(source_questions)
    # A rejected question's old gold page is replaced, so it no longer consumes a gold-page site slot.
    for row in rejected:
        site_counts[(str(row.get("page_type", "")), _site_key(row))] -= 1
    retained_gold_texts = [str(row.get("text", "")) for row in source_questions]
    gold_embeddings = index.encoder.encode(
        retained_gold_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float32)
    all_gold_titles = _gold_titles(source_questions, titles)
    generated_by_id: dict[str, dict[str, Any]] = {}
    used_candidate_titles = set(all_gold_titles)
    replacement_rows: list[dict[str, str]] = []
    sample_cursor = 0
    prompt_hash = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    print(f"Using teacher prompt SHA-256: {prompt_hash}", flush=True)

    for rejected_row in rejected:
        qid = str(rejected_row["id"])
        print(f"Replacing {qid} from the eligible-page seed-42 order after W050 (up to 10 candidates).", flush=True)
        generated, item, attempts, sample_cursor = _generate_replacement(
            rejected_row, candidates, index, titles, used_candidate_titles,
            page_texts, used_urls, site_counts, gold_embeddings,
            prompt=load_teacher_prompt(PROMPT_PATH), chunk_lookup=chunk_lookup,
            llm_call=llm_call, start_index=sample_cursor,
        )
        replacement_rows.extend(attempts)
        if not generated or not item:
            replacement_rows.append({
                "question_id": qid,
                "old_window_id": str(rejected_row.get("window_id", "")),
                "old_url": str(rejected_row.get("url", "")),
                "new_window_id": "", "new_url": "",
                "attempt_number": "",
                "candidate_source": "workflow_summary",
                "reason": "UNRESOLVED: no replacement generated within 10 candidates; original question removed.",
            })
            print(f"  No replacement for {qid}; it will be removed from the final set.", flush=True)
            continue
        generated["human_review"] = {}
        generated["review_decision"] = ""
        generated["replacement_source"] = "eligible_page_sample_order_after_W050"
        generated_by_id[qid] = generated
        used_urls.add(str(item["url"]))
        if item.get("partner"):
            used_urls.add(str(item["partner"]["url"]))
        site_counts[(str(item["page_type"]), _site_key(item))] += 1
        title = titles.get(str(item["url"]))
        if title:
            used_candidate_titles.add(title)
        gold_embeddings = np.vstack([gold_embeddings, index.encoder.encode(
            [str(item["text"])], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)])
    new_questions = list(generated_by_id.values())

    generator_system, closed_system = load_generator_systems(GENERATOR_PROMPT)
    if new_questions:
        new_questions = run_checks(
            new_questions, index, generator_system=generator_system,
            closed_book_system=closed_system, llm_call=llm_call,
        )
        new_questions = run_new_checks(new_questions, index, load_conflict_prompt(CONFLICT_PROMPT), llm_call=llm_call)

    new_by_id = {str(row["id"]): row for row in new_questions}
    final_rows = []
    for source in source_questions:
        qid = str(source["id"])
        if qid in ROUND2_IDS:
            replacement = new_by_id.get(qid)
            if replacement:
                final_rows.append(replacement)
            continue
        row = next((candidate for candidate in active if str(candidate["id"]) == qid), dict(source))
        final_rows.append(row)
    final_rows = mark_near_duplicates(final_rows, index.encoder, index.query_prefix, threshold=0.9)
    final_by_id = {str(row["id"]): row for row in final_rows}
    new_questions = [final_by_id[str(row["id"])] for row in new_questions]
    new_by_id = {str(row["id"]): row for row in new_questions}
    for row in final_rows:
        row["duplicate_title"] = find_duplicate_titles(row, titles, by_title)
    final_rows = run_template_answer_checks(final_rows, page_texts)
    usage_violations = page_usage_violations(final_rows)
    title_violations = partner_gold_title_violations(final_rows, titles)
    if usage_violations or title_violations:
        raise ValueError(f"Final page-use validation failed: duplicate pages={usage_violations}; partner/gold title matches={title_violations}")

    review_rows = []
    for row in new_questions:
        copy = dict(row)
        copy["human_review"] = {}
        copy["review_decision"] = ""
        review_rows.append(copy)
    final_template_flags = [row for row in final_rows if row["template_answer_check"]["flagged"]]
    return final_rows, review_rows, replacement_rows, {
        "baseline": baseline_template_flags, "final": final_template_flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_INPUT)
    parser.add_argument("--review", type=Path, default=REVIEW_INPUT)
    parser.add_argument("--pilot", type=Path, default=PILOT_INPUT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--out", type=Path, default=QUESTIONS_OUTPUT)
    parser.add_argument("--review-out", type=Path, default=REVIEW_OUTPUT)
    parser.add_argument("--replacements", type=Path, default=REPLACEMENTS_CSV)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        sample, candidates = _sample_candidate_pages()
        reviews = _read_review_sheet(args.review)
        counts = Counter(str(row.get(REVIEW_DECISION, "")).strip().casefold() for row in reviews.values())
        print(f"Review decisions: {dict(counts)}")
        print(f"Original sampled windows: {len(sample)} (quota and type counts verified)")
        print(f"Available post-W050 candidates: {len(candidates)}")
        for qid in sorted(ROUND2_IDS):
            eligible = [row for row in candidates if row["page_type"] == next(
                row["page_type"] for row in _read_jsonl(args.questions) if row["id"] == qid
            )]
            print(f"{qid}: {len(eligible)} same-type candidates; first={eligible[0]['url'] if eligible else 'none'}")
        return

    final_rows, review_rows, replacement_rows, flagged = prepare_round3(
        questions_path=args.questions, review_path=args.review, pilot_path=args.pilot,
        index_dir=args.index_dir,
    )
    write_jsonl(args.out, final_rows)
    _export_review(review_rows, args.review_out)
    _write_replacement_log(args.replacements, replacement_rows, append=True)
    counts = Counter(row.get("type") for row in final_rows)
    print(f"Final question count: {len(final_rows)}")
    print("Final question type distribution:")
    for question_type in ("list_all", "comparison", "paraphrased", "false_premise"):
        print(f"  {question_type}: {counts[question_type]}")
    print("Replacement attempts:")
    for row in replacement_rows:
        print(f"  {row['question_id']} attempt {row['attempt_number']} [{row['candidate_source']}]: {row['new_url']} — {row['reason']}")
    print(f"Page-use violations: {page_usage_violations(final_rows)}")
    print(f"Partner/gold title violations: {partner_gold_title_violations(final_rows, page_title_catalog(load_index(args.index_dir).chunk_records)[0])}")
    for label, rows in flagged.items():
        print(f"Template-answer flagged IDs ({label} set): {sorted({str(row['id']) for row in rows})}")
        print(f"Template-answer checks flagged {len({str(row['id']) for row in rows})} questions in the {label} set.")
    print(f"Round-three review questions: {len(review_rows)}")
    print(f"Wrote {args.out}, {args.review_out}, and {args.replacements}")


if __name__ == "__main__":
    main()
