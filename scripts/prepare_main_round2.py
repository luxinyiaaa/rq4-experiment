"""Apply formal question review, replace rejected items, and export round-two checks."""
from __future__ import annotations

import argparse
import csv
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.assign_types import OUTPUT_PLAN, _read_jsonl, write_jsonl
from scripts.build_index import DEFAULT_OUTDIR, load_index
from scripts.check_questions import (
    CONFLICT_PROMPT,
    GENERATOR_PROMPT,
    MODIFIED_ANSWER,
    MODIFIED_KEYPOINTS,
    MODIFIED_QUESTION,
    MODIFIED_QUOTES,
    REVIEW_DECISION,
    _edited_version,
    _parse_modified_keypoints,
    _read_review_sheet,
    _review_rows,
    find_duplicate_titles,
    load_conflict_prompt,
    load_generator_systems,
    mark_near_duplicates,
    page_title_catalog,
    page_usage_violations,
    partner_gold_title_violations,
    run_checks,
    run_new_checks,
    write_outputs,
)
from scripts.generate_questions import (
    CHUNKS_PATH,
    PROMPT_PATH,
    _teacher_response_issues,
    duplicate_title_note,
    generate_one,
    load_chunk_lookup,
    load_teacher_prompt,
)

MAIN_INPUT = PROJECT_ROOT / "outputs" / "main" / "questions_checked.jsonl"
PILOT_INPUT = PROJECT_ROOT / "outputs" / "pilot" / "questions_reviewed.jsonl"
REVIEW_INPUT = PROJECT_ROOT / "outputs" / "main" / "review_sheet_main.xlsx"
MAIN_OUTPUT = PROJECT_ROOT / "outputs" / "main" / "questions_main_round2.jsonl"
ROUND2_REVIEW = PROJECT_ROOT / "outputs" / "main" / "review_sheet_main_round2.xlsx"
REPLACEMENTS_CSV = PROJECT_ROOT / "outputs" / "main" / "window_replacements_main.csv"
PASSAGE_REJECTIONS = ["W059", "W020", "W073", "W010", "W026", "W034", "W080", "W035"]
PARTNER_REJECTIONS = ["W022", "W047", "W048", "W064", "W069"]
EXPECTED_DECISIONS = Counter({"accept": 16, "edit": 11, "reject": 13})
TYPE_SITE_CAP = {"dataset_record": 8, "about_science": 5, "vocabulary": 5}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["question_id", "old_window_id", "old_url", "new_window_id", "new_url", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_ranges(chunk_ids: list[str], chunk_lookup: dict[str, str]) -> tuple[str, list[tuple[int, int, str]]]:
    pieces, ranges = [], []
    offset = 0
    for chunk_id in chunk_ids:
        text = chunk_lookup.get(str(chunk_id), "")
        if not text.strip():
            continue
        pieces.append(text)
        ranges.append((offset, offset + len(text), str(chunk_id)))
        offset += len(text)
    return "".join(pieces), ranges


def _chunk_ids_for_span(start: int, end: int, ranges: list[tuple[int, int, str]]) -> list[str]:
    return [chunk_id for left, right, chunk_id in ranges if left < end and right > start]


def apply_edited_quotes(question: dict[str, Any], raw_lines: str, chunk_lookup: dict[str, str]) -> None:
    """Validate reviewer-supplied quote lines verbatim and replace source/chunk mapping."""
    lines = [line.strip() for line in raw_lines.splitlines() if line.strip()]
    if not lines:
        return
    sources = {"passage": question}
    if question.get("partner"):
        sources["partner"] = question["partner"]
    parsed: list[tuple[str, str, int, int, list[str]]] = []
    failures = []
    for line_number, line in enumerate(lines, start=1):
        match = re.match(r"^\[(passage|partner)\]\s+(.+)$", line, flags=re.IGNORECASE)
        if not match:
            failures.append(f"line {line_number}: expected [passage] or [partner] followed by quote text")
            continue
        source, quote = match.group(1).casefold(), match.group(2)
        if source not in sources:
            failures.append(f"line {line_number}: {source} quote supplied for a question without that source")
            continue
        record = sources[source]
        page_text = str(record.get("text", ""))
        start = page_text.find(quote)
        if start < 0:
            failures.append(f"line {line_number}: quote is not an exact substring of the {source} text")
            continue
        chunk_text, ranges = _source_ranges(record.get("chunk_ids", []), chunk_lookup)
        chunk_start = chunk_text.find(quote)
        if chunk_start < 0:
            failures.append(f"line {line_number}: quote matches window text but cannot be mapped exactly to its chunks")
            continue
        chunk_ids = _chunk_ids_for_span(chunk_start, chunk_start + len(quote), ranges)
        if not chunk_ids:
            failures.append(f"line {line_number}: quote did not overlap a source chunk")
            continue
        parsed.append((source, quote, start, start + len(quote), chunk_ids))
    if failures:
        raise ValueError(f"Edited supporting quote validation failed for {question.get('id')}: " + "; ".join(failures))
    question["supporting_quotes"] = [
        {"source": source, "text": quote, "matched": True, "char_start": start,
         "char_end": end, "chunk_ids": chunk_ids}
        for source, quote, start, end, chunk_ids in parsed
    ]
    question["gold_chunk_ids"] = list(dict.fromkeys(
        chunk_id for source, _quote, _start, _end, chunk_ids in parsed if source == "passage" for chunk_id in chunk_ids
    ))
    question["partner_chunk_ids"] = list(dict.fromkeys(
        chunk_id for source, _quote, _start, _end, chunk_ids in parsed if source == "partner" for chunk_id in chunk_ids
    ))
    question["quote_failed"] = False
    question.pop("quote_failure_reasons", None)


def apply_review_rows(
    questions: list[dict[str, Any]], reviews: dict[str, dict[str, str]], chunk_lookup: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = Counter(str(row.get(REVIEW_DECISION, "")).strip().casefold() for row in reviews.values())
    expected_ids = {str(row["id"]) for row in questions}
    missing = sorted(expected_ids - set(reviews))
    extra = sorted(set(reviews) - expected_ids)
    invalid = sorted(qid for qid in expected_ids if str(reviews.get(qid, {}).get(REVIEW_DECISION, "")).strip().casefold() not in {"accept", "edit", "reject"})
    if missing or extra or invalid:
        raise ValueError(f"Review sheet mismatch: missing={missing}, extra={extra}, blank/invalid decisions={invalid}")
    if decisions != EXPECTED_DECISIONS:
        raise ValueError(f"Review counts must be {dict(EXPECTED_DECISIONS)}, found {dict(decisions)}")

    active, rejected = [], []
    for source in questions:
        question = dict(source)
        review = reviews[str(question["id"])]
        decision = str(review[REVIEW_DECISION]).strip().casefold()
        question["human_review"] = review
        question["review_decision"] = decision
        if decision == "reject":
            rejected.append(question)
            continue
        if decision == "edit":
            effective = _edited_version(question)
            if effective is None:
                raise ValueError(f"Edit fields are missing for {question['id']}")
            for key in ("question", "reference_answer", "keypoints"):
                question[key] = effective[key]
            apply_edited_quotes(
                question,
                review.get(MODIFIED_QUOTES, ""),
                chunk_lookup,
            )
            question["review_applied"] = True
        active.append(question)
    return active, rejected


def _site_key(window: dict[str, Any]) -> str:
    domain = str(window.get("domain") or "").casefold()
    return domain or urlparse(str(window.get("url", ""))).netloc.casefold()


def _initial_page_use(
    pilot: list[dict[str, Any]], active: list[dict[str, Any]]
) -> tuple[set[str], Counter[tuple[str, str]]]:
    used_urls: set[str] = set()
    gold_site_counts: Counter[tuple[str, str]] = Counter()
    for question in [*pilot, *active]:
        url = str(question.get("url", ""))
        if url:
            used_urls.add(url)
            gold_site_counts[(str(question.get("page_type", "")), _site_key(question))] += 1
        partner = question.get("partner")
        if partner and partner.get("url"):
            used_urls.add(str(partner["url"]))
    return used_urls, gold_site_counts


def _gold_urls(pilot: list[dict[str, Any]], active: list[dict[str, Any]], replacing_id: str | None = None) -> set[str]:
    return {
        str(row.get("url", "")) for row in [*pilot, *active]
        if str(row.get("id", "")) != str(replacing_id or "") and row.get("url")
    }


def _select_partner(
    question: dict[str, Any], index: Any, used_urls: set[str],
    gold_urls: set[str], titles: dict[str, str],
) -> dict[str, Any] | None:
    for candidate in index.get_distractor_candidates(str(question["window_id"]), n=len(index.window_records)):
        if candidate.get("page_type") != "dataset_record":
            continue
        if candidate.get("url_template") != question.get("url_template"):
            continue
        url = str(candidate["url"])
        if url in used_urls or url in gold_urls:
            continue
        title = titles.get(url)
        if title and any(gold_url != str(question["url"]) and titles.get(gold_url) == title for gold_url in gold_urls):
            continue
        return {
            key: candidate.get(key)
            for key in ("window_id", "url", "domain", "page_type", "url_template", "text", "chunk_ids")
        }
    return None


def _candidate_gold_rows(index: Any, page_type: str, rng: random.Random) -> list[dict[str, Any]]:
    page_order = []
    pages_path = PROJECT_ROOT / "data" / "processed" / "pages.csv"
    with pages_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("eligible", "").casefold() == "true" and row.get("page_type") == page_type:
                page_order.append(row.get("url", ""))
    rank = {url: position for position, url in enumerate(page_order)}
    candidates = [
        row for row in index.window_records
        if row.get("page_type") == page_type
        and row.get("length_rule") == "900–1,500 chars"
    ]
    candidates.sort(key=lambda row: rank.get(str(row.get("url", "")), len(rank)))
    rng.shuffle(candidates)
    return candidates


def _is_within_site_cap(candidate: dict[str, Any], page_type: str, counts: Counter[tuple[str, str]]) -> bool:
    cap = TYPE_SITE_CAP.get(page_type)
    return cap is not None and counts[(page_type, _site_key(candidate))] < cap


def _title_note_for(
    item: dict[str, Any], titles: dict[str, str], by_title: dict[str, list[str]]
) -> str:
    return duplicate_title_note(item, titles, by_title)


def _generate_replacement(
    question: dict[str, Any], index: Any, pilot: list[dict[str, Any]], active: list[dict[str, Any]],
    titles: dict[str, str], by_title: dict[str, list[str]], used_urls: set[str],
    site_counts: Counter[tuple[str, str]], candidates: list[dict[str, Any]],
    *, prompt: str, chunk_lookup: dict[str, str], rejected_urls: set[str],
    llm_call: Any, replacement_log: list[dict[str, str]], replacements_path: Path,
    max_attempts: int = 3,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    tried = 0
    for candidate in candidates:
        if tried >= max_attempts:
            break
        url = str(candidate["url"])
        if url in used_urls or url in rejected_urls or not _is_within_site_cap(candidate, question["page_type"], site_counts):
            continue
        tried += 1
        item = {**question}
        for key in ("window_id", "url", "domain", "page_type", "url_template", "text", "chunk_ids"):
            item[key] = candidate.get(key)
        item["id"] = str(question["id"])
        if item.get("type") == "comparison":
            available = used_urls - {str(question.get("url", ""))}
            partner = _select_partner(
                item, index, available, _gold_urls(pilot, active, replacing_id=item["id"]), titles
            )
            if not partner:
                continue
            item["partner"] = partner
        note = _title_note_for(item, titles, by_title)
        print(f"Replacement attempt {tried}/{max_attempts} for {item['id']}: {item['url']}", flush=True)
        generated = generate_one(item, chunk_lookup, system_prompt=prompt, llm_call=llm_call, duplicate_note=note)
        if generated.get("status") == "skip":
            reason = str(generated.get("skip_reason", ""))
            print(f"  Teacher skipped: {reason}", flush=True)
            replacement_log.append({
                "question_id": str(item["id"]),
                "old_window_id": str(question.get("window_id", "")),
                "old_url": str(question.get("url", "")),
                "new_window_id": str(item.get("window_id", "")),
                "new_url": str(item.get("url", "")),
                "reason": f"SKIP attempt {tried}: {reason}",
            })
            _write_csv(replacements_path, replacement_log)
            continue
        if generated.get("status") != "generated":
            raise RuntimeError(f"Question generation failed for {item['id']}: {generated.get('generation_error')}")
        generated["review_decision"] = "replaced_reject"
        generated["replacement_reason"] = "Rejected item replaced using the same page type and sampling/window rules."
        return generated, item, None
    return None, None, f"No suitable non-skipped page found after {tried} attempts (limit {max_attempts})"


def prepare_round2(
    *, main_path: Path = MAIN_INPUT, pilot_path: Path = PILOT_INPUT,
    review_path: Path = REVIEW_INPUT, plan_path: Path = OUTPUT_PLAN,
    index_dir: Path = DEFAULT_OUTDIR, prompt_path: Path = PROMPT_PATH,
    generator_prompt_path: Path = GENERATOR_PROMPT,
    replacements_path: Path = REPLACEMENTS_CSV,
    llm_call: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    from rq4.llm import call_llm
    llm_call = llm_call or call_llm
    source_main = _read_jsonl(main_path)
    pilot = _read_jsonl(pilot_path)
    reviews = _read_review_sheet(review_path)
    chunk_lookup = load_chunk_lookup(CHUNKS_PATH)
    active, rejected = apply_review_rows(source_main, reviews, chunk_lookup)
    rejected_by_id = {str(row["id"]): row for row in rejected}
    reject_ids = set(rejected_by_id)
    if reject_ids != set(PASSAGE_REJECTIONS) | set(PARTNER_REJECTIONS):
        raise ValueError("Rejected IDs do not match the requested replacement list: " + ", ".join(sorted(reject_ids)))
    # A partner-only rejection keeps its gold page in the final question set.
    reserved_gold = []
    for question_id in PARTNER_REJECTIONS:
        retained = dict(rejected_by_id[question_id])
        retained.pop("partner", None)
        reserved_gold.append(retained)
    selection_active = [*active, *reserved_gold]
    index = load_index(index_dir)
    titles, by_title = page_title_catalog(index.chunk_records)
    generator_system, _closed_book_system = load_generator_systems(generator_prompt_path)
    prompt = load_teacher_prompt(prompt_path)
    used_urls, site_counts = _initial_page_use(pilot, selection_active)
    initial_collisions = page_usage_violations([*pilot, *selection_active])
    if initial_collisions:
        raise ValueError(f"Pre-existing retained questions violate page uniqueness: {initial_collisions}")
    title_collisions = partner_gold_title_violations([*pilot, *selection_active], titles)
    if title_collisions:
        raise ValueError(f"Pre-existing retained questions violate partner/gold title rule: {title_collisions}")

    # Re-run the existing edits using their reviewed question/answer/keypoints.
    edited_questions = [row for row in active if row.get("review_decision") == "edit"]
    checked_edits: list[dict[str, Any]] = []
    if edited_questions:
        checked_edits = run_checks(
            edited_questions, index, generator_system=generator_system,
            closed_book_system=load_generator_systems(generator_prompt_path)[1], llm_call=llm_call,
        )
        checked_edits = run_new_checks(checked_edits, index, load_conflict_prompt(CONFLICT_PROMPT), llm_call=llm_call)
    edit_by_id = {str(row["id"]): row for row in checked_edits}
    active = [edit_by_id.get(str(row["id"]), row) for row in active]
    selection_active = [*active, *reserved_gold]
    used_urls, site_counts = _initial_page_use(pilot, selection_active)

    plan_by_id = {str(row["id"]): row for row in _read_jsonl(plan_path)}
    index_by_url = {str(row["url"]): row for row in index.window_records}
    rejected_urls = {str(row.get("url", "")) for row in rejected}
    rng = random.Random(42)
    candidates_by_type = {
        page_type: _candidate_gold_rows(index, page_type, rng)
        for page_type in TYPE_SITE_CAP
    }
    new_questions: dict[str, dict[str, Any]] = {}
    replacement_log: list[dict[str, str]] = []
    unresolved: set[str] = set()
    _write_csv(replacements_path, replacement_log)

    def replace_gold_window(question_id: str) -> None:
        old = rejected_by_id[question_id]
        template = dict(plan_by_id[question_id])
        generated, item, error = _generate_replacement(
            template, index, pilot, selection_active, titles, by_title, used_urls, site_counts,
            candidates_by_type[template["page_type"]], prompt=prompt, chunk_lookup=chunk_lookup,
            rejected_urls=rejected_urls, llm_call=llm_call, replacement_log=replacement_log,
            replacements_path=replacements_path,
            max_attempts=10 if question_id in {"W034", "W073", "W080"} else 3,
        )
        if not generated or not item:
            if question_id in {"W034", "W073", "W080"}:
                unresolved.add(question_id)
                print(f"Unresolved replacement {question_id}: {error}; continue without this question.", flush=True)
                return
            raise RuntimeError(f"Could not replace rejected passage {question_id}: {error}")
        new_questions[question_id] = generated
        used_urls.add(str(item["url"]))
        site_counts[(str(item["page_type"]), _site_key(item))] += 1
        selection_active.append(item)
        if item.get("partner"):
            used_urls.add(str(item["partner"]["url"]))
        replacement_log.append({
            "question_id": question_id,
            "old_window_id": str(old.get("window_id", "")),
            "old_url": str(old.get("url", "")),
            "new_window_id": str(item.get("window_id", "")),
            "new_url": str(item.get("url", "")),
            "reason": "Rejected gold passage replaced by a fixed-seed same-type sample-style window; question regenerated.",
        })

    # W059 goes first because its old page is still used as another question's partner.
    replace_gold_window("W059")

    # Comparison items whose gold passage is retained only need a fresh partner.
    for question_id in PARTNER_REJECTIONS:
        old = rejected_by_id[question_id]
        template = dict(plan_by_id[question_id])
        partner = _select_partner(
            template, index, set(used_urls),
            _gold_urls(pilot, selection_active), titles,
        )
        if partner:
            item = {**template, "partner": partner}
            note = _title_note_for(item, titles, by_title)
            generated = generate_one(item, chunk_lookup, system_prompt=prompt, llm_call=llm_call, duplicate_note=note)
            if generated.get("status") == "generated":
                generated["review_decision"] = "replaced_partner"
                new_questions[question_id] = generated
                used_urls.add(str(partner["url"]))
                replacement_log.append({
                    "question_id": question_id,
                    "old_window_id": str((old.get("partner") or {}).get("window_id", "")),
                    "old_url": str((old.get("partner") or {}).get("url", "")),
                    "new_window_id": str(partner.get("window_id", "")),
                    "new_url": str(partner["url"]),
                    "reason": "Rejected comparison partner replaced; page-use and title rules enforced.",
                })
                continue
            if generated.get("status") == "skip":
                replacement_log.append({
                    "question_id": question_id,
                    "old_window_id": str((old.get("partner") or {}).get("window_id", "")),
                    "old_url": str((old.get("partner") or {}).get("url", "")),
                    "new_window_id": str(partner.get("window_id", "")),
                    "new_url": str(partner["url"]),
                    "reason": f"SKIP attempt 1: {generated.get('skip_reason', '')}",
                })
                _write_csv(replacements_path, replacement_log)
            if generated.get("status") != "skip":
                raise RuntimeError(f"Question generation failed for {question_id}: {generated.get('generation_error')}")
        # Missing partner or a skipped regenerated comparison falls back to a new gold window.
        template["partner"] = None
        old_gold_url = str(old.get("url", ""))
        if old_gold_url in used_urls:
            used_urls.remove(old_gold_url)
            site_counts[(str(old.get("page_type", "")), _site_key(old))] -= 1
        fallback_active = [row for row in selection_active if str(row.get("id")) != question_id]
        generated, item, error = _generate_replacement(
            template, index, pilot, fallback_active, titles, by_title, used_urls, site_counts,
            candidates_by_type[template["page_type"]], prompt=prompt, chunk_lookup=chunk_lookup,
            rejected_urls=rejected_urls, llm_call=llm_call, replacement_log=replacement_log,
            replacements_path=replacements_path,
        )
        if not generated or not item:
            raise RuntimeError(f"Could not replace partner for {question_id}: {error}")
        new_questions[question_id] = generated
        used_urls.add(str(item["url"]))
        site_counts[(str(item["page_type"]), _site_key(item))] += 1
        selection_active = [row for row in selection_active if str(row.get("id")) != question_id]
        selection_active.append(item)
        if item.get("partner"):
            used_urls.add(str(item["partner"]["url"]))
        replacement_log.append({
            "question_id": question_id,
            "old_window_id": str(old.get("window_id", "")),
            "old_url": str(old.get("url", "")),
            "new_window_id": str(item.get("window_id", "")),
            "new_url": str(item.get("url", "")),
            "reason": "New gold passage required because no usable replacement partner question was generated.",
        })

    # Finish the remaining gold replacements after partner availability has been resolved.
    for question_id in PASSAGE_REJECTIONS[1:]:
        replace_gold_window(question_id)

    new_rows = [new_questions[qid] for qid in rejected_by_id if qid in new_questions]
    new_rows = run_checks(
        new_rows, index, generator_system=generator_system,
        closed_book_system=load_generator_systems(generator_prompt_path)[1], llm_call=llm_call,
    )
    new_rows = run_new_checks(new_rows, index, load_conflict_prompt(CONFLICT_PROMPT), llm_call=llm_call)
    new_by_id = {str(row["id"]): row for row in new_rows}
    active = [new_by_id.get(str(row["id"]), row) for row in active]
    active.extend(new_rows)
    final_rows = [*pilot, *active]
    expected_total = 50 - len(unresolved)
    if len(final_rows) != expected_total:
        raise ValueError(f"Expected {expected_total} pilot + main questions after replacement, found {len(final_rows)}")
    final_rows = mark_near_duplicates(final_rows, index.encoder, index.query_prefix, threshold=0.9)
    usage_violations = page_usage_violations(final_rows)
    title_violations = partner_gold_title_violations(final_rows, titles)
    if usage_violations or title_violations:
        raise ValueError(f"Final page-use validation failed: duplicate pages={usage_violations}; partner/gold title matches={title_violations}")

    # Keep the fully processed new questions and any edited questions that failed at least one automatic check.
    edit_ids = {str(row["id"]) for row in checked_edits}
    final_by_id = {str(row["id"]): row for row in final_rows}
    round2_rows = list(new_by_id.values())
    for question_id in sorted(edit_ids):
        row = final_by_id[question_id]
        rt = row.get("round_trip_check", {})
        conflict = row.get("global_conflict", {})
        failed = (
            row.get("quote_failed") is True
            or rt.get("status") != "checked" or rt.get("pass") is not True
            or row.get("closed_book_status") != "checked" or row.get("closed_book_correct") is not True
            or row.get("duplicate_title", {}).get("duplicate_title") is True
            or conflict.get("status") != "checked" or conflict.get("conflict") is True
            or row.get("near_duplicate") is True
        )
        if failed:
            copy = dict(row)
            copy["human_review"] = {}
            copy["review_decision"] = ""
            round2_rows.append(copy)

    # Review round two must leave human decision fields empty, and its schema includes the new quote-edit field.
    for row in round2_rows:
        row["human_review"] = {}
        row["review_decision"] = ""
    replacements_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(replacements_path, replacement_log)
    for question_id in unresolved:
        replacement_log.append({
            "question_id": question_id,
            "old_window_id": str(rejected_by_id[question_id].get("window_id", "")),
            "old_url": str(rejected_by_id[question_id].get("url", "")),
            "new_window_id": "",
            "new_url": "",
            "reason": "UNRESOLVED after 10 skipped candidates; excluded from the 49-question final set.",
        })
    _write_csv(replacements_path, replacement_log)
    return final_rows, round2_rows, replacement_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=MAIN_INPUT)
    parser.add_argument("--pilot", type=Path, default=PILOT_INPUT)
    parser.add_argument("--review", type=Path, default=REVIEW_INPUT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--out", type=Path, default=MAIN_OUTPUT)
    parser.add_argument("--review-out", type=Path, default=ROUND2_REVIEW)
    parser.add_argument("--replacements", type=Path, default=REPLACEMENTS_CSV)
    args = parser.parse_args()
    final_rows, round2_rows, replacement_log = prepare_round2(
        main_path=args.main, pilot_path=args.pilot, review_path=args.review,
        index_dir=args.index_dir, replacements_path=args.replacements,
    )
    write_jsonl(args.out, final_rows)
    # Export a dedicated sheet with blank review decision columns.
    frame = _review_rows(round2_rows)
    try:
        import pandas as pd
        with pd.ExcelWriter(args.review_out, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="Main round 2")
            writer.sheets["Main round 2"].freeze_panes = "A2"
            writer.sheets["Main round 2"].auto_filter.ref = writer.sheets["Main round 2"].dimensions
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        from scripts.check_questions import _write_xlsx_minimal
        _write_xlsx_minimal(frame, args.review_out)
    counts = Counter(row["type"] for row in final_rows)
    print("Final question type counts (pilot + main):")
    for question_type in ("list_all", "comparison", "paraphrased", "false_premise"):
        print(f"  {question_type}: {counts[question_type]}")
    print("Replacement windows:")
    for entry in replacement_log:
        print(f"  {entry['question_id']}: {entry['old_url']} -> {entry['new_url']} ({entry['reason']})")
    print(f"Page-use violations: {page_usage_violations(final_rows)}")
    print(f"Partner/gold title violations: {partner_gold_title_violations(final_rows, page_title_catalog(load_index(args.index_dir).chunk_records)[0])}")
    print(f"Round-two review questions: {len(round2_rows)}")
    final_ids = {str(row.get("id")) for row in final_rows}
    missing_rejected = sorted(set(PASSAGE_REJECTIONS + PARTNER_REJECTIONS) - final_ids)
    print(f"Rejected IDs still unresolved: {missing_rejected}")
    print(f"Wrote {args.out}, {args.review_out}, and {args.replacements}")


if __name__ == "__main__":
    main()
