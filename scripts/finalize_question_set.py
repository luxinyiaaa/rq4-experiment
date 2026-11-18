"""Freeze the approved question set, retain review history, and write its report."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_questions import _read_review_sheet
from scripts.build_index import _read_jsonl


REVIEW_FILES = [
    ("pilot_review", PROJECT_ROOT / "outputs/pilot/review_sheet.xlsx"),
    ("main_round1", PROJECT_ROOT / "outputs/main/review_sheet_main.xlsx"),
    ("main_round2", PROJECT_ROOT / "outputs/main/review_sheet_main_round2.xlsx"),
    ("main_round3", PROJECT_ROOT / "outputs/main/review_sheet_main_round3.xlsx"),
]
DECISION_COLUMN = "decision (accept/edit/reject)"


def _history_for_questions(questions: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Counter[str]]]:
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: dict[str, Counter[str]] = {}
    for round_name, path in REVIEW_FILES:
        rows = _read_review_sheet(path)
        round_counts: Counter[str] = Counter()
        for question_id, review_row in rows.items():
            decision = str(review_row.get(DECISION_COLUMN, "")).strip().casefold()
            if decision:
                round_counts[decision] += 1
            if question_id not in {str(q["id"]) for q in questions}:
                continue
            history[question_id].append({
                "round": round_name,
                "source_file": str(path.relative_to(PROJECT_ROOT)),
                "decision": decision or None,
                "review": review_row,
            })
        counts[round_name] = round_counts
    return history, counts


def _question_checks(questions: list[dict[str, Any]]) -> dict[str, Any]:
    def check_value(row: dict[str, Any], field: str) -> bool:
        value = row.get(field)
        if isinstance(value, dict):
            return bool(value.get("pass", value.get("correct", False)))
        return str(value).strip().casefold() in {"true", "pass", "passed"}

    roundtrip_failed = [q["id"] for q in questions if (q.get("round_trip_check") or {}).get("status") == "checked" and not check_value(q, "round_trip_check")]
    roundtrip_unavailable = [q["id"] for q in questions if (q.get("round_trip_check") or {}).get("status") not in {None, "checked"}]
    closed_true = [q["id"] for q in questions if check_value(q, "closed_book_correct")]
    closed_unavailable = [q["id"] for q in questions if q.get("closed_book_status") == "unavailable" or str(q.get("closed_book_correct", "")).startswith("unavailable:")]
    duplicate_title = [q["id"] for q in questions if (q.get("duplicate_title") or {}).get("duplicate_title")]
    kb_conflicts = [q["id"] for q in questions if (q.get("global_conflict") or {}).get("conflict")]
    template = [q["id"] for q in questions if (q.get("template_answer_check") or {}).get("flagged")]
    return {
        "round_trip_failed": roundtrip_failed,
        "round_trip_unavailable": roundtrip_unavailable,
        "closed_book_correct": closed_true,
        "closed_book_unavailable": closed_unavailable,
        "duplicate_title": duplicate_title,
        "kb_conflict": kb_conflicts,
        "template_answer": template,
    }


def _replacement_report(targets: list[str]) -> list[dict[str, Any]]:
    import csv

    path = PROJECT_ROOT / "outputs/main/window_replacements_main.csv"
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for question_id in targets:
        attempts = [r for r in rows if r.get("question_id") == question_id and str(r.get("candidate_source", "")).startswith("seed42_eligible_sample_order_") and str(r.get("attempt_number", "")).strip()]
        urls = {r.get("new_url", "") for r in attempts if r.get("new_url")}
        sources = sorted({r.get("candidate_source", "") for r in attempts})
        selected = next((r for r in attempts if "Selected successfully" in r.get("reason", "")), None)
        result.append({
            "question_id": question_id,
            "attempted_windows": len(attempts),
            "distinct_pages": len(urls),
            "candidate_source": "; ".join(sources),
            "selected_window": selected.get("new_window_id") if selected else None,
            "outcome": "replaced" if selected else "deleted after 10 unsuccessful attempts",
            "reason_summary": "; ".join(dict.fromkeys(r.get("reason", "") for r in attempts if "Skipped:" in r.get("reason", "") or "Skipped by teacher" in r.get("reason", ""))),
        })
    return result


def _write_report(
    questions: list[dict[str, Any]], history_counts: dict[str, Counter[str]],
    checks: dict[str, Any], replacements: list[dict[str, Any]], path: Path,
) -> None:
    types = Counter(str(q.get("type", "")) for q in questions)
    page_types = Counter(str(q.get("page_type", "")) for q in questions)
    decisions = [
        ("Pilot review", "pilot_review"),
        ("Formal review, round 1", "main_round1"),
        ("Formal review, round 2", "main_round2"),
        ("Formal review, round 3", "main_round3"),
    ]
    lines = [
        "# Final Question Set Report",
        "",
        "## Sampling",
        "",
        "The initial sample contained 50 text windows: 25 `dataset_record`, 20 `about_science`, and 5 `vocabulary`. The per-site caps were 8 `dataset_record` pages and 5 pages each for `about_science` and `vocabulary`. One window was retained per sampled page; windows used 3–5 consecutive cleaned chunks, approximately 900–1,500 characters, and at least one concrete fact.",
        "",
        "## Review history",
        "",
        "| Review round | Accept | Edit | Reject |",
        "|---|---:|---:|---:|",
    ]
    for label, key in decisions:
        counts = history_counts.get(key, Counter())
        lines.append(f"| {label} | {counts.get('accept', 0)} | {counts.get('edit', 0)} | {counts.get('reject', 0)} |")
    lines.extend(["", "Review decisions and original review-sheet fields are retained in each question's `review_history` array in `questions_final.jsonl`.", "", "## Replacement attempts", "", "Each attempted sampled window is logged in `window_replacements_main.csv`; counts below include only the replacement workflow's eligible seed-42 sample order, not rejected distractor-pool candidates.", "", "| Question | Candidate windows tried | Distinct pages | Candidate source | Outcome |", "|---|---:|---:|---|---|"])
    for row in replacements:
        lines.append(f"| {row['question_id']} | {row['attempted_windows']} | {row['distinct_pages']} | `{row['candidate_source']}` | {row['outcome']} |")
    lines.extend(["", "### Deleted windows", "", "| Original window | Reason after up to 10 candidate attempts |", "|---|---|"])
    for row in replacements:
        if row["outcome"].startswith("deleted"):
            reason = row["reason_summary"].replace("|", "\\|") or "No suitable replacement generated within the 10-attempt limit."
            lines.append(f"| {row['question_id']} | {reason} |")
    lines.extend(["", "## Final set", "", f"The frozen set contains **{len(questions)} questions**: 10 pilot questions and {len(questions) - 10} formal questions.", "", "### Question types", "", "| Type | Count |", "|---|---:|"])
    for kind in ("list_all", "comparison", "paraphrased", "false_premise"):
        lines.append(f"| `{kind}` | {types.get(kind, 0)} |")
    lines.extend(["", "### Page types", "", "| Page type | Count |", "|---|---:|"])
    for kind, count in sorted(page_types.items()):
        lines.append(f"| `{kind}` | {count} |")
    lines.extend(["", "## Automated checks", "", f"- Reverse-answer check did not pass: {len(checks['round_trip_failed'])} questions ({', '.join(checks['round_trip_failed']) or 'none'}).", f"- Reverse-answer check unavailable: {len(checks['round_trip_unavailable'])} questions ({', '.join(checks['round_trip_unavailable']) or 'none'}).", f"- Closed-book answers marked correct: {len(checks['closed_book_correct'])} questions ({', '.join(checks['closed_book_correct']) or 'none'}).", f"- Closed-book check unavailable: {len(checks['closed_book_unavailable'])} questions ({', '.join(checks['closed_book_unavailable']) or 'none'}).", f"- Duplicate page titles: {len(checks['duplicate_title'])} questions ({', '.join(checks['duplicate_title']) or 'none'}).", f"- Knowledge-base conflicts: {len(checks['kb_conflict'])} questions ({', '.join(checks['kb_conflict']) or 'none'}).", f"- Template-answer check flagged: {len(checks['template_answer'])} questions ({', '.join(checks['template_answer']) or 'none'}).", "", "The template-answer flags were reviewed manually. W002 repeats the same news item on a listing page. W015, W022, W039, W040, and W043 use common field values, but each quote expresses a fact about a specific record. All six questions were retained.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def freeze_question_set(source: Path, output: Path, report: Path, manifest: Path) -> list[dict[str, Any]]:
    questions = _read_jsonl(source)
    if len(questions) != 44 or len({str(q.get("id")) for q in questions}) != 44:
        raise ValueError(f"Expected 44 unique questions; found {len(questions)} rows")
    if sum(q.get("status") == "generated" for q in questions) != 44:
        raise ValueError("All final questions must have generated status")
    history, counts = _history_for_questions(questions)
    for question in questions:
        qid = str(question["id"])
        question["review_history"] = history.get(qid, [])
        if qid == "W047":
            review = {DECISION_COLUMN: "accept", "review_note": "Accepted for the final question set."}
            question["human_review_round3"] = review
            question["review_decision"] = "accept"
            question["review_applied"] = True
    if not any(q.get("id") == "W047" and q.get("review_decision") == "accept" for q in questions):
        raise ValueError("W047 acceptance was not applied")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for question in questions:
            stream.write(json.dumps(question, ensure_ascii=False, sort_keys=True) + "\n")
    checks = _question_checks(questions)
    replacements = _replacement_report(["W010", "W020", "W026", "W034", "W047", "W069", "W080"])
    _write_report(questions, counts, checks, replacements, report)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest_data = {
        "question_file": str(output.relative_to(PROJECT_ROOT)),
        "sha256": digest,
        "question_count": len(questions),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(questions)} questions to {output}; SHA-256 {digest}")
    print(f"Review counts: { {key: dict(value) for key, value in counts.items()} }")
    print(f"Type distribution: {dict(Counter(q['type'] for q in questions))}")
    print(f"Page-type distribution: {dict(Counter(q['page_type'] for q in questions))}")
    print(f"Automated checks: {json.dumps(checks, ensure_ascii=False)}")
    for row in replacements:
        print(f"Replacement {row['question_id']}: tried={row['attempted_windows']}, distinct_pages={row['distinct_pages']}, outcome={row['outcome']}")
    print(f"Wrote {report} and {manifest}")
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "outputs/main/questions_main_round3.jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/final/questions_final.jsonl")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "outputs/final/question_set_report.md")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "outputs/final/manifest.json")
    args = parser.parse_args()
    freeze_question_set(args.source, args.out, args.report, args.manifest)


if __name__ == "__main__":
    main()
