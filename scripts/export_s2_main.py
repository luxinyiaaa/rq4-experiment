"""Assemble the formal S2 checked, rejected, review, and summary outputs."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/s2/main"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if value is not None else ""


def review_checks(checks: dict[str, Any]) -> dict[str, Any]:
    absence = checks.get("absence_check") or {}
    absence_summary = {}
    for variant, item in absence.items():
        if isinstance(item, dict):
            retrieval = item.get("retrieval") or {}
            absence_summary[variant] = {"passed": item.get("status") == "passed", "answerable": item.get("answerable"),
                "retrieved_bm25": len(retrieval.get("bm25_top_50", [])), "retrieved_vector": len(retrieval.get("vector_top_50", [])),
                "flagged_passage_ids": [p.get("passage_id") for p in item.get("answering_passages", [])],
                "reason": (item.get("judge") or {}).get("reason", "")}
    condition = checks.get("condition_semantics") or {}
    condition_summary = {key: condition.get(key) for key in ("passed", "needs_review", "retried_for_inconsistency", "type",
        "candidate_count", "catalogue_member_count", "missing_judge_decisions", "unexpected_matching_record_ids",
        "inconsistent_record_vs_matching_list_ids", "reason") if key in condition}
    quote_support = checks.get("quote_support") or {}
    single_page = checks.get("single_page_check") or {}
    unique_bridge = checks.get("unique_bridge") or {}
    return {"quote_verbatim": checks.get("quote_verbatim"),
        "quote_support": {"passed": quote_support.get("passed"), "non_decisive_failures": quote_support.get("non_decisive_failures"),
                          "keypoints": [{"keypoint_index": p.get("keypoint_index"), "decisive": p.get("decisive"),
                                         "applicable": p.get("applicable"), "passed": p.get("passed"),
                                         "evidence": [{"source": e.get("source"), "quote": e.get("quote"), "verbatim": e.get("verbatim"),
                                                       "supported": e.get("supported"), "reason": e.get("reason")}
                                                      for e in p.get("evidence", [])]} for p in quote_support.get("keypoints", [])]},
        "multi_page": checks.get("multi_page"), "single_page_check": {"passed": single_page.get("passed"), "pages": single_page.get("pages", [])},
        "unique_bridge": unique_bridge, "set_completeness": checks.get("set_completeness"),
        "absence_check": absence_summary, "conflict_check": checks.get("conflict_check"),
        "condition_semantics": condition_summary, "round_trip": checks.get("round_trip"),
        "closed_book": {"correct": (checks.get("closed_book") or {}).get("correct"), "reason": (checks.get("closed_book") or {}).get("judge", {}).get("reason", "")},
        "near_duplicate": checks.get("near_duplicate"), "rules": checks.get("rules"), "phrasing_overlap": checks.get("phrasing_overlap")}


def review_row(row: dict[str, Any]) -> dict[str, Any]:
    checks = row.get("checks", {})
    summary = review_checks(checks)
    qsupport = checks.get("quote_support") or {}
    return {
        "id": row.get("id"), "type": row.get("type"), "user_task": row.get("user_task"),
        "question_user": row.get("question_user"), "question_source": row.get("question_source"),
        "expected_behaviour": row.get("expected_behaviour"), "reference_answer": row.get("reference_answer"),
        "keypoints (decisive, sources, quotes)": compact(row.get("keypoints", [])),
        "evidence_pages (title, url)": compact([{"title": p.get("title"), "url": p.get("url")} for p in row.get("pages", [])]),
        "quote_verbatim": compact(summary.get("quote_verbatim")),
        "quote_support": compact(qsupport), "non_decisive_quote_support_failures": compact(qsupport.get("non_decisive_failures", [])),
        "multi_page": compact(summary.get("multi_page")), "single_page_check": compact(summary.get("single_page_check")),
        "unique_bridge": compact(summary.get("unique_bridge")), "set_completeness": compact(summary.get("set_completeness")),
        "absence_check": compact(summary.get("absence_check")), "conflict_check": compact(summary.get("conflict_check")),
        "condition_semantics": compact(summary.get("condition_semantics")), "round_trip": compact(summary.get("round_trip")),
        "closed_book": compact(summary.get("closed_book")), "near_duplicate": compact(summary.get("near_duplicate")),
        "rules": compact(summary.get("rules")), "phrasing_overlap": compact(summary.get("phrasing_overlap")),
        "needs_review": row.get("needs_review", False), "needs_review_reason": "; ".join(row.get("needs_review_reasons", [])),
        "auto_rejection_reasons": "; ".join(row.get("auto_rejection_reasons", [])),
        "difficulty": compact(row.get("difficulty", {})),
        "decision (accept / edit / reject)": "", "would_researcher_ask (yes / no)": "",
        "edited_question_user": "", "edited_question_source": "", "edited_reference_answer": "",
        "edited_keypoints (one per line; [D] = decisive)": "", "note": "",
    }


def main() -> None:
    checked_path = OUT / "questions_s2_main_checked.jsonl"
    checked_generated = [row for row in read_jsonl(checked_path)
                         if row.get("status") == "generated" and row.get("checks", {}).get("status") != "not_run_teacher_skip"]
    skips = read_jsonl(OUT / "generation_skips_s2.jsonl")
    # Keep each submitted bundle attempt in the all-candidates file, including teacher skips.
    skip_rows = []
    for row in skips:
        skip_rows.append({**row, "status": row.get("status", "skip"), "checks": {"status": "not_run_teacher_skip"},
                          "auto_rejection_reasons": [row.get("skip_reason") or "Teacher returned skip."],
                          "review_eligible": False})
    all_rows = [*checked_generated, *skip_rows]
    write_jsonl(checked_path, all_rows)
    rejected = [row for row in all_rows if not row.get("review_eligible", False)]
    write_jsonl(OUT / "questions_s2_main_rejected.jsonl", rejected)
    review = [row for row in all_rows if row.get("status") == "generated" and row.get("review_eligible") is True]
    payload = [review_row(row) for row in review]
    payload_path = Path("/private/tmp/s2_main_review_payload.json")
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    bundle_summary = json.loads((ROOT / "data/processed/s2/bundle_summary_s2.json").read_text(encoding="utf-8"))
    by_type: dict[str, dict[str, Any]] = {}
    for kind in ("ambiguity", "unanswerable", "set_all", "comparison_multi", "multi_hop"):
        candidates = [row for row in all_rows if row.get("type") == kind]
        generated = [row for row in candidates if row.get("status") == "generated"]
        passed = [row for row in generated if row.get("review_eligible") is True]
        reasons = Counter()
        for row in candidates:
            if row.get("status") == "skip":
                reasons["teacher_skip"] += 1
            elif row.get("status") == "generation_failed":
                reasons["generation_failed"] += 1
            else:
                for reason in row.get("auto_rejection_reasons", []):
                    reasons[str(reason)] += 1
        by_type[kind] = {"candidate_bundles": len(candidates), "generated": len(generated),
                         "passed": len(passed), "rejected_or_skipped": len(candidates) - len(passed),
                         "rejection_reasons": dict(sorted(reasons.items()))}

    pilot = read_jsonl(ROOT / "outputs/s2/pilot3/questions_s2_rechecked_newrules.jsonl")
    pilot_by_type = Counter()
    for row in pilot:
        pilot_by_type[row.get("type", "unknown")] += 1
    pilot_passed = sum(row.get("review_eligible") is True for row in pilot)
    pilot_reasons = Counter(reason for row in pilot for reason in row.get("auto_rejection_reasons", []))
    summary = ["# S2 Main Question Generation Summary", "",
               "Formal question generation used the corrected S2 checks and the updated teacher prompt.", "",
               "## Candidate and review counts", "",
               "| Type | Candidate bundles attempted | Generated | Passed to review | Rejected or skipped | Rejection reasons |",
               "|---|---:|---:|---:|---:|---|"]
    for kind in ("ambiguity", "unanswerable", "set_all", "comparison_multi", "multi_hop"):
        stats = by_type[kind]
        reason_text = "; ".join(f"{name}: {count}" for name, count in stats["rejection_reasons"].items()) or "None"
        summary.append(f"| {kind} | {stats['candidate_bundles']} | {stats['generated']} | {stats['passed']} | {stats['rejected_or_skipped']} | {reason_text} |")
    skip_counts = Counter(row.get("status", "skip") for row in skips)
    summary += ["", f"Selected bundle pool: {bundle_summary['selected_bundles']} packages. Generated questions: {sum(v['generated'] for v in by_type.values())}; non-generated attempts: {len(skips)} ({skip_counts.get('skip', 0)} teacher skips; {skip_counts.get('generation_failed', 0)} generation failures); review-sheet rows: {len(review)}; rejected/skipped rows: {len(rejected)}.",
                "", "## Pilot3 rechecked under the corrected rules", "",
                f"Questions rechecked: {len(pilot)}; passed to review: {pilot_passed}; rejected: {len(pilot) - pilot_passed}.",
                "", "| Type | Rechecked | Passed | Rejected | Rejection reasons |", "|---|---:|---:|---:|---|"]
    for kind in ("set_all", "multi_hop", "comparison_multi", "ambiguity", "unanswerable"):
        rows = [r for r in pilot if r.get("type") == kind]
        reasons = Counter(reason for row in rows for reason in row.get("auto_rejection_reasons", []))
        summary.append(f"| {kind} | {len(rows)} | {sum(r.get('review_eligible') is True for r in rows)} | {sum(r.get('review_eligible') is not True for r in rows)} | " +
                       ("; ".join(f"{name}: {count}" for name, count in sorted(reasons.items())) or "None") + " |")
    summary += ["", "## Generation pool limits", "",
                "| Type | Raw bundle candidates | Selected under diversity limits |", "|---|---:|---:|"]
    for kind in ("ambiguity", "unanswerable", "set_all", "comparison_multi", "multi_hop"):
        summary.append(f"| {kind} | {bundle_summary['raw_candidates'].get(kind, 0)} | {bundle_summary['by_type'].get(kind, 0)} |")
    (OUT / "s2_main_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_bundles_attempted": len(all_rows), "generated": sum(r.get("status") == "generated" for r in all_rows),
                      "review_rows": len(review), "rejected_or_skipped": len(rejected), "pilot3_rechecked": len(pilot),
                      "pilot3_passed": pilot_passed, "by_type": by_type}, indent=2))


if __name__ == "__main__":
    main()
