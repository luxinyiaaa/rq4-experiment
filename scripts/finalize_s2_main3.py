"""Refresh the main3 review payload and English summary from saved check results."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.build_kb_index import read_jsonl, write_jsonl
from scripts.export_s2_main import review_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/s2/main3"
MAIN2 = ROOT / "outputs/s2/main2/questions_s2_main2_checked.jsonl"


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if value is not None else ""


def main() -> None:
    rows = read_jsonl(OUT / "questions_s2_main3_checked.jsonl")
    main2 = read_jsonl(MAIN2)
    main2_review_all = [row for row in main2 if row.get("status") == "generated" and row.get("review_eligible") is True]
    if len(main2_review_all) != 10:
        raise RuntimeError(f"Expected 10 main2 review rows, found {len(main2_review_all)}")
    excluded_main2_multihop = [{**row, "status": "excluded_from_main3_review",
        "auto_rejection_reasons": ["existing_multi_hop_excluded_from_review_by_main3_scope"], "review_eligible": False}
        for row in main2_review_all if row.get("type") == "multi_hop"]
    main2_review = [row for row in main2_review_all if row.get("type") != "multi_hop"]
    for i, row in enumerate(rows, 1):
        if row.get("status") == "rejected_before_generation" and not row.get("id"):
            row["id"] = f"S2-MAIN3-REJECTED-{row.get('type', 'UNKNOWN').upper()}-{i:03d}"
    ambiguity = [row for row in rows if row.get("id") in {"S2-AMBIGUITY-001", "S2-AMBIGUITY-002"}]
    rescued = [row for row in ambiguity if row.get("review_eligible") is True]
    new_passed = [row for row in rows if row.get("id", "").startswith("S2-MAIN3-")
                  and row.get("status") == "generated" and row.get("review_eligible") is True]
    review = [*main2_review, *rescued, *new_passed]
    if sum(any("ETC L2 ARCHIVE" in str(page.get("title", "")) for page in row.get("pages", [])) for row in review) > 3:
        raise RuntimeError("Final review set exceeds the three-question ICOS ETC L2 ARCHIVE limit.")
    write_jsonl(OUT / "questions_s2_main3_checked.jsonl", rows)
    rejected = [*excluded_main2_multihop,
                *[row for row in rows if row.get("status") != "generated" or row.get("review_eligible") is not True]]
    write_jsonl(OUT / "questions_s2_main3_rejected.jsonl", rejected)

    payload = []
    for row in review:
        item = review_row(row)
        checks = row.get("checks", {})
        item["derivation_check"] = compact(checks.get("derivation_check"))
        item["comparison_multi_validity"] = compact(checks.get("comparison_multi_validity"))
        item["validation_retry"] = compact(row.get("validation_retry"))
        item["plausibility_check"] = compact(checks.get("plausibility_check"))
        payload.append(item)
    (OUT / "review_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    typed = {}
    for kind in ("ambiguity", "set_all", "unanswerable", "comparison_multi", "multi_hop"):
        candidates = [row for row in rows if row.get("type") == kind]
        if kind == "multi_hop":
            candidates = excluded_main2_multihop
        reasons = Counter()
        passed = 0
        for row in candidates:
            if row.get("status") == "generated" and row.get("review_eligible") is True:
                passed += 1
                continue
            values = row.get("auto_rejection_reasons", [])
            if values:
                reasons.update(map(str, values))
            elif row.get("status") == "skip":
                reasons["teacher_skip"] += 1
            elif row.get("status") == "generation_failed":
                reasons["generation_failed"] += 1
            elif row.get("generation_failed_reason"):
                reasons[str(row["generation_failed_reason"])] += 1
            elif row.get("skip_reason"):
                reasons[str(row["skip_reason"])] += 1
            else:
                reasons["not_eligible_before_generation"] += 1
        if kind == "ambiguity":
            # These are rechecks only; no new ambiguity candidates were generated.
            passed = sum(row.get("review_eligible") is True for row in candidates)
        typed[kind] = {"attempts": len(candidates), "passed": passed,
                       "rejected_or_skipped": len(candidates) - passed,
                       "rejection_reasons": dict(sorted(reasons.items()))}
    typed["multi_hop"] = {"attempts": len(excluded_main2_multihop), "passed": 0,
                          "rejected_or_skipped": len(excluded_main2_multihop),
                          "rejection_reasons": {"existing_multi_hop_excluded_from_review_by_main3_scope": len(excluded_main2_multihop)}}

    summary = ["# S2 Main3 Final Top-Up Summary", "",
        "Main3 started from the 10 questions admitted to the main2 review sheet, retained the eight non-multi-hop questions, and excluded its two existing multi-hop questions from review. It rechecked the two same-name ambiguity candidates using normalized record names from the index and added only eligible set_all, unanswerable, and comparison_multi questions. No S1 final output or raw source data was modified.", "",
        "## Candidate outcomes", "", "| Type | Attempts | Passed to review | Rejected or skipped | Rejection reasons |", "|---|---:|---:|---:|---|"]
    for kind, stats in typed.items():
        reason_text = "; ".join(f"{name}: {count}" for name, count in stats["rejection_reasons"].items()) or "None"
        summary.append(f"| {kind} | {stats['attempts']} | {stats['passed']} | {stats['rejected_or_skipped']} | {reason_text} |")
    summary += ["", "## Multi-hop outcome", "",
        "No multi-hop questions were generated in main3. Across the rounds, candidates built from text pages did not yield a natural question that necessarily depended on evidence from two pages.", "",
        f"Main2 review questions retained: {len(main2_review)}; pre-existing main2 multi-hop questions excluded from review: {len(excluded_main2_multihop)}; ambiguity candidates rescued: {len(rescued)}; new questions passed to review: {len(new_passed)}; final review rows: {len(review)}; rejected or skipped records: {len(rejected)}.", ""]
    (OUT / "s2_main3_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (OUT / "main3_counts.json").write_text(json.dumps({"by_type": typed, "review_rows": len(review),
        "rescued_ambiguity": [row.get("id") for row in rescued], "excluded_main2_multi_hop": len(excluded_main2_multihop), "new_passed": len(new_passed),
        "rejected_records": len(rejected)}, indent=2), encoding="utf-8")
    print(json.dumps({"by_type": typed, "main2_review_rows": len(main2_review), "rescued_ambiguity": len(rescued),
                      "new_passed": len(new_passed), "review_rows": len(review), "rejected_or_skipped": len(rejected)}, indent=2))


if __name__ == "__main__":
    main()
