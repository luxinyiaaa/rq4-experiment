"""Export the S2 main2 candidate audit, review workbook payload, and summary."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.build_kb_index import write_jsonl
from scripts.export_s2_main import compact, review_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/s2/main2"
ORDER = ("ambiguity", "unanswerable", "set_all", "comparison_multi", "multi_hop")
TARGETS = {"ambiguity": "all eligible", "unanswerable": 12, "set_all": 12, "comparison_multi": 9, "multi_hop": 4}


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def main() -> None:
    old_rechecked = read_rows(OUT / "rechecked_main21.jsonl")
    supplement = read_rows(OUT / "supplement_checked.jsonl")
    prior_skips = read_rows(OUT / "generation_skips_s2.jsonl")
    skip_rows = [{**row, "status": "skip", "checks": {"status": "not_run_teacher_skip"},
                  "auto_rejection_reasons": [row.get("skip_reason") or "Teacher returned skip."],
                  "review_eligible": False}
                 for row in prior_skips if row.get("status") == "skip"]
    rows = [*old_rechecked, *supplement, *skip_rows]

    # Recompute de-duplication and review eligibility over the complete combined pool.
    seen_conditions: dict[tuple[str, str], str] = {}
    for row in rows:
        if row.get("status") != "generated" or row.get("type") != "set_all" or not row.get("target_field"):
            continue
        key = (str(row.get("catalogue_id", "")), str(row.get("target_field", "")).casefold().strip())
        if key in seen_conditions:
            reason = "duplicate_catalogue_condition_field"
            reasons = row.setdefault("auto_rejection_reasons", [])
            if reason not in reasons:
                reasons.append(reason)
            row["duplicate_catalogue_condition_field"] = {"duplicate_of": seen_conditions[key]}
            row["review_eligible"] = False
        else:
            seen_conditions[key] = str(row.get("id", ""))

    write_jsonl(OUT / "questions_s2_main2_checked.jsonl", rows)
    rejected = [row for row in rows if row.get("status") != "generated" or row.get("review_eligible") is not True]
    review = [row for row in rows if row.get("status") == "generated" and row.get("review_eligible") is True]
    write_jsonl(OUT / "questions_s2_main2_rejected.jsonl", rejected)

    payload = []
    for row in review:
        item = review_row(row)
        checks = row.get("checks", {})
        item["derivation_check"] = compact(checks.get("derivation_check"))
        item["comparison_multi_validity"] = compact(checks.get("comparison_multi_validity"))
        item["validation_retry"] = compact(row.get("validation_retry"))
        payload.append(item)
    payload_path = Path("/private/tmp/s2_main2_review_payload.json")
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    bundle_summary_path = OUT / "bundle_summary_s2.json"
    bundle_summary = json.loads(bundle_summary_path.read_text(encoding="utf-8")) if bundle_summary_path.exists() else {}
    type_stats = {}
    for kind in ORDER:
        candidates = [row for row in rows if row.get("type") == kind]
        generated = [row for row in candidates if row.get("status") == "generated"]
        passed = [row for row in generated if row.get("review_eligible") is True]
        reasons = Counter()
        for row in candidates:
            if row.get("status") == "skip":
                reasons["teacher_skip"] += 1
            elif row.get("status") == "generation_failed":
                reasons[row.get("generation_failed_reason", "generation_failed")] += 1
            for reason in row.get("auto_rejection_reasons", []):
                reasons[str(reason)] += 1
        type_stats[kind] = {"candidate_attempts": len(candidates), "generated": len(generated),
                            "passed": len(passed), "rejected_or_skipped": len(candidates) - len(passed),
                            "rejection_reasons": dict(sorted(reasons.items()))}

    main_rechecks = {}
    for kind in ORDER:
        typed = [row for row in old_rechecked if row.get("type") == kind]
        reasons = Counter(reason for row in typed for reason in row.get("auto_rejection_reasons", []))
        main_rechecks[kind] = {"rechecked": len(typed), "passed": sum(row.get("review_eligible") is True for row in typed),
                               "rejected": sum(row.get("review_eligible") is not True for row in typed),
                               "rejection_reasons": dict(sorted(reasons.items()))}

    selected = bundle_summary.get("by_type", {})
    summary = ["# S2 Main2 Question Generation Summary", "",
        "The first S2 main run was rechecked with the main2 validation rules. Additional bundles were generated only for available shortfalls. API credentials were loaded from `.env`; mock tests ran before live calls. S1 outputs and raw source data were not modified.", "",
        "## Candidate outcomes", "",
        "| Type | Target | Rechecked main candidates | Bundle-pool selected (includes overlaps) | Total attempts | Passed to review | Rejected or skipped | Rejection reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for kind in ORDER:
        old_n = main_rechecks[kind]["rechecked"]
        stats = type_stats[kind]
        target = str(TARGETS[kind])
        reasons = "; ".join(f"{name}: {count}" for name, count in stats["rejection_reasons"].items()) or "None"
        summary.append(f"| {kind} | {target} | {old_n} | {selected.get(kind, 0)} | {stats['candidate_attempts']} | {stats['passed']} | {stats['rejected_or_skipped']} | {reasons} |")
    summary += ["", "## Existing main questions rechecked", "",
        f"Rechecked {len(old_rechecked)} previously generated questions; {sum(row.get('review_eligible') is True for row in old_rechecked)} passed and {sum(row.get('review_eligible') is not True for row in old_rechecked)} were rejected.", "",
        "| Type | Rechecked | Passed | Rejected | Rejection reasons |", "|---|---:|---:|---:|---|"]
    for kind in ORDER:
        stats = main_rechecks[kind]
        reasons = "; ".join(f"{name}: {count}" for name, count in stats["rejection_reasons"].items()) or "None"
        summary.append(f"| {kind} | {stats['rechecked']} | {stats['passed']} | {stats['rejected']} | {reasons} |")
    summary += ["", "## Candidate-pool constraints and shortfalls", "",
        f"The available bundle builder selected {sum(selected.values()) if isinstance(selected, dict) else 0} bundles. It produced {selected.get('ambiguity', 0)} unique ambiguity bundles, {selected.get('unanswerable', 0)} unanswerable bundles, {selected.get('set_all', 0)} set_all bundles, {selected.get('comparison_multi', 0)} comparison bundles, and {selected.get('multi_hop', 0)} multi-hop bundles.",
        "Bundle-pool counts include packages overlapping the rechecked main set. Set_all generation was limited by the one-item-per-(catalogue, condition field) rule and the available unique pairs; no additional unique set_all pair was available after de-duplication. Comparison generation yielded fewer bundles than its target after requiring an allowed shared relationship and parsed, explicit comparable values; five available replacement packages produced three passing questions. Multi-hop remained capped at four, all already present in main. Every eligible ambiguity package was included; the indexed pool contained two unique packages, both already present in main.",
        "", f"Review-sheet rows: {len(review)}. Rejected or skipped candidate rows: {len(rejected)}."]
    (OUT / "s2_main2_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_attempts": len(rows), "review_rows": len(review), "rejected_or_skipped": len(rejected),
                      "by_type": type_stats, "main_recheck": main_rechecks,
                      "bundle_pool_selected": selected, "summary": str(OUT / "s2_main2_summary.md")}, indent=2))


if __name__ == "__main__":
    main()
