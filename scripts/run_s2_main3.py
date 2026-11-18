"""Run the final S2 top-up round without touching S1 outputs or raw data."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scripts.build_bundles_s2 import build_bundles
from scripts.build_index import load_index
from scripts.build_kb_index import read_jsonl, write_jsonl
from scripts.check_questions_s2 import check_questions
from scripts.export_s2_main import review_row
from scripts.generate_questions_s2 import generate_one

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/s2/main3"
MAIN2 = ROOT / "outputs/s2/main2"
INDEX = ROOT / "data/processed/s2"
CLASSIFICATION = re.compile(r"^(country|geographical area|data holding centre|parameters?|station|network)$", re.I)


def is_icos_etc_bundle(row: dict[str, Any]) -> bool:
    name = str(row.get("catalogue_name", "")).casefold()
    return ("icos object catalogue" in name or "etc-archive" in name or
            any("etc l2 archive" in str(page.get("title", "")).casefold() for page in row.get("pages", [])))


def norm_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def answer_set(row: dict[str, Any]) -> set[str]:
    values = row.get("expected_items") or []
    if not values:
        values = [p.get("text", "") for p in row.get("keypoints", []) if isinstance(p, dict) and p.get("decisive")]
    return {norm_name(str(value)) for value in values if str(value).strip()}


def related_unanswerable(bundle: dict[str, Any], page_map: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    pages = [page_map.get(row.get("url"), {}) for row in bundle.get("pages", [])]
    relation_fields = ("geographical area", "parameter category", "parameters", "data holding centre", "holding institution", "part of", "programme")
    for field in relation_fields:
        vals = [norm_name(str(page.get("fields", {}).get(field, ""))) for page in pages]
        if len(vals) >= 2 and all(vals) and (len(set(vals)) == 1 or
                (field == "geographical area" and all(a in b or b in a for i, a in enumerate(vals) for b in vals[i + 1:]))):
            return True, f"Shared {field}: {vals[0]}"
    return False, "Evidence records share no allowed sea, topic, or holding-organisation context."


def annotate_program_verified(row: dict[str, Any], indexed_pages: list[dict[str, Any]]) -> bool:
    expected = norm_name(str(row.get("shared_name", "")))
    if not expected:
        return False
    matched = [page for page in indexed_pages if not page.get("name_truncated")
               and norm_name(str(page.get("record_name", ""))) == expected]
    urls = sorted({str(page.get("url", "")) for page in matched if page.get("url")})
    if len(urls) < 2:
        return False
    for point_index, point in enumerate(row.get("keypoints", []), 1):
        text = norm_name(str(point.get("text", "")))
        if "share the name" in text or "records named" in text or "records share" in text or "same name" in text:
            point["program_verified"] = True
            point["verification_method"] = "normalized_record_name_index_match"
            point["verified_record_name"] = row.get("shared_name")
            point["verified_record_urls"] = urls
            for check_name in ("quote_verbatim", "quote_support", "derivation_check"):
                section = row.get("checks", {}).get(check_name) or {}
                entries = section.get("keypoints", [])
                for entry in entries:
                    if entry.get("keypoint_index") == point_index:
                        entry.update({"applicable": False, "passed": True,
                                      "reason": "Programmatically verified from the normalized record-name index."})
                        entry["program_verified"] = True
                if check_name == "quote_verbatim":
                    section["passed"] = all(not entry.get("applicable") or entry.get("verbatim") for entry in entries)
                elif check_name == "quote_support":
                    section["passed"] = all(not entry.get("applicable") or entry.get("passed") for entry in entries)
                elif check_name == "derivation_check":
                    section["passed"] = all(not entry.get("applicable") or entry.get("derivable") for entry in entries)
            return True
    return False


def update_reasons(row: dict[str, Any]) -> None:
    checks = row.get("checks", {})
    reasons: list[str] = []
    if checks.get("absence_check") and not checks["absence_check"].get("passed"):
        reasons.append("absence_check_failed")
    condition = checks.get("condition_semantics") or {}
    if condition and not condition.get("passed"):
        if condition.get("needs_review"):
            row.setdefault("needs_review_reasons", []).append("condition_semantics_inconsistent_after_retry")
        else:
            reasons.append("semantic_condition_mismatch")
    for name, reason in (("quote_support", "quote_support_failed"), ("quote_verbatim", "quote_verbatim_failed")):
        section = checks.get(name) or {}
        if section and not section.get("passed"):
            reasons.append(reason)
    if (checks.get("derivation_check") or {}).get("applicable") and not checks["derivation_check"].get("passed"):
        reasons.append("derivation_check_failed")
    for name, reason in (("multi_page", "multi_page_requirement_failed"), ("single_page_check", "single_page_sufficient"),
                         ("unique_bridge", "bridge_not_unique"), ("comparison_multi_validity", "comparison_values_or_relation_invalid"),
                         ("plausibility_check", "implausible_research_question")):
        section = checks.get(name) or {}
        if section and section.get("passed") is False:
            reasons.append(reason)
    if checks.get("near_duplicate", {}).get("flagged"):
        reasons.append("near_duplicate")
    for name, value in (checks.get("rules") or {}).items():
        if isinstance(value, dict) and value.get("passed") is False:
            reasons.append(f"rule_failed:{name}")
    if any(isinstance(value, dict) and value.get("matches_reference") is False
           for value in (checks.get("round_trip") or {}).values()):
        reasons.append("round_trip_failed")
    if checks.get("set_completeness") and not checks["set_completeness"].get("passed"):
        reasons.append("set_completeness_failed")
    row["auto_rejection_reasons"] = list(dict.fromkeys(reasons))
    row["review_eligible"] = not reasons or bool(row.get("needs_review_reasons"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Run deterministic mock generation and checks only.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if not args.mock:
        load_dotenv(ROOT / ".env", override=True)

    main2_rows = read_jsonl(MAIN2 / "questions_s2_main2_checked.jsonl")
    main2_review = [row for row in main2_rows if row.get("status") == "generated" and row.get("review_eligible") is True]
    if len(main2_review) != 10:
        raise RuntimeError(f"Expected 10 main2 review questions; found {len(main2_review)}.")
    indexed_pages = read_jsonl(INDEX / "indexed_pages.jsonl")
    page_map = {str(page.get("url")): page for page in indexed_pages}

    ambiguity_rechecks = [dict(row) for row in main2_rows if row.get("id") in {"S2-AMBIGUITY-001", "S2-AMBIGUITY-002"}]
    for row in ambiguity_rechecks:
        verified = annotate_program_verified(row, indexed_pages)
        row["program_verified_name_claim"] = verified
        if verified:
            update_reasons(row)
        else:
            row["auto_rejection_reasons"] = list(row.get("auto_rejection_reasons", [])) + ["normalized_record_name_not_confirmed"]
            row["review_eligible"] = False

    # Build a reproducible candidate pool in mock mode first. Entity-link screening is
    # not used in main3 because this round explicitly excludes multi-hop.
    bundle_path = out / ("mock/bundles_main3.jsonl" if args.mock else "bundles_main3.jsonl")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundles, raw_counts, bundle_stats = build_bundles(42, 99, mock=True, output_dir=out / "mock", main3=True)
    write_jsonl(bundle_path, bundles)
    bundles = [row for row in bundles if row.get("type") in {"set_all", "unanswerable", "comparison_multi"}]
    existing_set_answers = [answer_set(row) for row in main2_rows if row.get("type") == "set_all"]
    excluded: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    per_type = Counter()
    for row in bundles:
        kind = row.get("type")
        reason = ""
        if is_icos_etc_bundle(row):
            reason = "icos_etc_catalogue_limit_reached"
        elif kind == "set_all":
            if not CLASSIFICATION.fullmatch(str(row.get("target_field", "")).strip()):
                reason = "set_all_field_not_classification_field"
            else:
                current = {norm_name(value) for value in row.get("expected_items", []) if value}
                for old in existing_set_answers:
                    denominator = min(len(current), len(old))
                    overlap = len(current & old) / denominator if denominator else 0.0
                    if overlap > 0.5:
                        reason = "set_all_answer_set_overlap_over_half"
                        row["answer_set_overlap"] = {"ratio": overlap, "basis": "intersection_over_smaller_set"}
                        break
        elif kind == "unanswerable":
            related, why = related_unanswerable(row, page_map)
            row["related_context_check"] = {"passed": related, "reason": why}
            if not related:
                reason = "unanswerable_conditions_not_related"
        cap = {"set_all": 8, "unanswerable": 10}.get(kind)
        if not reason and cap is not None and per_type[kind] >= cap:
            reason = f"main3_candidate_cap_{cap}"
        if reason:
            excluded.append({**row, "status": "rejected_before_generation", "auto_rejection_reasons": [reason], "review_eligible": False})
        else:
            filtered.append(row)
            per_type[kind] += 1
    type_ids = Counter()
    for row in filtered:
        type_ids[row["type"]] += 1
        row["bundle_id"] = f"S2-MAIN3-{row['type'].upper()}-{type_ids[row['type']]:03d}"

    # The smoke pass exercises generation, quote checks, retrieval checks, the new
    # plausibility judge path, and review eligibility without network calls.
    smoke_bundles = []
    for kind in ("set_all", "unanswerable", "comparison_multi"):
        candidate = next((row for row in filtered if row.get("type") == kind), None)
        if candidate:
            smoke_bundles.append(candidate)
    mock_generated = [generate_one(row, (ROOT / "prompts/teacher_generate_s2.md").read_text(encoding="utf-8").strip(),
                                   mock=True, output_root=out / "mock") for row in smoke_bundles]
    mock_checked = check_questions(mock_generated, output_root=out / "mock", mock=True)
    if any(row.get("status") not in {"generated", "skip", "generation_failed"} for row in mock_checked):
        raise RuntimeError("Mock smoke test returned an unexpected status.")
    if any((row.get("checks", {}).get("plausibility_check") or {}).get("passed") is not True for row in mock_checked if row.get("status") == "generated"):
        raise RuntimeError("Mock plausibility check did not pass for a generated smoke-test question.")
    write_jsonl(out / "mock/mock_checked.jsonl", mock_checked)

    if args.mock:
        print(json.dumps({"mock": True, "mock_smoke_rows": len(mock_checked), "candidate_bundles": len(filtered),
                          "mock_passed": sum(row.get("review_eligible") is True for row in mock_checked)}, indent=2))
        return

    generated_rows = [generate_one(row, (ROOT / "prompts/teacher_generate_s2.md").read_text(encoding="utf-8").strip(),
                                   mock=False, output_root=out) for row in filtered]
    generated = [row for row in generated_rows if row.get("status") == "generated"]
    checked_generated = check_questions(generated, output_root=out, mock=False, index=load_index(ROOT / "data/processed/index"))
    checked = [*checked_generated, *[row for row in generated_rows if row.get("status") != "generated"]]
    for row in checked:
        if row.get("status") == "generated":
            update_reasons(row)
    # Preserve all attempts, including static exclusions and teacher skips/failures.
    checked.extend(excluded)
    rescued = [row for row in ambiguity_rechecks if row.get("review_eligible") is True]
    review_rows = [*main2_review, *rescued,
                   *[row for row in checked if row.get("status") == "generated" and row.get("review_eligible") is True]]
    # Main2 already contains three ICOS ETC catalogue questions; never add an ICOS question.
    if any(is_icos_etc_bundle(row) for row in review_rows[10:]):
        raise RuntimeError("Main3 exceeded the ICOS ETC catalogue cap.")

    write_jsonl(out / "questions_s2_main3_checked.jsonl", [*ambiguity_rechecks, *checked])
    rejected = [row for row in [*ambiguity_rechecks, *checked]
                if row.get("status") != "generated" or row.get("review_eligible") is not True]
    write_jsonl(out / "questions_s2_main3_rejected.jsonl", rejected)
    payload = [review_row(row) for row in review_rows]
    (out / "review_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    by_type: dict[str, Any] = {}
    all_attempts = [*ambiguity_rechecks, *checked]
    for kind in ("ambiguity", "set_all", "unanswerable", "comparison_multi", "multi_hop"):
        typed = [row for row in all_attempts if row.get("type") == kind]
        reasons = Counter(reason for row in typed for reason in row.get("auto_rejection_reasons", []))
        if kind == "ambiguity":
            passed = sum(row.get("review_eligible") is True for row in ambiguity_rechecks)
        else:
            passed = sum(row.get("status") == "generated" and row.get("review_eligible") is True for row in typed)
        by_type[kind] = {"attempts": len(typed), "passed": passed, "rejected_or_skipped": len(typed) - passed,
                         "rejection_reasons": dict(sorted(reasons.items()))}
    by_type["multi_hop"] = {"attempts": 0, "passed": 0, "rejected_or_skipped": 0,
        "rejection_reasons": {"not_generated_this_round": 0}}
    summary = ["# S2 Main3 Final Top-Up Summary", "",
        "The main3 top-up reused the main2 checked questions, rechecked the two same-name ambiguity candidates with an index-based program verification, and generated only set_all, unanswerable, and comparison_multi questions. No S1 final output or raw source data was modified.", "",
        "## Candidate outcomes", "", "| Type | Attempts | Passed to review | Rejected or skipped | Rejection reasons |", "|---|---:|---:|---:|---|"]
    for kind in ("ambiguity", "set_all", "unanswerable", "comparison_multi", "multi_hop"):
        stats = by_type[kind]
        reason_text = "; ".join(f"{name}: {count}" for name, count in stats["rejection_reasons"].items()) or "None"
        summary.append(f"| {kind} | {stats['attempts']} | {stats['passed']} | {stats['rejected_or_skipped']} | {reason_text} |")
    summary += ["", "## Multi-hop outcome", "",
        "No multi-hop question was generated in main3. Across the rounds, candidates built from text pages did not yield a question that was both natural and necessarily dependent on evidence from two pages.", "",
        f"Main2 review questions retained: {len(main2_review)}; ambiguity candidates rescued: {len(rescued)}; new questions passed to review: {sum(row.get('status') == 'generated' and row.get('review_eligible') is True for row in checked)}; final review rows: {len(review_rows)}; rejected or skipped records: {len(rejected)}.", ""]
    (out / "s2_main3_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (out / "main3_counts.json").write_text(json.dumps({"by_type": by_type, "bundle_raw_counts": raw_counts,
        "bundle_selected_counts": bundle_stats.get("selected_by_type", {}), "review_rows": len(review_rows),
        "rescued_ambiguity": [row.get("id") for row in rescued], "rejected_records": len(rejected)}, indent=2), encoding="utf-8")
    print(json.dumps({"by_type": by_type, "review_rows": len(review_rows), "rescued_ambiguity": len(rescued),
                      "new_passed": sum(row.get("status") == "generated" and row.get("review_eligible") is True for row in checked),
                      "rejected_or_skipped": len(rejected), "summary": str(out / "s2_main3_summary.md")}, indent=2))


if __name__ == "__main__":
    main()
