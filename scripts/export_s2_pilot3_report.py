"""Export the S2 round-three review list, rejection log, and audit summaries."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def compact_checks(question: dict[str, Any]) -> dict[str, Any]:
    checks = question.get("checks", {})
    quote = checks.get("quote_verbatim") or {}
    quote_support = checks.get("quote_support") or {}
    multi = checks.get("multi_page") or {}
    complete = checks.get("set_completeness") or {}
    absence = checks.get("absence_check") or {}
    condition = checks.get("condition_semantics") or {}
    round_trip = checks.get("round_trip") or {}
    closed = checks.get("closed_book") or {}
    duplicate = checks.get("near_duplicate") or {}
    unsupported = [row.get("keypoint_index") for row in quote_support.get("keypoints", []) if not row.get("supported")]
    absence_summary = {key: {"passed": value.get("status") == "passed", "answerable": value.get("answerable"),
                             "bm25_top_50_count": len((value.get("retrieval") or {}).get("bm25_top_50", [])),
                             "vector_top_50_count": len((value.get("retrieval") or {}).get("vector_top_50", [])),
                             "merged_retrieved_passage_count": len((value.get("retrieval") or {}).get("merged", [])),
                             "flagged_passage_count": len(value.get("answering_passages", [])),
                             "reason": (value.get("judge") or {}).get("reason", "")}
                       for key, value in absence.items() if isinstance(value, dict)}
    condition_summary = None
    if condition:
        condition_summary = {key: condition.get(key) for key in (
            "passed", "type", "candidate_count", "catalogue_member_count", "expected_matching_record_ids",
            "judge_matching_record_ids", "missing_judge_decisions", "unexpected_matching_record_ids",
            "inconsistent_record_vs_matching_list_ids")}
    return {
        "quote_verbatim": {"passed": quote.get("passed"), "keypoints": len(quote.get("keypoints", [])),
                           "failed_keypoints": [r.get("keypoint_index") for r in quote.get("keypoints", []) if not r.get("verbatim")]},
        "quote_support": {"passed": quote_support.get("passed"), "keypoints": len(quote_support.get("keypoints", [])),
                          "unsupported_keypoints": unsupported},
        "multi_page": {"passed": multi.get("passed"), "decisive_source_page_count": multi.get("decisive_source_page_count"),
                       "required": multi.get("required")},
        "set_completeness": {"passed": complete.get("passed"), "missing_from_answer": complete.get("missing_from_answer", []),
                             "missing_from_catalogue_list": complete.get("missing_from_catalogue_list", [])},
        "absence_check": absence_summary or None,
        "conflict_check": checks.get("conflict_check"),
        "condition_semantics": condition_summary,
        "round_trip": {key: {"matches_reference": value.get("matches_reference"), "reason": value.get("reason", "")}
                       for key, value in round_trip.items() if isinstance(value, dict)},
        "closed_book": {"correct": closed.get("correct"), "judge_reason": (closed.get("judge") or {}).get("reason", "")},
        "near_duplicate": {"flagged": duplicate.get("flagged"), "matches": duplicate.get("matches", [])},
        "rules": checks.get("rules", {}),
        "phrasing_overlap": checks.get("phrasing_overlap", {}),
        "auto_rejection_reasons": question.get("auto_rejection_reasons", []),
    }


def review_row(question: dict[str, Any]) -> dict[str, Any]:
    pages = question.get("pages", [])
    return {
        "id": question.get("id"), "type": question.get("type"), "user_task": question.get("user_task"),
        "question_user": question.get("question_user"), "question_source": question.get("question_source"),
        "expected_behaviour": question.get("expected_behaviour"), "reference_answer": question.get("reference_answer"),
        "keypoints (decisive, source, quote)": question.get("keypoints", []),
        "evidence_pages (title, url)": [{"title": p.get("title"), "url": p.get("url"), "evidence_id": p.get("evidence_id")} for p in pages],
        "quote_verbatim": compact_checks(question)["quote_verbatim"],
        "quote_support": compact_checks(question)["quote_support"],
        "multi_page": compact_checks(question)["multi_page"],
        "set_completeness": compact_checks(question)["set_completeness"],
        "absence_check": compact_checks(question)["absence_check"],
        "conflict_check": compact_checks(question)["conflict_check"],
        "condition_semantics": compact_checks(question)["condition_semantics"],
        "round_trip": compact_checks(question)["round_trip"],
        "closed_book": compact_checks(question)["closed_book"],
        "near_duplicate": compact_checks(question)["near_duplicate"],
        "rules": compact_checks(question)["rules"],
        "phrasing_overlap": compact_checks(question)["phrasing_overlap"],
        "difficulty": question.get("difficulty", {}),
        "decision (accept / edit / reject)": "", "would_researcher_ask (yes / no)": "",
        "edited_question_user": "", "edited_question_source": "", "edited_reference_answer": "",
        "edited_keypoints": "", "note": "",
    }


def render_index_summary(out: Path) -> None:
    index_dir = ROOT / "data/processed/s2"
    metadata = json.loads((index_dir / "s2_index_metadata.json").read_text(encoding="utf-8"))
    catalogues = read_jsonl(index_dir / "catalogues.jsonl")
    diagnostics = read_jsonl(index_dir / "directory_diagnostics.jsonl")
    exclusions = read_jsonl(index_dir / "page_exclusions.jsonl")
    claims = read_jsonl(index_dir / "page_claims.jsonl")
    bridges = read_jsonl(index_dir / "claim_bridges.jsonl")
    link_screen = read_jsonl(index_dir / "entity_link_screening.jsonl")
    conflicts = read_jsonl(index_dir / "conflict_candidates.jsonl")
    failed_pages = sum(bool(row.get("error")) for row in claims)
    exclusion_totals = Counter(row.get("reason", "unspecified") for row in exclusions)
    summary = ["# S2 Index Summary — Round 3", "", f"- Indexed pages: {metadata.get('indexed_pages', 0)}",
               f"- Catalogues: {len(catalogues)}", f"- Eligible pages with extracted claims: {sum(bool(r.get('claims')) for r in claims)}",
               f"- Extracted factual claims: {sum(len(r.get('claims', [])) for r in claims)}",
               f"- Claim-extraction error pages: {failed_pages}", f"- Entity/fact bridges: {len(bridges)} ({sum(bool(r.get('cross_site')) for r in bridges)} cross-site)",
               f"- Entity links screened by judge: {len(link_screen)}; passed: {sum(r.get('passed') is True for r in link_screen)}",
               f"- Same-name groups with differing relevant fields: {sum(r.get('kind') == 'ambiguity' for r in conflicts)}",
               f"- S1 conflict observations carried forward: {sum(r.get('kind') == 'conflict' for r in conflicts)}", "",
               "## Page exclusions", "", "| Reason | Pages |", "|---|---:|"]
    summary.extend(f"| {reason} | {count} |" for reason, count in sorted(exclusion_totals.items()))
    summary += ["", "## Catalogue membership and field coverage", "",
                "Each row lists shared field labels and the per-field coverage within that URL-prefix group. Prefix diagnostics, including exclusions, follow the catalogue table.",
                "", "| Catalogue | Members | Shared fields | Field coverage |", "|---|---:|---|---|"]
    for cat in sorted(catalogues, key=lambda r: (-len(r.get("members", [])), r.get("name", ""))):
        coverage = cat.get("field_coverage", {})
        coverage_text = "; ".join(f"{field} {values.get('present', 0)}/{values.get('member_count', len(cat.get('members', [])))} ({values.get('coverage_pct', 0):.1f}%)"
                                   for field, values in sorted(coverage.items()))
        summary.append(f"| {cat.get('name', cat.get('catalogue_id', ''))} | {len(cat.get('members', []))} | {', '.join(cat.get('common_fields', []))} | {coverage_text} |")
    summary += ["", "## Prefix-level inclusion and exclusion diagnostics", "",
                "| URL prefix | Eligible | Common fields | Field coverage | Excluded | Exclusion reason |", "|---|---:|---|---|---:|---|"]
    for row in sorted(diagnostics, key=lambda r: r.get("url_prefix", "")):
        coverage = "; ".join(f"{field} {values.get('present', 0)}/{values.get('member_count', row.get('eligible_members', 0))} ({values.get('coverage_pct', 0):.1f}%)"
                              for field, values in sorted(row.get("field_coverage", {}).items()))
        excluded = "; ".join(f"{reason}: {count}" for reason, count in sorted(row.get("excluded_reasons", {}).items())) or "0"
        summary.append(f"| {row.get('url_prefix', '')} | {row.get('eligible_members', 0)} | {', '.join(row.get('common_fields', [])) or 'None'} | {coverage or 'None'} | {row.get('excluded_member_count', 0)} | {row.get('exclusion_reason', '')}; {excluded} |")
    summary += ["", "## Previously missed sources", "",
                "- NVS EXV current-term pages are included: 52 catalogue members; fields include URI, definition, identifier, date, and vocabulary relations.",
                "- eLTER external-dataset pages are eligible; a 6-member field-rich catalogue is formed, while the full 20-page prefix has only one field common to all pages and is not treated as one catalogue.",
                "- GOOS global-ocean-observing-network pages: 24 eligible pages, but only two sparse repeated labels (`network` and `programme`) and no three-field catalogue; prose facts are handled through page claims instead.",
                "- ETC L2 ARCHIVE catalogue membership includes five records, including San Rossore 2.", "",
                "## Examples of extracted claim bridges", ""]
    for bridge in bridges[:5]:
        a, b = bridge.get("fact_a", {}), bridge.get("fact_b", {})
        summary.append(f"- {bridge.get('entity', '')}: {a.get('url', '')} — {a.get('quote', '')} / {b.get('url', '')} — {b.get('quote', '')}")
    out.write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/s2/pilot3")
    args = parser.parse_args()
    output = args.output_root
    questions = read_jsonl(output / "questions_s2_checked.jsonl")
    generation_skips = read_jsonl(output / "generation_skips_s2.jsonl")
    review = [row for row in questions if row.get("status") == "generated" and row.get("review_eligible") is True]
    rejected = []
    for row in questions:
        if row.get("status") == "generated" and row.get("review_eligible") is False:
            rejected.append({"id": row.get("id"), "type": row.get("type"), "candidate_source": row.get("candidate_source"),
                             "status": "auto_rejected", "reasons": row.get("auto_rejection_reasons", []),
                             "question_user": row.get("question_user"), "check_summary": compact_checks(row)})
        elif row.get("status") != "generated":
            rejected.append({"id": row.get("id"), "type": row.get("type"), "status": row.get("status"),
                             "reasons": [row.get("skip_reason", row.get("status", "not_generated"))]})
    for row in generation_skips:
        rejected.append({"id": row.get("id"), "type": row.get("type"), "candidate_source": row.get("candidate_source"),
                         "status": "teacher_skipped_candidate", "reasons": [row.get("skip_reason", "Teacher skipped the candidate.")],
                         "bundle_id": row.get("id")})
    dump_jsonl(output / "questions_s2_rejected.jsonl", rejected)
    payload = [review_row(row) for row in review]
    (output / "review_sheet_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_index_summary(ROOT / "data/processed/s2/s2_index_summary.md")

    checked = [row for row in questions if row.get("status") == "generated"]
    counts = Counter(row.get("type") for row in checked)
    bundle_summary = json.loads((ROOT / "data/processed/s2/bundle_summary_s2.json").read_text(encoding="utf-8"))
    prompt_sha256 = next((row.get("teacher_prompt_sha256") for row in checked if row.get("teacher_prompt_sha256")), "")
    old_rows = read_jsonl(ROOT / "outputs/s2/pilot2/questions_s2_checked.jsonl")
    new_rows = read_jsonl(output / "pilot2_closed_book_rechecked.jsonl")
    old_correct = sum(row.get("closed_book", {}).get("correct") is True for row in old_rows if row.get("status") == "generated")
    new_correct = sum(row.get("closed_book", {}).get("correct") is True for row in new_rows if row.get("status") == "generated")
    checks_by_type = Counter()
    details = ["# S2 Pilot Round 3 Summary", "", f"Pilot questions generated: {len(checked)} of 10 slots; review eligible: {len(review)}; auto-rejected: {len(checked) - len(review)}.",
               f"Teacher prompt SHA-256: `{prompt_sha256}`.",
               "", "## Available package counts", "", "| Type | Raw candidates | Selected packages | Selected sources | Can reach 9? |", "|---|---:|---:|---|---|"]
    raw_by_source = bundle_summary["entity_links"]["raw_by_source"]
    selected_by_source = bundle_summary["entity_links"]["selected_by_source"]
    for kind in ("set_all", "multi_hop", "comparison_multi", "ambiguity", "unanswerable"):
        raw_source = "; ".join(f"{s}: {n}" for s, n in raw_by_source.get(kind, {}).items())
        selected_source = "; ".join(f"{s}: {n}" for s, n in selected_by_source.get(kind, {}).items())
        selected_count = bundle_summary["by_type"].get(kind, 0)
        details.append(f"| {kind} | {bundle_summary['raw_candidates'].get(kind, 0)} ({raw_source}) | {selected_count} | {selected_source} | {'Yes' if selected_count >= 9 else 'No'} |")
    details += ["", f"Total selected packages: {bundle_summary['selected_bundles']}. The current pool supports {bundle_summary['selected_bundles']} under the three-per-site/catalogue diversity cap, below the approximately 35-question target. Only `multi_hop` reaches nine; actual entity-link screening retained {bundle_summary['entity_links'].get('judge_passed', 0)} of {bundle_summary['entity_links'].get('eligible_for_judge_screen', 0)} candidate links.",
                "", "## Pilot question and check details", ""]
    for row in questions:
        details += [f"### {row.get('id')} — {row.get('type')} ({row.get('status')})", ""]
        if row.get("status") == "generated":
            details += [f"- User task: {row.get('user_task')}", f"- Question (user wording): {row.get('question_user')}",
                        f"- Question (source wording): {row.get('question_source')}", f"- Expected behaviour: {row.get('expected_behaviour')}",
                        f"- Reference answer: {row.get('reference_answer')}", "- Keypoints:"]
            details.extend(f"  - [{'D' if point.get('decisive') else 'context'}] {point.get('text')} (source {point.get('source')}; quote: `{point.get('quote')}`)" for point in row.get("keypoints", []))
            details.append("- Evidence pages:")
            details.extend(f"  - {page.get('title')} — {page.get('url')}" for page in row.get("pages", []))
            details += ["- Complete teacher JSON:", "```json", json.dumps(row.get("teacher_raw_json", {}), ensure_ascii=False, indent=2), "```"]
            checks = compact_checks(row)
            details += ["- Automated checks:"]
            details.extend(f"  - `{name}`: `{json.dumps(value, ensure_ascii=False)}`" for name, value in checks.items())
        else:
            details.append(f"- Skip reason: {row.get('skip_reason', 'No usable package remained for this slot.')}")
        details.append("")
    details += ["## Teacher-skipped candidate attempts", ""]
    if generation_skips:
        for row in generation_skips:
            details.append(f"- {row.get('id')} ({row.get('type')}, {row.get('candidate_source', 'catalogue_field')}): {row.get('skip_reason', 'No reason supplied.')}")
    else:
        details.append("- None.")
    details += ["", "## Strict closed-book recheck of pilot2", "",
                f"Generated pilot2 questions: {sum(row.get('status') == 'generated' for row in new_rows)}. Previously correct: {old_correct}; under the strict all-decisive-keypoints rule: {new_correct}.",
                "", "## Output files", "", "- Full machine-readable questions and all unabridged checks: `questions_s2_checked.jsonl`.",
                "- Questions removed from review and their reasons: `questions_s2_rejected.jsonl`.",
                "- Human-review worksheet contains only candidates that passed all automatic rejection gates.", ""]
    (output / "pilot3_summary.md").write_text("\n".join(details), encoding="utf-8")
    print(json.dumps({"generated_by_type": dict(counts), "review_eligible": len(review), "auto_rejected": len(checked)-len(review),
                      "teacher_skipped_candidates": len(generation_skips), "pilot2_closed_book_old_correct": old_correct,
                      "pilot2_closed_book_strict_correct": new_correct, "rejected_list": str(output / 'questions_s2_rejected.jsonl'),
                      "summary": str(output / 'pilot3_summary.md'), "review_rows": str(output / 'review_sheet_data.json')}, indent=2))


if __name__ == "__main__":
    main()
