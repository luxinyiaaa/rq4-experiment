"""Assemble audited S2 question-set v3 outputs from v2 plus the completed top-up."""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs/rawpage_questions/question_set_v2/final"
AUDIT_IN = ROOT / "outputs/rawpage_questions/question_set_v2/audit/audit.jsonl"
OUT = ROOT / "outputs/rawpage_questions/question_set_v3"
WORK = OUT / "work_questions_all.jsonl"
NEW_AUDIT = OUT / "audit_new.jsonl"
OLD_BOOK = V2 / "questions_v2_final.xlsx"
OLD_SUPERVISOR = V2 / "questions_for_supervisors.xlsx"
OLD_SAMPLE = V2 / "review_sample.xlsx"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def workbook_rows(path: Path, sheet: str) -> tuple[list[str], list[dict[str, Any]]]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    iterator = ws.iter_rows(values_only=True)
    headers = [str(x) if x is not None else "" for x in next(iterator)]
    rows = []
    for values in iterator:
        if not values or values[0] is None:
            continue
        rows.append({headers[i]: values[i] if i < len(values) else None for i in range(len(headers))})
    wb.close()
    return headers, rows


def all_question_headers() -> list[str]:
    headers, _ = workbook_rows(OLD_BOOK, "main")
    return headers


def as_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return value if value is not None else ""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(WORK)
    by_id = {q["id"]: q for q in questions}

    # Add IDs and page context omitted by the audit payload helper to its new rows.
    new_audit = read_jsonl(NEW_AUDIT)
    for row in new_audit:
        if not row.get("id"):
            q = next((x for x in questions if x.get("question") == row.get("question")), None)
            if q is None:
                raise RuntimeError("Could not map a new audit result back to its question")
            row["id"] = q["id"]
            row["url"] = q.get("url", "")
            row["page_id"] = q.get("page_id", "")
            row["set"] = "reserve" if row.get("flagged") else q.get("set", "main")
    write_jsonl(NEW_AUDIT, new_audit)

    # Preserve the source audit labels, then attach explicit human decisions.
    audit_rows = read_jsonl(AUDIT_IN) + new_audit
    for row in audit_rows:
        qid = row.get("id")
        row["manual_decision"] = ""
        row["manual_note"] = ""
        if qid in {"AS-113-d4277b7f44_1", "AS-005-98a245cac8_1", "AS-043-20aae7e822_1", "AS-074-2865827074_1"}:
            row["manual_decision"] = "remove"
            row["manual_note"] = by_id[qid].get("removal_reason", "")
        elif qid == "DR-027-aa63022fae_2":
            row["manual_decision"] = "edit"
            row["manual_note"] = "Corrected type, reference answer and keypoint decisiveness; cleared false_premise."
        elif qid == "AS-166-d10899302d_1":
            row["manual_decision"] = "keep"
            row["manual_note"] = "The page lists exactly these four items."
    write_jsonl(OUT / "audit.jsonl", audit_rows)

    main_rows = [q for q in questions if q.get("set") == "main"]
    extra_rows = [q for q in questions if q.get("set") == "extra"]
    reserve_rows = [q for q in questions if q.get("set") == "reserve"]
    pending_rows = [q for q in questions if q.get("set") == "pending"]
    removed_rows = [q for q in questions if q.get("set") == "removed"]
    if len(main_rows) != 300 or sum(q.get("page_type") == "dataset_record" for q in main_rows) != 60 or sum(q.get("page_type") == "about_science" for q in main_rows) != 240:
        raise RuntimeError("The final main set must contain 60 dataset_record and 240 about_science questions")
    if any(q.get("kb_label_final") != "in_kb" for q in main_rows):
        raise RuntimeError("A non-in_kb question entered the main set")
    about_site_counts = Counter(q.get("site") for q in questions if q.get("set") in {"main", "extra", "pending"} and q.get("page_type") == "about_science")
    if any(count > 60 for count in about_site_counts.values()):
        raise RuntimeError("The about_science 60-question site cap was exceeded")
    sets = [main_rows, extra_rows, reserve_rows, pending_rows, removed_rows]
    ids = [q["id"] for rows in sets for q in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("A question occurs in more than one set")
    if len(removed_rows) != 34:
        raise RuntimeError(f"Expected 34 removed questions, found {len(removed_rows)}")

    # Record all manual checks from v2 and the explicit v3 audit decisions.
    _, manual_rows = workbook_rows(OLD_BOOK, "manual_checks")
    manual_by_id = {str(r.get("id")): r for r in manual_rows}
    additions = {
        "AS-113-d4277b7f44_1": ("remove", by_id["AS-113-d4277b7f44_1"]["removal_reason"]),
        "AS-005-98a245cac8_1": ("remove", by_id["AS-005-98a245cac8_1"]["removal_reason"]),
        "AS-043-20aae7e822_1": ("remove", by_id["AS-043-20aae7e822_1"]["removal_reason"]),
        "AS-074-2865827074_1": ("remove", by_id["AS-074-2865827074_1"]["removal_reason"]),
        "DR-027-aa63022fae_2": ("edit", "Corrected type, reference answer and keypoint decisiveness; cleared false_premise."),
        "AS-166-d10899302d_1": ("keep", "The page lists exactly these four items."),
    }
    for qid, (decision, note) in additions.items():
        row = manual_by_id.setdefault(qid, {"id": qid, "set": "removed" if decision == "remove" else "main", "kb_label_final": by_id[qid].get("kb_label_final", ""), "kb_check_note": "", "table_answer_ok": "", "table_note": ""})
        row["decision"] = decision
        row["note"] = note
    for row in manual_by_id.values():
        row.setdefault("decision", "")
        row.setdefault("note", "")
    manual_rows_out = sorted(manual_by_id.values(), key=lambda x: str(x.get("id", "")))
    (OUT / "manual_checks.json").write_text(json.dumps(manual_rows_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for name, rows in (("questions_main.jsonl", main_rows), ("questions_extra.jsonl", extra_rows), ("questions_all.jsonl", questions)):
        write_jsonl(OUT / name, rows)

    # Keep the previous sample untouched except for removed items; add same-page-type replacements using seed 42.
    sample_headers, sample_rows = workbook_rows(OLD_SAMPLE, "review_sample")
    sample_ids = {str(r["id"]) for r in sample_rows}
    removed_sample = [r for r in sample_rows if str(r["id"]) in additions and additions[str(r["id"])][0] == "remove"]
    kept_sample = [r for r in sample_rows if r not in removed_sample]
    used_ids = {str(r["id"]) for r in kept_sample}
    rng = random.Random(42)
    replacement_ids: list[str] = []
    for old in removed_sample:
        old_q = by_id[str(old["id"])]
        candidates = [q for q in main_rows if q.get("page_type") == old_q.get("page_type") and q["id"] not in used_ids]
        candidates.sort(key=lambda q: q["id"])
        if not candidates:
            raise RuntimeError(f"No replacement sample question available for {old['id']}")
        replacement = rng.choice(candidates)
        used_ids.add(replacement["id"])
        replacement_ids.append(replacement["id"])
        kept_sample.append({"id": replacement["id"], "question": replacement["question"], "reference_answer": replacement["reference_answer"], "keypoints with quotes": format_keypoints_quotes(replacement), "url": replacement["url"], "question_ok": "", "answer_ok": "", "note": ""})
    # Update changed content while preserving the sampled question IDs and review entries.
    for row in kept_sample:
        current = by_id.get(str(row.get("id")))
        if current and current.get("id") == "DR-027-aa63022fae_2":
            row["question"] = current["question"]
            row["reference_answer"] = current["reference_answer"]
            row["keypoints with quotes"] = format_keypoints_quotes(current)
    (OUT / "review_sample_data.json").write_text(json.dumps({"headers": sample_headers, "rows": kept_sample, "replaced_ids": [str(r["id"]) for r in removed_sample], "replacement_ids": replacement_ids}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Prepare the supervisor-facing content while retaining the v2 column conventions.
    supervisor_headers, _ = workbook_rows(OLD_SUPERVISOR, "main")
    readme_headers, readme_rows = workbook_rows(OLD_SUPERVISOR, "README")
    main_types = Counter(q["type"] for q in main_rows)
    main_by_type_page = Counter((q["page_type"], q["type"]) for q in main_rows)
    extra_labels = Counter(q.get("kb_label_final", "") for q in extra_rows)
    main_manual_kb = sum(bool(manual_by_id.get(q["id"], {}).get("kb_check_note")) for q in main_rows)
    extra_manual_kb = sum(bool(manual_by_id.get(q["id"], {}).get("kb_check_note")) for q in extra_rows)
    table_main = sum(q.get("table_based") == "yes" for q in main_rows)
    edmed = sum(q.get("site") == "edmed.seadatanet.org" for q in main_rows if q.get("page_type") == "dataset_record")
    site_counts = Counter(q.get("site") for q in main_rows if q.get("page_type") == "about_science")
    fetch_dates = []
    page_csvs = [ROOT / "outputs/rawpage_questions/main/pages.csv", ROOT / "outputs/rawpage_questions/question_set/pages.csv", ROOT / "outputs/rawpage_questions/question_set_final/topup_pages.csv", V2 / "topup_pages.csv", OUT / "topup_pages.csv"]
    for page_csv in page_csvs:
        if page_csv.exists():
            for r in csv.DictReader(page_csv.open(encoding="utf-8", newline="")):
                if r.get("fetch_time_utc"):
                    fetch_dates.append(r["fetch_time_utc"][:10])
    import re
    previous_fetching = next((str(r.get("Details") or "") for r in readme_rows if r.get("Section") == "Fetching"), "")
    fetch_dates.extend(re.findall(r"\d{4}-\d{2}-\d{2}", previous_fetching))
    date_range = f"{min(fetch_dates)}–{max(fetch_dates)}" if fetch_dates else "2026-10-23–2026-10-25"
    readme_map = {str(r.get("Section")): str(r.get("Details") or "") for r in readme_rows}
    readme_map["Main and extra sets"] = f"Main set: {len(main_rows)} questions whose facts are on the same page in the knowledge base (in_kb): {sum(q['page_type']=='dataset_record' for q in main_rows)} dataset_record and {sum(q['page_type']=='about_science' for q in main_rows)} about_science; they are used for the main evaluation. Extra set: {len(extra_rows)} questions whose facts are on another page of the knowledge base (in_kb_elsewhere, {extra_labels.get('in_kb_elsewhere', 0)}) or not in it (not_in_kb, {extra_labels.get('not_in_kb', 0)}); they are answered in the same way but reported separately."
    sampling = readme_map.get("Sampling", "")
    sampling = sampling.replace("54 of the 60 come from it.", f"{edmed} of the 60 come from it.")
    readme_map["Sampling"] = sampling
    readme_map["Fetching"] = f"The current version of each page was downloaded on {date_range} (UTC) and its main text was extracted. Pages that could not be fetched or had less than 600 characters of text were skipped."
    readme_map["Question types"] = "list_all: asks for all items of a list on the page. paraphrased: asks for a fact in words different from the page. false_premise: contains a wrong detail that a good answer corrects before answering. Counts for the main set: " + ", ".join(f"{t} {main_types.get(t, 0)}" for t in ("list_all", "paraphrased", "false_premise")) + "."
    readme_map["Knowledge-base label"] = f"Whether the facts of the question are in the knowledge base (August 2025). in_kb: found in the knowledge-base text of the same page. in_kb_elsewhere: found on another page of the knowledge base. not_in_kb: not found; the page changed or the content is new since August 2025. Labels come from an automatic text check; {main_manual_kb} main-set and {extra_manual_kb} extra-set questions were checked manually. A not_in_kb label does not mean the system should refuse to answer: it is used to interpret results. Counts: main in_kb {sum(q.get('kb_label_final')=='in_kb' for q in main_rows)}; extra in_kb_elsewhere {extra_labels.get('in_kb_elsewhere', 0)}, not_in_kb {extra_labels.get('not_in_kb', 0)}."
    readme_map["Question audit"] = f"All {len(readme_rows) if False else len(read_jsonl(AUDIT_IN)) + len(new_audit)} main and extra questions were checked by a second model (Qwen3.7 Plus) for whether the reference answer gives what the question asks for, whether other pages could give a different correct answer, and whether false premises are clearly false. {len(REMOVE_REASONS_LOCAL)} questions were removed and 1 corrected after review."
    readme_map["Manual review"] = f"All {table_main} table-based questions in the main set were checked manually against the page. {len(removed_rows)} questions were removed after review: 21 earlier removals (10 job vacancies, 4 privacy-policy pages, 5 repeated-template dataset records, 1 answer not supported by its quote, 1 false premise unrelated to the question), 3 cross-page duplicates, 6 false-premise questions built on table values, and 4 questions removed after the question audit."
    # Replace in original order and insert the audit note after the KB label.
    readme_out = []
    for r in readme_rows:
        section = str(r.get("Section"))
        readme_out.append({"Section": section, "Details": readme_map.get(section, str(r.get("Details") or ""))})
        if section == "Knowledge-base label":
            readme_out.append({"Section": "Question audit", "Details": readme_map["Question audit"]})
    (OUT / "supervisor_data.json").write_text(json.dumps({"main_headers": supervisor_headers, "readme_headers": readme_headers, "readme": readme_out, "main": main_rows, "extra": extra_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Main report with verified counts and source-by-source tallies.
    topup_rows = list(csv.DictReader((OUT / "topup_pages.csv").open(encoding="utf-8", newline="")))
    lines = ["# Final Raw-Web Question Set V3", "", "## Counts by set", "", "| Set | Count |", "|---|---:|"]
    lines += [f"| {name} | {len(rows)} |" for name, rows in (("main", main_rows), ("extra", extra_rows), ("reserve", reserve_rows), ("pending", pending_rows), ("removed", removed_rows))]
    lines += ["", "## Counts by set, page type, type, site and KB label", "", "| Set | Page type | Type | Site | KB label | Count |", "|---|---|---|---|---|---:|"]
    lines += [f"| {set_name} | {pt} | {typ} | {site} | {label or '(blank)'} | {count} |" for (set_name, pt, typ, site, label), count in sorted(Counter((q.get("set"), q.get("page_type"), q.get("type"), q.get("site"), q.get("kb_label_final", "")) for q in questions).items())]
    lines += ["", "## Main by page type", "", "| Page type | Count |", "|---|---:|"]
    lines += [f"| {typ} | {sum(q.get('page_type') == typ for q in main_rows)} |" for typ in ("dataset_record", "about_science")]
    lines += ["", "## Main by question type", "", "| Type | Count |", "|---|---:|"]
    lines += [f"| {typ} | {count} |" for typ, count in sorted(Counter(q.get("type") for q in main_rows).items())]
    lines += ["", "## Main by site", "", "| Page type | Site | Count |", "|---|---|---:|"]
    lines += [f"| {pt} | {site} | {count} |" for (pt, site), count in sorted(Counter((q.get("page_type"), q.get("site")) for q in main_rows).items())]
    lines += ["", "## Extra by page type, type and site", "", "| Page type | Type | Site | Count |", "|---|---|---|---:|"]
    lines += [f"| {pt} | {typ} | {site} | {count} |" for (pt, typ, site), count in sorted(Counter((q.get("page_type"), q.get("type"), q.get("site")) for q in extra_rows).items())]
    lines += ["", "## Extra by label", "", "| KB label | Count |", "|---|---:|"]
    lines += [f"| {label or '(blank)'} | {count} |" for label, count in sorted(Counter(q.get("kb_label_final", "") for q in extra_rows).items())]
    lines += ["", f"Table-based questions in main: {table_main}", "", "## Manual checks", "", f"- Main KB notes: {main_manual_kb}", f"- Extra KB notes: {extra_manual_kb}", f"- Table-based main questions: {table_main}", f"- Question-level audit rows: {len(audit_rows)}", "- Explicit keep decisions: 1", "- Corrections after audit: 1", "- New question audit flags: " + str(sum(bool(a.get("flagged")) for a in new_audit)), "", "## Removals by reason", "", "| Reason code | Count |", "|---|---:|"]
    lines += [f"| {reason} | {count} |" for reason, count in sorted(Counter(q.get("reason_code", "unspecified") for q in removed_rows).items())]
    lines += ["", "## Top-up pages by status and skip reason", "", "| Status | Skip reason | Count |", "|---|---|---:|"]
    lines += [f"| {status} | {reason or '(none)'} | {count} |" for (status, reason), count in sorted(Counter((str(p.get("http_status")), str(p.get("skip_reason") or "")) for p in topup_rows).items())]
    teacher_calls = sum(int(p.get("teacher_calls") or 0) for p in topup_rows)
    prompt_tokens = sum(int(p.get("prompt_tokens") or 0) for p in topup_rows)
    completion_tokens = sum(int(p.get("completion_tokens") or 0) for p in topup_rows)
    judge_prompt = sum(int((a.get("usage") or {}).get("prompt_tokens") or 0) for a in new_audit)
    judge_completion = sum(int((a.get("usage") or {}).get("completion_tokens") or 0) for a in new_audit)
    lines += ["", "## New questions and model usage", "", f"- Generated: {sum(bool(q.get('v3_added')) for q in questions)}", f"- Added to main: {sum(bool(q.get('v3_added') and q.get('set') == 'main') for q in questions)}", f"- Sent to reserve: {len(reserve_rows)}", f"- Pending KB check: {sum(bool(q.get('v3_added') and q.get('set') == 'pending') for q in questions)}", f"- Teacher calls: {teacher_calls}; prompt tokens: {prompt_tokens}; completion tokens: {completion_tokens}", f"- Secondary judge calls: {len(new_audit)}; prompt tokens: {judge_prompt}; completion tokens: {judge_completion}", "", "## Audit and edits", "", "- Four audit-answer mismatches were removed from main and recorded with reason_code audit_answer_mismatch.", "- DR-027-aa63022fae_2 was corrected and logged as an edit in manual_checks.", "- AS-166-d10899302d_1 was kept with the requested note.", "", "## Review sample", "", f"- Removed sample questions replaced: {len(removed_sample)}", f"- Replacement IDs: {', '.join(replacement_ids) if replacement_ids else 'none'}", "", f"Top-up pages: {len(topup_rows)}. About-science site cap (60) checked against active main, extra and pending questions.", ""]
    (OUT / "summary_final.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Assembled v3: main={len(main_rows)}, extra={len(extra_rows)}, reserve={len(reserve_rows)}, pending={len(pending_rows)}, removed={len(removed_rows)}; sample replacements={len(removed_sample)}.")


def format_keypoints_quotes(q: dict[str, Any]) -> str:
    lines = []
    for point in q.get("keypoints", []):
        lines.append(("[decisive] " if point.get("decisive") else "[non-decisive] ") + str(point.get("text", "")))
    for quote in q.get("supporting_quotes", []):
        lines.append("[quote] " + str(quote.get("text", "")))
    return "\n".join(lines)


REMOVE_REASONS_LOCAL = {
    "AS-113-d4277b7f44_1", "AS-005-98a245cac8_1", "AS-043-20aae7e822_1", "AS-074-2865827074_1"
}

if __name__ == "__main__":
    main()
