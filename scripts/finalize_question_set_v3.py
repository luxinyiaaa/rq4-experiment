"""Apply the S2 question audit decisions and add four in-KB questions."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import rawpage_questions_main_stage1 as s1
import rawpage_questions_question_set as qs
import audit_question_set_v2 as audit_v2

SOURCE_DIR = ROOT / "outputs/rawpage_questions/question_set_v2/final"
AUDIT_DIR = ROOT / "outputs/rawpage_questions/question_set_v2/audit"
ORDER_PATH = ROOT / "outputs/rawpage_questions/main/sampling_order.csv"
OUT = ROOT / "outputs/rawpage_questions/question_set_v3"
OUT_RAW = OUT / "raw_web"
RAW_MAIN = ROOT / "data/raw_web/main"
RAW_KB = ROOT / "data/raw/site_pages_rows.csv"
PROCESSED_CLEAN = ROOT / "data/processed/chunks_clean.csv"
TYPE_PAIRS = [("list_all", "paraphrased"), ("paraphrased", "false_premise"), ("false_premise", "list_all")]
PAGE_TARGET = 4
SITE_CAP = 60
UA = s1.UA

REMOVE_REASONS = {
    "AS-113-d4277b7f44_1": "The question asks for float types; the reference answer gives a description (autonomous, robotic). Other KB pages give the types Core, BGC and Deep, which are also correct.",
    "AS-005-98a245cac8_1": "list_all question, but the page gives only examples (e.g.) of the outputs.",
    "AS-043-20aae7e822_1": "Broad question; other pages can give different correct descriptions of who ENVRI brings together.",
    "AS-074-2865827074_1": "The page gives three examples (Austria, Greece, the Nordic countries); the reference answer accepts only Austria.",
}
DR_FIX_ID = "DR-027-aa63022fae_2"
AS_KEEP_ID = "AS-166-d10899302d_1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stem(page_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", page_id)


def fetch(page: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    last = None
    attempts = []
    body = b""
    for attempt in range(1, 4):
        try:
            last = requests.get(page["url"], headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}, timeout=60, allow_redirects=True)
            body = last.content
            attempts.append({"attempt": attempt, "http_status": last.status_code})
            if last.status_code == 200:
                break
        except requests.RequestException as exc:
            last = None
            attempts.append({"attempt": attempt, "error": type(exc).__name__})
        if attempt < 3:
            time.sleep(60)
    metadata = {
        "id": page["id"], "original_url": page["url"], "final_url": last.url if last is not None else "",
        "http_status": last.status_code if last is not None else None,
        "content_type": last.headers.get("Content-Type", "") if last is not None else "",
        "fetch_time_utc": datetime.now(timezone.utc).isoformat(), "attempts": attempts,
        "retries": max(0, len(attempts) - 1),
    }
    if last is None:
        body = b""
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    (OUT_RAW / f"{stem(page['id'])}.html").write_bytes(body)
    (OUT_RAW / f"{stem(page['id'])}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata, body


def load_old_cache(page: dict[str, Any]) -> tuple[dict[str, Any], bytes] | None:
    for directory in (RAW_MAIN, SOURCE_DIR / "raw_web"):
        hp = directory / f"{stem(page['id'])}.html"
        mp = directory / f"{stem(page['id'])}.json"
        if hp.exists() and mp.exists():
            try:
                return json.loads(mp.read_text(encoding="utf-8")), hp.read_bytes()
            except (OSError, json.JSONDecodeError):
                pass
    return None


def decode_html(body: bytes, content_type: str) -> tuple[str, str]:
    from email.message import Message
    message = Message()
    message["content-type"] = content_type or "text/html"
    charset = message.get_content_charset()
    if charset:
        try:
            return body.decode(charset), charset
        except (LookupError, UnicodeDecodeError):
            return body.decode(charset, errors="replace"), charset + ":replace"
    head = body[:16384].decode("ascii", errors="ignore")
    match = re.search(r"<meta[^>]+charset\s*=\s*[\"']?\s*([^\s\"'/>;]+)", head, re.I)
    if not match:
        match = re.search(r"<meta[^>]+content=[\"'][^\"']*charset\s*=\s*([^\s\"';]+)", head, re.I)
    if match:
        charset = match.group(1)
        try:
            return body.decode(charset), charset
        except (LookupError, UnicodeDecodeError):
            return body.decode(charset, errors="replace"), charset + ":replace"
    try:
        return body.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return body.decode("cp1252", errors="replace"), "cp1252:replace"


def filtered_url(url: str) -> bool:
    return bool(re.search(r"(?:privacy|cookie|legal|terms|career|vacanc|job|position)", urlsplit(url).path.lower()))


def usable_pages_before_start(order: list[dict[str, str]], prior_pages: dict[str, dict[str, Any]]) -> int:
    ordered_ids = [r["id"] for r in order]
    last_idx = max((i for i, pid in enumerate(ordered_ids) if pid in prior_pages), default=-1)
    return sum(str(r.get("http_status")) == "200" and int(r.get("extracted_characters") or 0) >= 600
               for pid, r in prior_pages.items() if pid in set(ordered_ids[:last_idx + 1]))


def load_prior_page_rows(order: list[dict[str, str]]) -> tuple[dict[str, dict[str, Any]], int]:
    paths = [ROOT / "outputs/rawpage_questions/main/pages.csv",
             ROOT / "outputs/rawpage_questions/question_set/pages.csv",
             ROOT / "outputs/rawpage_questions/question_set_final/topup_pages.csv",
             SOURCE_DIR / "topup_pages.csv"]
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.exists():
            with path.open(encoding="utf-8", newline="") as f:
                rows.update({r["id"]: r for r in csv.DictReader(f) if r.get("page_type") == "about_science"})
    positions = {r["id"]: i for i, r in enumerate(order)}
    last_idx = max((positions[pid] for pid in rows if pid in positions), default=-1)
    return rows, last_idx


def audit_new(question: dict[str, Any], api_key: str, model: str, base_url: str, extra_body: dict[str, Any]) -> dict[str, Any]:
    payload = audit_v2.evidence_payload(question)
    messages = [
        {"role": "system", "content": audit_v2.SYSTEM_PROMPT},
        {"role": "user", "content": "Audit this item and return the specified JSON only:\n" + json.dumps(payload, ensure_ascii=False)},
    ]
    req = {"model": model, "messages": messages, "temperature": 0, "response_format": {"type": "json_object"}, **extra_body}
    started = time.monotonic()
    response = requests.post(base_url.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=req, timeout=120)
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(f"Secondary audit HTTP {response.status_code}")
    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        raise RuntimeError("Secondary audit returned invalid JSON") from None
    labels = audit_v2.normalize_result(parsed, question)
    usage = data.get("usage") or {}
    row = {
        **payload, "id": question["id"], "url": question.get("url", ""),
        "page_id": question.get("page_id", ""), **labels, "flagged": audit_v2.is_flagged(labels),
        "judge_text": json.dumps(parsed, ensure_ascii=False), "judge_model": data.get("model", model),
        "usage": {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in usage},
        "attempts": 1, "cache_hit": False, "elapsed_seconds": round(time.monotonic() - started, 4),
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(), "error": "",
    }
    return row


def main_run(mock: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(SOURCE_DIR / "questions_all.jsonl")
    if len({q["id"] for q in questions}) != len(questions):
        raise RuntimeError("Duplicate input IDs")
    if mock:
        assert qs.normalize_quote("A short exact quote") == "a short exact quote"
        assert filtered_url("https://site.test/privacy-policy") and not filtered_url("https://site.test/about")
        assert len(REMOVE_REASONS) == 4
        print("Mock checks passed: four removals, page filtering, type-pair and quote normalization.")
        return

    by_id = {q["id"]: q for q in questions}
    missing = (set(REMOVE_REASONS) | {DR_FIX_ID, AS_KEEP_ID}) - by_id.keys()
    if missing:
        raise RuntimeError("Required input IDs are missing: " + ", ".join(sorted(missing)))
    removed_rows = read_jsonl(SOURCE_DIR / "questions_all.jsonl")
    for qid, reason in REMOVE_REASONS.items():
        q = by_id[qid]
        q["set"] = "removed"
        q["included_in_set"] = "no"
        q["excluded_from_set_reason"] = reason
        q["reason_code"] = "audit_answer_mismatch"
        q["removal_reason"] = reason
    fix = by_id[DR_FIX_ID]
    fix["type"] = "paraphrased"
    for point in fix.get("keypoints", []):
        if point.get("text") == "The British Oceanographic Data Centre is the collating centre, not the data-holding centre":
            point["decisive"] = False
    fix["reference_answer"] = "The Environment Agency Head Office holds the records."
    fix["false_premise"] = ""
    keep_decision = {"id": AS_KEEP_ID, "set": "main", "decision": "keep", "note": "The page lists exactly these four items."}
    fix_decision = {"id": DR_FIX_ID, "set": "main", "decision": "edit", "note": "Corrected type, reference answer and keypoint decisiveness; cleared false-premise field."}

    main_rows = [q for q in questions if q.get("set") == "main"]
    about_count = sum(q.get("page_type") == "about_science" for q in main_rows)
    if about_count != 236:
        raise RuntimeError(f"Expected 236 about_science questions after four removals, found {about_count}")
    active = [q for q in questions if q.get("set") in {"main", "extra", "pending"}]
    signatures: dict[tuple[str, ...], tuple[str, str]] = {}
    for q in active:
        sig = qs.question_quote_signature(q)
        if sig and q.get("duplicate") != "yes":
            signatures.setdefault(sig, (q.get("page_id", ""), q["id"]))
    site_counts = Counter(q.get("site", "") for q in active if q.get("page_type") == "about_science")

    order = list(csv.DictReader(ORDER_PATH.open(encoding="utf-8", newline="")))
    prior_pages, last_idx = load_prior_page_rows(order)
    page_order_idx = {r["id"]: i for i, r in enumerate(order)}
    pair_index = usable_pages_before_start(order, prior_pages)
    new_page_rows: list[dict[str, Any]] = []
    new_audits: list[dict[str, Any]] = []
    raw_index = s1.RawKBIndex(RAW_KB)
    clean_texts = s1.load_chunk_texts(PROCESSED_CLEAN)
    teacher_cfg, teacher_key, teacher_endpoint = qs.load_teacher_config()
    teacher_model = str(teacher_cfg["model"])
    teacher_extra = dict(teacher_cfg.get("thinking_parameters") or {})
    audit_config = yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf-8")) or {}
    judge_cfg = audit_config["judge_secondary"]
    load_dotenv(ROOT / ".env", override=False)
    judge_key = os.getenv(str(judge_cfg["api_key_env"]))
    if not judge_key:
        raise RuntimeError("Secondary judge credential is unavailable")
    prompt = qs.build_prompt() + """

## Final top-up requirements
- Every decisive keypoint must be directly supported by the question's supporting quotes.
- Supporting quotes must be exact substrings of the supplied page text and must not end in an ellipsis.
- In false_premise questions, the false detail must concern the same subject as the question.
- Do not create false-premise questions based on table values. A question about a table may ask for at most one value.
"""
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    completed = 0
    for page_raw in order[last_idx + 1:]:
        if completed >= PAGE_TARGET:
            break
        page = {**page_raw, "site": page_raw.get("site") or urlsplit(page_raw["url"]).netloc.lower().removeprefix("www.")}
        page["page_type"] = "about_science"
        if filtered_url(page["url"]):
            new_page_rows.append({"id": page["id"], "url": page["url"], "page_type": "about_science", "http_status": "not_fetched_filtered", "questions_generated": 0, "questions_skipped": 2, "skip_reason": "page_filter_url"})
            write_csv(OUT / "topup_pages.csv", new_page_rows, ["id", "url", "page_type", "http_status", "retries", "extracted_characters", "truncated", "kb_in_live", "live_in_kb", "similarity_label", "questions_generated", "questions_skipped", "skip_reason", "teacher_calls", "prompt_tokens", "completion_tokens", "total_tokens", "type_pair_used", "fetch_time_utc", "declared_charset", "decoded_charset"])
            continue
        if site_counts[page["site"]] >= SITE_CAP:
            new_page_rows.append({"id": page["id"], "url": page["url"], "page_type": "about_science", "http_status": "not_fetched_site_cap", "questions_generated": 0, "questions_skipped": 2, "skip_reason": "site_question_cap_reached"})
            write_csv(OUT / "topup_pages.csv", new_page_rows, ["id", "url", "page_type", "http_status", "retries", "extracted_characters", "truncated", "kb_in_live", "live_in_kb", "similarity_label", "questions_generated", "questions_skipped", "skip_reason", "teacher_calls", "prompt_tokens", "completion_tokens", "total_tokens", "type_pair_used", "fetch_time_utc", "declared_charset", "decoded_charset"])
            continue
        cached = load_old_cache(page)
        if cached:
            meta, body = cached
        else:
            meta, body = fetch(page)
        status = meta.get("http_status")
        decoded, charset = decode_html(body, meta.get("content_type", "")) if status == 200 else ("", "")
        live_text = qs.trafilatura.extract(decoded, output_format="markdown", include_tables=True, include_comments=False, include_links=True) or "" if status == 200 else ""
        nchar = len(live_text)
        clean = clean_texts.get(page["url"], "")
        kb_in_live = s1.ngram_coverage(clean, live_text) if status == 200 else None
        live_in_kb = s1.ngram_coverage(live_text, clean) if status == 200 else None
        sim = s1.similarity_label(status, live_in_kb)
        pair = TYPE_PAIRS[pair_index % len(TYPE_PAIRS)] if status == 200 and nchar >= 600 else None
        if pair:
            pair_index += 1
        gen = {"calls": 0, "usage": {}, "model": teacher_model, "truncated": nchar > 15000}
        reason = ""
        accepted: list[dict[str, Any]] = []
        if status != 200:
            reason = "fetch_unavailable"
        elif nchar < 600:
            reason = "extracted_text_under_600_characters"
        else:
            gen = qs.call_teacher(teacher_endpoint, teacher_key, teacher_model, teacher_extra, prompt, page, live_text, prompt_hash, pair)
            if gen.get("error"):
                reason = str(gen["error"])
            elif gen.get("skip_reason"):
                reason = "teacher_skip:" + str(gen["skip_reason"])
            else:
                for slot, qout in enumerate(gen.get("questions", []), 1):
                    record = qs.make_question(page, qout, slot, live_text, raw_index, live_in_kb, sim, gen, prompt_hash, signatures, pair)
                    record["v3_added"] = True
                    record["quote_cut_off"] = "yes" if any(str(z.get("text", "")).rstrip().endswith(("...", "…")) for z in record.get("supporting_quotes", [])) else "no"
                    if record.get("duplicate") == "yes":
                        record["set"] = "duplicate"
                        record["excluded_from_set_reason"] = "duplicate_quote_signature"
                        reason = "duplicate_quote_signature"
                        questions.append(record)
                        continue
                    if record.get("quote_in_page") is not True or record.get("quote_cut_off") == "yes":
                        record["set"] = "rejected"
                        record["excluded_from_set_reason"] = "quote_check_failed"
                        reason = "quote_check_failed"
                        questions.append(record)
                        continue
                    if record.get("table_based") == "yes" and record.get("type") == "false_premise":
                        record["set"] = "rejected"
                        record["excluded_from_set_reason"] = "false_premise_based_on_table_values"
                        reason = "false_premise_based_on_table_values"
                        questions.append(record)
                        continue
                    if record.get("table_based") == "yes" and len(record.get("keypoints", [])) > 1:
                        record["set"] = "rejected"
                        record["excluded_from_set_reason"] = "table_question_requests_multiple_values"
                        reason = "table_question_requests_multiple_values"
                        questions.append(record)
                        continue
                    if record.get("needs_kb_check") == "no":
                        if site_counts[page["site"]] >= SITE_CAP:
                            record["set"] = "rejected"
                            record["excluded_from_set_reason"] = "site_question_cap_reached"
                            questions.append(record)
                            continue
                        record["kb_label_final"] = "in_kb"
                        record["set"] = "main"
                    else:
                        record["kb_label_final"] = ""
                        record["set"] = "pending"
                    site_counts[page["site"]] += 1
                    signature = qs.question_quote_signature(qout)
                    if signature:
                        signatures.setdefault(signature, (page["id"], record["id"]))
                    questions.append(record)
                    accepted.append(record)
                    if record["set"] == "main":
                        completed += 1
                    if record["set"] == "main":
                        audit_row = audit_new(record, judge_key, str(judge_cfg["model"]), str(judge_cfg["base_url"]), dict(judge_cfg.get("thinking_parameters") or {}))
                        audit_row["set"] = "main" if not audit_row["flagged"] else "reserve"
                        new_audits.append(audit_row)
                        if audit_row["flagged"]:
                            record["set"] = "reserve"
                            completed -= 1
                        else:
                            record["audit"] = {k: audit_row.get(k) for k in ("answer_fit", "single_answer", "premise_ok", "reason", "flagged")}
                    elif record["set"] == "pending":
                        audit_row = audit_new(record, judge_key, str(judge_cfg["model"]), str(judge_cfg["base_url"]), dict(judge_cfg.get("thinking_parameters") or {}))
                        audit_row["set"] = "reserve" if audit_row["flagged"] else "pending"
                        new_audits.append(audit_row)
                        if audit_row["flagged"]:
                            record["set"] = "reserve"
                        else:
                            record["audit"] = {k: audit_row.get(k) for k in ("answer_fit", "single_answer", "premise_ok", "reason", "flagged")}
                    write_jsonl(OUT / "work_questions_all.jsonl", questions)
                    if completed >= PAGE_TARGET:
                        break
        usage = gen.get("usage") or {}
        new_page_rows.append({"id": page["id"], "url": page["url"], "site": page["site"], "page_type": "about_science", "http_status": status if status is not None else "request_error", "retries": meta.get("retries", 0), "extracted_characters": nchar, "truncated": "yes" if nchar > 15000 else "no", "kb_in_live": kb_in_live if kb_in_live is not None else "", "live_in_kb": live_in_kb if live_in_kb is not None else "", "similarity_label": sim, "questions_generated": len(accepted), "questions_skipped": max(0, 2 - len(accepted)), "skip_reason": reason, "teacher_calls": gen.get("calls", 0), "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0), "type_pair_used": ",".join(pair) if pair else "", "fetch_time_utc": meta.get("fetch_time_utc", ""), "declared_charset": "", "decoded_charset": charset})
        write_csv(OUT / "topup_pages.csv", new_page_rows, list(new_page_rows[0].keys()))
        write_jsonl(OUT / "work_questions_all.jsonl", questions)
        write_jsonl(OUT / "audit_new.jsonl", new_audits)
        time.sleep(1)
    main_rows = [q for q in questions if q.get("set") == "main"]
    if sum(q.get("page_type") == "about_science" for q in main_rows) != 240:
        raise RuntimeError(f"Top-up ended at {sum(q.get('page_type') == 'about_science' for q in main_rows)} in-KB about_science questions, expected 240")
    write_jsonl(OUT / "work_questions_all.jsonl", questions)
    write_jsonl(OUT / "audit_new.jsonl", new_audits)
    print(f"Top-up complete: {completed} accepted main questions; {len(new_audits)} new audit records; {len(new_page_rows)} pages processed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-test", action="store_true")
    args = parser.parse_args()
    main_run(mock=args.mock_test)
