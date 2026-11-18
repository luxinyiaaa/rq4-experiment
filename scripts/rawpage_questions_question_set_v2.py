"""Build a 300-question in-KB raw-web set while preserving extra questions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import trafilatura
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rawpage_questions_main_stage1 as s1
import rawpage_questions_question_set as qs

SOURCE = ROOT / "outputs/rawpage_questions/question_set_final/questions_final.jsonl"
ORDER = ROOT / "outputs/rawpage_questions/main/sampling_order.csv"
RAW_MAIN = ROOT / "data/raw_web/main"
OUT = ROOT / "outputs/rawpage_questions/question_set_v2"
OUT_RAW = OUT / "raw_web"
REMOVED = ROOT / "outputs/rawpage_questions/question_set_final/removed_questions.jsonl"
CLEAN_KB = ROOT / "data/processed/chunks_clean.csv"
RAW_KB = ROOT / "data/raw/site_pages_rows.csv"
PAGE_TARGET = 240
SITE_CAP = 60
TYPE_PAIRS = [("list_all", "paraphrased"), ("paraphrased", "false_premise"), ("false_premise", "list_all")]
PAGE_FIELDS = ["id", "url", "site", "page_type", "http_status", "retries", "extracted_characters", "truncated", "kb_in_live", "live_in_kb", "similarity_label", "questions_generated", "questions_skipped", "skip_reason", "teacher_calls", "prompt_tokens", "completion_tokens", "total_tokens", "type_pair_used", "fetch_time_utc", "declared_charset", "decoded_charset"]
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def extract_markdown(text: str) -> str:
    return trafilatura.extract(text, output_format="markdown", include_tables=True, include_comments=False, include_links=True) or ""


def declared_charset(content_type: str, body: bytes) -> str:
    message = Message()
    message["content-type"] = content_type or "text/html"
    charset = message.get_content_charset()
    if charset:
        return charset.strip()
    head = body[:16384].decode("ascii", errors="ignore")
    match = re.search(r"<meta[^>]+charset\s*=\s*[\"']?\s*([^\s\"'/>;]+)", head, re.I)
    if not match:
        match = re.search(r"<meta[^>]+content=[\"'][^\"']*charset\s*=\s*([^\s\"';]+)", head, re.I)
    return match.group(1).strip() if match else ""


def decode_html(body: bytes, content_type: str) -> tuple[str, str, str]:
    declared = declared_charset(content_type, body)
    if declared:
        try:
            return body.decode(declared), declared, declared
        except (LookupError, UnicodeDecodeError):
            return body.decode(declared, errors="replace"), declared, declared + ":replace"
    try:
        return body.decode("utf-8"), "", "utf-8"
    except UnicodeDecodeError:
        detected = requests.utils.get_encoding_from_headers({"content-type": content_type}) if content_type else None
        encoding = detected or "cp1252"
        try:
            return body.decode(encoding, errors="replace"), "", encoding + ":replace"
        except LookupError:
            return body.decode("utf-8", errors="replace"), "", "utf-8:replace"


def page_stem(page_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", page_id)


def fetch(page: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    attempts: list[dict[str, Any]] = []
    last = None
    for attempt in range(1, 4):
        try:
            last = requests.get(page["url"], headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}, timeout=60, allow_redirects=True)
            attempts.append({"attempt": attempt, "http_status": last.status_code})
            if last.status_code == 200:
                break
        except requests.RequestException as exc:
            last = None
            attempts.append({"attempt": attempt, "error": type(exc).__name__})
        if attempt < 3:
            time.sleep(60)
    body = last.content if last is not None else b""
    metadata = {
        "id": page["id"], "original_url": page["url"], "final_url": last.url if last is not None else "",
        "http_status": last.status_code if last is not None else None,
        "content_type": last.headers.get("Content-Type", "") if last is not None else "",
        "fetch_time_utc": datetime.now(timezone.utc).isoformat(), "attempts": attempts,
        "retries": max(0, len(attempts) - 1),
    }
    stem = page_stem(page["id"])
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    (OUT_RAW / f"{stem}.html").write_bytes(body)
    (OUT_RAW / f"{stem}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata, body


def is_filtered_url(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return bool(re.search(r"(?:privacy|cookie|legal|terms|career|vacanc|job|position)", path))


def has_cutoff_quote(generated: list[dict[str, Any]]) -> bool:
    return any(str(quote.get("text", "")).rstrip().endswith(("...", "…")) for question in generated for quote in question.get("supporting_quotes", []))


def existing_cache(page_id: str) -> tuple[dict[str, Any], bytes] | None:
    stem = page_stem(page_id)
    for directory in (OUT_RAW, RAW_MAIN):
        html = directory / f"{stem}.html"
        metadata = directory / f"{stem}.json"
        if html.exists() and metadata.exists():
            try:
                return json.loads(metadata.read_text(encoding="utf-8")), html.read_bytes()
            except (OSError, json.JSONDecodeError):
                continue
    return None


def check_question_quotes(record: dict[str, Any], live_text: str, raw_index: Any) -> None:
    checks = []
    other_urls: list[str] = []
    for item in record.get("supporting_quotes", []):
        quote = str(item.get("text", ""))
        quote_tokens = s1.tokens(quote)
        in_url = raw_index.matches_url(quote_tokens, record["url"])
        others = raw_index.other_urls(quote_tokens, record["url"], limit=3) if len(quote_tokens) >= 8 and not in_url else []
        fuzzy, context = s1.fuzzy_score_and_context(quote, raw_index.chunks.get(record["url"], []))
        checks.append({"source": item.get("source", "passage"), "quote": quote, "quote_in_page": bool(quote) and quote in live_text, "in_kb": in_url, "quote_kb_fuzzy": round(fuzzy, 2), "kb_best_match": "" if in_url else context, "in_kb_any": in_url or bool(others), "kb_other_urls": others, "word_tokens": len(quote_tokens)})
        other_urls.extend(others)
    record["quote_checks"] = checks
    record["quote_in_page"] = all(row["quote_in_page"] for row in checks) if checks else False
    record["in_kb"] = s1.label_matches([row["in_kb"] for row in checks])
    record["in_kb_any"] = s1.label_matches([row["in_kb_any"] for row in checks])
    record["kb_other_urls"] = list(dict.fromkeys(other_urls))[:3]
    record["kb_best_match"] = "\n".join(f"{row['quote']} [fuzzy={row['quote_kb_fuzzy']:.2f}] {row['kb_best_match']}" for row in checks if not row["in_kb"])
    record["needs_kb_check"] = "yes" if record["in_kb"] != "all" or record["in_kb_any"] != "all" else "no"
    if record["needs_kb_check"] == "no":
        record["kb_label_final"] = "in_kb"
    else:
        record["kb_label_final"] = ""


def repair_encoding_questions(questions: list[dict[str, Any]], raw_index: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_page: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        by_page.setdefault(question["page_id"], []).append(question)
    findings: list[dict[str, Any]] = []
    audit_candidates: list[dict[str, Any]] = []
    for html_path in sorted(RAW_MAIN.glob("*.html")):
        body = html_path.read_bytes()
        try:
            text_utf8 = body.decode("utf-8")
        except UnicodeDecodeError:
            if html_path.stem == "DR-011-f45f8910a2":
                meta_path = html_path.with_suffix(".json")
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                decoded = body.decode("utf-8", errors="replace")
                live_text = extract_markdown(decoded)
                page_questions = by_page.get(html_path.stem, [])
                for record in page_questions:
                    for item in record.get("supporting_quotes", []):
                        quote = item.get("text", "")
                        if "LiĆØge" in quote:
                            line = next((line.strip() for line in live_text.splitlines() if "Data holding centre" in line), "")
                            if line:
                                item["text"] = line
                    check_question_quotes(record, live_text, raw_index)
                findings.append({"page_id": html_path.stem, "status": "UTF-8 decoded with replacement for isolated legacy byte; corrected text contains Liège", "questions_rechecked": len(page_questions), "quotes_updated": 1})
            continue
        byte_text = extract_markdown(body)
        utf8_text = extract_markdown(text_utf8)
        if byte_text != utf8_text:
            page_questions = by_page.get(html_path.stem, [])
            if page_questions:
                for record in page_questions:
                    check_question_quotes(record, utf8_text, raw_index)
                audit_candidates.append({"page_id": html_path.stem, "questions_rechecked": len(page_questions)})
    if audit_candidates:
        findings.extend({"page_id": row["page_id"], "status": "valid UTF-8 bytes produced different UTF-8 extraction", "questions_rechecked": row["questions_rechecked"]} for row in audit_candidates)
    return questions, findings


def settings() -> tuple[dict[str, Any], str, str]:
    load_dotenv(ROOT / ".env", override=False)
    config = yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf-8")) or {}
    teacher = config.get("judge_primary") or {}
    key_name = teacher.get("api_key_env")
    api_key = __import__("os").environ.get(str(key_name)) if key_name else None
    if not teacher.get("model") or not teacher.get("base_url") or not api_key:
        raise RuntimeError("Teacher configuration or credential is unavailable")
    return teacher, api_key, str(teacher["base_url"]).rstrip("/") + "/chat/completions"


def write_all(questions: list[dict[str, Any]]) -> None:
    write_jsonl(OUT / "questions_all.jsonl", questions)


def classify_sets(questions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    main = [q for q in questions if q.get("set") == "main"]
    extra = [q for q in questions if q.get("set") == "extra"]
    pending = [q for q in questions if q.get("set") == "pending"]
    return main, extra, pending


def update_set(q: dict[str, Any]) -> None:
    label = str(q.get("kb_label_final") or "")
    if label == "in_kb":
        q["set"] = "main"
    elif label in {"not_in_kb", "in_kb_elsewhere"}:
        q["set"] = "extra"
    else:
        q["set"] = "pending"


def run(mock: bool = False, prepare_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    questions = read_jsonl(SOURCE)
    raw_index = s1.RawKBIndex(RAW_KB)
    questions, encoding_findings = repair_encoding_questions(questions, raw_index)
    for question in questions:
        update_set(question)
    main, extra, pending = classify_sets(questions)
    current_as_main = sum(q["page_type"] == "about_science" for q in main)
    site_counts = Counter(q["site"] for q in main + extra if q["page_type"] == "about_science")
    signatures: dict[tuple[str, ...], tuple[str, str]] = {}
    for question in questions:
        sig = qs.question_quote_signature(question)
        if sig and question.get("duplicate") != "yes":
            signatures.setdefault(sig, (question["page_id"], question["id"]))
    with ORDER.open(newline="", encoding="utf-8") as handle:
        order = [row for row in csv.DictReader(handle) if row["page_type"] == "about_science"]
    prior_pages: dict[str, dict[str, Any]] = {}
    for path in (ROOT / "outputs/rawpage_questions/main/pages.csv", ROOT / "outputs/rawpage_questions/question_set/pages.csv", ROOT / "outputs/rawpage_questions/question_set_final/topup_pages.csv"):
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                prior_pages.update({row["id"]: row for row in csv.DictReader(handle) if row.get("page_type") == "about_science"})
    order_index = {row["id"]: i for i, row in enumerate(order)}
    previously_processed = [prior_pages[page_id] for page_id in sorted(prior_pages, key=lambda x: order_index.get(x, 10**9)) if page_id in order_index]
    processed_ids = set(prior_pages)
    usable_prior = [row for row in previously_processed if str(row.get("http_status")) == "200" and int(row.get("extracted_characters") or 0) >= 600]
    pair_index = len(usable_prior)
    if not mock and not prepare_only:
        teacher_cfg, api_key, endpoint = settings()
        model = str(teacher_cfg["model"])
        extra_body = dict(teacher_cfg.get("thinking_parameters") or {})
    else:
        teacher_cfg, api_key, endpoint, model, extra_body = {}, "", "", "mock-teacher", {}
    prompt = qs.build_prompt() + """

## Final top-up requirements
- Every decisive keypoint must be directly supported by the question's supporting quotes.
- Supporting quotes must be exact substrings of the supplied page text and must not end in an ellipsis.
- In false_premise questions, the false detail must concern the same subject as the question.
"""
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    clean_texts = s1.load_chunk_texts(CLEAN_KB)
    new_page_rows: list[dict[str, Any]] = []
    new_questions: list[dict[str, Any]] = [q for q in questions if q.get("v2_added")]
    mock_response = {"skip_page": False, "skip_reason": "", "questions": []}
    if mock:
        assert decode_html("<meta charset='utf-8'>A".encode(), "") == ("<meta charset='utf-8'>A", "utf-8", "utf-8")
        assert is_filtered_url("https://example.org/privacy-policy")
        assert not is_filtered_url("https://example.org/about/research")
        assert current_as_main == 212
        print("Mock checks passed: charset fallback, page filters, and 212 in-KB about-science starting questions.")
        return
    if prepare_only:
        remaining = [row for row in order if row["id"] not in processed_ids]
        print(f"Preflight passed: {len(order)} about-science pages in the frozen order; {len(previously_processed)} previously processed; next page {remaining[0]['id'] if remaining else 'none'}; {current_as_main} in-KB questions; site cap headroom {sum(max(0, SITE_CAP - n) for n in site_counts.values())}.")
        return
    write_jsonl(OUT / "questions_all.jsonl", questions)
    completed_v2 = set()
    pages_path = OUT / "topup_pages.csv"
    if pages_path.exists():
        with pages_path.open(newline="", encoding="utf-8") as handle:
            new_page_rows = list(csv.DictReader(handle))
        completed_v2 = {row["id"] for row in new_page_rows}
    prior_questions_path = OUT / "questions_all.jsonl"
    if prior_questions_path.exists():
        questions = read_jsonl(prior_questions_path)
        for q in questions:
            update_set(q)
        main, extra, pending = classify_sets(questions)
        current_as_main = sum(q["page_type"] == "about_science" for q in main)
        site_counts = Counter(q["site"] for q in main + extra + pending if q["page_type"] == "about_science")
        signatures = {}
        for question in questions:
            sig = qs.question_quote_signature(question)
            if sig and question.get("duplicate") != "yes":
                signatures.setdefault(sig, (question["page_id"], question["id"]))
        pair_index += sum(str(row.get("http_status")) == "200" and int(row.get("extracted_characters") or 0) >= 600 for row in new_page_rows)
    for p0 in order:
        if current_as_main >= PAGE_TARGET:
            break
        if p0["id"] in processed_ids or p0["id"] in completed_v2:
            continue
        page = {**p0, "site": p0.get("site") or urlsplit(p0["url"]).netloc.lower().removeprefix("www.")}
        if is_filtered_url(page["url"]):
            report = {"id": page["id"], "url": page["url"], "site": page["site"], "page_type": "about_science", "http_status": "not_fetched_filtered", "retries": 0, "extracted_characters": 0, "truncated": "no", "similarity_label": "", "questions_generated": 0, "questions_skipped": 2, "skip_reason": "page_filter_url", "teacher_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "type_pair_used": ""}
            new_page_rows.append(report); completed_v2.add(page["id"]); write_csv(pages_path, new_page_rows, PAGE_FIELDS); continue
        if site_counts[page["site"]] >= SITE_CAP:
            report = {"id": page["id"], "url": page["url"], "site": page["site"], "page_type": "about_science", "http_status": "not_fetched_site_cap", "retries": 0, "extracted_characters": 0, "truncated": "no", "similarity_label": "", "questions_generated": 0, "questions_skipped": 2, "skip_reason": "site_question_cap_reached", "teacher_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "type_pair_used": ""}
            new_page_rows.append(report); completed_v2.add(page["id"]); write_csv(pages_path, new_page_rows, PAGE_FIELDS); continue
        cached = existing_cache(page["id"])
        if cached:
            meta, body = cached
        else:
            meta, body = fetch(page)
        status = meta.get("http_status")
        content_type = meta.get("content_type", "")
        if status == 200:
            decoded, declared, used = decode_html(body, content_type)
            live_text = extract_markdown(decoded)
        else:
            decoded, declared, used, live_text = "", "", "", ""
        nchar = len(live_text)
        clean = clean_texts.get(page["url"], "")
        kb_in_live = s1.ngram_coverage(clean, live_text) if status == 200 else None
        live_in_kb = s1.ngram_coverage(live_text, clean) if status == 200 else None
        label = s1.similarity_label(status, live_in_kb)
        pair = TYPE_PAIRS[pair_index % 3] if status == 200 and nchar >= 600 else None
        if pair:
            pair_index += 1
        teacher = {"calls": 0, "usage": {}, "model": model, "truncated": nchar > 15000}
        generated: list[dict[str, Any]] = []
        reason = ""
        if status != 200:
            reason = "fetch_unavailable"
        elif nchar < 600:
            reason = "extracted_text_under_600_characters"
        elif pair is None:
            reason = "no_type_pair"
        else:
            page_prompt_hash = prompt_hash
            teacher = qs.call_teacher(endpoint, api_key, model, extra_body, prompt, page, live_text, prompt_hash, pair)
            if not teacher.get("error") and not teacher.get("skip_reason") and has_cutoff_quote(teacher.get("questions", [])):
                first = teacher
                retry_prompt = prompt + "\n\nCorrection: the previous response contained a supporting quote ending in an ellipsis. Return a corrected response with complete, exact quote spans, or skip the page with a reason."
                page_prompt_hash = hashlib.sha256(retry_prompt.encode("utf-8")).hexdigest()
                teacher = qs.call_teacher(endpoint, api_key, model, extra_body, retry_prompt, page, live_text, page_prompt_hash, pair)
                teacher["calls"] = int(teacher.get("calls", 0)) + int(first.get("calls", 0))
                combined_usage = Counter(first.get("usage") or {})
                combined_usage.update(teacher.get("usage") or {})
                teacher["usage"] = dict(combined_usage)
            if teacher.get("error"):
                reason = str(teacher["error"])
            elif teacher.get("skip_reason"):
                reason = "teacher_skip:" + str(teacher["skip_reason"])
            else:
                for slot, question in enumerate(teacher["questions"], 1):
                    record = qs.make_question(page, question, slot, live_text, raw_index, live_in_kb, label, teacher, page_prompt_hash, signatures, pair)
                    record["v2_added"] = True
                    record["quote_cut_off"] = "yes" if has_cutoff_quote([question]) else "no"
                    if record.get("duplicate") == "yes":
                        record["set"] = "duplicate"
                        reason = "duplicate_quote_signature"
                        continue
                    if record["needs_kb_check"] == "no":
                        if current_as_main >= PAGE_TARGET:
                            reason = "main_target_reached"
                            continue
                        record["kb_label_final"] = "in_kb"
                        record["set"] = "main"
                        current_as_main += 1
                    else:
                        record["kb_label_final"] = ""
                        record["set"] = "pending"
                    # Reserve site capacity for pending questions because manual review may later classify them as extra.
                    site_counts[page["site"]] += 1
                    generated.append(record)
                    new_questions.append(record)
                    questions.append(record)
                    signature = qs.question_quote_signature(question)
                    if signature:
                        signatures.setdefault(signature, (page["id"], record["id"]))
                    write_jsonl(OUT / "questions_all.jsonl", questions)
        usage = teacher.get("usage") or {}
        report = {"id": page["id"], "url": page["url"], "site": page["site"], "page_type": "about_science", "http_status": status if status is not None else "request_error", "retries": meta.get("retries", 0), "extracted_characters": nchar, "truncated": "yes" if nchar > 15000 else "no", "kb_in_live": kb_in_live if kb_in_live is not None else "", "live_in_kb": live_in_kb if live_in_kb is not None else "", "similarity_label": label, "questions_generated": len(generated), "questions_skipped": max(0, 2 - len(generated)), "skip_reason": reason, "teacher_calls": teacher.get("calls", 0), "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0), "type_pair_used": ",".join(pair) if pair else "", "fetch_time_utc": meta.get("fetch_time_utc", ""), "declared_charset": declared, "decoded_charset": used}
        new_page_rows.append(report); completed_v2.add(page["id"])
        write_csv(pages_path, new_page_rows, PAGE_FIELDS); write_jsonl(OUT / "questions_all.jsonl", questions)
        time.sleep(1)
    main, extra, pending = classify_sets(questions)
    if current_as_main < PAGE_TARGET:
        raise RuntimeError(f"Top-up stopped before the target: {current_as_main}/240 in-KB about-science questions")
    if sum(q["page_type"] == "dataset_record" for q in main) != 60 or sum(q["page_type"] == "about_science" for q in main) != PAGE_TARGET:
        raise RuntimeError("The final main set does not match the required 60/240 page-type allocation")
    write_jsonl(OUT / "questions_all.jsonl", questions)
    write_summary(main, extra, pending, new_page_rows, encoding_findings, time.monotonic() - started)
    print(f"Completed: main={len(main)} (dataset_record={sum(q['page_type']=='dataset_record' for q in main)}, about_science={sum(q['page_type']=='about_science' for q in main)}); extra={len(extra)}; pending={len(pending)}; new_questions={len(new_questions)}; pages_processed={len(new_page_rows)}.")


def write_summary(main: list[dict[str, Any]], extra: list[dict[str, Any]], pending: list[dict[str, Any]], pages: list[dict[str, Any]], encoding: list[dict[str, Any]], elapsed: float) -> None:
    lines = ["# Raw-Web Question Set V2 Summary", "", "## Main set by page type", "", "| Page type | Count |", "|---|---:|"]
    for typ in ("dataset_record", "about_science"):
        lines.append(f"| {typ} | {sum(q['page_type'] == typ for q in main)} |")
    lines += ["", "## Main set by question type", "", "| Type | Count |", "|---|---:|"]
    lines += [f"| {typ} | {count} |" for typ, count in sorted(Counter(q["type"] for q in main).items())]
    lines += ["", "## Main set by site", "", "| Page type | Site | Count |", "|---|---|---:|"]
    lines += [f"| {typ} | {site} | {count} |" for (typ, site), count in sorted(Counter((q["page_type"], q["site"]) for q in main).items())]
    lines += ["", "## Extra set by page type", "", "| Page type | Count |", "|---|---:|"]
    lines += [f"| {typ} | {sum(q['page_type'] == typ for q in extra)} |" for typ in ("dataset_record", "about_science")]
    lines += ["", "## Extra set by question type", "", "| Type | Count |", "|---|---:|"]
    lines += [f"| {typ} | {count} |" for typ, count in sorted(Counter(q["type"] for q in extra).items())]
    lines += ["", "## Extra set by site", "", "| Page type | Site | Count |", "|---|---|---:|"]
    lines += [f"| {typ} | {site} | {count} |" for (typ, site), count in sorted(Counter((q["page_type"], q["site"]) for q in extra).items())]
    lines += ["", "## Extra set by KB label", "", "| Label | Count |", "|---|---:|"]
    lines += [f"| {label} | {count} |" for label, count in sorted(Counter(q.get("kb_label_final", "") for q in extra).items())]
    lines += ["", f"Pending KB checks: {len(pending)}", "", "## Top-up pages by status", "", "| Status | Count |", "|---|---:|"]
    lines += [f"| {status} | {count} |" for status, count in sorted(Counter(str(p.get("http_status")) for p in pages).items())]
    lines += ["", "## Top-up pages by skip reason", "", "| Skip reason | Count |", "|---|---:|"]
    lines += [f"| {reason or '(none)'} | {count} |" for reason, count in sorted(Counter(str(p.get("skip_reason") or "") for p in pages).items())]
    added = [q for q in main + extra + pending if q.get("v2_added")]
    lines += ["", "## New questions", "", f"- Total: {len(added)}", f"- Main: {sum(q['set'] == 'main' for q in added)}", f"- Pending: {sum(q['set'] == 'pending' for q in added)}", f"- Table-based: {sum(q.get('table_based') == 'yes' for q in added)}"]
    lines += ["", "## Encoding findings", "", f"- Valid UTF-8 HTML pages whose extracted text changed under explicit UTF-8 decoding: {sum(f['status'].startswith('valid UTF-8') for f in encoding)}"]
    if encoding:
        lines += ["", "| Page ID | Finding | Questions rechecked | Quotes updated |", "|---|---|---:|---:|"]
        lines += [f"| {f['page_id']} | {f['status']} | {f['questions_rechecked']} | {f.get('quotes_updated', 0)} |" for f in encoding]
    else:
        lines.append("- No affected pages found.")
    lines += ["", "## Teacher usage", "", f"- Calls: {sum(int(p.get('teacher_calls') or 0) for p in pages)}", f"- Prompt tokens: {sum(int(p.get('prompt_tokens') or 0) for p in pages)}", f"- Completion tokens: {sum(int(p.get('completion_tokens') or 0) for p in pages)}", f"- Total tokens: {sum(int(p.get('total_tokens') or 0) for p in pages)}", f"- Run time seconds: {elapsed:.1f}", ""]
    (OUT / "summary_v2.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(mock=args.mock_test, prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
