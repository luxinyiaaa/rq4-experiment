"""Run evidence, retrieval, round-trip, closed-book, and rule checks for S2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
from rq4.llm import call_llm
from scripts.build_index import load_index
from scripts.build_bundles_s2 import _year_overlap, _parse_comparison_value, _shared_comparison_relation, _comparison_value_in_any_title
from scripts.build_kb_index import parse_time_interval, read_jsonl, write_jsonl
from scripts.generate_questions_s2 import EXPECTED_BEHAVIOUR, PROMPT as TEACHER_PROMPT, generate_one as teacher_generate_one, is_absence_keypoint, user_message as teacher_user_message, response_issues as teacher_response_issues

PAGES_PATH = ROOT / "data" / "processed" / "pages.csv"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks_clean.csv"
S1_QUESTIONS = ROOT / "outputs" / "final" / "questions_final.jsonl"
S1_ROUND3 = ROOT / "outputs" / "main" / "questions_main_round3.jsonl"
PROMPTS = ROOT / "prompts"
PILOT_DIR = ROOT / "outputs" / "s2" / "pilot2"
ABSENCE_PROMPT = PROMPTS / "judge_absence_check.md"
CONFLICT_PROMPT = PROMPTS / "judge_conflict_check.md"
CONDITION_PROMPT = PROMPTS / "judge_condition_check.md"
QUOTE_SUPPORT_PROMPT = PROMPTS / "judge_quote_support.md"
PLAUSIBILITY_PROMPT = PROMPTS / "judge_plausibility.md"
ROUNDTRIP_SYSTEM = """You are an independent answerability and answer-checking judge. Use only the supplied evidence pages. First answer the exact question from the evidence, following the stated expected behaviour, then compare it with the reference answer. For disambiguate, identify multiple records sharing the name and give each requested value separately. For flag_conflict, report both explicit values and say that they differ. For state_absence, explicitly state that no record with the requested combination is available. For list_all, include every matching record and no others. Return JSON only: {\"answer\":\"...\",\"matches_reference\":true|false,\"reason\":\"one sentence\"}. Derive the answer independently."""
CLOSED_BOOK_SYSTEM = "You answer questions for researchers. Answer the question using only your pretrained knowledge; you have not been given a knowledge base or passages. If you do not know, say so."
CLOSED_BOOK_JUDGE = "You strictly compare a closed-book answer with every decisive reference keypoint. Do not give credit for uncertainty, refusal, guessing, or a conclusion without required values. For ordinary answer types, every decisive fact must be present and correct; missing a number or value means incorrect. For state_absence, only an explicit statement that no such record/data combination exists is correct; 'I do not know' is incorrect. Return JSON only: {\"correct\":true|false,\"decisive_keypoints_present\":[true|false],\"reason\":\"one sentence\"}."
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")
DERIVED_CONCLUSION_RE = re.compile(r"\b(earliest|latest|longest|shortest|highest|lowest|largest|smallest|most recent|oldest|share (?:the )?same name|same name|values? differ|different values|two values? (?:are )?different|there are two|multiple (?:distinct )?records)\b", re.I)
TEMPLATE_TERMS = ("cookie settings", "privacy policy", "terms of use", "related links", "navigation", "skip to content", "manage cookie")
PILOT_TYPE_ORDER = ("set_all", "multi_hop", "comparison_multi", "ambiguity", "unanswerable")


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_derived_point(point: dict[str, Any]) -> bool:
    return point.get("derived") is True or bool(DERIVED_CONCLUSION_RE.search(str(point.get("text", ""))))


def _read_page_texts() -> dict[str, str]:
    chunks = pd.read_csv(CHUNKS_PATH, dtype=str, keep_default_na=False)
    return {url: "".join(frame.sort_values("chunk_number", kind="stable")["content"].tolist()) for url, frame in chunks.groupby("url", sort=False)}


def _support_quotes(question: dict[str, Any]) -> list[dict[str, Any]]:
    points = question.get("keypoints", [])
    rows = []
    for point in points:
        if not isinstance(point, dict):
            continue
        sources = point.get("sources") if isinstance(point.get("sources"), list) else [point.get("source", "")]
        quotes = point.get("quotes") if isinstance(point.get("quotes"), list) else [point.get("quote", "")]
        rows.extend({"text": str(q or ""), "source": str(s or ""), "keypoint_text": str(point.get("text", "")),
                     "decisive": point.get("decisive") is True} for s, q in zip(sources, quotes))
    return rows


def _build_bm25_index(chunks: list[dict[str, str]]) -> tuple[list[list[str]], Counter[str], list[str]]:
    tokenized = [_tokens(chunk["text"]) for chunk in chunks]
    dfs: Counter[str] = Counter()
    for tokens in tokenized:
        dfs.update(set(tokens))
    return tokenized, dfs, [str(c["chunk_id"]) for c in chunks]


def _bm25(query: str, chunks: list[dict[str, str]], tokenized: list[list[str]], dfs: Counter[str], k: int = 50) -> list[dict[str, Any]]:
    query_terms = set(_tokens(query))
    n_docs = max(1, len(tokenized))
    avg_len = sum(map(len, tokenized)) / n_docs
    scores = np.zeros(n_docs, dtype=np.float64)
    k1, b = 1.5, 0.75
    for index, terms in enumerate(tokenized):
        counts = Counter(terms)
        length = len(terms)
        for term in query_terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (n_docs - dfs[term] + 0.5) / (dfs[term] + 0.5))
            scores[index] += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * length / max(avg_len, 1)))
    indices = np.argsort(-scores, kind="stable")[: min(k, n_docs)]
    return [{**chunks[int(i)], "score": float(scores[int(i)])} for i in indices]


def _passage_refs(pages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(page["evidence_id"]): page for page in pages}


def _call_json(role: str, system: str, user: str, output_root: Path, *, mock: bool,
               mock_json: dict[str, Any] | None = None) -> dict[str, Any]:
    if mock:
        return {"json": mock_json or {}, "text": json.dumps(mock_json or {}), "model": f"mock-{role}", "cache_hit": False}
    return call_llm(role, [{"role": "system", "content": system}, {"role": "user", "content": user}],
        json_mode=True, temperature=0.0, prompt_sha256=_hash(system), log_messages=True,
        cache_dir=output_root / "cache" / "llm", log_dir=output_root / "logs")


def _quote_chunk_ids(page: dict[str, Any], quote: str, chunks_by_id: dict[str, str]) -> list[str]:
    joined = ""
    ranges = []
    cursor = 0
    for chunk_id in page.get("chunk_ids", []):
        text = chunks_by_id.get(str(chunk_id), "")
        if not text.strip():
            continue
        ranges.append((cursor, cursor + len(text), str(chunk_id)))
        joined += text
        cursor += len(text)
    start = joined.find(quote) if quote else -1
    if start < 0:
        return []
    end = start + len(quote)
    return [chunk_id for left, right, chunk_id in ranges if left < end and right > start]


def quote_verbatim_check(question: dict[str, Any], chunks_by_id: dict[str, str]) -> dict[str, Any]:
    pages = _passage_refs(question.get("pages", []))
    records = []
    all_exact = True
    for index, point in enumerate(question.get("keypoints", []), 1):
        if _is_derived_point(point):
            records.append({"keypoint_index": index, "applicable": False, "reason": "Derived conclusion."})
            continue
        sources = point.get("sources") if isinstance(point.get("sources"), list) else [point.get("source", "")]
        quotes = point.get("quotes") if isinstance(point.get("quotes"), list) else [point.get("quote", "")]
        no_quote_needed = question.get("expected_behaviour") == "state_absence" and is_absence_keypoint(point.get("text", ""))
        evidence = []
        if no_quote_needed:
            records.append({"keypoint_index": index, "applicable": False, "reason": "Absence keypoint."})
            continue
        point_chunks = []
        for source, quote_value in zip(sources, quotes):
            source, quote = str(source or ""), str(quote_value or "")
            page = pages.get(source)
            exact = bool(page and quote and quote in page.get("text", ""))
            chunk_ids = _quote_chunk_ids(page, quote, chunks_by_id) if exact and page else []
            evidence.append({"source": source, "quote": quote, "verbatim": exact, "gold_chunk_ids": chunk_ids})
            point_chunks.extend(chunk_ids)
            all_exact = all_exact and exact and bool(chunk_ids)
        point["gold_chunk_ids"] = list(dict.fromkeys(point_chunks))
        records.append({"keypoint_index": index, "applicable": True, "evidence": evidence,
                        "verbatim": bool(evidence) and all(x["verbatim"] for x in evidence),
                        "gold_chunk_ids": point["gold_chunk_ids"]})
    return {"passed": bool(all_exact and records), "keypoints": records}


def multi_page_check(question: dict[str, Any]) -> dict[str, Any]:
    decisive = [p for p in question.get("keypoints", []) if p.get("decisive") is True]
    pages = _passage_refs(question.get("pages", []))
    urls = sorted({pages[source]["url"] for p in decisive
                   for source in (p.get("sources") if isinstance(p.get("sources"), list) else [p.get("source", "")])
                   if source in pages})
    exempt = question.get("type") == "unanswerable"
    return {"required": not exempt, "passed": True if exempt else len(urls) >= 2, "decisive_source_page_count": len(urls), "source_urls": urls}


def set_completeness_check(question: dict[str, Any], chunks: list[dict[str, str]], bm25_cache: Any) -> dict[str, Any] | None:
    if question.get("type") != "set_all":
        return None
    expected = [str(x) for x in question.get("expected_items", [])]
    answer = str(question.get("reference_answer", "")).casefold()
    points = " ".join(str(p.get("text", "")) for p in question.get("keypoints", [])).casefold()
    material = answer + " " + points
    def item_alias(name: str) -> str:
        parts = re.split(r"\bfrom\b", name, maxsplit=1, flags=re.I)
        return parts[-1].strip() if len(parts) == 2 else name
    present = [name for name in expected if name.casefold() in material or item_alias(name).casefold() in material]
    missing = [name for name in expected if name not in present]
    field = str(question.get("target_field", "")); value = str(question.get("target_value", ""))
    cat_name = str(question.get("catalogue_name", "")); query = f"{value} {cat_name} {field}"
    tokenized, dfs, _ = bm25_cache
    bm25_rows = _bm25(query, chunks, tokenized, dfs, 50)
    retrieved_urls = list(dict.fromkeys(row["url"] for row in bm25_rows))
    field_rows = read_jsonl(ROOT / "data/processed/s2/catalogue_fields.jsonl")
    if question.get("time_condition"):
        condition=question["time_condition"]
        global_matches=[row for row in field_rows if _year_overlap(
            parse_time_interval(str(row.get("fields", {}).get(field,""))),
            int(condition["start_year"]),int(condition.get("end_year") or condition["start_year"]))]
    else:
        global_matches = [row for row in field_rows if str(row.get("fields", {}).get(field, "")).casefold().strip() == value.casefold().strip()]
    listed_urls = set(question.get("field_match_urls", []))
    missing_global = [{"url": row["url"], "title": row.get("record_name") or row.get("title"),
                       "found_by_bm25": row["url"] in retrieved_urls} for row in global_matches if row["url"] not in listed_urls]
    return {"expected_items": expected, "items_found_in_answer_or_keypoints": present,
            "missing_from_answer": missing, "extra_in_answer": [],
            "catalogue_listed_urls": sorted(listed_urls),
            "global_field_value_matches": [{"url": r["url"], "title": r.get("record_name") or r.get("title")} for r in global_matches],
            "missing_from_catalogue_list": missing_global, "bm25_query": query, "bm25_top_urls": retrieved_urls,
            "passed": not missing and not missing_global}


def _retrieval_evidence(question_text: str, index: Any, chunks: list[dict[str, str]], bm25_cache: tuple[list[list[str]], Counter[str], list[str]]) -> dict[str, Any]:
    tokenized, dfs, _ = bm25_cache
    bm25_rows = _bm25(question_text, chunks, tokenized, dfs, 50)
    vector_rows = index.retrieve(question_text, k=50)
    merged = []
    seen = set()
    for method, rows in (("bm25", bm25_rows), ("vector", vector_rows)):
        for row in rows:
            chunk_id = str(row["chunk_id"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append({"passage_id": f"R{len(merged) + 1}", "chunk_id": chunk_id, "url": row["url"], "text": row["text"], "retrieval_methods": [method]})
    # Preserve both retrieval methods for passages returned by both rankers.
    methods_by_id: dict[str, set[str]] = {}
    for method, rows in (("bm25", bm25_rows), ("vector", vector_rows)):
        for row in rows:
            methods_by_id.setdefault(str(row["chunk_id"]), set()).add(method)
    for row in merged:
        row["retrieval_methods"] = sorted(methods_by_id[row["chunk_id"]])
    return {"bm25_top_50": bm25_rows, "vector_top_50": vector_rows, "merged": merged}


def _check_absence(question: dict[str, Any], index: Any, chunks: list[dict[str, str]], bm25_cache: Any,
                   output_root: Path, *, mock: bool) -> dict[str, Any] | None:
    if question.get("type") != "unanswerable":
        return None
    prompt = ABSENCE_PROMPT.read_text(encoding="utf-8").strip()
    checks = {}
    for variant in ("question_user", "question_source"):
        retrieval = _retrieval_evidence(question[variant], index, chunks, bm25_cache) if not mock else {"bm25_top_50": [], "vector_top_50": [], "merged": []}
        passages = "\n\n".join(f"[{p['passage_id']}] ({p['url']})\n{p['text']}" for p in retrieval["merged"])
        user = f"Question: {question[variant]}\n\nRetrieved passages:\n{passages}"
        result = _call_json("judge", prompt.split("## User template")[0], user, output_root, mock=mock,
            mock_json={"answerable": False, "passage_ids": [], "reason": "Mock absence-check response."})
        checks[variant] = {"answerable": bool((result.get("json") or {}).get("answerable")),
            "status": "failed" if (result.get("json") or {}).get("answerable") is True else "passed",
            "judge": result.get("json"), "retrieval": retrieval,
            "answering_passages": [p for p in retrieval["merged"] if p["passage_id"] in (result.get("json") or {}).get("passage_ids", [])]}
    return {"question_user": checks["question_user"], "question_source": checks["question_source"],
            "passed": all(v["status"] == "passed" for v in checks.values())}


def _check_conflict(question: dict[str, Any], output_root: Path, *, mock: bool) -> dict[str, Any] | None:
    if question.get("type") != "ambiguity" or question.get("ambiguity_kind") != "confirmed_conflict":
        return None
    candidate = question.get("conflict_candidate", {})
    pages = question.get("pages", [])
    prompt = CONFLICT_PROMPT.read_text(encoding="utf-8").strip().split("## User template")[0]
    user = (f"Candidate attribute: {candidate.get('attribute')}\nCandidate value A: {candidate.get('value_a')}\n"
        f"Page A: {pages[0]['title']} ({pages[0]['url']})\nExact excerpt A:\n{candidate.get('quote_a')}\n\n"
        f"Candidate value B: {candidate.get('value_b')}\nPage B: {pages[1]['title']} ({pages[1]['url']})\nExact excerpt B:\n{candidate.get('quote_b')}")
    result = _call_json("judge", prompt, user, output_root, mock=mock,
        mock_json={"confirmed": True, "same_object": True, "same_attribute": True, "values_differ": True, "reason": "Mock conflict confirmation."})
    judgement = result.get("json") or {}
    pages_by_url = {page["url"]: page for page in pages}
    full_text_by_url = _read_page_texts()
    quote_a = str(candidate.get("quote_a", "")); quote_b = str(candidate.get("quote_b", ""))
    exact_a = bool(quote_a and quote_a in full_text_by_url.get(candidate.get("url_a"), ""))
    exact_b = bool(quote_b and quote_b in full_text_by_url.get(candidate.get("url_b"), ""))
    confirmed = all(judgement.get(k) is True for k in ("confirmed", "same_object", "same_attribute", "values_differ"))
    return {"quotes_verbatim": {"value_a": exact_a, "value_b": exact_b}, "judge": judgement,
        "passed": exact_a and exact_b and confirmed}


def _plausibility_check(question: dict[str, Any], output_root: Path, *, mock: bool) -> dict[str, Any] | None:
    if question.get("type") not in {"set_all", "unanswerable", "comparison_multi"}:
        return None
    prompt = PLAUSIBILITY_PROMPT.read_text(encoding="utf-8").strip().split("## User template")[0]
    context = {
        "type": question.get("type"), "user_task": question.get("user_task"),
        "question_user": question.get("question_user"), "question_source": question.get("question_source"),
        "expected_behaviour": question.get("expected_behaviour"), "target_field": question.get("target_field"),
        "target_value": question.get("target_value"), "target_values": question.get("target_values"),
        "related_context": question.get("related_context"), "catalogue_name": question.get("catalogue_name"),
        "evidence_pages": [{"title": page.get("title"), "url": page.get("url"),
                            "text": page.get("text", "")[:5000]} for page in question.get("pages", [])],
    }
    result = _call_json("judge", prompt, json.dumps(context, ensure_ascii=False), output_root,
        mock=mock, mock_json={"plausible": True, "reason": "Mock plausibility check."})
    judgement = result.get("json") or {}
    return {"passed": judgement.get("plausible") is True, "judge": judgement,
            "model": result.get("model"), "cache_hit": result.get("cache_hit")}


def _round_trip(question: dict[str, Any], variant: str, output_root: Path, *, mock: bool) -> dict[str, Any]:
    prompt_pages = "\n\n".join(f"[{p['evidence_id']}] {p['title']} ({p['url']})\n{p['text']}" for p in question.get("pages", []))
    user = (f"Question: {question[variant]}\nExpected behaviour: {question['expected_behaviour']}\n"
        f"Evidence pages:\n{prompt_pages}\n\nReference answer for the comparison after you answer independently:\n{question['reference_answer']}")
    mock_answer = question.get("reference_answer", "")
    result = _call_json("judge", ROUNDTRIP_SYSTEM, user, output_root, mock=mock,
        mock_json={"answer": mock_answer, "matches_reference": True, "reason": "Mock round-trip response."})
    parsed = result.get("json") or {}
    return {"answer": parsed.get("answer", ""), "matches_reference": parsed.get("matches_reference") is True,
        "reason": parsed.get("reason", ""), "model": result.get("model"), "cache_hit": result.get("cache_hit")}


def _closed_book(question: dict[str, Any], output_root: Path, *, mock: bool) -> dict[str, Any]:
    messages = [{"role": "system", "content": CLOSED_BOOK_SYSTEM}, {"role": "user", "content": question["question_user"]}]
    if mock:
        answer = question["reference_answer"]
        generation = {"text": answer, "model": "mock-generator", "cache_hit": False}
    else:
        generation = call_llm("generator", messages, temperature=0.0, prompt_sha256=_hash(CLOSED_BOOK_SYSTEM),
            cache_dir=output_root / "cache" / "llm", log_dir=output_root / "logs")
    decisive = [str(p.get("text", "")) for p in question.get("keypoints", []) if p.get("decisive") is True]
    judge_prompt = (f"Question: {question['question_user']}\nExpected behaviour: {question['expected_behaviour']}\n"
        f"Decisive keypoints, all required:\n- " + "\n- ".join(decisive) +
        f"\n\nReference answer: {question['reference_answer']}\n\nClosed-book answer:\n{generation.get('text', '')}")
    judged = _call_json("judge", CLOSED_BOOK_JUDGE, judge_prompt, output_root, mock=mock,
        mock_json={"correct": True, "decisive_keypoints_present": [True for _ in decisive], "reason": "Mock closed-book result."})
    parsed = judged.get("json") or {}
    decisive_results = parsed.get("decisive_keypoints_present", [])
    if not isinstance(decisive_results, list):
        decisive_results = []
    explicit_absence = is_absence_keypoint(generation.get("text", ""))
    strict_correct = (parsed.get("correct") is True and len(decisive_results) == len(decisive)
                      and all(value is True for value in decisive_results)
                      and (question.get("expected_behaviour") != "state_absence" or explicit_absence))
    return {"answer": generation.get("text", ""), "correct": strict_correct,
        "judge": parsed, "model": generation.get("model"), "cache_hit": generation.get("cache_hit")}


def _semantic_condition_check(question: dict[str, Any], index: Any, chunks: list[dict[str, str]],
                              bm25_cache: Any, output_root: Path, *, mock: bool) -> dict[str, Any] | None:
    qtype = question.get("type")
    if qtype not in {"set_all", "unanswerable"}:
        return None
    field_rows = read_jsonl(ROOT / "data/processed/s2/catalogue_fields.jsonl")
    catalogue_rows = read_jsonl(ROOT / "data/processed/s2/catalogues.jsonl")
    catalogue = next((row for row in catalogue_rows if row.get("catalogue_id") == question.get("catalogue_id")), None)
    catalogue_urls = {m["url"] for m in catalogue.get("members", [])} if catalogue else set(question.get("field_match_urls", []))
    target_field = str(question.get("target_field", ""))
    temporal_condition = bool(question.get("time_condition") or re.search(r"time|temporal|coverage|period", target_field, re.I))
    if qtype == "set_all" and temporal_condition:
        incomplete = []
        for url in catalogue_urls:
            row = next((r for r in field_rows if r.get("url") == url), {})
            interval = parse_time_interval(str(row.get("fields", {}).get(target_field, "")))
            if interval.get("start_year") is None or (interval.get("end_year") is None and not interval.get("open_end")):
                incomplete.append(url)
        if incomplete:
            return {"passed": False, "auto_reject": True, "reason": "Temporal condition depends on unknown or incomplete catalogue intervals.",
                    "unknown_interval_urls": incomplete, "type": qtype}
    if question.get("time_condition"):
        unknown = []
        category_field = question["time_condition"].get("category_field")
        for page in question.get("pages", []):
            fields = next((r.get("fields", {}) for r in field_rows if r.get("url") == page.get("url")), {})
            temporal_field = target_field if qtype == "set_all" else question["time_condition"].get("temporal_field") or question.get("second_field", "")
            interval = parse_time_interval(str(fields.get(temporal_field, "")))
            if not interval.get("start_year"):
                unknown.append(page.get("url"))
        if unknown:
            return {"passed": False, "auto_reject": True, "reason": "Temporal condition relies on a record with unknown start year.",
                    "unknown_interval_urls": unknown, "type": qtype}
    query = str(question.get("question_user", ""))
    tokenized, dfs, _ = bm25_cache
    bm_rows = _bm25(query, chunks, tokenized, dfs, 50)
    vec_rows = [] if mock else index.retrieve(query, k=50)
    retrieved: dict[str, str] = {}
    for row in [*bm_rows, *vec_rows]:
        retrieved.setdefault(str(row["url"]), str(row.get("text", "")))
    rows_by_url = {str(row["url"]): row for row in field_rows}
    urls = catalogue_urls | set(retrieved)
    candidate_records = []
    for n, url in enumerate(sorted(urls), 1):
        metadata = rows_by_url.get(url, {})
        candidate_records.append({
            "record_id": url,
            "title": metadata.get("record_name") or metadata.get("title") or url,
            "fields": metadata.get("fields", {}),
            "time_intervals": metadata.get("time_intervals", {
                field: parse_time_interval(str(value)) for field, value in metadata.get("fields", {}).items()
                if re.search(r"time|period|temporal|coverage", field, re.I)}),
            "catalogue_member": url in catalogue_urls,
            "bm25_or_vector_excerpt": retrieved.get(url, "")[:360],
        })
    expected = set(question.get("field_match_urls", [])) if qtype == "set_all" else set()
    system = CONDITION_PROMPT.read_text(encoding="utf-8").split("## User template")[0].strip()
    if mock:
        judgement = {"records": [{"record_id": row["record_id"], "satisfies": row["record_id"] in expected} for row in candidate_records],
                     "matching_record_ids": sorted(expected), "condition_is_semantically_clear": True,
                     "reason": "Mock condition check."}
    else:
        # Bound each request: a catalogue plus two top-50 retrieval lists can
        # otherwise exceed provider request/response limits. Every candidate
        # still receives an explicit semantic judgement and is merged below.
        judged_records: list[dict[str, Any]] = []
        matching_ids: set[str] = set()
        batch_size = 12
        clear_flags: list[bool] = []
        reasons: list[str] = []
        for offset in range(0, len(candidate_records), batch_size):
            batch = candidate_records[offset:offset + batch_size]
            user = (f"Question: {query}\nQuestion type: {qtype}\n\nEvery catalogue member and each unique page retrieved in the BM25/vector top-50 is being checked across batches. "
                    "Decide every record in this batch individually; use interval overlap for time conditions. Do not infer decisions for other batches.\n"
                    + json.dumps(batch, ensure_ascii=False))
            response = _call_json("judge", system, user, output_root, mock=False)
            part = response.get("json") or {}
            judged_records.extend(row for row in part.get("records", []) if isinstance(row, dict))
            matching_ids.update(map(str, part.get("matching_record_ids", [])))
            clear_flags.append(part.get("condition_is_semantically_clear") is True)
            if part.get("reason"):
                reasons.append(str(part["reason"]))
        judgement = {"records": judged_records, "matching_record_ids": sorted(matching_ids),
                     "condition_is_semantically_clear": bool(clear_flags) and all(clear_flags),
                     "reason": "; ".join(reasons)}
    def has_inconsistency(item: dict[str, Any]) -> bool:
        ids = set(map(str, item.get("matching_record_ids", [])))
        record_rows = [row for row in item.get("records", []) if isinstance(row, dict)]
        complete = {str(row.get("record_id")) for row in record_rows}
        true_ids = {str(row.get("record_id")) for row in record_rows if row.get("satisfies") is True}
        return true_ids != ids or complete != {str(row["record_id"]) for row in candidate_records}
    retried = False
    if not mock and has_inconsistency(judgement):
        retried = True
        judged_records = []
        matching_ids = set()
        retry_flags = []
        for offset in range(0, len(candidate_records), 12):
            batch = candidate_records[offset:offset + 12]
            user = (f"RETRY: Your prior per-record judgments and matching_record_ids did not agree. Re-evaluate all records in this batch carefully and ensure the summary exactly matches records marked satisfies=true.\n"
                    f"Question: {query}\nQuestion type: {qtype}\nRecords:\n" + json.dumps(batch, ensure_ascii=False))
            response = _call_json("judge", system, user, output_root, mock=False)
            part = response.get("json") or {}
            judged_records.extend(row for row in part.get("records", []) if isinstance(row, dict))
            matching_ids.update(map(str, part.get("matching_record_ids", [])))
            retry_flags.append(part.get("condition_is_semantically_clear") is True)
        judgement = {"records": judged_records, "matching_record_ids": sorted(matching_ids),
                     "condition_is_semantically_clear": bool(retry_flags) and all(retry_flags),
                     "reason": "Retried once after record/list inconsistency."}
    judged_ids = set(map(str, judgement.get("matching_record_ids", [])))
    complete_ids = {str(row.get("record_id")) for row in judgement.get("records", []) if isinstance(row, dict)}
    required_ids = {str(row["record_id"]) for row in candidate_records}
    true_decision_ids = {str(row.get("record_id")) for row in judgement.get("records", [])
                         if isinstance(row, dict) and row.get("satisfies") is True}
    inconsistent_decisions = sorted(true_decision_ids ^ judged_ids)
    result_match = judged_ids == expected and complete_ids == required_ids and not inconsistent_decisions
    if qtype == "unanswerable":
        result_match = not judged_ids and complete_ids == required_ids and not inconsistent_decisions
    unresolved = has_inconsistency(judgement)
    return {"passed": result_match and judgement.get("condition_is_semantically_clear") is True,
            "needs_review": unresolved and retried, "retried_for_inconsistency": retried,
            "type": qtype, "expected_matching_record_ids": sorted(expected), "judge_matching_record_ids": sorted(judged_ids),
            "missing_judge_decisions": sorted(required_ids - complete_ids),
            "unexpected_matching_record_ids": sorted(judged_ids - expected),
            "inconsistent_record_vs_matching_list_ids": inconsistent_decisions,
            "candidate_count": len(candidate_records), "catalogue_member_count": len(catalogue_urls),
            "bm25_top_50_urls": list(dict.fromkeys(row["url"] for row in bm_rows)),
            "vector_top_50_urls": list(dict.fromkeys(str(row["url"]) for row in vec_rows)),
            "judge": judgement}


def _quote_support_check(question: dict[str, Any], output_root: Path, *, mock: bool) -> dict[str, Any]:
    system = QUOTE_SUPPORT_PROMPT.read_text(encoding="utf-8").split("## User template")[0].strip()
    pages = _passage_refs(question.get("pages", []))
    rows=[]
    absence_expected = question.get("expected_behaviour") == "state_absence"
    for i, point in enumerate(question.get("keypoints", []), 1):
        point_text = str(point.get("text", ""))
        if _is_derived_point(point):
            rows.append({"keypoint_index": i, "decisive": point.get("decisive") is True,
                         "applicable": False, "reason": "Derived conclusions are assessed by derivation_check."})
            continue
        if absence_expected and is_absence_keypoint(point_text):
            rows.append({"keypoint_index": i, "decisive": point.get("decisive") is True,
                         "applicable": False, "reason": "Absence claims are assessed by absence_check and condition_semantics."})
            continue
        sources = point.get("sources") if isinstance(point.get("sources"), list) else [point.get("source", "")]
        quotes = point.get("quotes") if isinstance(point.get("quotes"), list) else [point.get("quote", "")]
        evidence_results = []
        for source, quote_value in zip(sources, quotes):
            page = pages.get(str(source), {})
            quote = str(quote_value or "")
            if mock:
                result = {"json": {"supported": bool(quote and quote in page.get("text", "")), "reason": "Mock support check."}}
            else:
                user = (f"Keypoint: {point_text}\nSource quote: {quote}\nSource passage:\n{page.get('text', '')}")
                result = _call_json("judge", system, user, output_root, mock=False)
            parsed = result.get("json") or {}
            verbatim = bool(quote and quote in page.get("text", ""))
            evidence_results.append({"source": source, "quote": quote, "verbatim": verbatim,
                "supported": parsed.get("supported") is True, "reason": parsed.get("reason", "")})
        supported = bool(evidence_results) and all(r["verbatim"] and r["supported"] for r in evidence_results)
        rows.append({"keypoint_index": i, "decisive": point.get("decisive") is True,
                     "applicable": True, "passed": supported, "evidence": evidence_results})
    decisive_failed = [row for row in rows if row.get("applicable") and row.get("decisive") and not row.get("passed")]
    return {"passed": not decisive_failed, "keypoints": rows,
            "non_decisive_failures": [row["keypoint_index"] for row in rows if row.get("applicable") and not row.get("decisive") and not row.get("passed")]}


def _derivation_check(question: dict[str, Any], output_root: Path, *, mock: bool) -> dict[str, Any]:
    points = question.get("keypoints", [])
    derived = [(i, point) for i, point in enumerate(points, 1) if _is_derived_point(point)]
    rows = []
    system = ("You are a strict logical entailment judge. Determine whether the derived conclusion follows from the other supplied keypoints alone, without using outside facts. A ranking such as earliest must follow from the stated values; a conclusion that two records share a name or that two values differ must follow from explicit facts. Return JSON only: {\"derivable\":true|false,\"reason\":\"...\"}.")
    for index, point in derived:
        others = [{"text": item.get("text", ""), "decisive": item.get("decisive") is True}
                  for i, item in enumerate(points, 1) if i != index]
        if mock:
            parsed = {"derivable": bool(others), "reason": "Mock derivation check."}
        else:
            user = f"Derived conclusion: {point.get('text', '')}\nOther keypoints:\n{json.dumps(others, ensure_ascii=False)}"
            parsed = _call_json("judge", system, user, output_root, mock=False).get("json") or {}
        rows.append({"keypoint_index": index, "text": point.get("text"), "decisive": point.get("decisive") is True,
                     "derivable": parsed.get("derivable") is True, "reason": parsed.get("reason", "")})
    return {"applicable": bool(rows), "passed": all(row["derivable"] for row in rows), "keypoints": rows}


def _single_page_check(question: dict[str, Any], output_root: Path, *, mock: bool) -> dict[str, Any] | None:
    if question.get("type") not in {"set_all", "comparison_multi", "multi_hop", "ambiguity"} or len(question.get("pages", [])) < 2:
        return None
    system = ("You are a strict single-page answerability judge. First answer the question using only the one supplied page. Then compare your answer with the reference answer and check every decisive keypoint. Use the same strict rule as closed-book scoring: every decisive fact must be present and correct; missing a number or value is incorrect. For an absence question this check is not applicable. Return JSON only: {\"answer\":\"...\",\"decisive_keypoints_present\":[true|false],\"correct\":true|false,\"reason\":\"...\"}.")
    decisive = [str(point.get("text", "")) for point in question.get("keypoints", []) if point.get("decisive") is True]
    results = []
    for page in question.get("pages", []):
        if mock:
            parsed = {"answer": "", "decisive_keypoints_present": [False for _ in decisive], "correct": False,
                      "reason": "Mock check: no single page supplies every decisive keypoint."}
        else:
            user = (f"Question: {question.get('question_user', '')}\n\nReference answer:\n{question.get('reference_answer', '')}\n\n"
                    f"All decisive keypoints:\n- " + "\n- ".join(decisive) +
                    f"\n\nOnly page {page.get('title')} ({page.get('url')}) is available:\n{page.get('text', '')}")
            parsed = _call_json("judge", system, user, output_root, mock=False).get("json") or {}
        present = parsed.get("decisive_keypoints_present", [])
        if not isinstance(present, list):
            present = []
        sufficient = (parsed.get("correct") is True and len(present) == len(decisive) and all(value is True for value in present))
        results.append({"evidence_id": page.get("evidence_id"), "url": page.get("url"),
                        "sufficient": sufficient, "answer": parsed.get("answer", ""),
                        "decisive_keypoints_present": present, "reason": parsed.get("reason", "")})
    return {"passed": not any(row["sufficient"] for row in results), "pages": results}


def _unique_bridge_check(question: dict[str, Any], output_root: Path, *, mock: bool) -> dict[str, Any] | None:
    if question.get("type") != "multi_hop":
        return None
    system = ("You are a strict entity-resolution judge. Given the question and only its evidence pages, list the distinct entities matching the bridge description in the question. It must identify exactly one entity. Ignore connections that rely only on events, news, workshops, or project lists. Return JSON only: {\"entities\":[\"...\"],\"unique\":true|false,\"reason\":\"...\"}.")
    if mock:
        parsed = {"entities": [str(question.get("entity", "mock entity"))], "unique": True, "reason": "Mock bridge check."}
    else:
        pages = "\n\n".join(f"[{p.get('evidence_id')}] {p.get('title')}\n{p.get('text', '')}" for p in question.get("pages", []))
        user = f"Question: {question.get('question_user', '')}\n\nEvidence pages:\n{pages}"
        parsed = _call_json("judge", system, user, output_root, mock=False).get("json") or {}
    return {"passed": parsed.get("unique") is True and len(parsed.get("entities", [])) == 1,
            "entities": parsed.get("entities", []), "reason": parsed.get("reason", "")}


def _near_duplicate_check(questions: list[dict[str, Any]], *, mock: bool, index: Any | None) -> list[dict[str, Any]]:
    generated = [q for q in questions if q.get("status") == "generated"]
    current_ids = {str(row.get("id")) for row in generated}
    s1 = read_jsonl(S1_QUESTIONS)
    references = [{"id": row["id"], "question": row["question"], "set": "S1"} for row in s1]
    prior_main = read_jsonl(ROOT / "outputs/s2/main/questions_s2_main_checked.jsonl")
    references.extend({"id": row["id"], "question": row.get("question_user", ""), "set": "S2-main"}
                      for row in prior_main if row.get("status") == "generated" and str(row.get("id")) not in current_ids)
    prior_main2 = read_jsonl(ROOT / "outputs/s2/main2/questions_s2_main2_checked.jsonl")
    references.extend({"id": row["id"], "question": row.get("question_user", ""), "set": "S2-main2"}
                      for row in prior_main2 if row.get("status") == "generated" and str(row.get("id")) not in current_ids)
    references.extend({"id": row["id"], "question": row.get("question_user", ""), "set": "S2"} for row in generated)
    if not generated:
        return questions
    all_texts = [QUERY_PREFIX + str(row["question"]) for row in references]
    if mock:
        matrix = np.eye(len(all_texts), dtype=np.float32)
    else:
        vectors = index.encoder.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        matrix = np.asarray(vectors, dtype=np.float32) @ np.asarray(vectors, dtype=np.float32).T
    s1n = (len(s1)
           + sum(1 for row in prior_main if row.get("status") == "generated" and str(row.get("id")) not in current_ids)
           + sum(1 for row in prior_main2 if row.get("status") == "generated" and str(row.get("id")) not in current_ids))
    def year_normalized(text: str) -> str:
        text = re.sub(r"\b(?:18|19|20|21)\d{2}(?:\s*[-–]\s*(?:18|19|20|21)?\d{2})?\b", "<year>", text.casefold())
        return re.sub(r"[^a-z0-9]+", " ", text).strip()
    for i, question in enumerate(generated):
        position = s1n + i
        matches = []
        for j, ref in enumerate(references):
            if j == position:
                continue
            score = float(matrix[position, j])
            year_variant = bool(year_normalized(str(question.get("question_user", "")))
                                and year_normalized(str(question.get("question_user", ""))) == year_normalized(str(ref.get("question", ""))))
            if score > 0.9 or year_variant:
                matches.append({"id": ref["id"], "set": ref["set"], "similarity": score})
        question["near_duplicate"] = {"flagged": bool(matches), "threshold": 0.9, "matches": matches}
    return questions


def _rules(question: dict[str, Any], page_texts: dict[str, str]) -> dict[str, Any]:
    both = f"{question.get('question_user', '')} {question.get('question_source', '')}"
    lower = both.casefold()
    contact = bool(re.search(r"\b(e-?mail|phone number|telephone|postal address|home address|contact details)\b", lower)
                   or re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", both))
    identifier_used = bool(re.search(r"https?://\S+|\b(?:hash|sha-?256)\b|\b(?:ID|identifier)\s*[:#]?\s*[A-Z0-9_-]{4,}", both, re.I))
    named_page = any(page.get("title", "").casefold() in lower for page in question.get("pages", []) if len(page.get("title", "")) >= 5)
    identifier_only = identifier_used and not named_page
    answer_in_question = False
    if question.get("type") == "comparison_multi":
        # Comparison titles sometimes contain the compared dates; flag those explicit values.
        values = question.get("record_values", {})
        if isinstance(values, dict):
            expected_values = " ".join(str(x) for x in values.values())
        else:
            expected_values = " ".join(str(x.get("value", "")) for x in values if isinstance(x, dict))
        answer_values = set(re.findall(r"\b\d+(?:\.\d+)?\b", expected_values))
        question_values = set(re.findall(r"\b\d+(?:\.\d+)?\b", both))
        answer_in_question = bool(answer_values & question_values)
    if not answer_in_question:
        for point in question.get("keypoints", []):
            text = re.sub(r"\s+", " ", str(point.get("text", ""))).strip()
            if len(text) >= 35 and text.casefold() in re.sub(r"\s+", " ", both).casefold():
                answer_in_question = True
                break
    template_terms = [term for term in TEMPLATE_TERMS if term in lower]
    template_quotes = []
    known_template = re.compile(r"\b(licen[cs]e|keywords?|downloads?|previews?|no online assets found)\b", re.I)
    catalogue = next((row for row in read_jsonl(ROOT / "data/processed/s2/catalogues.jsonl")
                      if row.get("catalogue_id") == question.get("catalogue_id")), None)
    catalogue_urls = [m.get("url") for m in (catalogue or {}).get("members", [])]
    catalogue_size = len(catalogue_urls)
    for point in question.get("keypoints", []):
        quotes = point.get("quotes") if isinstance(point.get("quotes"), list) else [point.get("quote", "")]
        for quote_value in quotes:
            quote = str(quote_value or "")
            if not quote:
                continue
            known_field = bool(known_template.search(quote))
            occurrences = sum(quote.casefold() in page_texts.get(url, "").casefold() for url in catalogue_urls)
            majority = catalogue_size > 0 and occurrences > catalogue_size / 2
            if known_field or majority:
                template_quotes.append({"quote": quote, "catalogue_occurrences": occurrences,
                                        "catalogue_size": catalogue_size, "known_template_field": known_field})
    bridge_named = False
    if question.get("type") == "multi_hop" and question.get("entity"):
        bridge_named = str(question["entity"]).casefold() in lower
    return {
        "no_contact_or_private_information": {"passed": not contact, "flagged": contact},
        "not_identifier_only": {"passed": not identifier_only, "flagged": identifier_only},
        "question_does_not_include_answer": {"passed": not answer_in_question, "flagged": answer_in_question},
        "not_template_text": {"passed": not template_terms and not template_quotes, "flagged_terms": template_terms, "repeated_quotes": template_quotes},
        "multi_hop_bridge_not_named": {"applicable": question.get("type") == "multi_hop", "passed": not bridge_named},
    }


def _comparison_multi_check(question: dict[str, Any]) -> dict[str, Any] | None:
    if question.get("type") != "comparison_multi":
        return None
    field = str(question.get("target_field", ""))
    values_obj = question.get("record_values", {})
    values = list(values_obj.values()) if isinstance(values_obj, dict) else [row.get("value", "") for row in values_obj if isinstance(row, dict)]
    fields = read_jsonl(ROOT / "data/processed/s2/catalogue_fields.jsonl")
    field_rows = {row.get("url"): row.get("fields", {}) for row in fields}
    page_rows = [{"url": page.get("url"), "title": page.get("title"), "fields": field_rows.get(page.get("url"), {})}
                 for page in question.get("pages", [])]
    parsed = [_parse_comparison_value(field, str(value)) for value in values]
    actual_values = [str(page.get("fields", {}).get(field, "")).strip() for page in page_rows]
    source_values_match = len(actual_values) == len(values) and all(
        str(value).strip().casefold() == actual.casefold() for value, actual in zip(values, actual_values))
    relation = _shared_comparison_relation(page_rows)
    title_leak = _comparison_value_in_any_title([str(value) for value in values], [str(page.get("title", "")) for page in page_rows])
    passed = len(page_rows) == 3 and len(values) == 3 and all(parsed) and source_values_match and bool(relation) and not title_leak
    return {"passed": passed, "field": field, "record_values": values, "parsed_values": parsed,
            "source_values_match": source_values_match, "shared_allowed_relation": relation,
            "compared_value_in_title": title_leak}


def _phrasing_overlap(question: dict[str, Any]) -> dict[str, float]:
    evidence = " ".join(str(page.get("text", "")) for page in question.get("pages", []))
    evidence_tokens = set(_tokens(evidence))
    result = {}
    for variant in ("question_user", "question_source"):
        tokens = set(_tokens(str(question.get(variant, ""))))
        result[variant] = len(tokens & evidence_tokens) / max(1, len(tokens))
    return result


def _mock_retrieval_index() -> Any:
    class FakeIndex:
        encoder = None
        query_prefix = QUERY_PREFIX
        chunk_records = []

        @staticmethod
        def retrieve(query: str, k: int = 50) -> list[dict[str, Any]]:
            return []
    return FakeIndex()


def check_questions(questions: list[dict[str, Any]], *, output_root: Path = PILOT_DIR,
                    mock: bool = False, index: Any | None = None, retry_unanswerable: bool = True) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    chunks_df = pd.read_csv(CHUNKS_PATH, dtype=str, keep_default_na=False)
    chunks_by_id = {str(row["id"]): str(row["content"]) for row in chunks_df.to_dict("records")}
    page_texts = _read_page_texts()
    chunks = [{"chunk_id": str(row["id"]), "url": row["url"], "text": row["content"]} for row in chunks_df.to_dict("records")]
    bm25_cache = _build_bm25_index(chunks)
    if index is None:
        index = _mock_retrieval_index() if mock else load_index(ROOT / "data" / "processed" / "index")
    teacher_hash = _hash(TEACHER_PROMPT.read_text(encoding="utf-8").strip())
    for question in questions:
        if question.get("status") != "generated":
            question["checks"] = {"status": "not_run_generation_failed_or_skip"}
            if question.get("status") == "skip" and question.get("type") == "ambiguity":
                question["checks"]["conflict_check"] = _check_conflict(question, output_root, mock=mock)
            continue
        quote = quote_verbatim_check(question, chunks_by_id)
        multi = multi_page_check(question)
        completeness = set_completeness_check(question, chunks, bm25_cache)
        absence = _check_absence(question, index, chunks, bm25_cache, output_root, mock=mock)
        conflict = _check_conflict(question, output_root, mock=mock)
        condition = _semantic_condition_check(question, index, chunks, bm25_cache, output_root, mock=mock)
        quote_support = _quote_support_check(question, output_root, mock=mock)
        derivation = _derivation_check(question, output_root, mock=mock)
        single_page = _single_page_check(question, output_root, mock=mock)
        unique_bridge = _unique_bridge_check(question, output_root, mock=mock)
        comparison = _comparison_multi_check(question)
        plausibility = _plausibility_check(question, output_root, mock=mock)
        question["round_trip"] = {
            "question_user": _round_trip(question, "question_user", output_root, mock=mock),
            "question_source": _round_trip(question, "question_source", output_root, mock=mock),
        }
        question["closed_book"] = _closed_book(question, output_root, mock=mock)
        question["rules"] = _rules(question, page_texts)
        question["phrasing_overlap"] = _phrasing_overlap(question)
        question["checks"] = {
            "quote_verbatim": quote,
            "multi_page": multi,
            "set_completeness": completeness,
            "absence_check": absence,
            "conflict_check": conflict,
            "condition_semantics": condition,
            "quote_support": quote_support,
            "derivation_check": derivation,
            "single_page_check": single_page,
            "unique_bridge": unique_bridge,
            "comparison_multi_validity": comparison,
            "plausibility_check": plausibility,
            "round_trip": question["round_trip"],
            "closed_book": question["closed_book"],
            "near_duplicate": None,
            "rules": question["rules"],
            "phrasing_overlap": question["phrasing_overlap"],
            "teacher_prompt_sha256": question.get("teacher_prompt_sha256", teacher_hash),
        }
    _near_duplicate_check(questions, mock=mock, index=index)
    seen_catalogue_fields: dict[tuple[str, str], str] = {}
    for question in questions:
        if question.get("checks", {}).get("status"):
            continue
        question["checks"]["near_duplicate"] = question.get("near_duplicate")
        checks = question["checks"]
        reasons=[]
        if checks.get("absence_check") and not checks["absence_check"].get("passed"):
            reasons.append("absence_check_failed")
        if checks.get("condition_semantics") and not checks["condition_semantics"].get("passed"):
            if checks["condition_semantics"].get("needs_review"):
                question.setdefault("needs_review_reasons", []).append("condition_semantics_inconsistent_after_retry")
            else:
                reasons.append("semantic_condition_mismatch")
        if checks.get("quote_support") and not checks["quote_support"].get("passed"):
            reasons.append("quote_support_failed")
        if checks.get("derivation_check", {}).get("applicable") and not checks["derivation_check"].get("passed"):
            reasons.append("derivation_check_failed")
        if checks.get("quote_verbatim") and not checks["quote_verbatim"].get("passed"):
            reasons.append("quote_verbatim_failed")
        if checks.get("multi_page") and not checks["multi_page"].get("passed"):
            reasons.append("multi_page_requirement_failed")
        if checks.get("single_page_check") and not checks["single_page_check"].get("passed"):
            reasons.append("single_page_sufficient")
        if checks.get("unique_bridge") and not checks["unique_bridge"].get("passed"):
            reasons.append("bridge_not_unique")
        if checks.get("comparison_multi_validity") and not checks["comparison_multi_validity"].get("passed"):
            reasons.append("comparison_values_or_relation_invalid")
        if checks.get("plausibility_check") and not checks["plausibility_check"].get("passed"):
            reasons.append("implausible_research_question")
        if question.get("type") == "set_all" and question.get("target_field"):
            condition_key = (str(question.get("catalogue_id", "")), str(question.get("target_field", "")).casefold().strip(),
                             str(question.get("target_value", "")).casefold().strip())
            if condition_key in seen_catalogue_fields:
                reasons.append("duplicate_catalogue_condition_field")
                question["duplicate_catalogue_condition_field"] = {"duplicate_of": seen_catalogue_fields[condition_key]}
            else:
                seen_catalogue_fields[condition_key] = str(question.get("id", ""))
        if checks.get("near_duplicate", {}).get("flagged"):
            reasons.append("near_duplicate")
        rule_checks=checks.get("rules", {})
        failed_rules=[name for name,value in rule_checks.items() if isinstance(value,dict) and value.get("passed") is False]
        if failed_rules:
            reasons.extend(f"rule_failed:{name}" for name in failed_rules)
        round_trip=checks.get("round_trip",{})
        if any(not value.get("matches_reference") for value in round_trip.values() if isinstance(value,dict)):
            reasons.append("round_trip_failed")
        question["auto_rejection_reasons"]=reasons
        if retry_unanswerable and not mock and question.get("type") == "unanswerable" and reasons:
            feedback = ("VALIDATION RETRY: The previous question failed these checks: " + "; ".join(reasons)
                        + ". Correct the question, reference answer, and keypoints so every statement follows from the supplied evidence and the absence claim matches the semantic condition. A decisive state_absence keypoint must not include a quote.")
            bundle = {**question, "bundle_id": str(question.get("bundle_id") or question.get("id"))}
            regenerated = teacher_generate_one(bundle, TEACHER_PROMPT.read_text(encoding="utf-8").strip(),
                mock=False, output_root=output_root, validation_feedback=feedback)
            if regenerated.get("status") == "generated":
                retried = check_questions([regenerated], output_root=output_root, mock=False, index=index,
                                          retry_unanswerable=False)[0]
                retried["validation_retry"] = {"attempted": True, "feedback": feedback,
                    "first_attempt_rejection_reasons": reasons,
                    "generation_attempts": regenerated.get("generation_attempts", [])}
                if retried.get("auto_rejection_reasons"):
                    retried["status"] = "generation_failed"
                    retried["generation_failed_reason"] = "Validation still failed after the one allowed unanswerable retry."
                    retried["review_eligible"] = False
                question.update(retried)
                continue
            question["validation_retry"] = {"attempted": True, "feedback": feedback,
                "first_attempt_rejection_reasons": reasons,
                "generation_attempts": regenerated.get("generation_attempts", [])}
            question["status"] = "generation_failed"
            question["generation_failed_reason"] = "Teacher output remained invalid after the one allowed unanswerable retry."
            question["review_eligible"] = False
            continue
        question["auto_rejection_reasons"] = reasons
        question["needs_review"] = bool(question.get("needs_review_reasons"))
        question["review_eligible"] = not reasons
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PILOT_DIR / "questions_s2_pilot.jsonl")
    parser.add_argument("--output", type=Path, default=PILOT_DIR / "questions_s2_checked.jsonl")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--closed-book-only", action="store_true")
    args = parser.parse_args()
    if not args.mock:
        # Enforce the project .env as the source of credential values.
        load_dotenv(ROOT / ".env", override=True)
    source_rows=read_jsonl(args.input)
    if args.closed_book_only:
        rows=[]
        for row in source_rows:
            if row.get("status") == "generated":
                row["closed_book"]=_closed_book(row,args.output.parent,mock=args.mock)
            rows.append(row)
    else:
        rows = check_questions(source_rows, output_root=args.output.parent, mock=args.mock)
    write_jsonl(args.output, rows)
    print(json.dumps({"checked": len(rows), "generated": sum(r.get("status") == "generated" for r in rows),
        "quote_verbatim_passed": sum((r.get("checks", {}).get("quote_verbatim") or {}).get("passed", False) for r in rows),
        "multi_page_passed": sum((r.get("checks", {}).get("multi_page") or {}).get("passed", False) for r in rows),
        "review_eligible": sum(r.get("review_eligible",False) for r in rows),
        "rejected": sum(bool(r.get("auto_rejection_reasons")) for r in rows),
        "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
