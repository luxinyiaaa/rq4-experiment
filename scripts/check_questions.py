"""Run pilot answerability, closed-book, and near-duplicate checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from datetime import datetime
from zipfile import ZIP_DEFLATED, ZipFile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rq4.llm import call_llm
from scripts.build_index import DEFAULT_OUTDIR, RetrievalIndex, load_index, _read_jsonl
from scripts.generate_questions import PILOT_OUTPUT, PROMPT_PATH

OUTPUT_JSONL = PROJECT_ROOT / "outputs" / "pilot" / "questions_pilot.jsonl"
OUTPUT_REVIEW = PROJECT_ROOT / "outputs" / "pilot" / "review_sheet.xlsx"
GENERATOR_PROMPT = PROJECT_ROOT / "prompts" / "generator_answer.md"
CONFLICT_PROMPT = PROJECT_ROOT / "prompts" / "judge_kb_conflict.md"
MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"
GENERIC_PAGE_TITLES = {
    "edmed", "concept", "home", "document",
    "european directory of marine environmental data (edmed)",
}
REVIEW_DECISION = "decision (accept/edit/reject)"
REVIEW_NOTE = "review_note"
MODIFIED_QUESTION = "edited_question"
MODIFIED_ANSWER = "edited_reference_answer"
MODIFIED_KEYPOINTS = "edited_keypoints (one per line, [D] = decisive)"
MODIFIED_QUOTES = "edited_supporting_quotes (one per line, [passage] or [partner] prefix)"


def _extract_system(text: str, start_heading: str, end_heading: str) -> str:
    start = text.index(start_heading) + len(start_heading)
    end = text.index(end_heading, start)
    return text[start:end].strip()


def load_generator_systems(path: Path = GENERATOR_PROMPT) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    ordinary = _extract_system(text, "## System", "## User template")
    closed_section = text.split("# Closed-book variant", 1)[1]
    closed = _extract_system(closed_section, "## System", "## User template")
    return ordinary, closed


def load_conflict_prompt(path: Path = CONFLICT_PROMPT) -> str:
    return path.read_text(encoding="utf-8").strip()


def page_title_catalog(chunks: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Get each page's first non-generic Markdown H1 and index titles to URLs."""
    by_url: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_url.setdefault(str(chunk["url"]), []).append(chunk)
    titles: dict[str, str] = {}
    for url, page_chunks in by_url.items():
        page_chunks.sort(key=lambda row: int(row.get("chunk_id", 0)))
        found = None
        for chunk in page_chunks:
            for line in str(chunk.get("text", "")).splitlines():
                if line.startswith("# "):
                    candidate = line[2:].strip()
                    if candidate.casefold() not in GENERIC_PAGE_TITLES:
                        found = candidate
                        break
            if found:
                break
        if found:
            titles[url] = found
    by_title: dict[str, list[str]] = {}
    for url, title in titles.items():
        by_title.setdefault(title, []).append(url)
    return titles, by_title


def find_duplicate_titles(question: dict[str, Any], titles: dict[str, str], by_title: dict[str, list[str]]) -> dict[str, Any]:
    target_urls = [str(question["url"])]
    partner = question.get("partner")
    if partner:
        target_urls.append(str(partner["url"]))
    excluded = set(target_urls)
    matches = []
    for target_url in target_urls:
        title = titles.get(target_url)
        if not title:
            continue
        for other_url in by_title.get(title, []):
            if other_url not in excluded:
                matches.append({"title": title, "for_url": target_url, "url": other_url})
    return {"duplicate_title": bool(matches), "matches": matches}


def page_text_catalog(chunks: list[dict[str, Any]]) -> dict[str, str]:
    """Join cleaned chunks into one whitespace-normalizable text per URL."""
    pages: dict[str, list[tuple[int, str]]] = {}
    for chunk in chunks:
        url = str(chunk.get("url", ""))
        if not url:
            continue
        try:
            order = int(chunk.get("chunk_id", chunk.get("id", 0)))
        except (TypeError, ValueError):
            order = len(pages.get(url, []))
        pages.setdefault(url, []).append((order, str(chunk.get("text", chunk.get("content", "")))))
    return {url: "".join(text for _, text in sorted(rows)) for url, rows in pages.items()}


def template_answer_check(
    question: dict[str, Any], page_texts: dict[str, str], *, min_other_pages: int = 2
) -> dict[str, Any]:
    """Flag supporting quotes repeated verbatim on multiple other KB pages.

    Whitespace runs are collapsed on both quote and page text before matching.
    A question is flagged when any one of its supporting quotes appears on at
    least ``min_other_pages`` pages besides the question's gold and partner.
    """
    normalize = lambda value: re.sub(r"\s+", " ", str(value)).strip()
    excluded = {str(question.get("url", ""))}
    partner = question.get("partner")
    if partner and partner.get("url"):
        excluded.add(str(partner["url"]))
    normalized_pages = {
        url: normalize(text) for url, text in page_texts.items() if url not in excluded
    }
    matches = []
    for quote in question.get("supporting_quotes", []) or []:
        if isinstance(quote, dict):
            text = str(quote.get("text", ""))
            source = str(quote.get("source", "passage"))
        else:
            text, source = str(quote), "passage"
        normalized_quote = normalize(text)
        if not normalized_quote:
            continue
        urls = [url for url, page_text in normalized_pages.items() if normalized_quote in page_text]
        if len(urls) >= min_other_pages:
            matches.append({"source": source, "quote": text, "other_page_count": len(urls), "urls": urls})
    return {
        "status": "checked",
        "flagged": bool(matches),
        "matching_quotes": matches,
        "minimum_other_pages": min_other_pages,
        "whitespace_normalized": True,
    }


def run_template_answer_checks(
    questions: list[dict[str, Any]], page_texts: dict[str, str]
) -> list[dict[str, Any]]:
    output = []
    for source in questions:
        question = dict(source)
        question["template_answer_check"] = template_answer_check(question, page_texts)
        output.append(question)
    return output


def page_usage_violations(questions: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """Return page URLs used as gold/partner by more than one question."""
    uses: dict[str, list[dict[str, str]]] = {}
    for question in questions:
        for role, record in (("gold", question), ("partner", question.get("partner"))):
            if record and record.get("url"):
                uses.setdefault(str(record["url"]), []).append({
                    "question_id": str(question.get("id", "")),
                    "role": role,
                })
    return {url: rows for url, rows in uses.items() if len(rows) > 1}


def partner_gold_title_violations(
    questions: list[dict[str, Any]], titles: dict[str, str]
) -> list[dict[str, str]]:
    """Find partner titles that equal a different question's gold-page title."""
    gold_titles = {
        str(question.get("id", "")): titles.get(str(question.get("url", "")))
        for question in questions
    }
    violations = []
    for question in questions:
        partner = question.get("partner")
        if not partner:
            continue
        partner_title = titles.get(str(partner.get("url", "")))
        if not partner_title:
            continue
        for other_id, gold_title in gold_titles.items():
            if other_id != str(question.get("id", "")) and gold_title == partner_title:
                violations.append({
                    "question_id": str(question.get("id", "")),
                    "partner_url": str(partner.get("url", "")),
                    "title": partner_title,
                    "other_gold_id": other_id,
                })
    return violations


def _conflict_user_message(question: dict[str, Any], passages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    keypoints = "\n".join(
        f"{'[D]' if point.get('decisive') else '[ ]'} {point.get('text', '')}"
        for point in question.get("keypoints", [])
    ) or "(none provided)"
    blocks = []
    mapping = []
    for number, passage in enumerate(passages, start=1):
        passage_id = f"P{number}"
        chunk_id = str(passage["chunk_id"])
        url = str(passage["url"])
        blocks.append(f"[{passage_id}] (from {url}; chunk_id={chunk_id})\n{passage['text']}")
        mapping.append({"passage_id": passage_id, "chunk_id": chunk_id, "url": url})
    context = "\n\n".join(blocks) if blocks else "(No other-page chunks remained after removing gold and partner pages.)"
    message = (
        f"Question: {question.get('question', '')}\n\n"
        f"Reference answer: {question.get('reference_answer', '')}\n\n"
        f"Keypoints (D = decisive):\n{keypoints}\n\n"
        f"Passages from other pages of the knowledge base:\n\n{context}"
    )
    return message, mapping


def check_global_conflict(
    question: dict[str, Any],
    index: RetrievalIndex,
    conflict_system: str,
    *,
    llm_call=call_llm,
) -> dict[str, Any]:
    retrieved = index.retrieve(str(question.get("question", "")), k=10)
    excluded_urls = {str(question.get("url", ""))}
    if question.get("partner"):
        excluded_urls.add(str(question["partner"]["url"]))
    passages = [row for row in retrieved if str(row["url"]) not in excluded_urls]
    user_message, passage_map = _conflict_user_message(question, passages)
    prompt_hash = hashlib.sha256(conflict_system.encode("utf-8")).hexdigest()
    try:
        response = llm_call(
            "judge",
            [
                {"role": "system", "content": conflict_system},
                {"role": "user", "content": user_message},
            ],
            json_mode=True,
            temperature=0.0,
            extra_body={"enable_thinking": False},
            prompt_sha256=prompt_hash,
            log_messages=True,
        )
        result = response.get("json")
        if not isinstance(result, dict) or type(result.get("conflict")) is not bool or not isinstance(result.get("conflicts"), list):
            raise ValueError("Judge response must contain boolean conflict and list conflicts")
        by_passage = {row["passage_id"]: row for row in passage_map}
        conflicts = []
        for entry in result["conflicts"]:
            if not isinstance(entry, dict):
                continue
            mapped = by_passage.get(str(entry.get("passage_id", "")), {})
            conflicts.append({
                "passage_id": entry.get("passage_id"),
                "chunk_id": mapped.get("chunk_id"),
                "url": mapped.get("url"),
                "what_differs": str(entry.get("what_differs", "")),
            })
        return {
            "status": "checked",
            "conflict": result["conflict"],
            "conflicts": conflicts,
            "passage_map": passage_map,
            "retrieved_top_k": len(retrieved),
            "checked_passage_count": len(passages),
            "raw_json": result,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "passage_map": passage_map,
            "retrieved_top_k": len(retrieved),
            "checked_passage_count": len(passages),
        }


def answer_user_message(question: str, passages: list[tuple[str, str]]) -> str:
    blocks = [f"[{passage_id}] {text}" for passage_id, text in passages]
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def _answer_from_passages(
    role: str,
    question: str,
    system: str,
    passages: list[tuple[str, str]],
    *,
    llm_call=call_llm,
) -> str:
    response = llm_call(
        role,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": answer_user_message(question, passages)},
        ],
        temperature=0.0,
        extra_body={"enable_thinking": False} if role == "judge" else None,
    )
    return str(response.get("text", "")).strip()


def check_decisive_labels(keypoints: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    if len(labels) != len(keypoints):
        raise ValueError(f"Received {len(labels)} labels for {len(keypoints)} keypoints")
    normalized = [str(label).strip().casefold() for label in labels]
    decisive_indices = [i for i, point in enumerate(keypoints) if bool(point.get("decisive"))]
    decisive_all_entailment = bool(decisive_indices) and all(
        normalized[i] == "entailment" for i in decisive_indices
    )
    no_contradiction = all(label != "contradiction" for label in normalized)
    return {
        "labels": labels,
        "decisive_all_entailment": decisive_all_entailment,
        "no_contradiction": no_contradiction,
        "pass": decisive_all_entailment and no_contradiction,
    }


def ragchecker_response2answer(
    *,
    question_id: str,
    question: str,
    reference_answer: str,
    answer: str,
    keypoints: list[dict[str, Any]],
    context: list[tuple[str, str]],
    llm_call=call_llm,
    models_config_path: Path = MODELS_CONFIG,
    trace: list[dict[str, Any]] | None = None,
    check_type: str = "response2answer",
) -> list[str]:
    """Use RAGChecker's response2answer labels through the cached rq4 LLM client."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from ragchecker import RAGChecker
    from ragchecker.container import RAGResult, RAGResults, RetrievedDoc

    with models_config_path.open(encoding="utf-8") as stream:
        judge_settings = yaml.safe_load(stream)["judge"]
    result = RAGResult(
        query_id=question_id,
        query=question,
        gt_answer=reference_answer,
        response=answer,
        retrieved_context=[RetrievedDoc(doc_id=pid, text=text) for pid, text in context],
        gt_answer_claims=[str(point["text"]) for point in keypoints],
    )
    results = RAGResults(results=[result])

    def cached_judge_calls(prompts: list[str]) -> list[str]:
        labels = []
        for prompt in prompts:
            is_claim_judgment = "If the claim is supported by ANY passage" in prompt
            response = llm_call(
                "judge",
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                extra_body={"enable_thinking": False},
                max_tokens=10 if is_claim_judgment else 1024,
                log_messages=True,
            )
            raw_label = str(response["text"]).strip()
            if raw_label not in {"Entailment", "Neutral", "Contradiction"}:
                raise ValueError(f"RAGChecker judge returned an invalid label: {raw_label!r}")
            labels.append(raw_label)
            if trace is not None:
                trace.append({
                    "prompt": prompt,
                    "raw_response": raw_label,
                    "model_id": response.get("api_model", response.get("model")),
                    "token_usage": response.get("usage"),
                    "request_parameters": response.get("request_parameters"),
                    "cache_hit": response.get("cache_hit"),
                })
        return labels

    checker = RAGChecker(
        extractor_name=f"openai/{judge_settings['model']}",
        checker_name=f"openai/{judge_settings['model']}",
        extractor_api_base=judge_settings["base_url"],
        checker_api_base=judge_settings["base_url"],
        joint_check=False,
        custom_llm_api_func=cached_judge_calls,
    )
    checker.check_claims(results, check_type=check_type)
    labels = getattr(result, check_type)
    if not isinstance(labels, list):
        raise ValueError(f"RAGChecker returned unexpected labels for {question_id}: {labels!r}")
    return [str(label) for label in labels]


def mark_near_duplicates(
    questions: list[dict[str, Any]],
    encoder: Any,
    query_prefix: str,
    threshold: float = 0.9,
) -> list[dict[str, Any]]:
    valid = [row for row in questions if row.get("status") == "generated" and row.get("question")]
    if not valid:
        return questions
    vectors = encoder.encode(
        [query_prefix + str(row["question"]) for row in valid],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarities = np.asarray(vectors, dtype=np.float32) @ np.asarray(vectors, dtype=np.float32).T
    matches: dict[str, list[dict[str, Any]]] = {str(row["id"]): [] for row in valid}
    for left in range(len(valid)):
        for right in range(left + 1, len(valid)):
            score = float(similarities[left, right])
            if score > threshold:
                left_id, right_id = str(valid[left]["id"]), str(valid[right]["id"])
                matches[left_id].append({"id": right_id, "similarity": score})
                matches[right_id].append({"id": left_id, "similarity": score})
    for row in questions:
        pair_matches = matches.get(str(row["id"]), [])
        row["near_duplicate"] = bool(pair_matches)
        row["near_duplicate_matches"] = pair_matches
    return questions


def _question_context(question: dict[str, Any]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    gold = [("P1", str(question["text"]))]
    roundtrip = list(gold)
    rag_context = [(str(question.get("window_id", question["id"])), str(question["text"]))]
    if question.get("type") == "comparison" and question.get("partner"):
        partner = question["partner"]
        roundtrip.append(("P2", str(partner["text"])))
        rag_context.append((str(partner.get("window_id", "partner")), str(partner["text"])))
    return roundtrip, rag_context


def _read_review_sheet(path: Path) -> dict[str, dict[str, str]]:
    """Read decisions and edit fields from the workbook, including inline-string XLSX files."""
    if not path.exists():
        return {}
    try:
        import openpyxl
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        values = list(workbook.active.iter_rows(values_only=True))
        workbook.close()
        if not values:
            return {}
        headers = [str(value or "") for value in values[0]]
        records = [dict(zip(headers, (str(v or "") for v in row))) for row in values[1:]]
    except ImportError:
        with zipfile.ZipFile(path) as archive:
            shared = []
            if "xl/sharedStrings.xml" in archive.namelist():
                shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                shared = ["".join(node.itertext()) for node in shared_root.findall("m:si", ns)]
            root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            xml_rows = root.findall(".//m:sheetData/m:row", ns)
            matrix: list[list[str]] = []
            for row in xml_rows:
                cells: dict[int, str] = {}
                for cell in row.findall("m:c", ns):
                    ref = str(cell.attrib.get("r", ""))
                    letters = re.match(r"([A-Z]+)", ref)
                    if not letters:
                        continue
                    column = 0
                    for char in letters.group(1):
                        column = column * 26 + ord(char) - 64
                    cell_type = cell.attrib.get("t")
                    if cell_type == "inlineStr":
                        value = "".join(cell.find("m:is", ns).itertext()) if cell.find("m:is", ns) is not None else ""
                    else:
                        raw = cell.findtext("m:v", default="", namespaces=ns)
                        value = shared[int(raw)] if cell_type == "s" and raw else raw
                    cells[column - 1] = value
                matrix.append([cells.get(i, "") for i in range(max(cells, default=-1) + 1)])
        if not matrix:
            return {}
        headers = [str(value) for value in matrix[0]]
        records = [dict(zip(headers, row)) for row in matrix[1:]]
    result = {}
    for record in records:
        qid = str(record.get("id", "")).strip()
        if qid:
            result[qid] = record
    return result


def _parse_modified_keypoints(value: str) -> list[dict[str, Any]]:
    points = []
    for line in value.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\[(D| )\]\s*(.+)$", line, flags=re.IGNORECASE)
        if match:
            points.append({"text": match.group(2).strip(), "decisive": match.group(1).upper() == "D"})
        else:
            points.append({"text": line, "decisive": False})
    if not points:
        raise ValueError("Modified keypoints column is empty")
    return points


def _edited_version(question: dict[str, Any]) -> dict[str, Any] | None:
    review = question.get("human_review", {})
    decision = str(review.get(REVIEW_DECISION, "")).strip().casefold()
    if decision != "edit":
        return None
    missing = [name for name in (MODIFIED_QUESTION, MODIFIED_ANSWER, MODIFIED_KEYPOINTS) if not str(review.get(name, "")).strip()]
    if missing:
        raise ValueError("Decision is edit, but these edit columns are empty: " + ", ".join(missing))
    return {
        **question,
        "question": str(review[MODIFIED_QUESTION]).strip(),
        "reference_answer": str(review[MODIFIED_ANSWER]).strip(),
        "keypoints": _parse_modified_keypoints(str(review[MODIFIED_KEYPOINTS])),
    }


def run_new_checks(
    questions: list[dict[str, Any]],
    index: RetrievalIndex,
    conflict_system: str,
    *,
    llm_call=call_llm,
) -> list[dict[str, Any]]:
    titles, by_title = page_title_catalog(index.chunk_records)
    output = []
    for position, source in enumerate(questions, start=1):
        question = dict(source)
        question["duplicate_title"] = find_duplicate_titles(question, titles, by_title)
        if question.get("status") == "generated":
            print(f"Global conflict check {position}/{len(questions)}: {question['id']}", flush=True)
            question["global_conflict"] = check_global_conflict(
                question, index, conflict_system, llm_call=llm_call
            )
        else:
            question["global_conflict"] = {"status": "not_run", "reason": "Question was not generated"}
        output.append(question)
    return output


def run_edited_question_checks(
    questions: list[dict[str, Any]],
    index: RetrievalIndex,
    conflict_system: str,
    generator_system: str,
    *,
    llm_call=call_llm,
) -> None:
    for question in questions:
        try:
            edited = _edited_version(question)
        except ValueError as exc:
            question["edited_checks"] = {"status": "unavailable", "error": str(exc)}
            continue
        if edited is None:
            continue
        qid = str(question["id"])
        context, rag_context = _question_context(edited)
        try:
            answer = _answer_from_passages("judge", edited["question"], generator_system, context, llm_call=llm_call)
            labels = ragchecker_response2answer(
                question_id=f"{qid}-edited-roundtrip",
                question=edited["question"],
                reference_answer=edited["reference_answer"],
                answer=answer,
                keypoints=edited["keypoints"],
                context=rag_context,
                llm_call=llm_call,
            )
            roundtrip = {"status": "checked", **check_decisive_labels(edited["keypoints"], labels), "answer": answer}
        except Exception as exc:
            roundtrip = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
        conflict = check_global_conflict(edited, index, conflict_system, llm_call=llm_call)
        question["edited_checks"] = {"status": "checked", "round_trip_check": roundtrip, "global_conflict": conflict}


def run_closed_book_only(
    questions: list[dict[str, Any]],
    closed_book_system: str,
    *,
    llm_call=call_llm,
) -> list[dict[str, Any]]:
    """Rerun only closed-book answer generation and its RAGChecker assessment."""
    output = []
    for position, source in enumerate(questions, start=1):
        row = dict(source)
        if row.get("status") != "generated":
            output.append(row)
            continue
        try:
            effective = _edited_version(row) or row
            answer = llm_call(
                "generator",
                [
                    {"role": "system", "content": closed_book_system},
                    {"role": "user", "content": f"Question: {effective['question']}"},
                ],
                temperature=0.0,
            )
            answer_text = str(answer.get("text", "")).strip()
            labels = ragchecker_response2answer(
                question_id=f"{row['id']}-closed-book",
                question=effective["question"],
                reference_answer=effective["reference_answer"],
                answer=answer_text,
                keypoints=effective["keypoints"],
                context=[],
                llm_call=llm_call,
            )
            judged = check_decisive_labels(effective["keypoints"], labels)
            row["closed_book_answer"] = answer_text
            row["closed_book_labels"] = judged["labels"]
            row["closed_book_correct"] = judged["pass"]
            row["closed_book_status"] = "checked"
            row["closed_book_version"] = "edited" if effective is not source else "original"
            row.pop("closed_book_error", None)
        except Exception as exc:
            row["closed_book_correct"] = None
            row["closed_book_status"] = "unavailable"
            row["closed_book_error"] = f"{type(exc).__name__}: {exc}"
        print(f"Closed-book check {position}/{len(questions)}: {row['id']} ({row.get('closed_book_status')})", flush=True)
        output.append(row)
    return output


def run_checks(
    questions: list[dict[str, Any]],
    index: RetrievalIndex,
    *,
    generator_system: str,
    closed_book_system: str,
    llm_call=call_llm,
) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    for position, source in enumerate(questions, start=1):
        question = dict(source)
        if question.get("status") != "generated":
            question.update({
                "round_trip_check": {"status": "not_run", "reason": question.get("skip_reason", "Question skipped")},
                "closed_book_correct": None,
                "near_duplicate": False,
                "near_duplicate_matches": [],
            })
            checked.append(question)
            continue

        qid = str(question["id"])
        qtext = str(question["question"])
        keypoints = question["keypoints"]
        roundtrip_passages, rag_context = _question_context(question)
        print(f"Answerability check {position}/{len(questions)}: {qid}", flush=True)
        try:
            roundtrip_answer = _answer_from_passages(
                "judge", qtext, generator_system, roundtrip_passages, llm_call=llm_call
            )
            roundtrip_labels = ragchecker_response2answer(
                question_id=f"{qid}-roundtrip",
                question=qtext,
                reference_answer=str(question.get("reference_answer", "")),
                answer=roundtrip_answer,
                keypoints=keypoints,
                context=rag_context,
                llm_call=llm_call,
            )
            roundtrip_result = check_decisive_labels(keypoints, roundtrip_labels)
            question["round_trip_answer"] = roundtrip_answer
            question["round_trip_check"] = {"status": "checked", **roundtrip_result}
        except Exception as exc:
            question["round_trip_check"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}

        print(f"Closed-book check {position}/{len(questions)}: {qid}", flush=True)
        try:
            closed_response = llm_call(
                "generator",
                [
                    {"role": "system", "content": closed_book_system},
                    {"role": "user", "content": f"Question: {qtext}"},
                ],
                temperature=0.0,
            )
            closed_book_answer = str(closed_response.get("text", "")).strip()
            closed_labels = ragchecker_response2answer(
                question_id=f"{qid}-closed-book",
                question=qtext,
                reference_answer=str(question.get("reference_answer", "")),
                answer=closed_book_answer,
                keypoints=keypoints,
                context=[],
                llm_call=llm_call,
            )
            closed_result = check_decisive_labels(keypoints, closed_labels)
            question["closed_book_answer"] = closed_book_answer
            question["closed_book_labels"] = closed_result["labels"]
            question["closed_book_correct"] = closed_result["pass"]
            question["closed_book_status"] = "checked"
        except Exception as exc:
            question["closed_book_correct"] = None
            question["closed_book_status"] = "unavailable"
            question["closed_book_error"] = f"{type(exc).__name__}: {exc}"
        checked.append(question)

    return mark_near_duplicates(checked, index.encoder, index.query_prefix, threshold=0.9)


def _review_rows(questions: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for question in questions:
        keypoints = question.get("keypoints", [])
        formatted_keypoints = "\n".join(
            f"{'[D]' if point.get('decisive') else '[ ]'} {point.get('text', '')}"
            for point in keypoints
        )
        quotes = "\n".join(
            f"[{quote.get('source', '?')}] {quote.get('text', '')}"
            + (" (QUOTE FAILED)" if not quote.get("matched") else "")
            for quote in question.get("supporting_quotes", [])
        )
        if question.get("quote_failed") and not quotes:
            quotes = "QUOTE FAILED: teacher returned no supporting quote."
        original = f"PASSAGE ({question.get('url', '')}):\n{question.get('text', '')}"
        if question.get("type") == "comparison" and question.get("partner"):
            partner = question["partner"]
            original += f"\n\nPARTNER ({partner.get('url', '')}):\n{partner.get('text', '')}"
        if question.get("type") == "false_premise" and question.get("similar"):
            similar = question["similar"]
            original += f"\n\nSIMILAR PASSAGE ({similar.get('url', '')}):\n{similar.get('text', '')}"
        roundtrip = question.get("round_trip_check", {})
        if roundtrip.get("status") == "checked":
            roundtrip_summary = (
                f"{'PASS' if roundtrip.get('pass') else 'FAIL'}; labels="
                + ", ".join(str(label) for label in roundtrip.get("labels", []))
            )
        else:
            roundtrip_summary = roundtrip.get("status", "not_run")
            if roundtrip.get("reason"):
                roundtrip_summary += f": {roundtrip['reason']}"
        closed_summary: Any = question.get("closed_book_correct")
        if question.get("closed_book_status") == "unavailable":
            closed_summary = f"unavailable: {question.get('closed_book_error', '')}"
        title_check = question.get("duplicate_title", {})
        if title_check.get("matches"):
            title_summary = "\n".join(
                f"{row.get('title')} (same as {row.get('for_url')}): {row.get('url')}"
                for row in title_check["matches"]
            )
        else:
            title_summary = "No duplicate title" if title_check.get("duplicate_title") is False else title_check.get("status", "not_run")
        conflict_check = question.get("global_conflict", {})
        if conflict_check.get("status") == "checked":
            conflict_rows = conflict_check.get("conflicts", [])
            conflict_summary = "\n".join(
                f"{row.get('passage_id')} / chunk {row.get('chunk_id')} — {row.get('what_differs')} ({row.get('url')})"
                for row in conflict_rows
            ) if conflict_rows else "No conflicting passage identified"
        else:
            conflict_summary = conflict_check.get("error", conflict_check.get("reason", conflict_check.get("status", "not_run")))
        human_review = question.get("human_review", {})
        template_check = question.get("template_answer_check", {})
        template_summary = "not_run"
        if template_check.get("status") == "checked":
            if not template_check.get("flagged"):
                template_summary = "No repeated supporting quote found on two or more other pages"
            else:
                template_summary = "\n".join(
                    f"[{entry.get('source')}] {entry.get('quote')} ({entry.get('other_page_count')} other pages): "
                    + ", ".join(entry.get("urls", []))
                    for entry in template_check.get("matching_quotes", [])
                )
        edited_checks = question.get("edited_checks", {})
        roundtrip_cell = roundtrip_summary
        if edited_checks.get("round_trip_check"):
            edited_rt = edited_checks["round_trip_check"]
            edit_status = "unavailable" if edited_rt.get("status") != "checked" else ("PASS" if edited_rt.get("pass") else "FAIL")
            roundtrip_cell += f"\nEdited version: {edit_status}"
        if edited_checks.get("error"):
            roundtrip_cell += f"\nEdited version: {edited_checks['error']}"
        if edited_checks.get("global_conflict", {}).get("status") == "checked":
            edit_conflicts = edited_checks["global_conflict"].get("conflicts", [])
            conflict_summary += "\nEdited version: " + (
                "; ".join(f"chunk {c.get('chunk_id')}: {c.get('what_differs')}" for c in edit_conflicts)
                if edit_conflicts else "No conflicting passage identified"
            )
        rows.append({
            "id": question.get("id"),
            "type": question.get("type"),
            "question": question.get("question", ""),
            "reference_answer": question.get("reference_answer", ""),
            "keypoints ([D] = decisive)": formatted_keypoints,
            "supporting_quotes": quotes,
            "passage (comparison: with partner)": original,
            "round_trip_check": roundtrip_cell,
            "closed_book_correct": closed_summary,
            "near_duplicate (>0.9)": question.get("near_duplicate", False),
            "near_duplicate_with": json.dumps(question.get("near_duplicate_matches", []), ensure_ascii=False),
            "false_premise_note": question.get("false_premise", "") or "",
            "gold_chunk_ids": ", ".join(str(value) for value in question.get("gold_chunk_ids", [])),
            "duplicate_title": title_summary,
            "kb_conflict (chunk IDs and difference)": conflict_summary,
            "template_answer_check": template_summary,
            MODIFIED_QUESTION: human_review.get(MODIFIED_QUESTION, ""),
            MODIFIED_ANSWER: human_review.get(MODIFIED_ANSWER, ""),
            MODIFIED_KEYPOINTS: human_review.get(MODIFIED_KEYPOINTS, ""),
            MODIFIED_QUOTES: human_review.get(MODIFIED_QUOTES, ""),
            REVIEW_DECISION: human_review.get(REVIEW_DECISION, ""),
            REVIEW_NOTE: human_review.get(REVIEW_NOTE, ""),
        })
    return pd.DataFrame(rows)


def _safe_review_export_path(path: Path) -> Path:
    if not path.exists():
        return path
    existing = _read_review_sheet(path)
    if not any(str(row.get(REVIEW_DECISION, "")).strip() for row in existing.values()):
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{stamp}_{suffix}{path.suffix}")
        suffix += 1
    return candidate


def write_outputs(questions: list[dict[str, Any]], jsonl_path: Path, xlsx_path: Path) -> Path:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    actual_xlsx_path = _safe_review_export_path(xlsx_path)
    actual_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in questions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    frame = _review_rows(questions)
    try:
        with pd.ExcelWriter(actual_xlsx_path, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="Pilot questions")
            sheet = writer.sheets["Pilot questions"]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            widths = {"A": 12, "B": 18, "C": 55, "D": 45, "E": 48, "F": 48, "G": 80,
                      "H": 40, "I": 16, "J": 22, "K": 40, "L": 30, "M": 40, "N": 50,
                      "O": 24, "P": 60, "Q": 60, "R": 55, "S": 55, "T": 48, "U": 30, "V": 40}
            for column, width in widths.items():
                sheet.column_dimensions[column].width = width
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        _write_xlsx_minimal(frame, actual_xlsx_path)
    return actual_xlsx_path


def _write_xlsx_minimal(frame: pd.DataFrame, path: Path) -> None:
    """Write a standards-compliant, dependency-free XLSX when openpyxl is absent."""
    def column_name(number: int) -> str:
        name = ""
        while number:
            number, remainder = divmod(number - 1, 26)
            name = chr(65 + remainder) + name
        return name

    widths = [12, 18, 55, 45, 48, 48, 80, 40, 16, 22, 40, 30, 40, 50, 24, 60, 60, 55, 55, 48, 30, 40]
    rows = [list(frame.columns)] + frame.fillna("").astype(str).values.tolist()
    sheet_rows = []
    for row_number, values in enumerate(rows, start=1):
        cells = []
        for column_number, value in enumerate(values, start=1):
            address = f"{column_name(column_number)}{row_number}"
            cells.append(f'<c r="{address}" t="inlineStr"><is><t xml:space="preserve">{escape(str(value))}</t></is></c>')
        sheet_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    cols_xml = "".join(f'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>' for i, width in enumerate(widths, 1))
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<cols>{cols_xml}</cols><sheetData>{"".join(sheet_rows)}</sheetData>'
        f'<autoFilter ref="A1:{column_name(len(frame.columns))}{len(rows)}"/>'
        '</worksheet>'
    )
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                '<sheets><sheet name="Pilot questions" sheetId="1" r:id="rId1"/></sheets></workbook>')
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        archive.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=PILOT_OUTPUT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--generator-prompt", type=Path, default=GENERATOR_PROMPT)
    parser.add_argument("--out", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--review-xlsx", type=Path, default=OUTPUT_REVIEW)
    parser.add_argument("--comparison-questions", type=Path, help="include another question JSONL when checking near-duplicates")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--new-checks-only", action="store_true", help="run only title and whole-KB conflict checks")
    modes.add_argument("--closed-book-only", action="store_true", help="rerun only closed-book answer checks")
    args = parser.parse_args()

    questions = _read_jsonl(args.questions)
    human_reviews = _read_review_sheet(args.review_xlsx)
    for question in questions:
        question["human_review"] = human_reviews.get(str(question.get("id")), {})

    generator_system, closed_book_system = load_generator_systems(args.generator_prompt)
    if args.closed_book_only:
        checked = run_closed_book_only(questions, closed_book_system)
    else:
        index = load_index(args.index_dir)
        conflict_system = load_conflict_prompt()
        if args.new_checks_only:
            checked = run_new_checks(questions, index, conflict_system)
        else:
            checked = run_checks(
                questions,
                index,
                generator_system=generator_system,
                closed_book_system=closed_book_system,
            )
            checked = run_new_checks(checked, index, conflict_system)
        run_edited_question_checks(checked, index, conflict_system, generator_system)
        if args.comparison_questions:
            comparison_questions = _read_jsonl(args.comparison_questions)
            combined = mark_near_duplicates(
                checked + comparison_questions, index.encoder, index.query_prefix, threshold=0.9
            )
            checked = combined[:len(checked)]
    actual_review_path = write_outputs(checked, args.out, args.review_xlsx)
    generated = [row for row in checked if row.get("status") == "generated"]
    print(f"Pilot questions: {len(checked)} total; {len(generated)} generated; {len(checked)-len(generated)} skipped")
    if args.new_checks_only:
        print(f"Duplicate titles: {sum(row.get('duplicate_title', {}).get('duplicate_title') is True for row in generated)} / {len(generated)}")
        print(f"Whole-KB conflict checks completed: {sum(row.get('global_conflict', {}).get('status') == 'checked' for row in generated)} / {len(generated)}")
        print(f"Whole-KB conflicts flagged: {sum(row.get('global_conflict', {}).get('conflict') is True for row in generated)}")
    else:
        print(f"Round-trip passed: {sum(row.get('round_trip_check', {}).get('pass') is True for row in generated)} / {len(generated)}")
        print(f"Duplicate-title questions: {sum(row.get('duplicate_title', {}).get('duplicate_title') is True for row in generated)} / {len(generated)}")
        print(f"Whole-KB conflicts flagged: {sum(row.get('global_conflict', {}).get('conflict') is True for row in generated)} / {len(generated)}")
    if args.closed_book_only or not args.new_checks_only:
        completed = sum(row.get("closed_book_status") == "checked" for row in generated)
        print(f"Closed-book correct: {sum(row.get('closed_book_correct') is True for row in generated)} / {completed} completed")
        print(f"Closed-book unavailable: {sum(row.get('closed_book_status') == 'unavailable' for row in generated)}")
    if not args.new_checks_only:
        print(f"Near-duplicate questions: {sum(row.get('near_duplicate') is True for row in generated)}")
        print(f"Quote failures: {sum(row.get('quote_failed') is True for row in checked)}")
    print(f"Wrote {args.out}")
    print(f"Wrote {actual_review_path}")
    if not args.new_checks_only:
        print("\nTwo pilot examples:")
        for row in generated[:2]:
            print(f"- {row['id']} [{row['type']}]: {row.get('question', '')}")
            print(f"  Answer: {row.get('reference_answer', '')}")
            print(f"  Round-trip: {row.get('round_trip_check', {}).get('pass', 'unavailable')}; "
                  f"closed-book: {row.get('closed_book_correct', 'unavailable')}; "
                  f"quote_failed: {row.get('quote_failed', False)}")


if __name__ == "__main__":
    main()
