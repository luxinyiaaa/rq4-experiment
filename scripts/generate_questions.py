"""Generate RQ4 questions with the configured teacher model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rq4.llm import call_llm
from scripts.assign_types import INPUT_WINDOWS, OUTPUT_PLAN, TYPE_COUNTS, _read_jsonl

PROMPT_PATH = PROJECT_ROOT / "prompts" / "teacher_generate.md"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks_clean.csv"
PILOT_OUTPUT = PROJECT_ROOT / "outputs" / "pilot" / "questions_pilot.jsonl"
PILOT_TYPE_COUNTS = {
    "list_all": 3,
    "comparison": 2,
    "paraphrased": 3,
    "false_premise": 2,
}


def _section(text: str, heading: str, following_heading: str) -> str:
    start = text.index(heading) + len(heading)
    end = text.index(following_heading, start)
    return text[start:end].strip()


def load_teacher_prompt(path: Path = PROMPT_PATH) -> str:
    # Send the entire file verbatim so the general rules, type rules, schema,
    # and all four examples are present in the actual model messages.
    return path.read_text(encoding="utf-8").strip()


def select_pilot(plan: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected_ids: set[str] = set()
    for question_type, count in PILOT_TYPE_COUNTS.items():
        choices = [row for row in plan if row["type"] == question_type]
        if len(choices) < count:
            raise ValueError(f"Pilot needs {count} {question_type} items; plan has {len(choices)}")
        rng.shuffle(choices)
        selected_ids.update(str(row["id"]) for row in choices[:count])
    return [row for row in plan if str(row["id"]) in selected_ids]


def load_chunk_lookup(path: Path = CHUNKS_PATH) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {str(row["id"]): row["content"] for row in csv.DictReader(stream)}


def _source_ranges(chunk_ids: list[str], chunk_lookup: dict[str, str]) -> tuple[str, list[tuple[int, int, str]]]:
    pieces: list[str] = []
    ranges: list[tuple[int, int, str]] = []
    cursor = 0
    for chunk_id in chunk_ids:
        text = chunk_lookup.get(str(chunk_id), "")
        if not text.strip():
            continue
        pieces.append(text)
        ranges.append((cursor, cursor + len(text), str(chunk_id)))
        cursor += len(text)
    return "".join(pieces), ranges


def _chunk_ids_for_span(start: int, end: int, ranges: list[tuple[int, int, str]]) -> list[str]:
    return [chunk_id for left, right, chunk_id in ranges if left < end and right > start]


def duplicate_title_note(
    item: dict[str, Any], titles: dict[str, str], by_title: dict[str, list[str]]
) -> str:
    """Render the teacher-prompt note when a gold/partner page title is duplicated."""
    matches = []
    sources = [("passage", item)]
    if item.get("partner"):
        sources.append(("partner passage", item["partner"]))
    used = {str(item.get("url", ""))}
    if item.get("partner"):
        used.add(str(item["partner"].get("url", "")))
    for label, source in sources:
        url = str(source.get("url", ""))
        title = titles.get(url)
        if title and any(other != url and other not in used for other in by_title.get(title, [])):
            matches.append(label)
    return "\n".join(
        "NOTE: Another record in the knowledge base has the same title as the "
        f"{label}. Identify this record in the question by a detail from the "
        f"{label} that the other record does not share, such as the file name "
        "or the period covered."
        for label in matches
    )


def _user_message(item: dict[str, Any]) -> str:
    partner_block = ""
    similar_block = ""
    if item["type"] == "comparison":
        partner = item["partner"]
        partner_block = (
            f"\nPARTNER PASSAGE (from {partner['url']}):\n<<<\n{partner['text']}\n>>>\n"
        )
    elif item["type"] == "false_premise":
        similar = item["similar"]
        similar_block = (
            f"\nSIMILAR PASSAGE (from {similar['url']}). Use it only as a source of a plausible but wrong detail:\n"
            f"<<<\n{similar['text']}\n>>>\n"
        )
    return (
        f"QUESTION TYPE: {item['type']}\n\n"
        f"PASSAGE (from {item['url']}):\n<<<\n{item['text']}\n>>>\n"
        f"{partner_block}{similar_block}{item.get('duplicate_title_note', '')}\n"
        "Write one question of this type. Follow all rules. Return JSON only."
    )


def _map_supporting_quotes(
    generated: dict[str, Any], item: dict[str, Any], chunk_lookup: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[str]]:
    sources: dict[str, dict[str, Any]] = {
        "passage": item,
    }
    if item.get("partner"):
        sources["partner"] = item["partner"]
    if item.get("similar"):
        sources["similar"] = item["similar"]
    quote_records = []
    ids_by_source: dict[str, list[str]] = {source: [] for source in sources}
    failures = []
    quotes = generated.get("supporting_quotes", [])
    if not isinstance(quotes, list):
        return [], ids_by_source, ["supporting_quotes is not a list"]
    if not quotes:
        return [], ids_by_source, ["supporting_quotes is empty"]

    source_info = {}
    for source, record in sources.items():
        joined, ranges = _source_ranges(record.get("chunk_ids", []), chunk_lookup)
        source_info[source] = (str(record.get("text", "")), ranges, joined)

    for quote_index, quote in enumerate(quotes, start=1):
        if not isinstance(quote, dict):
            failures.append(f"Quote {quote_index} is not an object")
            quote_records.append({"matched": False, "raw": quote})
            continue
        source = str(quote.get("source", "passage")).strip().lower()
        if source in {"gold", "main"}:
            source = "passage"
        text = str(quote.get("text", ""))
        if source not in source_info:
            failures.append(f"Quote {quote_index} refers to unavailable source {source!r}")
            quote_records.append({"source": source, "text": text, "matched": False, "chunk_ids": []})
            continue
        source_text, ranges, chunk_text = source_info[source]
        start = source_text.find(text) if text else -1
        if start < 0:
            # Verify that the passage is reconstructible from its chunk IDs, so
            # source formatting mismatches are visible rather than guessed.
            start = chunk_text.find(text) if text else -1
        if start < 0:
            failures.append(f"Quote {quote_index} did not exactly match source {source}")
            quote_records.append({"source": source, "text": text, "matched": False, "chunk_ids": []})
            continue
        end = start + len(text)
        chunk_ids = _chunk_ids_for_span(start, end, ranges)
        if not chunk_ids:
            failures.append(f"Quote {quote_index} matched source text but no chunk IDs covered its span")
        for chunk_id in chunk_ids:
            if chunk_id not in ids_by_source[source]:
                ids_by_source[source].append(chunk_id)
        quote_records.append({
            "source": source,
            "text": text,
            "matched": bool(chunk_ids),
            "char_start": start,
            "char_end": end,
            "chunk_ids": chunk_ids,
        })
    return quote_records, ids_by_source, failures


def generate_one(
    item: dict[str, Any],
    chunk_lookup: dict[str, str],
    *,
    system_prompt: str,
    llm_call=call_llm,
    duplicate_note: str | None = None,
) -> dict[str, Any]:
    if duplicate_note is not None:
        item = {**item, "duplicate_title_note": duplicate_note}
    prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _user_message(item)},
    ]
    base = {
        "id": item["id"],
        "window_id": item["window_id"],
        "type": item["type"],
        "url": item["url"],
        "page_type": item["page_type"],
        "text": item["text"],
        "chunk_ids": item["chunk_ids"],
        "partner": item.get("partner"),
        "similar": item.get("similar"),
    }
    attempts: list[dict[str, Any]] = []
    messages = base_messages
    last_response: dict[str, Any] = {}
    for attempt_number in (1, 2):
        try:
            response = llm_call(
                "teacher",
                messages,
                json_mode=True,
                temperature=0.0,
                prompt_sha256=prompt_sha256,
                log_messages=True,
            )
        except Exception as exc:
            attempts.append({"attempt": attempt_number, "error": f"{type(exc).__name__}: {exc}"})
            return {
                **base,
                "status": "generation_failed",
                "generation_error": attempts[-1]["error"],
                "generation_attempts": attempts,
            }

        last_response = response
        result = response.get("json")
        issues = _teacher_response_issues(result, item, chunk_lookup)
        attempts.append({
            "attempt": attempt_number,
            "cache_hit": response.get("cache_hit"),
            "raw_json": result,
            "raw_text": response.get("text", ""),
            "validation_errors": issues,
        })
        if not issues:
            if result.get("skip") is True:
                return {
                    **base,
                    "teacher_model": response.get("model"),
                    "status": "skip",
                    "skip_reason": str(result["reason"]),
                    "teacher_raw_json": result,
                    "generation_attempts": attempts,
                    "quote_failed": False,
                }
            quote_records, ids_by_source, failures = _map_supporting_quotes(result, item, chunk_lookup)
            return {
                **base,
                "teacher_model": response.get("model"),
                "status": "generated",
                "question": str(result["question"]),
                "reference_answer": str(result["reference_answer"]),
                "keypoints": result["keypoints"],
                "supporting_quotes": quote_records,
                "false_premise": result.get("false_premise"),
                "gold_chunk_ids": ids_by_source.get("passage", []),
                "partner_chunk_ids": ids_by_source.get("partner", []),
                "similar_chunk_ids": ids_by_source.get("similar", []),
                "quote_failed": bool(failures),
                "quote_failure_reasons": failures,
                "teacher_raw_json": result,
                "generation_attempts": attempts,
                "teacher_cache_hit": response.get("cache_hit"),
            }
        if attempt_number == 1:
            messages = [
                *base_messages,
                {"role": "assistant", "content": str(response.get("text", ""))},
                {
                    "role": "user",
                    "content": (
                        "Your previous JSON was invalid for these reasons: "
                        + "; ".join(issues)
                        + ". Regenerate the complete JSON object. Include every required field, "
                          "including reference_answer, keypoints with a boolean decisive value, "
                          "and non-empty supporting_quotes with exact source text. Do not use an "
                          "alternate answer field or omit fields. Return JSON only."
                    ),
                },
            ]
    return {
        **base,
        "teacher_model": last_response.get("model"),
        "status": "generation_failed",
        "generation_error": "; ".join(attempts[-1].get("validation_errors", [])),
        "generation_attempts": attempts,
        "teacher_raw_json": attempts[-1].get("raw_json"),
        "quote_failed": True,
    }


def _teacher_response_issues(result: Any, item: dict[str, Any], chunk_lookup: dict[str, str]) -> list[str]:
    if not isinstance(result, dict):
        return ["response must be a JSON object"]
    if type(result.get("skip")) is not bool:
        skip_issues = ["skip must be a boolean"]
    else:
        skip_issues = []
    if result.get("skip") is True:
        return skip_issues + ([] if str(result.get("reason", "")).strip() else ["skip response is missing reason"])
    issues = skip_issues
    if result.get("type") != item["type"]:
        issues.append(f"type must equal planned type {item['type']!r}")
    for field in ("question", "reference_answer"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            issues.append(f"missing or empty {field}")
    keypoints = result.get("keypoints")
    if not isinstance(keypoints, list) or not keypoints:
        issues.append("keypoints must be a non-empty list")
    else:
        for i, point in enumerate(keypoints, start=1):
            if not isinstance(point, dict) or not isinstance(point.get("text"), str) or not point["text"].strip():
                issues.append(f"keypoint {i} needs non-empty text")
            if not isinstance(point, dict) or type(point.get("decisive")) is not bool:
                issues.append(f"keypoint {i} needs a boolean decisive field")
    quotes = result.get("supporting_quotes")
    if not isinstance(quotes, list) or not quotes:
        issues.append("supporting_quotes must be a non-empty list")
    else:
        for i, quote in enumerate(quotes, start=1):
            if not isinstance(quote, dict) or quote.get("source") not in {"passage", "partner"} or not str(quote.get("text", "")).strip():
                issues.append(f"supporting quote {i} needs source passage/partner and non-empty text")
    if item["type"] == "false_premise" and not str(result.get("false_premise", "")).strip():
        issues.append("false_premise question is missing its false_premise explanation")
    if not issues:
        _, _, quote_failures = _map_supporting_quotes(result, item, chunk_lookup)
        issues.extend(quote_failures)
    return issues


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=OUTPUT_PLAN)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    parser.add_argument("--out", type=Path, default=PILOT_OUTPUT)
    parser.add_argument("--pilot", action="store_true", help="generate only 3/2/3/2 questions by type")
    parser.add_argument("--exclude-questions", type=Path, help="exclude IDs found in an existing questions JSONL file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    plan = _read_jsonl(args.plan)
    if args.pilot and args.exclude_questions:
        parser.error("--pilot and --exclude-questions cannot be used together")
    if args.pilot:
        selected = select_pilot(plan, args.seed)
    elif args.exclude_questions:
        excluded_rows = _read_jsonl(args.exclude_questions)
        excluded_ids = {str(row["id"]) for row in excluded_rows}
        missing_ids = sorted(excluded_ids - {str(row["id"]) for row in plan})
        if missing_ids:
            raise ValueError("Excluded question IDs are not in the plan: " + ", ".join(missing_ids))
        selected = [row for row in plan if str(row["id"]) not in excluded_ids]
        if len(selected) != 40:
            raise ValueError(f"Excluding {len(excluded_ids)} IDs should leave 40 questions, found {len(selected)}")
        print(f"Excluded {len(excluded_ids)} existing questions; remaining type counts: {dict(Counter(row['type'] for row in selected))}")
    else:
        selected = plan
    if not args.pilot and not args.exclude_questions and Counter(row["type"] for row in selected) != Counter(TYPE_COUNTS):
        raise ValueError("Full plan does not match the configured 12/11/14/13 allocation")
    chunk_lookup = load_chunk_lookup(args.chunks)
    system_prompt = load_teacher_prompt(args.prompt)
    questions = []
    for index, item in enumerate(selected, start=1):
        print(f"Teacher question {index}/{len(selected)}: {item['id']} ({item['type']})", flush=True)
        questions.append(generate_one(item, chunk_lookup, system_prompt=system_prompt))
    write_jsonl(args.out, questions)
    print(f"Generated status counts: {dict(Counter(item['status'] for item in questions))}")
    print(f"Quote failures: {sum(bool(item.get('quote_failed')) for item in questions)}")
    print(f"Wrote {args.out}")

    valid_question = next((row for row in questions if row.get("status") == "generated"), None)
    raw_json = valid_question.get("teacher_raw_json") if valid_question else None
    if raw_json is None:
        failed = next((row for row in questions if row.get("teacher_raw_json") is not None), None)
        raw_json = failed.get("teacher_raw_json") if failed else None
    print("\nFull raw teacher JSON for inspection:")
    print(json.dumps(raw_json, ensure_ascii=False, indent=2) if raw_json is not None else "No teacher JSON was returned.")
    complete = bool(
        isinstance(raw_json, dict)
        and isinstance(raw_json.get("reference_answer"), str)
        and isinstance(raw_json.get("keypoints"), list)
        and all(isinstance(point, dict) and type(point.get("decisive")) is bool for point in raw_json["keypoints"])
        and isinstance(raw_json.get("supporting_quotes"), list)
    )
    print(f"Required fields present (reference_answer, keypoints.decisive, supporting_quotes): {complete}")

    log_path = PROJECT_ROOT / "outputs" / "logs" / "teacher.jsonl"
    matching_events = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("prompt_sha256") == hashlib.sha256(system_prompt.encode("utf-8")).hexdigest() and event.get("api_request_sent") and event.get("messages"):
                matching_events.append(event)
    if matching_events:
        audit = matching_events[0]
        sent_messages = audit["messages"]
        sent_system = next((m.get("content", "") for m in sent_messages if m.get("role") == "system"), "")
        print(f"\nFull messages from outputs/logs/teacher.jsonl (request hash {audit.get('request_hash')}):")
        print(json.dumps(sent_messages, ensure_ascii=False, indent=2))
        print(f"Logged system message matches complete prompt file: {sent_system == system_prompt}")
        print(f"Example headings in logged prompt: {sum(f'### Example {i}:' in sent_system for i in range(1, 5))}/4")


if __name__ == "__main__":
    main()
