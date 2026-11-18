"""Generate S2 pilot questions using the dedicated teacher prompt."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from rq4.llm import call_llm
from scripts.build_kb_index import read_jsonl, write_jsonl
from scripts.build_bundles_s2 import TYPE_ORDER

PROMPT = ROOT / "prompts" / "teacher_generate_s2.md"
DEFAULT_BUNDLES = ROOT / "data" / "processed" / "s2" / "bundles_s2.jsonl"
PILOT_DIR = ROOT / "outputs" / "s2" / "pilot2"
EXPECTED_BEHAVIOUR = {
    "set_all": "list_all", "multi_hop": "answer", "comparison_multi": "answer",
    "ambiguity": {"same_name": "disambiguate", "confirmed_conflict": "flag_conflict"},
    "unanswerable": "state_absence",
}
TYPE_ORDER = ("set_all", "multi_hop", "comparison_multi", "ambiguity", "unanswerable")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def is_absence_keypoint(text: Any) -> bool:
    value = str(text).strip()
    return bool(re.search(r"^no\b.{0,180}\b(?:record|dataset|entry|data)\b", value, re.I)
                or re.search(r"\b(?:there\s+(?:is|are)\s+no|none\s+(?:is|are)\s+available|not\s+available|does\s+not\s+exist|not\s+in\s+the\s+knowledge\s+base)\b", value, re.I))


def select_pilot(bundles: list[dict[str, Any]], seed: int = 42, count_per_type: int = 2) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    result = []
    catalogue_use: dict[tuple[str, str], int] = {}
    for kind in TYPE_ORDER:
        candidates = [row for row in bundles if row["type"] == kind]
        if not candidates:
            continue
        rng.shuffle(candidates)
        picked = []
        for row in candidates:
            cat = (row.get("catalogue_id") or row.get("pages", [{}])[0].get("url", "") if row.get("pages") else "no_catalogue", kind)
            if catalogue_use.get(cat, 0) >= 2:
                continue
            if kind == "ambiguity" and picked and row.get("ambiguity_kind") == picked[0].get("ambiguity_kind"):
                if any(x.get("ambiguity_kind") != picked[0].get("ambiguity_kind") for x in candidates):
                    continue
            picked.append(row)
            catalogue_use[cat] = catalogue_use.get(cat, 0) + 1
            if len(picked) == count_per_type:
                break
        result.extend(picked)
    return result


def user_message(bundle: dict[str, Any], validation_feedback: str = "") -> str:
    pages = []
    for page in bundle["pages"]:
        pages.append(f"[{page['evidence_id']}] {page['title']} ({page['url']})\n<<<\n{page['text']}\n>>>")
    tail = "\n".join(part for part in (bundle.get("group_note", ""), bundle.get("s1_note", "")) if part)
    set_all_scope = ""
    return (
        f"QUESTION TYPE: {bundle['type']}\nUSER TASK: {bundle['user_task']}\n\n"
        "EVIDENCE PAGES:\n" + "\n\n".join(pages) + ("\n\n" + tail if tail else "") + set_all_scope
        + ("\n\n" + validation_feedback.strip() if validation_feedback.strip() else "")
        + "\n\nWrite one question of this type. Follow all rules. Return JSON only."
    )


def response_issues(result: Any, bundle: dict[str, Any]) -> list[str]:
    if not isinstance(result, dict):
        return ["Output must be a JSON object"]
    if not isinstance(result.get("skip"), bool):
        return ["skip must be a boolean"]
    if result.get("skip") is True:
        return [] if str(result.get("reason", "")).strip() else ["skip=true requires a reason"]
    required_strings = ("type", "user_task", "question_user", "question_source", "expected_behaviour", "reference_answer", "construction_note")
    issues = [f"{key} must be a non-empty string" for key in required_strings if not isinstance(result.get(key), str) or not result[key].strip()]
    if result.get("type") != bundle["type"]:
        issues.append("type does not match the evidence bundle")
    if result.get("user_task") != bundle["user_task"]:
        issues.append("user_task does not match the evidence bundle")
    expected = EXPECTED_BEHAVIOUR[bundle["type"]]
    if isinstance(expected, dict):
        expected = expected.get(bundle.get("ambiguity_kind"), "disambiguate")
    if result.get("expected_behaviour") != expected:
        issues.append("expected_behaviour does not match the fixed behaviour for this type")
    page_ids = {page["evidence_id"] for page in bundle["pages"]}
    page_map = {page["evidence_id"]: page for page in bundle["pages"]}
    used = result.get("pages_used")
    if not isinstance(used, list) or len(set(used)) < 2 or any(source not in page_ids for source in used):
        issues.append("pages_used must identify at least two supplied evidence pages")
    keypoints = result.get("keypoints")
    if not isinstance(keypoints, list) or not keypoints:
        issues.append("keypoints must be a non-empty array")
    else:
        for i, point in enumerate(keypoints, 1):
            if not isinstance(point, dict):
                issues.append(f"keypoint {i} must be an object")
                continue
            if not isinstance(point.get("text"), str) or not point["text"].strip():
                issues.append(f"keypoint {i} needs text")
            if not isinstance(point.get("decisive"), bool):
                issues.append(f"keypoint {i} needs a boolean decisive value")
            sources = point.get("sources") if isinstance(point.get("sources"), list) else [point.get("source")]
            quotes = point.get("quotes") if isinstance(point.get("quotes"), list) else [point.get("quote", "")]
            if point.get("derived") is True:
                if any(str(q or "").strip() for q in quotes):
                    issues.append(f"keypoint {i} is derived and must not include quotes")
                continue
            absence = result.get("expected_behaviour") == "state_absence" and is_absence_keypoint(point.get("text", ""))
            if absence:
                continue
            if not sources or len(sources) != len(quotes):
                issues.append(f"keypoint {i} sources and quotes must be non-empty lists of equal length")
                continue
            for source, quote in zip(sources, quotes):
                if source not in page_ids:
                    issues.append(f"keypoint {i} source must be one of {sorted(page_ids)}")
                elif not isinstance(quote, str) or not quote or quote not in page_map[source]["text"]:
                    issues.append(f"keypoint {i} quote is not an exact substring of its source page")
    return issues


def _mock_quote(page: dict[str, Any]) -> str:
    for line in page["text"].splitlines():
        text = line.strip()
        if len(text) >= 24 and not text.startswith(("#", "!", "http")):
            return text[:160]
    return page["text"][:120]


def _mock_field_quote(page: dict[str, Any], field: str, value: str) -> str:
    lines = page["text"].splitlines()
    for i, line in enumerate(lines):
        if field.casefold() in line.casefold() and value.casefold() in (line + (" " + lines[i + 1] if i + 1 < len(lines) else "")).casefold():
            candidate = line if value.casefold() in line.casefold() else line + "\n" + lines[i + 1]
            if candidate and candidate in page["text"]: return candidate
    return value if value and value in page["text"] else _mock_quote(page)


def mock_response(bundle: dict[str, Any]) -> dict[str, Any]:
    kind = bundle["type"]
    pages = bundle["pages"]
    all_used = [page["evidence_id"] for page in pages]
    if kind == "set_all":
        expected = bundle.get("expected_items") or [page["title"] for page in pages]
        answer = "Matching records: " + "; ".join(expected) + "."
        points = [{"text": f"{page['title']} meets the selected record criterion", "decisive": True,
                   "source": page["evidence_id"], "quote": _mock_quote(page)} for page in pages]
        user_q = f"Which records in this series match the stated {bundle.get('target_field') or 'data'} requirement?"
        source_q = f"Which series records have {bundle.get('target_field') or 'the required field'}?"
    elif kind == "multi_hop":
        answer = f"The referenced entity is {bundle.get('entity', pages[1]['title'])}; the second page describes its related record or service."
        points = [{"text": f"The first record refers to {bundle.get('entity', pages[1]['title'])}", "decisive": True,
                   "source": pages[0]["evidence_id"], "quote": _mock_quote(pages[0])},
                  {"text": f"The second page provides details about {bundle.get('entity', pages[1]['title'])}", "decisive": True,
                   "source": pages[1]["evidence_id"], "quote": _mock_quote(pages[1])}]
        user_q = f"Which organization or service is linked to the {pages[0]['title']} record, and what does its dedicated page say?"
        source_q = f"What does {pages[0]['title']} state about {bundle.get('entity', pages[1]['title'])}, and what does its page describe?"
    elif kind == "comparison_multi":
        vals = bundle.get("record_values", {})
        answer = "; ".join(f"{title}: {value}" for title, value in vals.items())
        points = [{"text": f"{title} has the {bundle.get('target_field')} value {value}", "decisive": True,
                   "source": pages[i]["evidence_id"], "quote": _mock_quote(pages[i])}
                  for i, (title, value) in enumerate(vals.items())]
        user_q = f"How do these three records compare in {bundle.get('target_field')}?"
        source_q = f"What {bundle.get('target_field')} value does each of the three records state?"
    elif kind == "ambiguity" and bundle.get("ambiguity_kind") == "confirmed_conflict":
        c = bundle["conflict_candidate"]
        answer = f"The pages give different {c['attribute']} values: {c['value_a']} at the first source and {c['value_b']} at the second source."
        points = [{"text": f"The first page gives {c['value_a']} for {c['attribute']}", "decisive": True, "source": pages[0]["evidence_id"], "quote": c["quote_a"]},
                  {"text": f"The second page gives {c['value_b']} for {c['attribute']}", "decisive": True, "source": pages[1]["evidence_id"], "quote": c["quote_b"]},
                  {"text": "The knowledge base gives different values", "decisive": True, "source": pages[0]["evidence_id"], "quote": c["quote_a"]}]
        user_q = f"What {c['attribute']} is recorded for {pages[0]['title']}?"
        source_q = f"What value is given for {c['attribute']} for {pages[0]['title']}?"
    elif kind == "ambiguity":
        values = bundle.get("record_values", [])
        answer = "There are multiple records with this name: " + "; ".join(f"{item['title']} has {item['value']}" for item in values) + "."
        points = [{"text": "Multiple distinct records share the requested name", "decisive": True,
                   "source": pages[0]["evidence_id"], "quote": _mock_quote(pages[0])}]
        points.extend({"text": f"{item['title']}: {item['value']}", "decisive": True, "source": pages[i]["evidence_id"],
                       "quote": _mock_field_quote(pages[i], bundle.get("target_field", ""), item["value"])}
                      for i,item in enumerate(values))
        user_q = f"What is recorded for {bundle.get('shared_name')}?"
        source_q = f"What {bundle.get('target_field')} value is given for {bundle.get('shared_name')}?"
    else:
        answer = f"The knowledge base has no record matching that combination. The closest records are {pages[0]['title']} and {pages[1]['title']}."
        points = [{"text": "The requested record combination is not in the knowledge base", "decisive": True, "source": pages[0]["evidence_id"], "quote": _mock_quote(pages[0])},
                  {"text": f"The closest records are {pages[0]['title']} and {pages[1]['title']}", "decisive": False, "source": pages[1]["evidence_id"], "quote": _mock_quote(pages[1])}]
        user_q = f"Is there a record combining a measurement or feature from {pages[0]['title']} with a different series member?"
        source_q = f"Does this series contain a record with the requested combination of attributes?"
    expected_behaviour = EXPECTED_BEHAVIOUR[kind]
    if isinstance(expected_behaviour, dict):
        expected_behaviour = expected_behaviour.get(bundle.get("ambiguity_kind"), "disambiguate")
    # Keep conflict mock quotations inside each supplied evidence window.
    if kind == "ambiguity" and bundle.get("ambiguity_kind") == "confirmed_conflict":
        c = bundle["conflict_candidate"]
        answer = f"The two sources give different {c.get('attribute', 'attribute')} values: {c.get('value_a')} and {c.get('value_b')}."
        points = [{"text": f"The first source gives {c.get('value_a')}", "decisive": True, "source": pages[0]["evidence_id"], "quote": _mock_quote(pages[0])},
                  {"text": f"The second source gives {c.get('value_b')}", "decisive": True, "source": pages[1]["evidence_id"], "quote": _mock_quote(pages[1])},
                  {"text": "The two values differ", "decisive": True, "source": pages[0]["evidence_id"], "quote": _mock_quote(pages[0])}]
        user_q = f"What {c.get('attribute', 'value')} is reported for {pages[0]['title']}?"
        source_q = f"What value is stated for {c.get('attribute', 'value')} on these two sources?"
    return {
        "skip": False, "type": kind, "user_task": bundle["user_task"],
        "question_user": user_q, "question_source": source_q,
        "expected_behaviour": expected_behaviour, "reference_answer": answer,
        "keypoints": points, "pages_used": all_used,
        "construction_note": "The question combines evidence from multiple indexed knowledge-base pages.",
    }


def generate_one(bundle: dict[str, Any], system_prompt: str, *, llm_call: Callable[..., dict[str, Any]] | None = None,
                 mock: bool = False, output_root: Path = PILOT_DIR, validation_feedback: str = "") -> dict[str, Any]:
    digest = prompt_hash(system_prompt)
    attempts = []
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message(bundle, validation_feedback)}]
    for attempt in range(1, 3):
        try:
            response = ({"json": mock_response(bundle), "text": json.dumps(mock_response(bundle), ensure_ascii=False), "model": "mock-teacher", "cache_hit": False}
                        if mock else (llm_call or call_llm)("teacher", messages, json_mode=True, temperature=0.0,
                            prompt_sha256=digest, log_messages=True,
                            cache_dir=output_root / "cache" / "llm", log_dir=output_root / "logs"))
            parsed = response.get("json")
            issues = response_issues(parsed, bundle)
            attempts.append({"attempt": attempt, "cache_hit": response.get("cache_hit"), "validation_errors": issues, "raw_json": parsed})
            if not issues:
                base = {key: value for key, value in bundle.items() if key not in {"pages"}}
                generated = {
                    **base, "id": bundle["bundle_id"], "status": "skip" if parsed.get("skip") else "generated",
                    "teacher_model": response.get("model"), "teacher_prompt_sha256": digest,
                    "teacher_raw_json": parsed, "generation_attempts": attempts,
                }
                if parsed.get("skip"):
                    generated["skip_reason"] = parsed.get("reason", "")
                else:
                    generated.update({key: parsed[key] for key in ("type", "user_task", "question_user", "question_source", "expected_behaviour", "reference_answer", "keypoints", "pages_used", "construction_note")})
                generated["pages"] = bundle["pages"]
                return generated
            if attempt == 1:
                messages = [*messages, {"role": "assistant", "content": response.get("text", "")},
                            {"role": "user", "content": "Your previous JSON is incomplete or invalid: " + "; ".join(issues) + ". Regenerate the complete JSON object following the exact output format. Do not omit required fields or invent evidence. Return JSON only."}]
        except Exception as exc:
            attempts.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
            if attempt == 1:
                messages = [*messages, {"role": "user", "content": f"The previous response failed to produce valid JSON ({type(exc).__name__}). Retry once and return the complete required JSON object only."}]
    return {"id": bundle["bundle_id"], "type": bundle["type"], "status": "generation_failed",
            "teacher_prompt_sha256": digest, "generation_attempts": attempts, "pages": bundle["pages"]}


def generate_pilot(input_path: Path = DEFAULT_BUNDLES, output_path: Path = PILOT_DIR / "questions_s2_pilot.jsonl",
                   *, mock: bool = False, seed: int = 42, selected_types: list[str] | None = None,
                   merge_existing: bool = False, count_per_type: int = 2,
                   fill_missing_slots: bool = True) -> list[dict[str, Any]]:
    if not mock:
        load_dotenv(ROOT / ".env", override=True)
    prompt = PROMPT.read_text(encoding="utf-8").strip()
    all_bundles = read_jsonl(input_path)
    if selected_types is not None:
        selected_set = set(selected_types)
        unknown = selected_set - set(TYPE_ORDER)
        if unknown:
            raise ValueError(f"Unknown S2 types requested: {sorted(unknown)}")
        selected_kind_order = [kind for kind in TYPE_ORDER if kind in selected_set]
    else:
        selected_kind_order = list(TYPE_ORDER)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    new_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for kind in selected_kind_order:
        candidates = [row for row in all_bundles if row.get("type") == kind]
        rng.shuffle(candidates)
        successes: list[dict[str, Any]] = []
        for bundle in candidates:
            if len(successes) >= count_per_type:
                break
            generated = generate_one(bundle, prompt, mock=mock, output_root=output_path.parent)
            if generated.get("status") == "generated":
                successes.append(generated)
            else:
                skipped_rows.append({**generated, "candidate_source": bundle.get("candidate_source", "catalogue_field"),
                                     "skip_or_failure_reason": generated.get("skip_reason") or generated.get("generation_attempts")})
        new_rows.extend(successes)
        for missing_number in range(len(successes), count_per_type if fill_missing_slots else len(successes)):
            new_rows.append({"id": f"S2-{kind.upper()}-MISSING-{missing_number+1:03d}", "type": kind,
                "status": "skip", "skip_reason": "No further eligible package produced a usable question after trying all seeded candidates.",
                "generation_attempts": [], "teacher_prompt_sha256": prompt_hash(prompt)})
    skip_path = output_path.with_name("generation_skips_s2.jsonl")
    write_jsonl(skip_path, skipped_rows)
    if merge_existing and output_path.exists():
        selected_set = set(selected_kind_order)
        retained = [row for row in read_jsonl(output_path) if row.get("type") not in selected_set]
        by_type = {row["type"]: row for row in [*retained, *new_rows]}
        rows = [by_type[kind] for kind in TYPE_ORDER if kind in by_type]
    else:
        rows = new_rows
    write_jsonl(output_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_BUNDLES)
    parser.add_argument("--output", type=Path, default=PILOT_DIR / "questions_s2_pilot.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count-per-type", type=int, default=2)
    parser.add_argument("--types", nargs="+", choices=TYPE_ORDER, help="generate selected type(s) only")
    parser.add_argument("--merge-existing", action="store_true", help="replace selected types while retaining the other pilot items")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-fill-missing-slots", action="store_true", help="do not invent placeholder rows when fewer bundles exist")
    args = parser.parse_args()
    rows = generate_pilot(args.input, args.output, mock=args.mock, seed=args.seed, selected_types=args.types,
                          merge_existing=args.merge_existing, count_per_type=args.count_per_type,
                          fill_missing_slots=not args.no_fill_missing_slots)
    skip_path = args.output.with_name("generation_skips_s2.jsonl")
    print(json.dumps({"generated": sum(r.get("status") == "generated" for r in rows), "missing_slots": sum(r.get("status") == "skip" for r in rows), "failed_slots": sum(r.get("status") == "generation_failed" for r in rows), "skipped_candidate_attempts": len(read_jsonl(skip_path)), "skip_log": str(skip_path), "output": str(args.output), "prompt_sha256": prompt_hash(PROMPT.read_text(encoding="utf-8").strip())}, indent=2))


if __name__ == "__main__":
    main()
