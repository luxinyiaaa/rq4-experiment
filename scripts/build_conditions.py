"""Select distractors and build the four evidence conditions for reviewed questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rq4.llm import call_llm
from scripts.build_index import DEFAULT_OUTDIR, RetrievalIndex, _domain, _same_dataset, _text_similarity, _url_template, _read_jsonl, _write_jsonl, load_index
from scripts.check_questions import GENERIC_PAGE_TITLES, page_title_catalog

QUESTIONS_PATH = PROJECT_ROOT / "outputs" / "pilot" / "questions_reviewed.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "pilot" / "conditions_pilot.jsonl"
DISTRACTOR_PROMPT = PROJECT_ROOT / "prompts" / "judge_distractor_check.md"


def _system_prompt(path: Path = DISTRACTOR_PROMPT) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index("## System") + len("## System")
    end = text.index("## User template", start)
    return text[start:end].strip()


def _title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            value = line[2:].strip()
            if value.casefold() not in GENERIC_PAGE_TITLES:
                return value
    return ""


def _conflict_urls(question: dict[str, Any]) -> set[str]:
    urls = {
        str(row.get("url"))
        for row in question.get("global_conflict", {}).get("conflicts", [])
        if row.get("url")
    }
    edited = question.get("edited_checks", {}).get("global_conflict", {})
    urls.update(str(row.get("url")) for row in edited.get("conflicts", []) if row.get("url"))
    return urls


def _candidate_rankings(
    question: dict[str, Any], index: RetrievalIndex, titles: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    gold = question
    gold_url = str(gold["url"])
    partner = gold.get("partner") or {}
    partner_url = str(partner.get("url", ""))
    blocked_urls = {url for url in (gold_url, partner_url) if url}
    blocked_urls.update(_conflict_urls(question))
    target_titles = {title for title in (titles.get(gold_url), titles.get(partner_url), _title(gold.get("text", "")), _title(partner.get("text", ""))) if title}
    target_titles.update(
        str(row.get("title")) for row in question.get("duplicate_title", {}).get("matches", []) if row.get("title")
    )

    query_vector = index._query_embedding(str(gold["text"]))
    scores = np.asarray(index.window_embeddings, dtype=np.float32) @ query_vector
    gold_template = _url_template(gold_url)
    gold_domain = _domain(gold_url)
    kept: list[tuple[int, float, str, int, dict[str, Any]]] = []
    exclusions: Counter[str] = Counter()
    for offset, candidate in enumerate(index.window_records):
        url = str(candidate["url"])
        reason = None
        if candidate.get("page_type") in {"api_docs", "search_page"}:
            reason = "excluded_page_type"
        elif url in blocked_urls:
            reason = "gold_partner_or_conflict_page"
        elif _same_dataset(gold, candidate):
            reason = "same_dataset"
        elif _text_similarity(str(gold.get("text", "")), str(candidate.get("text", ""))) > 0.7:
            reason = "text_similarity_over_0.7"
        elif (titles.get(url) or _title(str(candidate.get("text", "")))) in target_titles and (titles.get(url) or _title(str(candidate.get("text", "")))):
            reason = "same_title_as_gold_or_partner"
        if reason:
            exclusions[reason] += 1
            continue
        template = str(candidate.get("url_template") or _url_template(url))
        domain = str(candidate.get("domain") or _domain(url))
        priority = 0 if template == gold_template else 1 if domain == gold_domain else 2
        value = {
            **candidate,
            "similarity": float(scores[offset]),
            "candidate_rank": 0,
        }
        kept.append((priority, -float(scores[offset]), url, offset, value))
    kept.sort(key=lambda row: (row[0], row[1], row[2]))
    ranked = []
    for rank, (_, _, _, _, row) in enumerate(kept, start=1):
        ranked.append({**row, "candidate_rank": rank})
    return ranked, dict(exclusions)


def _judge_message(question: dict[str, Any], batch: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    keypoints = "\n".join(
        f"{'[D]' if point.get('decisive') else '[ ]'} {point.get('text', '')}"
        for point in question.get("keypoints", [])
    )
    blocks = []
    mapping = []
    for position, candidate in enumerate(batch, start=1):
        passage_id = f"D{position}"
        blocks.append(f"[{passage_id}] (from {candidate['url']})\n{candidate['text']}")
        mapping.append({"passage_id": passage_id, "window_id": str(candidate["window_id"]), "url": str(candidate["url"])})
    message = (
        f"Question: {question['question']}\n\nReference answer: {question['reference_answer']}\n\n"
        f"Keypoints (D = decisive):\n{keypoints}\n\nCandidate passages:\n\n" + "\n\n".join(blocks)
    )
    return message, mapping


def _judge_batch(
    question: dict[str, Any], batch: list[dict[str, Any]], system_prompt: str,
    llm_call: Callable[..., dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    user_message, mapping = _judge_message(question, batch)
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    response = llm_call(
        "judge",
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
        json_mode=True, temperature=0.0, extra_body={"enable_thinking": False},
        prompt_sha256=prompt_hash, log_messages=True,
    )
    result = response.get("json")
    labels = result.get("labels") if isinstance(result, dict) else None
    if not isinstance(labels, list):
        raise ValueError(f"Judge response for {question['id']} must contain a labels list")
    by_id = {str(row.get("passage_id")): row for row in labels if isinstance(row, dict)}
    reviewed = []
    for candidate, map_row in zip(batch, mapping):
        label = by_id.get(map_row["passage_id"])
        if not label or label.get("label") not in {"safe", "answers"}:
            raise ValueError(f"Judge omitted or returned an invalid label for {map_row['passage_id']} ({question['id']})")
        reviewed.append({
            **candidate,
            "judge_label": label["label"],
            "judge_reason": str(label.get("reason", "")),
            "judge_passage_id": map_row["passage_id"],
        })
    metadata = {
        "model_id": response.get("api_model", response.get("model")),
        "usage": response.get("usage"),
        "request_parameters": response.get("request_parameters"),
        "cache_hit": response.get("cache_hit"),
        "candidate_map": mapping,
    }
    return reviewed, metadata


def choose_distractors(
    question: dict[str, Any], index: RetrievalIndex, system_prompt: str,
    *, llm_call: Callable[..., dict[str, Any]] = call_llm,
) -> dict[str, Any]:
    titles, _ = page_title_catalog(index.chunk_records)
    ranked, exclusions = _candidate_rankings(question, index, titles)
    reviewed: list[dict[str, Any]] = []
    judge_calls = []
    for start in range(0, min(len(ranked), 20), 10):
        batch = ranked[start:start + 10]
        if not batch:
            break
        batch_reviewed, metadata = _judge_batch(question, batch, system_prompt, llm_call)
        reviewed.extend(batch_reviewed)
        judge_calls.append(metadata)
        if sum(row["judge_label"] == "safe" for row in reviewed) >= 4:
            break
    safe = [row for row in reviewed if row["judge_label"] == "safe"][:4]
    if len(safe) < 4:
        raise RuntimeError(
            f"Only {len(safe)} safe distractors found among the first {min(20, len(ranked))} candidates for {question['id']}"
        )
    return {
        "selected": safe,
        "checked_candidates": reviewed,
        "judge_removed": sum(row["judge_label"] == "answers" for row in reviewed),
        "judge_calls": judge_calls,
        "prefilter_excluded_count": sum(exclusions.values()),
        "prefilter_exclusion_reasons": exclusions,
    }


def _passage(row: dict[str, Any], passage_id: str, role: str) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "role": role,
        "url": row["url"],
        "window_id": row.get("window_id"),
        "chunk_ids": [str(value) for value in row.get("chunk_ids", [])],
        "text": row["text"],
        **({"similarity": row["similarity"]} if "similarity" in row else {}),
    }


def build_question_conditions(
    questions: list[dict[str, Any]], index: RetrievalIndex, system_prompt: str,
    *, llm_call: Callable[..., dict[str, Any]] = call_llm,
) -> list[dict[str, Any]]:
    type_positions: Counter[str] = Counter()
    output = []
    for question in questions:
        if question.get("status") != "generated":
            continue
        kind = str(question["type"])
        rotation = ("first", "middle", "last")[(type_positions[kind]) % 3]
        type_positions[kind] += 1
        chosen = choose_distractors(question, index, system_prompt, llm_call=llm_call)
        distractors = chosen["selected"]
        gold = _passage(question, "P1", "gold")
        partner = _passage(question["partner"], "P2", "partner") if kind == "comparison" else None
        c1 = [gold] + ([partner] if partner else [])

        distractor_passages = [_passage(row, f"P{position}", "distractor") for position, row in enumerate(distractors, start=1)]
        gold_group = [gold] + ([partner] if partner else [])
        start = {"first": 0, "middle": 2, "last": 4}[rotation]
        c2_list = distractor_passages[:start] + gold_group + distractor_passages[start:]
        c2 = [{**row, "passage_id": f"P{position}"} for position, row in enumerate(c2_list, start=1)]
        c3 = [{**row, "passage_id": f"P{position}"} for position, row in enumerate(distractor_passages, start=1)]

        retrieved = index.retrieve(str(question["question"]), k=10)
        c4 = [
            {"passage_id": f"P{position}", "role": "retrieved_chunk", "url": row["url"],
             "chunk_id": str(row["chunk_id"]), "similarity": float(row["similarity"]), "text": row["text"]}
            for position, row in enumerate(retrieved, start=1)
        ]
        gold_ids = {str(value) for value in question.get("gold_chunk_ids", [])}
        hit_ranks = [position for position, row in enumerate(retrieved, start=1) if str(row["chunk_id"]) in gold_ids]
        c4_gold_rank = min(hit_ranks) if hit_ranks else None
        gold_url = str(question.get("url", ""))
        partner_url = str(question.get("partner", {}).get("url", "")) if partner else ""
        gold_page_ranks = {
            "gold": next((position for position, row in enumerate(retrieved, start=1)
                          if str(row.get("url", "")) == gold_url), None),
            "partner": (next((position for position, row in enumerate(retrieved, start=1)
                              if str(row.get("url", "")) == partner_url), None)
                        if partner else None),
        }
        output.append({
            "id": question["id"], "type": kind, "question": question["question"],
            "reference_answer": question["reference_answer"], "gold_position": rotation,
            "selected_distractors": [
                {key: row[key] for key in ("window_id", "url", "similarity", "chunk_ids", "judge_reason", "candidate_rank")}
                for row in distractors
            ],
            "distractor_selection": {
                "judge_removed": chosen["judge_removed"],
                "prefilter_excluded_count": chosen["prefilter_excluded_count"],
                "prefilter_exclusion_reasons": chosen["prefilter_exclusion_reasons"],
                "checked_candidates": chosen["checked_candidates"],
                "judge_calls": chosen["judge_calls"],
            },
            "c4_gold_hit_rank": c4_gold_rank,
            "c4_gold_page_hit_rank": gold_page_ranks,
            "conditions": {
                "C1": {"passages": c1, "passage_map": [{"passage_id": row["passage_id"], "window_id": row.get("window_id"), "chunk_ids": row["chunk_ids"], "url": row["url"]} for row in c1]},
                "C2": {"passages": c2, "passage_map": [{"passage_id": row["passage_id"], "window_id": row.get("window_id"), "chunk_ids": row["chunk_ids"], "url": row["url"]} for row in c2]},
                "C3": {"passages": c3, "passage_map": [{"passage_id": row["passage_id"], "window_id": row.get("window_id"), "chunk_ids": row["chunk_ids"], "url": row["url"]} for row in c3]},
                "C4": {"passages": c4, "passage_map": [{"passage_id": row["passage_id"], "chunk_id": row["chunk_id"], "url": row["url"]} for row in c4]},
            },
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--judge-prompt", type=Path, default=DISTRACTOR_PROMPT)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    questions = _read_jsonl(args.questions)
    index = load_index(args.index_dir)
    result = build_question_conditions(questions, index, _system_prompt(args.judge_prompt))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out, result)
    print(f"Built four conditions for {len(result)} reviewed questions: {args.out}")
    for row in result:
        urls = ", ".join(f"{item['url']} ({item['similarity']:.4f})" for item in row["selected_distractors"])
        print(f"{row['id']}: gold={row['gold_position']}; distractors={urls}; judge removed={row['distractor_selection']['judge_removed']}; C4 gold rank={row['c4_gold_hit_rank']}")


if __name__ == "__main__":
    main()
