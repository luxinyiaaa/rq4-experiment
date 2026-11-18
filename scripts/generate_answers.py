"""Generate unscored answers for all four conditions of the pilot questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rq4.llm import call_llm
from scripts.build_index import _read_jsonl, _write_jsonl
from scripts.check_questions import _write_xlsx_minimal

CONDITIONS_PATH = PROJECT_ROOT / "outputs" / "pilot" / "conditions_pilot.jsonl"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "generator_answer.md"
ANSWERS_PATH = PROJECT_ROOT / "outputs" / "pilot" / "answers_pilot.jsonl"
REVIEW_PATH = PROJECT_ROOT / "outputs" / "pilot" / "answers_review.xlsx"


def load_generator_system(path: Path = PROMPT_PATH) -> tuple[str, str]:
    prompt_file = path.read_text(encoding="utf-8")
    start = prompt_file.index("## System") + len("## System")
    end = prompt_file.index("## User template", start)
    return prompt_file[start:end].strip(), hashlib.sha256(prompt_file.encode("utf-8")).hexdigest()


def make_user_message(question: str, passages: list[dict[str, Any]]) -> str:
    sections = [f"[{row['passage_id']}] {row['text']}" for row in passages]
    return "Passages:\n\n" + "\n\n".join(sections) + f"\n\nQuestion: {question}"


def generate_answers(
    conditions: list[dict[str, Any]], system_prompt: str, prompt_sha256: str,
    *, llm_call: Callable[..., dict[str, Any]] = call_llm,
    progress: bool = True,
) -> list[dict[str, Any]]:
    outputs = []
    total = sum(len(row.get("conditions", {})) for row in conditions)
    ordinal = 0
    for question in conditions:
        for condition_name in ("C1", "C2", "C3", "C4"):
            condition = question["conditions"][condition_name]
            ordinal += 1
            if progress:
                print(f"Generating answer {ordinal}/{total}: {question['id']} {condition_name}", flush=True)
            user_message = make_user_message(question["question"], condition["passages"])
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            try:
                response = llm_call(
                    "generator", messages, temperature=0.0,
                    prompt_sha256=prompt_sha256, log_messages=True,
                )
                status = "generated"
                answer = str(response.get("text", ""))
                error = None
            except Exception as exc:
                response = {}
                status = "error"
                answer = ""
                error = f"{type(exc).__name__}: {exc}"
            outputs.append({
                "id": question["id"],
                "type": question["type"],
                "question": question["question"],
                "reference_answer": question.get("reference_answer", ""),
                "condition": condition_name,
                "gold_position": question.get("gold_position"),
                "passages": condition["passages"],
                "passage_map": condition["passage_map"],
                "prompt_sha256": prompt_sha256,
                "model_id": response.get("api_model", response.get("model")),
                "token_usage": response.get("usage"),
                "request_parameters": response.get("request_parameters"),
                "elapsed_seconds": response.get("elapsed_seconds"),
                "cache_hit": response.get("cache_hit"),
                "answer": answer,
                "status": status,
                **({"error": error} if error else {}),
            })
    return outputs


def review_frame(questions: list[dict[str, Any]], answers: list[dict[str, Any]]) -> pd.DataFrame:
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in answers:
        by_id.setdefault(str(row["id"]), {})[str(row["condition"])] = row
    output = []
    for question in questions:
        variants = by_id.get(str(question["id"]), {})
        c4 = variants.get("C4", {})
        distractors = question.get("selected_distractors", [])
        output.append({
            "id": question["id"],
            "type": question["type"],
            "question": question["question"],
            "reference_answer": question.get("reference_answer", ""),
            "C1_answer": variants.get("C1", {}).get("answer", ""),
            "C2_answer (gold position)": (
                f"Gold position: {question.get('gold_position')}\n" + str(variants.get("C2", {}).get("answer", ""))
            ),
            "C3_answer": variants.get("C3", {}).get("answer", ""),
            "C4_answer": c4.get("answer", ""),
            "C4_gold_hit_rank": question.get("c4_gold_hit_rank"),
            "C4_gold_page_hit_rank": (question.get("c4_gold_page_hit_rank") or {}).get("gold"),
            "C4_partner_page_hit_rank": (question.get("c4_gold_page_hit_rank") or {}).get("partner"),
            "distractor_urls_and_similarities": "\n".join(
                f"{row['url']} | {row['similarity']:.6f}" for row in distractors
            ),
            "excluded_candidate_count": question.get("distractor_selection", {}).get("prefilter_excluded_count", 0)
            + question.get("distractor_selection", {}).get("judge_removed", 0),
            "exclusion_reasons": json.dumps({
                "prefilter": question.get("distractor_selection", {}).get("prefilter_exclusion_reasons", {}),
                "judge_labeled_answers": question.get("distractor_selection", {}).get("judge_removed", 0),
            }, ensure_ascii=False),
        })
    return pd.DataFrame(output)


def write_review_sheet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="Answers review")
            sheet = writer.sheets["Answers review"]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column, width in {
                "A": 12, "B": 18, "C": 55, "D": 50, "E": 65, "F": 65, "G": 65,
                "H": 65, "I": 18, "J": 18, "K": 18, "L": 80, "M": 22, "N": 50,
            }.items():
                sheet.column_dimensions[column].width = width
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        _write_xlsx_minimal(frame, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, default=CONDITIONS_PATH)
    parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    parser.add_argument("--out", type=Path, default=ANSWERS_PATH)
    parser.add_argument("--review-xlsx", type=Path, default=REVIEW_PATH)
    args = parser.parse_args()
    conditions = _read_jsonl(args.conditions)
    system_prompt, prompt_hash = load_generator_system(args.prompt)
    answers = generate_answers(conditions, system_prompt, prompt_hash)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out, answers)
    frame = review_frame(conditions, answers)
    write_review_sheet(frame, args.review_xlsx)
    successful = sum(row["status"] == "generated" for row in answers)
    print(f"Generated {successful}/{len(answers)} unscored answers.")
    answers_by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in answers:
        answers_by_id.setdefault(str(row["id"]), {})[str(row["condition"])] = row
    print("\nPer-question generation summary:")
    for question in conditions:
        distractor_summary = ", ".join(
            f"{row['url']} ({row['similarity']:.4f})" for row in question.get("selected_distractors", [])
        )
        by_condition = answers_by_id.get(str(question["id"]), {})
        lengths = ", ".join(
            f"{name}={len(by_condition.get(name, {}).get('answer', ''))} chars"
            for name in ("C1", "C2", "C3", "C4")
        )
        page_ranks = question.get("c4_gold_page_hit_rank", {})
        print(
            f"{question['id']}: distractors=[{distractor_summary}]; "
            f"judge removed={question.get('distractor_selection', {}).get('judge_removed', 0)}; "
            f"C4 gold chunk rank={question.get('c4_gold_hit_rank')}; "
            f"C4 gold/partner page rank={page_ranks.get('gold')}/{page_ranks.get('partner')}; {lengths}"
        )
    print(f"Wrote {args.out}")
    print(f"Wrote {args.review_xlsx}")


if __name__ == "__main__":
    main()
