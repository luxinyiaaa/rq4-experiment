"""Apply review-sheet edits and rerun checks for questions marked edit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_index import DEFAULT_OUTDIR, _read_jsonl, load_index
from scripts.check_questions import (
    CONFLICT_PROMPT,
    GENERATOR_PROMPT,
    OUTPUT_JSONL,
    OUTPUT_REVIEW,
    _edited_version,
    _read_review_sheet,
    load_conflict_prompt,
    load_generator_systems,
    run_edited_question_checks,
    write_outputs,
)

REVIEWED_OUTPUT = PROJECT_ROOT / "outputs" / "pilot" / "questions_reviewed.jsonl"


def prepare_reviewed_questions(
    questions: list[dict],
    reviews: dict[str, dict[str, str]],
    *,
    index_dir: Path = DEFAULT_OUTDIR,
    conflict_prompt_path: Path = CONFLICT_PROMPT,
    generator_prompt_path: Path = GENERATOR_PROMPT,
) -> list[dict]:
    allowed_decisions = {"accept", "edit", "reject"}
    missing_or_invalid = []
    for question in questions:
        qid = str(question.get("id"))
        decision = str(reviews.get(qid, {}).get("decision (accept/edit/reject)", "")).strip().casefold()
        if decision not in allowed_decisions:
            missing_or_invalid.append(qid)
    if missing_or_invalid:
        raise ValueError("Missing or invalid review decisions for question IDs: " + ", ".join(missing_or_invalid))

    edits = 0
    for question in questions:
        question["human_review"] = reviews.get(str(question.get("id")), {})
        decision = str(question["human_review"]["decision (accept/edit/reject)"]).strip().casefold()
        question["review_decision"] = decision
        edited = _edited_version(question)
        if edited:
            edits += 1
        elif decision == "reject":
            question["status"] = "rejected"
            question["review_applied"] = True
    if edits:
        index = load_index(index_dir)
        conflict_system = load_conflict_prompt(conflict_prompt_path)
        generator_system, _ = load_generator_systems(generator_prompt_path)
        run_edited_question_checks(questions, index, conflict_system, generator_system)
        for question in questions:
            edited = _edited_version(question)
            if edited:
                question["question"] = edited["question"]
                question["reference_answer"] = edited["reference_answer"]
                question["keypoints"] = edited["keypoints"]
                question["review_applied"] = True
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--review-xlsx", type=Path, default=OUTPUT_REVIEW)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--out", type=Path, default=REVIEWED_OUTPUT)
    args = parser.parse_args()
    questions = _read_jsonl(args.input)
    reviews = _read_review_sheet(args.review_xlsx)
    reviewed = prepare_reviewed_questions(questions, reviews, index_dir=args.index_dir)
    actual_review_path = write_outputs(reviewed, args.out, args.review_xlsx)
    applied = sum(row.get("review_applied") is True for row in reviewed)
    print(f"Prepared {len(reviewed)} reviewed questions; applied edits/rejections: {applied}; output: {args.out}; review sheet: {actual_review_path}")


if __name__ == "__main__":
    main()
