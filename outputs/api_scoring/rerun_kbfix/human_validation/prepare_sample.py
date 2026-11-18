"""Draw a reproducible, model-balanced blind sample from the rerun workbook."""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "final" / "all_answers_and_scores_v4.xlsx"
SEED = 20261116
MODELS = ("EGI gpt-oss-120b", "GPT-4.1", "GPT-5.6 Sol", "GPT-6 Astra")
KEYPOINT = re.compile(r"^(\d+)\. (\[DECISIVE\] )?(.*)$")


def main() -> None:
    sheet = load_workbook(SOURCE, read_only=True, data_only=True)["All answers and scores"]
    values = iter(sheet.values)
    headers = next(values)
    rows = [dict(zip(headers, values)) for values in values]
    if len(rows) != 1200 or len({(r["question_id"], r["model"]) for r in rows}) != 1200:
        raise ValueError("Source workbook does not contain 1,200 unique question-model pairs")
    by_model = {model: sorted((r for r in rows if r["model"] == model),
                              key=lambda r: r["question_id"]) for model in MODELS}
    if any(len(v) != 300 for v in by_model.values()):
        raise ValueError("Each model must have 300 answers")

    rng = random.Random(SEED)
    model_order = list(MODELS)
    rng.shuffle(model_order)
    selected = []
    used_questions = set()
    for model in model_order:
        eligible = [r for r in by_model[model] if r["question_id"] not in used_questions]
        choice = rng.sample(eligible, 10)
        selected.extend(choice)
        used_questions.update(r["question_id"] for r in choice)
    rng.shuffle(selected)

    blind = []
    key = []
    for i, row in enumerate(selected, 1):
        item_id = f"HV-R{i:03d}"
        keypoints = []
        for j, line in enumerate(row["keypoints"].splitlines(), 1):
            match = KEYPOINT.fullmatch(line)
            if not match or int(match.group(1)) != j:
                raise ValueError(f"Unexpected keypoint format for {row['question_id']}: {line}")
            keypoints.append({"number": j, "decisive": bool(match.group(2)),
                              "text": match.group(3)})
        if not keypoints or len(keypoints) > 8 or not any(k["decisive"] for k in keypoints):
            raise ValueError(f"Invalid keypoints for {row['question_id']}")
        blind.append({"blind_item_id": item_id, "question": row["question"],
                      "reference_answer": row["reference_answer"],
                      "keypoints": keypoints, "full_answer_text": row["full_llm_answer"]})
        key.append({"blind_item_id": item_id, "question_id": row["question_id"],
                    "model": row["model"]})

    if len(blind) != 40 or len({r["question_id"] for r in key}) != 40 or \
       Counter(r["model"] for r in key) != {model: 10 for model in MODELS}:
        raise ValueError("Sample is not balanced or contains duplicate questions")
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "blind_sample_data.json").write_text(
        json.dumps({"seed": SEED, "items": blind}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    with (HERE / "human_validation_key.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["blind_item_id", "question_id", "model"])
        writer.writeheader()
        writer.writerows(key)
    (HERE / "sample_manifest.json").write_text(json.dumps({
        "seed": SEED, "sampling": "Python random.Random; 10 answers per model; unique questions; final shuffle",
        "source": str(SOURCE.relative_to(HERE.parents[3])),
        "sample_size": 40, "per_model": 10, "unique_questions": 40}, indent=2), encoding="utf-8")
    print(f"Prepared 40 blind items, 10 per model, seed {SEED}")


if __name__ == "__main__":
    main()
