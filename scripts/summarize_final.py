"""Create the final experiment tables and blinded human-check workbooks."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.build_index import _read_jsonl
from scripts.check_questions import _write_xlsx_minimal
from scripts.score_answers import _score_number

SEED = 42
BOOTSTRAP_REPLICATES = 2000


def _metric_value(score: dict[str, Any], metric: str) -> float | None:
    if metric == "correctness":
        value = (score.get("ragchecker") or {}).get("correctness")
        return float(value) if isinstance(value, bool) else None
    if metric == "completeness":
        value = (score.get("ragchecker") or {}).get("completeness")
        return float(value) if value is not None else None
    if metric == "flask":
        value = (score.get("flask") or {}).get("score")
        return float(value) if isinstance(value, (int, float)) else None
    if metric == "abstention_rate":
        value = (score.get("abstention") or {}).get("label")
        return float(value == "abstains") if value else None
    if metric == "faithfulness":
        return _score_number(score.get("faithfulness"))
    raise ValueError(metric)


def bootstrap_mean(values: Iterable[float], seed: int = SEED, replicates: int = BOOTSTRAP_REPLICATES) -> tuple[float | None, float | None, float | None]:
    values_array = np.asarray(list(values), dtype=float)
    if not len(values_array):
        return None, None, None
    mean = float(np.mean(values_array))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values_array), size=(replicates, len(values_array)))
    sampled = np.mean(values_array[indices], axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return mean, float(low), float(high)


def _row_for_group(table: str, condition: str, subgroup: str, metric: str, values: list[float]) -> dict[str, Any]:
    mean, low, high = bootstrap_mean(values)
    return {
        "table": table, "condition": condition, "subgroup": subgroup,
        "metric": metric, "n": len(values), "estimate": mean,
        "ci95_lower": low, "ci95_upper": high,
        "bootstrap_seed": SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def build_summary(scores: list[dict[str, Any]], conditions: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_id_condition = {(str(row["id"]), str(row["condition"])): row for row in scores}
    conditions_by_id = {str(row["id"]): row for row in conditions}
    metric_names = ("correctness", "completeness", "flask", "abstention_rate", "faithfulness")
    rows: list[dict[str, Any]] = []

    def add_group(table: str, condition: str, subgroup: str, grouped_scores: list[dict[str, Any]]) -> None:
        for metric in metric_names:
            values = [value for score in grouped_scores if (value := _metric_value(score, metric)) is not None]
            rows.append(_row_for_group(table, condition, subgroup, metric, values))
        if condition == "C4":
            c4_rows = [r for r in grouped_scores]
            chunk_hits = [float(conditions_by_id[str(r["id"])].get("c4_gold_hit_rank") is not None) for r in c4_rows]
            page_hits = [float((conditions_by_id[str(r["id"])].get("c4_gold_page_hit_rank") or {}).get("gold") is not None) for r in c4_rows]
            recall = [v for r in c4_rows if (v := _score_number(r.get("context_recall"))) is not None]
            rows.append(_row_for_group(table, condition, subgroup, "gold_chunk_hit_at_10", chunk_hits))
            rows.append(_row_for_group(table, condition, subgroup, "gold_page_hit_at_10", page_hits))
            rows.append(_row_for_group(table, condition, subgroup, "context_recall", recall))

    for condition in ("C1", "C2", "C3", "C4"):
        group = [r for r in scores if r["condition"] == condition]
        add_group("condition", condition, "all", group)
    for condition in ("C1", "C2", "C3", "C4"):
        for question_type in ("list_all", "comparison", "paraphrased", "false_premise"):
            group = [r for r in scores if r["condition"] == condition and r.get("type") == question_type]
            add_group("condition_by_type", condition, question_type, group)
    c2_positions = sorted({str(r.get("gold_position") or "unknown") for r in scores if r.get("condition") == "C2"})
    for position in c2_positions:
        group = [r for r in scores if r.get("condition") == "C2" and str(r.get("gold_position") or "unknown") == position]
        add_group("C2_by_gold_position", "C2", position, group)
    summary = pd.DataFrame(rows)

    paired_rows = []
    by_question: dict[str, dict[str, dict[str, Any]]] = {}
    for row in scores:
        by_question.setdefault(str(row["id"]), {})[str(row["condition"])] = row
    for metric in ("correctness", "completeness", "faithfulness"):
        for pair_name, left_name, right_name in (("C1_vs_C2", "C1", "C2"), ("C1_vs_C4", "C1", "C4")):
            pairs = []
            for qid, group in by_question.items():
                left = group.get(left_name)
                right = group.get(right_name)
                if left is None or right is None:
                    continue
                left_value, right_value = _metric_value(left, metric), _metric_value(right, metric)
                if left_value is not None and right_value is not None:
                    pairs.append((left_value, right_value))
            pvalue = None
            statistic = None
            discordant = None
            if metric == "correctness":
                b = sum(left == 1.0 and right == 0.0 for left, right in pairs)
                c = sum(left == 0.0 and right == 1.0 for left, right in pairs)
                discordant = b + c
                if discordant:
                    test = binomtest(min(b, c), discordant, 0.5, alternative="two-sided")
                    pvalue = float(test.pvalue)
                statistic = float(min(b, c)) if discordant else 0.0
                method = "McNemar exact binomial test"
            else:
                differences = np.asarray([left - right for left, right in pairs], dtype=float)
                if len(differences) and np.any(differences != 0):
                    test = wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
                    statistic, pvalue = float(test.statistic), float(test.pvalue)
                method = "Wilcoxon signed-rank test"
            paired_rows.append({"metric": metric, "comparison": pair_name, "method": method, "n_pairs": len(pairs), "statistic": statistic, "discordant_pairs": discordant, "p_value": pvalue})
    paired = pd.DataFrame(paired_rows)

    c4 = [r for r in scores if r.get("condition") == "C4"]
    cross = []
    for refusal in ("answers", "partially_answers", "abstains", "missing"):
        for recall_zero in (True, False, None):
            count = 0
            for row in c4:
                label = (row.get("abstention") or {}).get("label") or "missing"
                recall = _score_number(row.get("context_recall"))
                is_zero = None if recall is None else bool(np.isclose(recall, 0.0))
                if label == refusal and is_zero is recall_zero:
                    count += 1
            cross.append({"refusal_label": refusal, "context_recall_zero": "missing" if recall_zero is None else str(recall_zero).lower(), "count": count})
    cross_frame = pd.DataFrame(cross, columns=["refusal_label", "context_recall_zero", "count"])

    needs = []
    for row in scores:
        if row.get("needs_manual_check") is True:
            needs.append({
                "id": row["id"], "condition": row["condition"], "type": row.get("type"),
                "question": row.get("question", ""),
                "reference_answer": row.get("reference_answer", ""),
                "ragchecker_correctness": (row.get("ragchecker") or {}).get("correctness"),
                "keypoint_labels": "; ".join((row.get("ragchecker") or {}).get("keypoint_labels", [])),
                "flask_score": (row.get("flask") or {}).get("score"),
                "abstention": (row.get("abstention") or {}).get("label"),
                "answer": row.get("answer", ""),
            })
    needs_frame = pd.DataFrame(needs, columns=["id", "condition", "type", "question", "reference_answer", "ragchecker_correctness", "keypoint_labels", "flask_score", "abstention", "answer"])
    return summary, paired, cross_frame, needs_frame


def create_human_check(scores: list[dict[str, Any]], questions: list[dict[str, Any]], output_dir: Path) -> None:
    rng = random.Random(SEED)
    by_condition = {c: sorted([s for s in scores if s.get("condition") == c], key=lambda r: str(r["id"])) for c in ("C1", "C2", "C3", "C4")}
    if any(len(rows) < 5 for rows in by_condition.values()):
        raise ValueError("Each condition needs at least five answers for the blinded sample")
    selected = []
    for condition, rows in by_condition.items():
        selected.extend((rng.sample(rows, 5)))
    rng.shuffle(selected)
    question_map = {str(q["id"]): q for q in questions}
    blind_rows, key_rows = [], []
    for number, answer in enumerate(selected, start=1):
        item_id = f"H{number:03d}"
        question = question_map[str(answer["id"])]
        keypoints = "\n".join(("[D] " if point.get("decisive") else "[ ] ") + str(point.get("text", "")) for point in question.get("keypoints", []))
        blind_rows.append({
            "item_id": item_id,
            "question": question["question"],
            "reference_answer": question["reference_answer"],
            "keypoints": keypoints,
            "answer": answer.get("answer", ""),
            "human_correct": "",
            "human_abstention": "",
            "human_note": "",
        })
        key_rows.append({
            "item_id": item_id,
            "question_id": answer["id"],
            "condition": answer["condition"],
            "type": answer.get("type"),
            "ragchecker_correctness": (answer.get("ragchecker") or {}).get("correctness"),
            "ragchecker_completeness": (answer.get("ragchecker") or {}).get("completeness"),
            "flask_score": (answer.get("flask") or {}).get("score"),
            "abstention": (answer.get("abstention") or {}).get("label"),
            "faithfulness": _score_number(answer.get("faithfulness")),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_xlsx_minimal(pd.DataFrame(blind_rows), output_dir / "human_check_blind.xlsx")
    # Write the answer key without reading it back; it stays unopened until human grading is done.
    pd.DataFrame(key_rows).to_csv(output_dir / "human_check_key.csv", index=False)


def run(scores_path: Path, questions_path: Path, conditions_path: Path, outdir: Path) -> None:
    scores = _read_jsonl(scores_path)
    questions = _read_jsonl(questions_path)
    conditions = _read_jsonl(conditions_path)
    if len(scores) != len(questions) * 4 or len(questions) != 44:
        raise ValueError(f"Expected 44 questions and 176 scores, found {len(questions)} and {len(scores)}")
    summary, paired, cross, needs = build_summary(scores, conditions)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "score_summary_final.csv"
    paired_path = outdir / "paired_tests_final.csv"
    cross_path = outdir / "c4_abstention_contextrecall_final.csv"
    needs_path = outdir / "needs_manual_check_final.csv"
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    cross.to_csv(cross_path, index=False)
    needs.to_csv(needs_path, index=False)
    _write_xlsx_minimal(needs, outdir / "manual_check_disagreements.xlsx")
    create_human_check(scores, questions, outdir)
    print("Condition summary with bootstrap 95% CIs (percentile, 2,000 resamples, seed 42):")
    print(summary[summary["table"] == "condition"].to_string(index=False))
    print("Condition × question type:")
    print(summary[summary["table"] == "condition_by_type"].to_string(index=False))
    print("C2 by gold position:")
    print(summary[summary["table"] == "C2_by_gold_position"].to_string(index=False))
    print("C4 refusal × ContextRecall=0:")
    print(cross.to_string(index=False))
    print("Paired tests:")
    print(paired.to_string(index=False))
    print("Needs manual check:")
    print(needs[["id", "condition", "ragchecker_correctness", "flask_score"]].to_string(index=False))
    print(f"Wrote {summary_path}, {paired_path}, {cross_path}, {needs_path}")
    print(f"Wrote {outdir / 'human_check_blind.xlsx'} and {outdir / 'manual_check_disagreements.xlsx'}")
    print("Wrote human_check_key.csv; it was not reopened.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "outputs/final/scores_final.jsonl")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "outputs/final/questions_final.jsonl")
    parser.add_argument("--conditions", type=Path, default=PROJECT_ROOT / "outputs/final/conditions_final.jsonl")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs/final")
    args = parser.parse_args()
    run(args.scores, args.questions, args.conditions, args.outdir)


if __name__ == "__main__":
    main()
