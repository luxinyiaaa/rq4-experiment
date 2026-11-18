"""Compute Table 6.7-style human–judge agreement after blind labels are filled.

Usage: python3 compute_agreement.py [--sheet human_validation_blind.xlsx]
The script refuses incomplete or altered review sheets. It writes agreement.csv and
agreement.md only after all 40 items have valid human labels.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

if importlib.util.find_spec("openpyxl") is None:
    system_python = shutil.which("python3")
    if system_python is None or Path(system_python).resolve() == Path(sys.executable).resolve():
        raise SystemExit("openpyxl is unavailable; install it or run with a Python that has it")
    print(f"openpyxl is unavailable in {sys.executable}; using {system_python}", file=sys.stderr)
    os.execv(system_python, [system_python, __file__, *sys.argv[1:]])

from openpyxl import load_workbook
from openpyxl.worksheet.formula import ArrayFormula
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "final" / "all_answers_and_scores_v4.xlsx"
LABELS = {"Entailment", "Neutral", "Contradiction"}
ABSTENTIONS = {"answers", "partly_answers", "abstains"}


def kappa(left: list, right: list, weighted: bool = False) -> float | None:
    """Match the category and quadratic weighting used in full/final_analysis.py."""
    if not left:
        return None
    levels = sorted(set(left) | set(right), key=str)
    a, b, n = Counter(left), Counter(right), len(left)

    def weight(x, y):
        if not weighted:
            return float(x != y)
        return ((levels.index(x) - levels.index(y)) / max(1, len(levels) - 1)) ** 2

    observed = sum(weight(x, y) for x, y in zip(left, right)) / n
    expected = sum(weight(x, y) * a[x] * b[y] / n**2 for x in levels for y in levels)
    return 1 - observed / expected if expected else None


def stats(left: list, right: list, relevance: bool = False) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("Agreement requires nonempty paired labels")
    result = {"n": len(left), "exact_agreement": sum(x == y for x, y in zip(left, right)) / len(left)}
    if relevance:
        result["within_one"] = sum(abs(x - y) <= 1 for x, y in zip(left, right)) / len(left)
        result["quadratic_weighted_kappa"] = kappa(left, right, weighted=True)
        result["spearman"] = (float(spearmanr(left, right).statistic)
                              if len(set(left)) > 1 and len(set(right)) > 1 else None)
    else:
        result["cohen_kappa"] = kappa(left, right)
    return result


def read_review(path: Path, sample: dict) -> dict:
    sheet = load_workbook(path, read_only=True, data_only=False)["Blind review"]
    values = iter(sheet.values)
    header = next(values)
    expected = ("Blind item ID", "Question", "Reference answer", "Keypoints", "Full answer text")
    if tuple(header[:5]) != expected:
        raise ValueError("Unexpected blind review layout")
    rows = list(values)
    if len(rows) != 40:
        raise ValueError(f"Expected 40 review rows, found {len(rows)}")
    result, problems = {}, []
    for cells in rows:
        blind_id = cells[0]
        if blind_id not in sample or blind_id in result:
            raise ValueError(f"Unexpected or duplicate blind ID: {blind_id}")
        original = sample[blind_id]
        kp_text = "\n".join(f"{k['number']}. [{'DECISIVE' if k['decisive'] else 'NOT DECISIVE'}] {k['text']}"
                            for k in original["keypoints"])
        if (cells[1], cells[2], cells[3], cells[4]) != (
            original["question"], original["reference_answer"], kp_text, original["full_answer_text"]):
            raise ValueError(f"Source text changed in {blind_id}")
        n = len(original["keypoints"])
        human = list(cells[5:5+n])
        if any(x not in LABELS for x in human):
            problems.append(f"{blind_id}: keypoint labels")
        if any(x not in (None, "") for x in cells[5+n:13]):
            problems.append(f"{blind_id}: unused keypoint label")
        abstention = cells[13]
        relevance = cells[14]
        if abstention not in ABSTENTIONS:
            problems.append(f"{blind_id}: abstention")
        if isinstance(relevance, bool) or str(relevance) not in {"1", "2", "3", "4", "5"}:
            problems.append(f"{blind_id}: relevance")
        formula = cells[15].text if isinstance(cells[15], ArrayFormula) else cells[15]
        if not isinstance(formula, str) or not formula.startswith("=IF("):
            problems.append(f"{blind_id}: correctness formula")
        result[blind_id] = {"labels": human, "abstention": abstention, "relevance": relevance,
                            "correct": all(l == "Entailment" for l, k in zip(human, original["keypoints"])
                                           if k["decisive"]) and "Contradiction" not in human,
                            "decisive_indices": [i for i, k in enumerate(original["keypoints"]) if k["decisive"]]}
    if problems:
        raise ValueError("Review is incomplete or invalid:\n" + "\n".join(problems))
    return result


def calculate(sheet_path: Path) -> list[dict]:
    sample_data = json.loads((HERE / "blind_sample_data.json").read_text(encoding="utf-8"))
    sample = {x["blind_item_id"]: x for x in sample_data["items"]}
    with (HERE / "human_validation_key.csv").open(newline="", encoding="utf-8") as f:
        key = list(csv.DictReader(f))
    if len(sample) != 40 or len(key) != 40 or {r["blind_item_id"] for r in key} != set(sample):
        raise ValueError("Blind sample and key do not match")
    human = read_review(sheet_path, sample)
    workbook = load_workbook(SOURCE, read_only=True, data_only=True)["All answers and scores"]
    values = iter(workbook.values)
    headers = next(values)
    source = {(r["question_id"], r["model"]): r for r in (dict(zip(headers, x)) for x in values)}
    if len(source) != 1200:
        raise ValueError("Judge source does not have 1,200 unique answers")

    paired = []
    for entry in key:
        blind_id = entry["blind_item_id"]
        judge = source[entry["question_id"], entry["model"]]
        h = human[blind_id]
        if judge["question"] != sample[blind_id]["question"] or \
           judge["full_llm_answer"] != sample[blind_id]["full_answer_text"]:
            raise ValueError(f"Source answer changed for {blind_id}")
        record = {"blind_item_id": blind_id, "model": entry["model"], "human": h, "judges": {}}
        for name in ("primary", "secondary"):
            labels = [x.strip() for x in judge[f"{name}_keypoint_labels"].split(";")]
            if len(labels) != len(h["labels"]):
                raise ValueError(f"Keypoint count mismatch for {blind_id} {name}")
            record["judges"][name] = {
                "labels": labels, "correct": bool(judge[f"{name}_correct_judge_only"]),
                "abstention": judge[f"{name}_abstention"],
                "relevance": int(judge[f"{name}_relevance"])}
        paired.append(record)

    results = []
    groups = [("overall", paired)] + [(model, [r for r in paired if r["model"] == model])
                                     for model in sorted({r["model"] for r in paired})]
    for group, subset in groups:
        for name in ("primary", "secondary"):
            def add(metric, left, right, relevance=False):
                results.append({"group": group, "judge": name, "metric": metric,
                                **stats(left, right, relevance)})
            add("correctness", [r["human"]["correct"] for r in subset],
                [r["judges"][name]["correct"] for r in subset])
            kp_human, kp_judge = [], []
            for r in subset:
                for i in r["human"]["decisive_indices"]:
                    kp_human.append(r["human"]["labels"][i] == "Entailment")
                    kp_judge.append(r["judges"][name]["labels"][i] == "Entailment")
            add("decisive_keypoint_present", kp_human, kp_judge)
            for position in range(max(len(r["human"]["decisive_indices"]) for r in subset)):
                relevant = [r for r in subset if len(r["human"]["decisive_indices"]) > position]
                add(f"decisive_keypoint_{position + 1}_present",
                    [r["human"]["labels"][r["human"]["decisive_indices"][position]] == "Entailment"
                     for r in relevant],
                    [r["judges"][name]["labels"][r["human"]["decisive_indices"][position]] == "Entailment"
                     for r in relevant])
            add("abstention", [r["human"]["abstention"] for r in subset],
                [r["judges"][name]["abstention"] for r in subset])
            add("abstention_partly_merged", ["answers" if r["human"]["abstention"] == "partly_answers"
                                              else r["human"]["abstention"] for r in subset],
                ["answers" if r["judges"][name]["abstention"] == "partly_answers"
                 else r["judges"][name]["abstention"] for r in subset])
            add("relevance", [int(r["human"]["relevance"]) for r in subset],
                [r["judges"][name]["relevance"] for r in subset], relevance=True)
    return results


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=HERE / "human_validation_blind.xlsx")
    parser.add_argument("--output-dir", type=Path, default=HERE,
                        help="Where to write agreement.csv and agreement.md")
    args = parser.parse_args()
    results = calculate(args.sheet)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["group", "judge", "metric", "n", "exact_agreement", "cohen_kappa",
              "within_one", "quadratic_weighted_kappa", "spearman"]
    with (args.output_dir / "agreement.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    lines = ["# Human–judge agreement for the rerun sample", "",
             "The 40 human-labelled answers were selected with seed 20261116, 10 per model. "
             "Correctness is calculated from human keypoint labels. Decisive-keypoint agreement "
             "uses Entailment versus Neutral/Contradiction, pooled across decisive keypoints. "
             "Cohen's kappa is unweighted for correctness, keypoints and abstention; relevance "
             "uses quadratic weighted kappa, matching the earlier Table 6.7 calculation.", "",
             "| Group | Judge | Metric | n | Exact agreement | Cohen κ | Within one | Quadratic weighted κ | Spearman ρ |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['group']} | {r['judge']} | {r['metric']} | {r['n']} | "
                     f"{fmt(r['exact_agreement'])} | {fmt(r.get('cohen_kappa'))} | "
                     f"{fmt(r.get('within_one'))} | {fmt(r.get('quadratic_weighted_kappa'))} | "
                     f"{fmt(r.get('spearman'))} |")
    (args.output_dir / "agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote agreement.csv and agreement.md")


if __name__ == "__main__":
    main()
