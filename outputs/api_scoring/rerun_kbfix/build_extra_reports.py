"""Question-type and old/new descriptive tables; no model calls."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from outputs.api_scoring.rerun_kbfix import build_reports as reports
from outputs.api_scoring.rerun_kbfix.run_rerun import ANSWER_SOURCE, complete_jsonl, answer_checks
from outputs.api_scoring.full import report_full as rep
from scripts.analyze_metrics_v2 import wilson

OUT = reports.OUT
MODELS = rep.MODELS
TYPE_DEFINITIONS = {
    "list_all": "Asks for every item in a clearly bounded list in the source passage; every listed item is a decisive keypoint.",
    "paraphrased": "Asks for one or two related facts using wording that differs from the source sentence or field.",
    "false_premise": "Includes a plausible wrong detail contradicted by the source and asks a separate factual question; the answer must correct the premise and answer the question.",
}


def table(rows: list[dict], columns: list[str]) -> str:
    return reports.md(rows, columns)


def rate(hits: int, n: int) -> str:
    if not n:
        return "—"
    low, high = wilson(hits, n)
    return f"{100*hits/n:.1f}% [{low*100:.1f}, {high*100:.1f}] (n={n})"


def mean_sd(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values) if len(values)>1 else 0:.3f} (n={len(values)})"


def by_type(items: list[dict]) -> None:
    lines = ["# Results by question type", "",
             "Correctness and false abstention are rates with Wilson 95% intervals. Other cells give mean ± sample SD; citation metrics show their applicable n.", ""]
    columns = ["question_type", "model", "n", "correctness", "false abstention", "completeness",
               "relevance", "citation support", "citation completeness"]
    for judge in ("primary", "secondary"):
        rows = []
        for qtype in TYPE_DEFINITIONS:
            for model in MODELS:
                subset = [x for x in items if x["type"] == qtype and x["model"] == model]
                n = len(subset)
                correct = sum(bool(x["metrics"]["answer_correctness"][judge]) for x in subset)
                abst = sum(x["metrics"]["appropriate_abstention"][judge]["label"] == "abstains" for x in subset)
                def vals(metric):
                    return [float(v) for x in subset if (v := rep.value(x, metric, judge)) is not None]
                rows.append({"question_type": qtype, "model": model, "n": n,
                             "correctness": rate(correct, n), "false abstention": rate(abst, n),
                             "completeness": mean_sd(vals("completeness")),
                             "relevance": mean_sd(vals("response_relevance")),
                             "citation support": mean_sd(vals("citation_support")),
                             "citation completeness": mean_sd(vals("citation_completeness"))})
        lines += [f"## {judge.title()} judge", "", table(rows, columns), "",
                  "Question types (from the question-generation prompt):", ""]
        lines += [f"- **{name}:** {definition}" for name, definition in TYPE_DEFINITIONS.items()]
        lines.append("")
    path = OUT / "final/results_by_question_type.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def old_vs_new(items: list[dict]) -> None:
    old_answers = {(x["id"], x["model_name"]): x for x in
                   complete_jsonl(reports.ROOT / "outputs/api_answers/full/answers.jsonl")
                   if answer_checks.is_success(x)}
    new_answers = {(x["id"], x["model_name"]): x for x in complete_jsonl(ANSWER_SOURCE)
                   if answer_checks.is_success(x)}
    old_items = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in
                 (reports.ROOT / "outputs/api_scoring/full300/items").glob("*.json")}
    current = {reports.item_key(x): x for x in items}
    rows = []
    for model in MODELS:
        for group in ("overall", *TYPE_DEFINITIONS, "EDMED/SeaDataNet", "other pages"):
            keys = [key for key, item in current.items() if item["model"] == model and (
                group == "overall" or item["type"] == group or
                (group == "EDMED/SeaDataNet" and any(s in item["source_url"].lower() for s in ("edmed", "seadatanet"))) or
                (group == "other pages" and not any(s in item["source_url"].lower() for s in ("edmed", "seadatanet")))
            )]
            old_c = sum(bool(old_items[k]["metrics"]["answer_correctness"]["primary"]) for k in keys)
            new_c = sum(bool(current[k]["metrics"]["answer_correctness"]["primary"]) for k in keys)
            old_nf = sum(bool(answer_checks.UNAVAILABLE_RE.search(old_answers[current[k]["id"], model]["answer"])) for k in keys)
            new_nf = sum(bool(answer_checks.UNAVAILABLE_RE.search(new_answers[current[k]["id"], model]["answer"])) for k in keys)
            rows.append({"model": model, "group": group, "n": len(keys),
                         "old correct": f"{old_c}/{len(keys)} ({old_c/len(keys):.1%})",
                         "new correct": f"{new_c}/{len(keys)} ({new_c/len(keys):.1%})",
                         "old could not find": f"{old_nf}/{len(keys)} ({old_nf/len(keys):.1%})",
                         "new could not find": f"{new_nf}/{len(keys)} ({new_nf/len(keys):.1%})"})
    lines = ["# Old versus KB-fix rerun", "",
             "Descriptive comparison of the same 300 questions per model. Correctness is the primary judge's original label; no manual alternative-answer decisions are applied. The 'could not find' count uses the answer runner's keyword rule. No new statistical tests were run.", "",
             table(rows, ["model", "group", "n", "old correct", "new correct", "old could not find", "new could not find"]), ""]
    (OUT / "final/old_vs_new.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    snap = json.loads((OUT / "input_snapshot.json").read_text(encoding="utf-8"))
    items = [json.loads((OUT / "items" / (key + ".json")).read_text(encoding="utf-8"))
             for key in snap["answer_ids"]]
    by_type(items)
    old_vs_new(items)


if __name__ == "__main__":
    main()
