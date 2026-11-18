"""Recompute false-premise sensitivity from cached labels only."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "final"
SUPPLEMENT = OUT / "supplementary/false_premise_sensitivity.md"
MODELS = ("egi", "openai", "openai_gpt56_sol", "openai_gpt6_astra")
JUDGES = ("primary", "secondary")
DIRECT_ANSWER_IDS = {"AS-055-e61cb36aa7_2"}


def labels_for(item, judge):
    return next(x["labels"] for x in item["judgements"]
        if x.get("metric") == "answer_correctness_completeness" and x.get("judge") == judge)


def md(rows, keys):
    return "| " + " | ".join(keys) + " |\n|" + "|".join("---" for _ in keys) + "|\n" + "\n".join(
        "| " + " | ".join(str(row.get(k, "")).replace("|", "\\|").replace("\n", " ") for k in keys) + " |" for row in rows)


def main():
    keypoints = list(csv.DictReader((OUT / "false_premise_correcting_keypoints.csv").open(newline="", encoding="utf-8")))
    if len(keypoints) != 27 or len({x["question_id"] for x in keypoints}) != 27:
        raise ValueError("Expected 27 false-premise questions")
    lookup = {}
    for row in keypoints:
        qid = row["question_id"]
        for model in MODELS:
            item = json.loads((BASE / "items" / (qid + "__" + model + ".json")).read_text(encoding="utf-8"))
            lookup[qid, model] = item
    comparison = []
    outcomes = []
    changed_cases = []
    for model in MODELS:
        for judge in JUDGES:
            old_count = intended_count = strict_count = answered_without = 0
            category_counts = Counter()
            for row in keypoints:
                qid = row["question_id"]
                item = lookup[qid, model]
                labels = labels_for(item, judge)
                correction_index = int(row["correcting_keypoint_index"]) - 1
                correction = labels[correction_index]
                if correction not in ("Entailment", "Neutral", "Contradiction"):
                    raise ValueError(f"Unexpected correction label: {correction}")
                decisive = [i for i, kp in enumerate(item["keypoints"]) if kp.get("decisive")]
                if correction_index not in decisive:
                    raise ValueError(f"Correction keypoint not decisive: {qid}")
                other_entailed = all(labels[i] == "Entailment" for i in decisive if i != correction_index)
                no_contradiction = "Contradiction" not in labels
                strict = bool(item["metrics"]["answer_correctness"][judge])
                old = other_entailed and no_contradiction
                intended = (strict if qid in DIRECT_ANSWER_IDS else
                    correction in ("Entailment", "Neutral") and other_entailed and no_contradiction)
                strict_count += strict
                old_count += old
                intended_count += intended
                if old != intended:
                    changed_cases.append((qid, model, judge, old, intended))
                abst = item["primary_abstention_label"].replace("_", " ")
                if correction == "Entailment":
                    category = "corrects the premise"
                elif correction == "Contradiction":
                    category = "accepts the premise"
                elif abst in ("answers", "partly answers"):
                    category = "silent on the premise"
                elif abst == "abstains":
                    category = "abstains"
                else:
                    raise ValueError(f"Unexpected abstention label: {abst}")
                category_counts[category] += 1
                answered_without += (abst != "abstains" and correction != "Entailment" and
                    any(labels[i] == "Entailment" for i in decisive if i != correction_index))
            n = len(keypoints)
            if sum(category_counts.values()) != n:
                raise ValueError("Outcome counts do not partition all questions")
            comparison.append({"model": model, "judge": judge, "n": n,
                "strict": f"{strict_count}/{n} ({strict_count/n:.1%})",
                "previous_lenient": f"{old_count}/{n} ({old_count/n:.1%})",
                "intended_lenient": f"{intended_count}/{n} ({intended_count/n:.1%})",
                "answers_without_correction": f"{answered_without}/{n} ({answered_without/n:.1%})"})
            outcomes.append({"model": model, "judge": judge, "n": n,
                **{name: f"{category_counts[name]}/{n} ({category_counts[name]/n:.1%})"
                   for name in ("corrects the premise", "accepts the premise", "silent on the premise", "abstains")}})
    if changed_cases:
        changed_text = "; ".join(f"{qid} {model} {judge}: {old} → {new}"
            for qid, model, judge, old, new in changed_cases)
    else:
        changed_text = "No cached answer changes lenient classification under the intended definition."
    lines = ["# False-premise sensitivity (supplementary)", "",
        "This analysis uses the 27 false-premise questions in the 94-question API cohort, with four model answers per question and both cached judges. No judge labels were changed.", "",
        "## Lenient-correctness rule", "",
        "The previous calculation required every decisive keypoint other than the premise correction to be Entailment and required **no keypoint** to be Contradiction. Thus, a Contradiction on the premise-correcting keypoint already failed lenient correctness: an answer that repeats or accepts the false premise could not pass. The correction keypoint could be Neutral or Entailment.", "",
        "The intended calculation makes that condition explicit: the correction keypoint may be Neutral but not Contradiction; every other decisive keypoint must be Entailment; and no keypoint may be Contradiction. When the correction keypoint is itself the requested answer, intended lenient correctness equals strict correctness.", "",
        "### Question where correction is the requested answer", "",
        "- **AS-055-e61cb36aa7_2:** The question asks which city hosted the IOC Assembly session. The correction keypoint gives Paris, rather than the question's false Maputo premise. No other question in this 27-question cohort has its correction keypoint as the requested answer. In the cached labels, three model answers entail this correction and one contradicts it under each judge; none is Neutral, so the exception does not change the observed totals.", "",
        "## Strict, previous lenient, and intended lenient correctness", "",
        md(comparison, ["model", "judge", "n", "strict", "previous_lenient", "intended_lenient", "answers_without_correction"]), "",
        changed_text, "",
        "The final column counts non-abstaining answers that entail at least one other decisive keypoint while missing the premise correction; it is a separate descriptive measure.", "",
        "## Mutually exclusive premise outcomes", "",
        "The outcome uses the premise-correcting keypoint label from each judge. Neutral answers are split using the primary judge's abstention label, which is shared for both judges' applicability decisions.", "",
        md(outcomes, ["model", "judge", "n", "corrects the premise", "accepts the premise", "silent on the premise", "abstains"]), "",
        "Each row sums to 27 answers (100%). 'Corrects the premise' means Entailment of the correction keypoint, regardless of the answer's other keypoints; 'accepts' means Contradiction; 'silent' and 'abstains' partition Neutral labels.", "",
        "## Premise-correction keypoints", "",
        md([{**row, "also_requested_answer": "yes" if row["question_id"] in DIRECT_ANSWER_IDS else "no"}
            for row in keypoints], ["question_id", "correcting_keypoint", "also_requested_answer"]), ""]
    SUPPLEMENT.parent.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT.write_text("\n".join(lines), encoding="utf-8")
    # Move only this section from the main analysis; preserve the surrounding text.
    main = OUT / "final_results.md"
    content = main.read_text(encoding="utf-8")
    start = "## False-premise sensitivity\n"
    end = "## Human-label normalisation and HV004 correction\n"
    if start in content:
        prefix, remainder = content.split(start, 1)
        _, suffix = remainder.split(end, 1)
        main.write_text(prefix.rstrip() + "\n\n" + end + suffix, encoding="utf-8")
    elif end not in content:
        raise ValueError("Could not locate human-validation section in final_results.md")
    print(json.dumps({"questions": len(keypoints), "changed_lenient_cases": len(changed_cases),
        "outcome_rows": len(outcomes)}, sort_keys=True))


if __name__ == "__main__":
    main()
