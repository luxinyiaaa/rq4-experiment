"""Aggregate the completed 300-question scoring run without changing source results."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from outputs.api_scoring.full import report_full as rep
from scripts.analyze_metrics_v2 import wilson

OUT = Path(os.environ.get("RERUN_SCORING_DIR", Path(__file__).resolve().parent))
FINAL = OUT / "final"
OLD = ROOT / "outputs/api_scoring/full"
MODELS = rep.MODELS
METRICS = (*rep.METRICS, "false_abstention_rate", "overall_abstention_rate", "partly_answers_rate")


def md(rows, keys):
    return rep.markdown_table([{k: ("—" if row.get(k) is None else str(row.get(k, "")))
                                .replace("|", "\\|").replace("\n", " ")
                                for k in keys} for row in rows], keys)


def item_key(item):
    return item["id"] + "__" + item["model"]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fixed_summary(items):
    rep.OUT = OUT
    rows = rep.summarise(items)
    for row in rows:
        if row["judge"] != "secondary":
            continue
        subset = [x for x in items if x["model"] == row["model"] and
                  (row["grouping"] == "model" or x[row["grouping"]] == row["group"])]
        labels = [x["metrics"]["appropriate_abstention"]["secondary"]["label"] for x in subset]
        for name, label in (("false_abstention_rate", "abstains"),
                            ("overall_abstention_rate", "abstains"),
                            ("partly_answers_rate", "partly_answers")):
            hits = labels.count(label)
            row[name] = hits / len(labels)
            row[name + "_n"] = len(labels)
            row[name + "_ci95"] = wilson(hits, len(labels)) if len(labels) >= 10 else None
    rep.write_csv(OUT / "summary.csv", rows)
    return rows


def usage():
    events = [json.loads(line) for line in (OUT / "judge_calls.jsonl").open() if line.strip()]
    by_role = defaultdict(Counter)
    for event in events:
        role = event["role"]
        by_role[role]["calls"] += 1
        by_role[role]["input_tokens"] += event["input_tokens"]
        by_role[role]["output_tokens"] += event["output_tokens"]
    result = {"by_judge": dict(by_role), "new_calls": len(events),
              "new_tokens": sum(x["input_tokens"] + x["output_tokens"] for x in events)}
    write_json(OUT / "usage.json", result)
    return result


def alternatives(items, snapshot):
    rows = []
    flagged = Counter()
    for item in items:
        alt = item.get("alternative_answer") or {}
        if alt.get("alternative_answer") != "yes":
            continue
        flagged[item["model"]] += 1
        rows.append({"answer_id": item["id"], "model": item["model"],
                     "question": item["question"], "reference_answer": item["reference_answer"],
                     "decisive_keypoints": "\n".join(p["text"] for p in item["keypoints"] if p.get("decisive")),
                     "answer": item["answer"], "judge_reason": alt.get("reason", ""), "my_decision": ""})
    write_json(OUT / "alternative_answers_to_review_data.json", rows)
    return rows, [], flagged


def false_premise(items, snapshot):
    questions = {x["id"]: x for x in (json.loads(line) for line in
        (ROOT / "outputs/rawpage_questions/question_set_v3/questions_main.jsonl").open())}
    ids = [qid for qid in snapshot["question_ids"] if str(questions[qid].get("false_premise", "")).strip()]
    lookup = {(x["id"], x["model"]): x for x in items}
    correcting = {}
    kp_rows = []
    for qid in ids:
        q = questions[qid]
        decisive = [i for i, kp in enumerate(q["keypoints"]) if kp.get("decisive")]
        candidates = [i for i in decisive if any(s in q["keypoints"][i]["text"].lower()
                      for s in (" not ", " while ", " instead", " rather than ", "incorrect", "wrong"))]
        if not decisive:
            raise ValueError("False-premise item lacks decisive keypoint: " + qid)
        correcting[qid] = candidates[0] if candidates else decisive[0]
        kp_rows.append({"question_id": qid, "question": q["question"],
                        "false_premise": q["false_premise"],
                        "correcting_keypoint_index": correcting[qid] + 1,
                        "correcting_keypoint": q["keypoints"][correcting[qid]]["text"]})
    rep.write_csv(OUT / "false_premise_correcting_keypoints.csv", kp_rows)
    strict_rows, sensitivity, outcomes = [], [], []
    for judge in ("primary", "secondary"):
        for model in MODELS:
            strict = old_lenient = intended = without = 0
            cats = Counter()
            for qid in ids:
                item = lookup[qid, model]
                labels = next(x["labels"] for x in item["judgements"] if
                              x.get("metric") == "answer_correctness_completeness" and x.get("judge") == judge)
                idx = correcting[qid]
                decisive = [i for i, kp in enumerate(item["keypoints"]) if kp.get("decisive")]
                correction = labels[idx]
                other_entailed = all(labels[i] == "Entailment" for i in decisive if i != idx)
                no_contradiction = "Contradiction" not in labels
                correct = bool(item["metrics"]["answer_correctness"][judge])
                strict += correct
                old_lenient += other_entailed and no_contradiction
                intended += (correct if qid == "AS-055-e61cb36aa7_2" else
                             correction in ("Entailment", "Neutral") and other_entailed and no_contradiction)
                abst = item["primary_abstention_label"]
                cats[("corrects the premise" if correction == "Entailment" else
                      "accepts the premise" if correction == "Contradiction" else
                      "abstains" if abst == "abstains" else "silent on the premise")] += 1
                without += (abst != "abstains" and correction != "Entailment" and
                            any(labels[i] == "Entailment" for i in decisive if i != idx))
            n = len(ids)
            strict_rows.append({"model": model, "judge": judge, "n": n,
                                "strict_correct_n": strict, "strict_correct_rate": strict / n})
            sensitivity.append({"model": model, "judge": judge, "n": n,
                                "strict": f"{strict}/{n} ({strict/n:.1%})",
                                "previous_lenient": f"{old_lenient}/{n} ({old_lenient/n:.1%})",
                                "intended_lenient": f"{intended}/{n} ({intended/n:.1%})",
                                "answers_without_correction": f"{without}/{n} ({without/n:.1%})"})
            outcomes.append({"model": model, "judge": judge, "n": n, **{
                cat: f"{cats[cat]}/{n} ({cats[cat]/n:.1%})" for cat in
                ("corrects the premise", "accepts the premise", "silent on the premise", "abstains")}})
    rep.write_csv(OUT / "false_premise_strict.csv", strict_rows)
    sup = FINAL / "supplementary/false_premise_sensitivity.md"
    sup.parent.mkdir(exist_ok=True)
    sup.write_text("\n".join(["# False-premise sensitivity", "",
        f"This analysis covers {len(ids)} false-premise questions. The primary abstention label separates silent from abstaining answers for both judges.", "",
        "## Strict and lenient correctness", "",
        "The lenient rule permits a Neutral premise-correction keypoint if every other decisive keypoint is entailed and no keypoint is contradicted. When the correction is itself the requested answer (AS-055-e61cb36aa7_2), lenient correctness equals strict correctness.", "",
        md(sensitivity, ["model", "judge", "n", "strict", "previous_lenient", "intended_lenient", "answers_without_correction"]), "",
        "## Four premise outcomes", "",
        md(outcomes, ["model", "judge", "n", "corrects the premise", "accepts the premise", "silent on the premise", "abstains"]), "",
        "## Correcting keypoints", "",
        md(kp_rows, ["question_id", "correcting_keypoint", "false_premise"]), ""]), encoding="utf-8")
    return strict_rows


def relevance(items):
    low = []
    nonabst = []
    for item in items:
        p = item["metrics"]["response_relevance"]["primary"]
        s = item["metrics"]["response_relevance"]["secondary"]
        if min(p["score"], s["score"]) <= 3:
            low.append({"experiment": "API", "answer_id": item["id"], "model_or_condition": item["model"],
                        "question": item["question"], "full_answer_text": item["answer"],
                        "primary_score": p["score"], "primary_reason": p["reason"],
                        "secondary_score": s["score"], "secondary_reason": s["reason"],
                        "primary_abstention_label": item["primary_abstention_label"]})
    for model in MODELS:
        subset = [x for x in items if x["model"] == model and
                  x["primary_abstention_label"] in ("answers", "partly_answers")]
        for judge in ("primary", "secondary"):
            vals = [x["metrics"]["response_relevance"][judge]["score"] for x in subset]
            rng = np.random.default_rng(42)
            draws = np.asarray(vals)[rng.integers(0, len(vals), (10000, len(vals)))].mean(axis=1) if len(vals) >= 10 else []
            ci = np.quantile(draws, [.025, .975]) if len(vals) >= 10 else None
            nonabst.append({"experiment": "API", "model_or_condition": model, "judge": judge, "n": len(vals),
                            "mean_relevance": float(np.mean(vals)) if vals else None,
                            "ci95_lower": float(ci[0]) if ci is not None else None,
                            "ci95_upper": float(ci[1]) if ci is not None else None})
    prior_low = json.loads((OLD / "final/relevance_low_scores_data.json").read_text())
    low.extend(x for x in prior_low if x["experiment"] == "S1")
    prior_nonabst = list(csv.DictReader((OLD / "final/relevance_nonabstaining.csv").open(newline="", encoding="utf-8")))
    nonabst.extend(x for x in prior_nonabst if x["experiment"] == "S1")
    write_json(OUT / "relevance_low_scores_data.json", low)
    rep.write_csv(OUT / "relevance_nonabstaining.csv", nonabst)
    return low, nonabst


def main():
    snapshot = json.loads((OUT / "input_snapshot.json").read_text())
    if len(snapshot["answer_ids"]) != 1200 or len(set(snapshot["answer_ids"])) != 1200:
        raise ValueError("Refusing final reports: fewer than 1,200 distinct answers are snapshotted")
    from outputs.api_scoring.rerun_kbfix.run_rerun import complete_jsonl, ANSWER_SOURCE, answer_checks
    answers = complete_jsonl(ANSWER_SOURCE)
    valid = {(x.get("id"), x.get("model_name")): x for x in answers if answer_checks.is_success(x)}
    if (len(answers) != 1200 or len(valid) != 1200 or
        {qid + "__" + model for qid, model in valid} != set(snapshot["answer_ids"]) or
        any(hashlib.sha256(row["answer"].encode("utf-8")).hexdigest() !=
            snapshot["answer_text_sha256"].get(qid + "__" + model)
            for (qid, model), row in valid.items())):
        raise ValueError("Refusing final reports: collection is incomplete or differs from the snapshot")
    keys = snapshot["answer_ids"]
    paths = [OUT / "items" / (key + ".json") for key in keys]
    missing = [key for key, path in zip(keys, paths) if not path.exists()]
    if missing:
        raise ValueError(f"Scoring incomplete: {len(missing)} items missing")
    items = [json.loads(path.read_text()) for path in paths]
    if len(items) != 1200 or {item_key(x) for x in items} != set(keys):
        raise ValueError("Item IDs do not match the input snapshot")
    for item in items:
        if hashlib.sha256(item["answer"].encode("utf-8")).hexdigest() != snapshot["answer_text_sha256"][item_key(item)]:
            raise ValueError("Scored item text differs from the input snapshot: " + item_key(item))
        if set(item["metrics"]) != set((*rep.METRICS, "appropriate_abstention")):
            raise ValueError("Incomplete metrics: " + item_key(item))
        if any(judge not in item["metrics"][metric] for metric in (*rep.METRICS, "appropriate_abstention")
               for judge in ("primary", "secondary")):
            raise ValueError("Missing judge result: " + item_key(item))
    FINAL.mkdir(exist_ok=True)
    summary = fixed_summary(items)
    comp = rep.comparisons(items)
    agree = rep.agreement(items)
    family = rep.bias(items)
    outside = rep.outside_urls(items)
    alt_rows, carried, flagged = alternatives(items, snapshot)
    strict = false_premise(items, snapshot)
    low, nonabst = relevance(items)
    used = usage()
    main_rows = [x for x in summary if x["grouping"] == "model"]
    lines = ["# Full API-answer scoring: 300 main-set questions", "",
             "The 1,200 answer texts are fixed by `input_snapshot.json`. The primary judge is GPT-6 Sol and the secondary judge is Qwen3.7 Plus, both at temperature 0. Judging uses the final relevance rubric and pilot v2 keypoint rules. Citation support groups statements by cited URL set and uses matching full-page KB rows as evidence, with selected chunks and neighbours for long pages. Correctness is judge-only until the new alternative answers are reviewed. Citation applicability is gated by the primary judge's abstention label; secondary false abstention is calculated from the secondary judge's own labels. Intervals are Wilson 95% for rates and bootstrap 95% for means.", ""]
    for grouping, title in (("model", "By model"), ("type", "By question type"), ("page_type", "By page type")):
        lines.extend(["## " + title, ""])
        for judge in ("primary", "secondary"):
            rows = [{"Model": r["model"], "Group": r["group"], "n": r["n"], **{
                rep.LABELS[m]: rep.cell(r, m) for m in METRICS}} for r in summary
                if r["grouping"] == grouping and r["judge"] == judge]
            lines.extend(["### " + judge.title() + " judge", "",
                          md(rows, ["Model", "Group", "n", *(rep.LABELS[m] for m in METRICS)]), ""])
    (OUT / "main_table.md").write_text("\n".join(lines[:lines.index("## By question type")]), encoding="utf-8")
    comparison_rows = [{"Judge": x["judge"], "Metric": x["metric"],
        "Pair": x["model_a"] + " − " + x["model_b"], "n": x["n_pairs"],
        "Difference (95% CI)": f"{x['mean_paired_difference']:+.3f}" +
            (f" [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" if x["ci95"] else ""),
        "p": f"{x['p']:.4g}", "Holm p": f"{x['p_holm']:.4g}"} for x in comp]
    lines.extend(["## Paired model comparisons", "",
        "Positive differences favor the first model. Correctness uses paired Newcombe intervals and exact McNemar tests; other metrics use paired bootstrap intervals and Wilcoxon signed-rank tests. Holm adjustment is across six pairs per judge and metric.", "",
        md(comparison_rows, ["Judge", "Metric", "Pair", "n", "Difference (95% CI)", "p", "Holm p"]), ""])
    lines.extend(["## Judge agreement", "", md(agree, ["metric", "n_pairs", "exact_agreement", "exact_agreement_ci95", "cohen_kappa", "quadratic_weighted_kappa", "spearman", "agreement_within_one_point", "mean_absolute_difference"]), "",
                  "## Family-bias check", "", md(family, ["metric", "model", "n", "primary_minus_secondary", "ci95"]), ""])
    counts = Counter(x["model"] for x in outside)
    lines.extend(["## Cited URLs outside the KB", "",
                  md([{"model": m, "instances": counts[m], "unique_urls": len({x["cited_url"] for x in outside if x["model"] == m})} for m in MODELS], ["model", "instances", "unique_urls"]), "",
                  "## Alternative answers and usage", "",
                  f"Flagged answers: {dict(flagged)}. {len(alt_rows)} candidates need review. No previous manual decisions were carried over or applied; correctness is judge-only.", "",
                  f"Judge calls in this rerun: {used['new_calls']:,}; returned input and output tokens: {used['new_tokens']:,}.", "",
                  "The 40-answer human validation was not repeated: it assesses agreement between the human labels and the judges on the original answers, not correctness of these new answers.", ""])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    human_source = OLD / "final/human_agreement_as_filed_vs_corrected.csv"
    human = list(csv.DictReader(human_source.open(newline="", encoding="utf-8")))
    shutil.copyfile(human_source, FINAL / "human_agreement_as_filed_vs_corrected.csv")
    final = ["# Final API-answer analysis: 300 main-set questions", "",
             "## Main results (primary judge)", "",
             md([{"model": r["model"], "n": r["n"], **{m: rep.cell(r, m) for m in METRICS}}
                 for r in main_rows if r["judge"] == "primary"], ["model", "n", *METRICS]), "",
             "Correctness is judge-only pending review of the new flagged alternative answers.", "",
             "## Main results (secondary judge)", "",
             md([{"model": r["model"], "n": r["n"], **{m: rep.cell(r, m) for m in METRICS}}
                 for r in main_rows if r["judge"] == "secondary"], ["model", "n", *METRICS]), "",
             "Secondary false abstention uses the secondary judge's own abstention labels.", "",
             "## False-premise correctness (strict)", "",
             md([{"model": r["model"], "judge": r["judge"], "n": r["n"], "strict_correct": f"{r['strict_correct_n']}/{r['n']} ({r['strict_correct_rate']:.1%})"} for r in strict], ["model", "judge", "n", "strict_correct"]), "",
             "Lenient false-premise analysis and four outcomes are in `supplementary/false_premise_sensitivity.md`.", "",
             "## Human-validation agreement", "",
             "The 40-answer sample and its human labels are unchanged from the 94-question analysis. This validation was not repeated; agreement refers to the original answers and validates the judges, not the new answers.", "",
             md(human, ["group", "judge", "metric", "n", "as_filed_exact", "corrected_exact", "as_filed_within_one", "corrected_within_one", "as_filed_kappa", "corrected_kappa", "as_filed_quadratic_kappa", "corrected_quadratic_kappa", "as_filed_spearman", "corrected_spearman"]), "",
             "## Judge agreement", "",
             md(agree, ["metric", "n_pairs", "exact_agreement", "cohen_kappa", "quadratic_weighted_kappa", "spearman", "agreement_within_one_point"]), "",
             "## Low relevance scores", "",
             md([{"experiment": exp, "model_or_condition": m,
                  "n_low": sum(x["experiment"] == exp and x["model_or_condition"] == m for x in low)}
                 for exp, names in (("API", MODELS), ("S1", ("C1", "C2", "C3", "C4"))) for m in names],
                ["experiment", "model_or_condition", "n_low"]), "",
             f"Either judge scored relevance at 3 or below for {len(low)} answers. The item-level list is in `relevance_low_scores.xlsx`.", "",
             "## Relevance among non-abstaining answers", "",
             md([{"experiment": r["experiment"], "model_or_condition": r["model_or_condition"],
                  "judge": r["judge"], "n": r["n"],
                  "mean_relevance": f"{float(r['mean_relevance']):.3f}" if r["mean_relevance"] not in (None, "") else "—",
                  "95% CI": f"[{float(r['ci95_lower']):.3f}, {float(r['ci95_upper']):.3f}]" if r["ci95_lower"] not in (None, "") else "—"} for r in nonabst],
                ["experiment", "model_or_condition", "judge", "n", "mean_relevance", "95% CI"]), "",
             "Non-abstaining means `answers` or `partly_answers` under the primary judge's shared gate. Intervals use 10,000 bootstrap draws (seed 42).", ""]
    (FINAL / "final_results.md").write_text("\n".join(final), encoding="utf-8")
    print(json.dumps({"items": len(items), "new_calls": used["new_calls"], "new_tokens": used["new_tokens"],
                      "alternatives_to_review": len(alt_rows), "carried_decisions": len(carried),
                      "low_relevance": len(low)}, sort_keys=True))


if __name__ == "__main__":
    main()
