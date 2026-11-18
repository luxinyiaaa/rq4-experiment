"""Final API scoring analyses from reviewed human labels and cached judge items."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openpyxl
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/api_scoring/full"
OUT = BASE / "final"
MODELS = ("egi", "openai", "openai_gpt56_sol", "openai_gpt6_astra")


def yes_no(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    canonical = str(value).strip().lower().replace("_", " ")
    if canonical in ("yes", "true"):
        return "yes"
    if canonical in ("no", "false"):
        return "no"
    if canonical in ("n/a", "none", ""):
        return "N/A"
    raise ValueError(f"Unexpected yes/no label: {value!r}")


def abstention(value):
    canonical = " ".join(str(value).strip().lower().replace("_", " ").split())
    if canonical not in ("answers", "abstains", "partly answers"):
        raise ValueError(f"Unexpected abstention label: {value!r}")
    return canonical


def rows_xlsx(path):
    sheet = openpyxl.load_workbook(path, read_only=True, data_only=True).active
    values = sheet.values
    heads = next(values)
    return [dict(zip(heads, row)) for row in values]


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def kappa(a, b, weighted=False):
    if not a:
        return None
    levels = sorted(set(a) | set(b), key=str)
    left, right = Counter(a), Counter(b)
    n = len(a)
    def weight(x, y):
        if not weighted:
            return float(x != y)
        return ((levels.index(x) - levels.index(y)) / max(1, len(levels) - 1)) ** 2
    observed = sum(weight(x, y) for x, y in zip(a, b)) / n
    expected = sum(weight(x, y) * left[x] * right[y] / n**2 for x in levels for y in levels)
    return 1 - observed / expected if expected else None


def agreement(a, b, metric):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"n": 0}
    a, b = zip(*pairs)
    out = {"n": len(a), "exact": sum(x == y for x, y in pairs) / len(a)}
    if metric == "relevance":
        out["within_one"] = sum(abs(int(x) - int(y)) <= 1 for x, y in pairs) / len(a)
        out["quadratic_kappa"] = kappa(a, b, weighted=True)
        out["spearman"] = float(spearmanr(a, b).statistic) if len(set(a)) > 1 and len(set(b)) > 1 else None
    else:
        out["kappa"] = kappa(a, b)
    return out


def labels_for(item, judge):
    result = next((x.get("labels") for x in item.get("judgements", []) if x.get("metric") == "answer_correctness_completeness" and x.get("judge") == judge), None)
    if result is None:
        raise ValueError(f"No keypoint labels: {item['id']} {item['model']} {judge}")
    return result


def main():
    OUT.mkdir(exist_ok=True)
    snapshot = json.loads((BASE / "input_snapshot.json").read_text())
    items = [json.loads((BASE / "items" / (key + ".json")).read_text()) for key in snapshot["answer_ids"]]
    if len(items) != 376:
        raise ValueError("Expected 376 full items")
    lookup = {(x["id"], x["model"]): x for x in items}
    key = list(csv.DictReader((BASE / "human_validation/human_validation_key.csv").open(newline="", encoding="utf-8")))
    human_path = Path(os.environ.get("HUMAN_LABEL_FILE", str(BASE / "human_validation/human_validation_filled.xlsx")))
    human = rows_xlsx(human_path)
    if len(key) != 40 or {x["sample_id"] for x in key} != {x["sample_id"] for x in human}:
        raise ValueError("Human-validation sample mismatch")
    human_by_id = {x["sample_id"]: x for x in human}
    wide = []
    for entry in key:
        row = human_by_id[entry["sample_id"]]
        item = lookup[(entry["answer_id"], entry["model"])]
        if row["question"] != item["question"] or row["answer_text"] != item["answer"]:
            raise ValueError("Human-validation question/answer mismatch")
        result = {"sample_id": entry["sample_id"], "model": entry["model"], "question": row["question"],
                  "my_correct": yes_no(row["correct"]), "my_relevance": int(row["relevance"]),
                  "my_abstention": abstention(row["abstention"]), "my_note": row.get("note") or ""}
        differences = []
        decisive = [i for i, kp in enumerate(item["keypoints"]) if kp.get("decisive")]
        for j, pos in enumerate(decisive, 1):
            value = yes_no(row[f"decisive_keypoint_{j}_present"])
            result[f"my_decisive_keypoint_{j}_present"] = value
        for judge in ("primary", "secondary"):
            result[f"{judge}_correct"] = "yes" if item["metrics"]["answer_correctness"][judge] else "no"
            rel = item["metrics"]["response_relevance"][judge]
            result[f"{judge}_relevance"] = rel["score"]
            result[f"{judge}_relevance_reason"] = rel["reason"]
            result[f"{judge}_abstention"] = abstention(item["metrics"]["appropriate_abstention"][judge]["label"])
            result[f"{judge}_abstention_reason"] = item["metrics"]["appropriate_abstention"][judge].get("reason", "")
            kp_labels = labels_for(item, judge)
            kp_trace = next(x.get("trace", []) for x in item["judgements"]
                if x.get("metric") == "answer_correctness_completeness" and x.get("judge") == judge)
            result[f"{judge}_correctness_reason"] = " | ".join(
                f"Keypoint {i + 1}: {entry.get('reason', '')}" for i, entry in enumerate(kp_trace)
                if (i in decisive and kp_labels[i] != "Entailment") or kp_labels[i] == "Contradiction")
            if not result[f"{judge}_correctness_reason"]:
                result[f"{judge}_correctness_reason"] = "All decisive keypoints entailed; no keypoint contradicted."
            for j, pos in enumerate(decisive, 1):
                result[f"{judge}_decisive_keypoint_{j}_present"] = "yes" if kp_labels[pos] == "Entailment" else "no"
                result[f"{judge}_decisive_keypoint_{j}_label"] = kp_labels[pos]
                result[f"{judge}_decisive_keypoint_{j}_reason"] = kp_trace[pos].get("reason", "")
            if result["my_correct"] != result[f"{judge}_correct"]: differences.append(f"{judge}:correct")
            if result["my_relevance"] != result[f"{judge}_relevance"]: differences.append(f"{judge}:relevance")
            if result["my_abstention"] != result[f"{judge}_abstention"]: differences.append(f"{judge}:abstention")
            for j in range(1, len(decisive) + 1):
                if result[f"my_decisive_keypoint_{j}_present"] != result[f"{judge}_decisive_keypoint_{j}_present"]:
                    differences.append(f"{judge}:decisive_keypoint_{j}")
        result["disagreement"] = ", ".join(differences)
        wide.append(result)
    (OUT / "human_vs_judges_data.json").write_text(json.dumps(wide, ensure_ascii=False, indent=2))

    human_stats = []
    for subset_name, subset in [("overall", wide)] + [(m, [r for r in wide if r["model"] == m]) for m in MODELS]:
        for judge in ("primary", "secondary"):
            for metric in ("correct", "abstention", "relevance"):
                stats = agreement([r[f"my_{metric}"] for r in subset], [r[f"{judge}_{metric}"] for r in subset], metric)
                human_stats.append({"group": subset_name, "judge": judge, "metric": metric, **stats})
            merged = lambda x: "answers" if x == "partly answers" else x
            stats = agreement([merged(r["my_abstention"]) for r in subset], [merged(r[f"{judge}_abstention"]) for r in subset], "abstention")
            human_stats.append({"group": subset_name, "judge": judge, "metric": "abstention_partly_merged", **stats})
            pairs = []
            for r in subset:
                n = sum(k.startswith("my_decisive_keypoint_") and k.endswith("_present") and r[k] in ("yes", "no") for k in r)
                for i in range(1, n + 1):
                    pairs.append((r[f"my_decisive_keypoint_{i}_present"], r[f"{judge}_decisive_keypoint_{i}_present"]))
            stats = agreement([x for x, _ in pairs], [y for _, y in pairs], "keypoint")
            human_stats.append({"group": subset_name, "judge": judge, "metric": "decisive_keypoint_present", **stats})
            for j in range(1, 9):
                keyname = f"my_decisive_keypoint_{j}_present"
                positioned = [r for r in subset if r.get(keyname) in ("yes", "no")]
                if positioned:
                    stats = agreement([r[keyname] for r in positioned],
                        [r[f"{judge}_decisive_keypoint_{j}_present"] for r in positioned], "keypoint")
                    human_stats.append({"group": subset_name, "judge": judge,
                        "metric": f"decisive_keypoint_{j}_present", **stats})
    write_csv(OUT / "human_agreement.csv", human_stats)
    distributions = []
    for abstaining, subset in (("abstaining", [r for r in wide if r["my_abstention"] == "abstains"]),
                               ("non_abstaining", [r for r in wide if r["my_abstention"] != "abstains"])):
        for label in ("my", "primary", "secondary"):
            counts = Counter(int(r[f"{label}_relevance"]) for r in subset)
            distributions.append({"human_abstention_group": abstaining, "rater": label,
                                  "n": len(subset), **{f"score_{v}": counts[v] for v in range(1, 6)}})
    write_csv(OUT / "relevance_distribution.csv", distributions)

    # Reviewed alternative decisions are applied only to the primary correctness label.
    reviewed = rows_xlsx(BASE / "alternative_answers_reviewed.xlsx")
    audit = {(r["answer_id"], r["model"]): r for r in csv.DictReader((BASE / "alternative_answer_audit.csv").open(newline="", encoding="utf-8"))}
    alt_rows = []
    decisions = {}
    for row in reviewed:
        keypair = row["answer_id"], row["model"]
        item = lookup[keypair]
        decision = str(row["my_decision"] or "").strip().lower()
        if decision not in ("correct", "incorrect"):
            raise ValueError(f"Missing alternative-answer decision: {keypair}")
        decisions[keypair] = decision == "correct"
        a = audit.get(keypair, {})
        if item.get("alternative_answer_source") == "full pilot recheck":
            source = "full run (pilot-answer recheck after pilot v2)"
        elif item.get("reused_from_pilot_v2"):
            source = ("original pilot, superseded after pilot v2 correctness" if
                      a.get("initial_alt_source") == "original pilot" and
                      not item.get("alternative_answer", {}).get("alternative_answer") else "pilot v2")
        else:
            source = "full run"
        alt_rows.append({"answer_id": row["answer_id"], "model": row["model"],
                         "primary_correct": item["metrics"]["answer_correctness"]["primary"],
                         "secondary_correct": item["metrics"]["answer_correctness"]["secondary"],
                         "current_alternative_answer": item.get("alternative_answer", {}).get("alternative_answer", ""),
                         "alternative_answer_source": source, "original_pilot_label": a.get("initially_listed", ""),
                         "my_decision": decision})
    write_csv(OUT / "alternative_label_check.csv", alt_rows)
    correction = []
    for model in MODELS:
        subset = [x for x in items if x["model"] == model]
        hits = sum(bool(x["metrics"]["answer_correctness"]["primary"]) for x in subset)
        changed = sum((decisions.get((x["id"], model), x["metrics"]["answer_correctness"]["primary"])) for x in subset)
        correction.append({"model": model, "n": len(subset), "judge_only_correct": hits,
                           "judge_only_rate": hits / len(subset), "after_manual_review_correct": changed,
                           "after_manual_review_rate": changed / len(subset)})
    write_csv(OUT / "alternative_correctness.csv", correction)

    # Use the first decisive keypoint with explicit correction language; otherwise the
    # first decisive keypoint, which is how this question set normally encodes the repair.
    questions = {x["id"]: x for x in (json.loads(line) for line in
        (ROOT / "outputs/rawpage_questions/question_set_v3/questions_main.jsonl").open(encoding="utf-8"))}
    false_ids = [qid for qid in snapshot["question_ids"] if str(questions[qid].get("false_premise", "")).strip()]
    correcting = {}
    for qid in false_ids:
        kps = questions[qid]["keypoints"]
        decisive = [i for i, kp in enumerate(kps) if kp.get("decisive")]
        candidates = [i for i in decisive if any(s in kps[i]["text"].lower() for s in (" not ", " while ", " instead", " rather than ", "incorrect", "wrong"))]
        if not decisive:
            raise ValueError(f"No decisive keypoint for {qid}")
        correcting[qid] = candidates[0] if candidates else decisive[0]
    write_csv(OUT / "false_premise_correcting_keypoints.csv", [
        {"question_id": qid, "question": questions[qid]["question"],
         "false_premise": questions[qid]["false_premise"], "correcting_keypoint_index": correcting[qid] + 1,
         "correcting_keypoint": questions[qid]["keypoints"][correcting[qid]]["text"]} for qid in false_ids])
    fp_rows = []
    for judge in ("primary", "secondary"):
        for model in MODELS:
            subset = [lookup[(qid, model)] for qid in false_ids]
            strict = lenient = answered_without = 0
            for item in subset:
                labels = labels_for(item, judge)
                idx = correcting[item["id"]]
                decisive = [i for i, kp in enumerate(item["keypoints"]) if kp.get("decisive")]
                others = [i for i in decisive if i != idx]
                no_contradiction = "Contradiction" not in labels
                other_entailed = all(labels[i] == "Entailment" for i in others)
                strict += bool(item["metrics"]["answer_correctness"][judge])
                lenient += no_contradiction and other_entailed
                answered_without += (item["primary_abstention_label"] != "abstains" and
                                     any(labels[i] == "Entailment" for i in others) and
                                     labels[idx] != "Entailment")
            n = len(subset)
            fp_rows.append({"model": model, "judge": judge, "n": n,
                            "strict_correct_n": strict, "strict_correct_rate": strict / n,
                            "lenient_correct_n": lenient, "lenient_correct_rate": lenient / n,
                            "answered_without_correction_n": answered_without,
                            "answered_without_correction_rate": answered_without / n})
    write_csv(OUT / "false_premise_sensitivity.csv", fp_rows)

    summary = list(csv.DictReader((BASE / "summary.csv").open(newline="", encoding="utf-8")))
    judge_agree = list(csv.DictReader((BASE / "judge_agreement.csv").open(newline="", encoding="utf-8")))
    def md(rows, keys):
        return "| " + " | ".join(keys) + " |\n|" + "|".join("---" for _ in keys) + "|\n" + "\n".join(
            "| " + " | ".join(str(r.get(k, "")).replace("|", "\\|").replace("\n", " ").strip() for k in keys) + " |" for r in rows)
    primary = [r for r in summary if r["grouping"] == "model" and r["judge"] == "primary"]
    def formatted_metric(row, metric):
        value = row.get(metric)
        if value in ("", "None", None):
            return "—"
        ci = row.get(metric + "_ci95", "")
        interval = ""
        if ci not in ("", "None"):
            values = json.loads(ci.replace("(", "[").replace(")", "]"))
            interval = f" [{values[0]:.3f}, {values[1]:.3f}]"
        return f"{float(value):.3f} (n={row[metric+'_n']})" + interval
    main_rows = []
    for row in primary:
        main_rows.append({"model": row["model"], "n": row["n"],
             **{metric: formatted_metric(row, metric)
                for metric in ("answer_correctness", "completeness", "response_relevance", "citation_support",
                               "citation_completeness", "false_abstention_rate", "overall_abstention_rate",
                               "partly_answers_rate")}})
    filed_path = OUT / "human_agreement_as_filed.csv"
    if filed_path.exists():
        filed_stats = list(csv.DictReader(filed_path.open(newline="", encoding="utf-8")))
        filed_lookup = {(r["group"], r["judge"], r["metric"]): r for r in filed_stats}
        comparison_rows = []
        for corrected in human_stats:
            filed = filed_lookup[(corrected["group"], corrected["judge"], corrected["metric"])]
            comparison_rows.append({"group": corrected["group"], "judge": corrected["judge"],
                "metric": corrected["metric"], "n": corrected["n"],
                **{f"as_filed_{k}": filed.get(k, "") for k in ("exact", "within_one", "kappa", "quadratic_kappa", "spearman")},
                **{f"corrected_{k}": corrected.get(k, "") for k in ("exact", "within_one", "kappa", "quadratic_kappa", "spearman")}})
        write_csv(OUT / "human_agreement_as_filed_vs_corrected.csv", comparison_rows)
    else:
        comparison_rows = []
    original_hv004 = next(r for r in rows_xlsx(BASE / "human_validation/human_validation_filled.xlsx") if r["sample_id"] == "HV004")
    revised_hv004 = human_by_id["HV004"]
    hv004_changes = [{"field": k, "as_filed": original_hv004[k], "corrected": revised_hv004[k]}
        for k in ("correct", "decisive_keypoint_2_present", "note")]
    def display(row):
        return {k: (f"{float(v):.3f}" if k.startswith(("as_filed_", "corrected_")) and v not in ("", None, "None")
                    else "—" if v in ("", None, "None") else v)
                for k, v in row.items()}
    lines = ["# Final API-answer analysis", "", "## Main results (primary judge)", "",
             md(main_rows, ["model", "n", "answer_correctness", "completeness", "response_relevance", "citation_support", "citation_completeness", "false_abstention_rate", "overall_abstention_rate", "partly_answers_rate"]), "",
             "These are the aggregate scores for the 94 complete questions; relevance uses the final rubric.", "",
             "## Correctness after alternative-answer review", "",
             md([{**r, "judge_only_rate": f"{r['judge_only_rate']:.3f}", "after_manual_review_rate": f"{r['after_manual_review_rate']:.3f}"} for r in correction],
                ["model", "n", "judge_only_correct", "judge_only_rate", "after_manual_review_correct", "after_manual_review_rate"]), "",
             "The reviewed decisions replace only the primary correctness label for the listed answers.", "",
             "## Reviewed alternative-answer label check", "",
             md(alt_rows, ["answer_id", "model", "primary_correct", "secondary_correct",
                           "current_alternative_answer", "original_pilot_label", "alternative_answer_source", "my_decision"]), "",
             "Pilot answers use pilot v2 correctness; where a pilot answer was rechecked for an alternative answer after pilot v2, its current alternative label is from that later full-run recheck.", "",
             "## Human-label normalisation and HV004 correction", "",
             "Normalised the judges' `partly_answers` to the human label `partly answers` in agreement, kappa and disagreement calculations (3 primary and 1 secondary labels in the sample). Correctness and decisive-keypoint labels already used the same yes/no meaning after their existing mappings; relevance scores were numeric. No other case, space, underscore or true/false mismatch changed a label.", "",
             md(hv004_changes, ["field", "as_filed", "corrected"]), "",
             "HV004 was corrected from yes to no for correctness and decisive keypoint 2, following the original human note that the answer did not identify international PhD students as the primary target group.", "",
             "## Human-validation agreement: as filed and corrected", "",
             md([display(r) for r in comparison_rows],
                ["group", "judge", "metric", "n", "as_filed_exact", "corrected_exact", "as_filed_within_one", "corrected_within_one", "as_filed_kappa", "corrected_kappa", "as_filed_quadratic_kappa", "corrected_quadratic_kappa", "as_filed_spearman", "corrected_spearman"]), "",
             "Agreement compares 40 blind human labels with each judge after formatting normalisation; decisive-keypoint agreement pools applicable keypoints within each group and also reports each position separately.", "",
             "## Relevance score distribution by human abstention status", "",
             md(distributions, ["human_abstention_group", "rater", "n", "score_1", "score_2", "score_3", "score_4", "score_5"]), "",
             "These distributions separate answers the human marked as abstaining from the other answers.", "",
             "## Judge agreement", "",
             md([{k: (f"{float(v):.3f}" if k in ("exact_agreement", "cohen_kappa", "quadratic_weighted_kappa", "spearman", "agreement_within_one_point") and v not in ("", "None") else v) for k, v in r.items()} for r in judge_agree],
                ["metric", "n_pairs", "exact_agreement", "cohen_kappa", "quadratic_weighted_kappa", "spearman", "agreement_within_one_point"]), "",
             "Judge agreement describes the consistency of the two independent scoring passes across all 376 answers.", ""]
    (OUT / "final_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"human_rows": len(wide), "reviewed_alternatives": len(alt_rows),
                      "false_premise_questions": len(false_ids)}, sort_keys=True))


if __name__ == "__main__":
    main()
