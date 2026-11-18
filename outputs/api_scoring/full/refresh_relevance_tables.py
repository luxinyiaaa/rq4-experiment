"""Refresh relevance-bearing reports without regenerating blind sample material."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from outputs.api_scoring.full import report_full as api
from outputs.metrics_v2 import revise_outputs as s1


def usage(path):
    totals = {"judge_primary": Counter(), "judge_secondary": Counter()}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            event = json.loads(line)
            role = event["role"]
            totals[role]["calls"] += 1
            totals[role]["input_tokens"] += event.get("input_tokens", 0)
            totals[role]["output_tokens"] += event.get("output_tokens", 0)
    return {k: dict(v) for k, v in totals.items()}


def refresh_api():
    out = api.OUT
    snapshot = json.loads((out / "input_snapshot.json").read_text())
    items = [json.loads((api.ITEMS / (key + ".json")).read_text()) for key in snapshot["answer_ids"]]
    if len(items) != 376:
        raise ValueError("Expected 376 API items")
    # Every reused pilot answer must still carry the pilot v2 correctness result.
    pilot = ROOT / "outputs/api_scoring/pilot_v2/items"
    mismatches = []
    for item in items:
        source = pilot / (item["id"] + "__" + item["model"] + ".json")
        if not source.exists():
            continue
        old = json.loads(source.read_text())
        for judge in ("primary", "secondary"):
            if item["metrics"]["answer_correctness"][judge] != old["metrics"]["answer_correctness"][judge]:
                mismatches.append((item, judge, old["metrics"]["answer_correctness"][judge]))
    for item, judge, correction in mismatches:
        item["metrics"]["answer_correctness"][judge] = correction
        api.ITEMS.joinpath(item["id"] + "__" + item["model"] + ".json").write_text(json.dumps(item, ensure_ascii=False, indent=2))
    summary = api.summarise(items)
    comp = api.comparisons(items)
    agreement = api.agreement(items)
    bias = api.bias(items)
    outside = list(csv.DictReader((out / "cited_urls_outside_kb.csv").open(newline="", encoding="utf-8")))
    alternatives = [x for x in items if x.get("alternative_answer", {}).get("alternative_answer") == "yes"]
    previous_usage = json.loads((out / "usage.json").read_text())
    sample_count = len(list(csv.DictReader((out / "human_validation/human_validation_key.csv").open(newline="", encoding="utf-8"))))
    api.report(snapshot, summary, comp, agreement, bias, outside, alternatives, previous_usage, [None] * sample_count)
    new_usage = usage(out / "relevance_v2_progress/judge_calls.jsonl")
    (out / "relevance_rescore_usage.json").write_text(json.dumps(new_usage, indent=2))
    with (out / "report.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Relevance rubric revision\n\n")
        stream.write("Relevance was re-scored for all 376 answers by both judges with refined anchors after a first human pass showed that the original anchors were ambiguous for abstentions and for additional content. Other metric labels were retained.\n\n")
        stream.write("New relevance calls and returned tokens: " + ", ".join(
            f"{role} {data.get('calls', 0)} calls, {data.get('input_tokens', 0) + data.get('output_tokens', 0)} tokens"
            for role, data in new_usage.items()) + ".\n")
    # The key is a label record for the existing blind sample. Refresh relevance only.
    path = out / "human_validation/human_validation_key.csv"
    key = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    lookup = {(x["id"], x["model"]): x for x in items}
    for row in key:
        labels = json.loads(row["judge_labels"])
        item = lookup[(row["answer_id"], row["model"])]
        for judge in ("primary", "secondary"):
            labels[judge]["response_relevance"] = item["metrics"]["response_relevance"][judge]["score"]
        row["judge_labels"] = json.dumps(labels, ensure_ascii=False)
    api.write_csv(path, key, ["sample_id", "answer_id", "model", "judge_labels"])
    return {"correctness_fixes": len(mismatches), "relevance_usage": new_usage}


def refresh_s1():
    records = s1.read_jsonl(s1.ITEMS / "judgements.jsonl")
    cond, by_type = s1.condition_tables(records)
    robust = s1.paired_table(records, "C2")
    loss = s1.paired_table(records, "C4")
    agree = s1.agreement_table(records)
    discr = s1.discrimination_table(records)
    comparison = s1.correct_vs_incorrect(records)
    for name, rows in (("metric_by_condition_S1.csv", cond), ("metric_by_type_C4_S1.csv", by_type),
                       ("robustness_paired_S1.csv", robust), ("retrieval_loss_S1.csv", loss),
                       ("judge_agreement.csv", agree), ("discrimination.csv", discr),
                       ("correct_vs_incorrect_C4.csv", comparison)):
        s1.write_csv(s1.SUMMARY / name, rows)
    s1.write_csv(s1.ITEMS / "answer_metrics_wide.csv", s1.build_wide(records))
    (s1.SUMMARY / "main_table_S1.md").write_text(s1.condition_main_table(records), encoding="utf-8")
    outliers = s1.read_jsonl(s1.ITEMS / "faithfulness_outliers.jsonl")
    outlier_counts = {"total": len(outliers),
        "by_condition": dict(Counter(str(x.get("condition", "")) for x in outliers)),
        "by_rule": dict(Counter(rule for x in outliers for rule in x.get("outlier_rules", []))),
        "trace_api_calls": 0}
    calls, tokens, parse_failures = s1.usage()
    attribution = s1.read_csv_rows(s1.SUMMARY / "c4_error_attribution_S1.csv")
    summary = s1.report(records, cond, by_type, robust, loss, agree, discr, attribution,
                        comparison, outlier_counts, calls, tokens, parse_failures)
    (s1.SUMMARY / "metrics_v2_summary.md").write_text(summary, encoding="utf-8")
    old = s1.read_csv_rows(s1.OUT / "relevance_v1_archive/metric_by_condition_S1.csv")
    old_map = {(r["judge"], r["metric"]): r for r in old}
    comparison_rows = []
    for row in cond:
        if row["metric"] != "Response relevance":
            continue
        prior = old_map[(row["judge"], row["metric"])]
        for condition in s1.CONDITIONS:
            comparison_rows.append({"judge": row["judge"], "condition": condition,
                "n": row[f"{condition}_n_applicable"],
                "old_relevance": prior[f"{condition}_estimate"],
                "new_relevance": row[f"{condition}_estimate"],
                "difference": float(row[f"{condition}_estimate"]) - float(prior[f"{condition}_estimate"])})
    s1.write_csv(s1.SUMMARY / "relevance_old_vs_new_S1.csv", comparison_rows)
    new_usage = usage(s1.OUT / "relevance_v2_progress/judge_calls.jsonl")
    (s1.OUT / "relevance_rescore_usage.json").write_text(json.dumps(new_usage, indent=2))
    with (s1.SUMMARY / "metrics_v2_summary.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Relevance rubric revision\n\nAll 176 S1 answers were re-scored for relevance by both judges with the final rubric. The prior values are archived under `relevance_v1_archive/`.\n\n")
        stream.write("| Judge | Condition | Old relevance | New relevance | Change |\n|---|---|---:|---:|---:|\n")
        for row in comparison_rows:
            stream.write(f"| {row['judge']} | {row['condition']} | {float(row['old_relevance']):.3f} | {float(row['new_relevance']):.3f} | {row['difference']:+.3f} |\n")
        stream.write("\nNew relevance calls and returned tokens: " + ", ".join(
            f"{role} {data.get('calls', 0)} calls, {data.get('input_tokens', 0) + data.get('output_tokens', 0)} tokens"
            for role, data in new_usage.items()) + ".\n")
    return {"relevance_old_vs_new": comparison_rows, "relevance_usage": new_usage}


def main():
    print(json.dumps({"api": refresh_api(), "s1": refresh_s1()}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
