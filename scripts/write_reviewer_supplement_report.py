"""Write aggregate reviewer notes and their separate item-level appendix."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final"
sys.path.insert(0, str(ROOT))
from scripts.complete_supplemental_analysis import read_inline_xlsx


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "No rows."
    values = [columns] + [["—" if pd.isna(row[c]) else str(row[c]) for c in columns] for _, row in frame.iterrows()]
    widths = [max(len(str(row[i])) for row in values) for i in range(len(columns))]
    lines = ["| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values[0])) + " |",
             "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"]
    lines.extend("| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |" for row in values[1:])
    return "\n".join(lines)


def manual_overlap_counts() -> tuple[dict[str, int], int]:
    """Read the key only for aggregate overlap counts; never return its contents."""
    key_path = OUT / "human_validation" / "answers_key.csv"
    with key_path.open(encoding="utf-8", newline="") as stream:
        key_rows = list(csv.DictReader(stream))
    blind_answer_rows = [r for r in key_rows if not str(r.get("form", "")).strip()]
    blind_pairs = {(r.get("question_id", ""), r.get("condition", "")) for r in blind_answer_rows}
    groups = {
        "needs_manual_check": {(r["id"], r["condition"]) for r in csv.DictReader((OUT / "needs_manual_check_final.csv").open(encoding="utf-8", newline=""))},
        "correctness_v2_disagreement": {(r["id"], r["condition"]) for r in csv.DictReader((OUT / "correctness_disagreements.csv").open(encoding="utf-8", newline=""))},
        "faithfulness_retry": {(r["id"], r["condition"]) for r in (json.loads(line) for line in (OUT / "faithfulness_retry_4096.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())},
    }
    counts = {name: len(pairs & blind_pairs) for name, pairs in groups.items()}
    return counts, len(blind_answer_rows)


def build_reports() -> tuple[str, str, dict[str, int]]:
    summary = pd.read_csv(OUT / "score_summary_final.csv")
    paired = pd.read_csv(OUT / "paired_tests_final.csv")
    retrieval = pd.read_csv(OUT / "paraphrase_retrieval_summary.csv")
    closed = pd.read_csv(OUT / "closed_book_correctness_crosscheck.csv")
    disagreements = pd.read_csv(OUT / "correctness_disagreements.csv")
    needs = pd.read_csv(OUT / "needs_manual_check_final.csv")
    faith = pd.read_json(OUT / "faithfulness_retry_4096.jsonl", lines=True)
    quote_split = pd.read_csv(OUT / "quote_split_by_c4_accuracy.csv")
    rule_counts: Counter[str] = Counter()
    for rules in needs["rules"].fillna(""):
        rule_counts.update(rule.strip() for rule in str(rules).split(";") if rule.strip())
    unique_needs = len(needs)
    overlap, blind_n = manual_overlap_counts()

    condition_rows = []
    for analysis_set in ("all_44", "main_34"):
        part = summary[(summary.analysis_set == analysis_set) & (summary.table == "condition") & summary.condition.isin(["C1", "C2", "C4"])]
        for condition in ("C1", "C2", "C4"):
            rows = part[part.condition == condition].set_index("metric")
            row: dict[str, Any] = {"analysis_set": analysis_set, "condition": condition}
            for metric in ("correctness", "correctness_v2", "abstention_rate", "completeness", "flask", "faithfulness", "gold_chunk_hit_at_10", "gold_page_hit_at_10", "context_recall"):
                if metric in rows.index:
                    row[metric] = f"{rows.loc[metric, 'estimate']:.3f}" if pd.notna(rows.loc[metric, "estimate"]) else "—"
                else:
                    row[metric] = "—"
            row["faithfulness_n"] = int(rows.loc["faithfulness", "n"]) if "faithfulness" in rows.index else 0
            condition_rows.append(row)

    main: list[str] = [
        "# Reviewer Supplement: Final Experiment",
        "",
        "All analyses use the frozen 44-question set and existing 176 generated answers. No answers were regenerated or rescored for this supplement. Proportion intervals are Wilson 95% intervals; means use 2,000 bootstrap resamples (seed 42). Paired correctness intervals use Newcombe (1998) method 10, with two Wilson intervals combined using the paired phi correlation, and no continuity correction. Exact McNemar tests are retained. Robustness denotes C1 vs C2; retrieval effect denotes C4 vs C1.",
        "",
        "## Condition summaries",
        "",
        "`correctness` is the original RAGChecker result; `correctness_v2` adds answer2response contradiction checks. Estimates are shown below; full intervals and subgroup tables are in `score_summary_final.csv`.",
        "",
        markdown_table(pd.DataFrame(condition_rows), list(condition_rows[0].keys())),
        "",
        "## Paired differences",
        "",
        "Differences are C2 − C1 for robustness and C4 − C1 for retrieval effect. See `paired_tests_final.csv` for confidence intervals, discordant counts, and p-values.",
        "",
        markdown_table(paired[paired.metric.isin(["correctness_difference", "correctness_v2_difference", "completeness_difference", "faithfulness_difference"])], ["analysis_set", "comparison", "metric", "estimate", "ci95_lower", "ci95_upper", "test", "p_value"]),
        "",
        "## Needs-manual-check flags",
        "",
        "| Rule | Flagged answers |",
        "| --- | ---: |",
        f"| Correctness/FLASK disagreement | {rule_counts['correctness_flask_disagreement']} |",
        f"| Abstention label is abstains but RAGChecker is correct | {rule_counts['abstains_but_ragchecker_correct']} |",
        f"| C4 label is answers but ContextRecall is zero | {rule_counts['C4_answers_context_recall_zero']} |",
        f"| `correctness_v2` differs from original correctness | {rule_counts['correctness_v2_disagreement']} |",
        f"| Unique answers flagged | {unique_needs} |",
        "",
        "## Paraphrase retrieval contrast",
        "",
        "Top-10 retrieval only; no answers were generated or scored. Gold chunk and gold page hit rates use the same membership definitions as the C4 retrieval cross-tab. The table reports all 14 questions, the 13-question set excluding the answer-in-question item, and the quote-split groups.",
        "",
        markdown_table(retrieval, list(retrieval.columns)),
        "",
        "## Faithfulness, closed-book, and other checks",
        "",
        "Faithfulness retry sample sizes by condition are shown in the condition table (`faithfulness_n`). Five previously truncated Faithfulness calls were rerun with a larger token limit; item-level results are in `reviewer_supplement_item_details.md`.",
        "",
        f"Closed-book checks completed for {int((closed.closed_book_status == 'checked').sum())}/44 questions; {int(((closed.closed_book_status == 'checked') & (closed.closed_book_correct == True)).sum())} were correct. Their C1–C4 correctness cross-check remains in `closed_book_correctness_crosscheck.csv`.",
        "",
        "C4 retrieval by gold page hit × gold chunk hit is in `c4_retrieval_cross_tab.csv`. Keypoint-form and quote-splitting diagnostics are in `keypoint_form_diagnostic.csv` and `quote_split_by_c4_accuracy.csv`.",
        "",
        "## Blind-sample overlap and kappa plan",
        "",
        f"The blind-answer key contains {blind_n} sampled answers. Overlap counts are: needs-manual-check answers, {overlap['needs_manual_check']}; correctness-v2 discrepancies, {overlap['correctness_v2_disagreement']}; Faithfulness retries, {overlap['faithfulness_retry']}. Key contents are not included in this report.",
        "",
        "After human ratings are complete, report agreement and kappa for all 60 blind answers, then report the same statistics after removing the union of these flagged overlap items. This is a sensitivity analysis; the two results should be presented together.",
        "",
        "## Item-level appendix",
        "",
        "All per-answer review flags, correctness-v2 discrepancies and Faithfulness retry details are kept separately in `reviewer_supplement_item_details.md`.",
        "",
    ]

    details: list[str] = ["# Reviewer Supplement: Item Details", "", "This appendix contains item-level automatic-analysis details for reviewer audit. The aggregate report intentionally excludes these lists.", "", "## Needs-manual-check answers", ""]
    details.append(markdown_table(needs[["id", "condition", "rules"]], ["id", "condition", "rules"]))
    details.extend(["", "## Correctness-v2 discrepancies", ""])
    discrepancy_rows = []
    for _, row in disagreements.iterrows():
        try:
            evidence = json.loads(row["contradiction_evidence"])
        except (TypeError, json.JSONDecodeError):
            evidence = []
        statements = "<br>".join(f"{e.get('answer_statement')}: {e.get('judge_raw_return')}" for e in evidence)
        discrepancy_rows.append({"id": row["id"], "condition": row["condition"], "original_correctness": row["original_correctness"], "correctness_v2": row["correctness_v2"], "contradicted_statements_and_raw_returns": statements})
    details.append(markdown_table(pd.DataFrame(discrepancy_rows), ["id", "condition", "original_correctness", "correctness_v2", "contradicted_statements_and_raw_returns"]))
    details.extend(["", "## Faithfulness retries", ""])
    details.append(markdown_table(faith[["id", "condition", "value"]], ["id", "condition", "value"]))
    details.extend(["", "## Quote-split accuracy", "", markdown_table(quote_split, list(quote_split.columns)), ""])
    return "\n".join(main), "\n".join(details), overlap


def main() -> None:
    summary, details, overlap = build_reports()
    (OUT / "reviewer_supplement_summary.md").write_text(summary, encoding="utf-8")
    (OUT / "reviewer_supplement_item_details.md").write_text(details, encoding="utf-8")
    print("Blind sample overlap counts:")
    print(f"Needs-manual-check: {overlap['needs_manual_check']}")
    print(f"Correctness-v2 discrepancies: {overlap['correctness_v2_disagreement']}")
    print(f"Faithfulness retries: {overlap['faithfulness_retry']}")
    print("Reports written: reviewer_supplement_summary.md; reviewer_supplement_item_details.md")


if __name__ == "__main__":
    main()
