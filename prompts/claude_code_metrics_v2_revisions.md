# Metrics v2: revisions after the full-run review

Replace `prompts/metric_definitions_rq1.md` with the new version first, and read it fully. It now has 10 metrics, numbered 1–10. The changes below use existing judgements and the cache; no new judge calls are needed except where stated. Do not change any judge prompt or score.

Constraints as before: write only under `outputs/metrics_v2/`; English only; the blind human-validation sample stays blind (no item IDs, questions, answers or scores in the terminal or in summary files).

## 1. Remove two metrics from all reported outputs

- Remove Factual accuracy (reason: the gold pages are not external factual evidence, which the thesis definition requires).
- Remove Instruction fulfilment (reason: its three requirements are measured by Citation completeness, Appropriate abstention and Evidence faithfulness).
- Keep their raw judgements in `items/judgements.jsonl` and the cache, but drop them from every summary table, the wide item file, and the markdown summary. Renumber metrics as in the definitions file.

## 2. Split robustness and retrieval loss

- `robustness_paired_S1.csv`: only C1 → C2, for metrics 1–4 and 7–8, Holm adjustment across these metrics.
- New `retrieval_loss_S1.csv`: C1 → C4 with the same statistics, Holm adjustment within this table.
- In the markdown summary, title them "Controlled robustness (C1 → C2)" and "Retrieval loss (C1 → C4)".

## 3. Abstention: descriptive C4 rows

Add two rows for C4 to `metric_by_condition_S1.csv`: overall abstention rate (abstains / 44) and partial-answer rate (partly answers / 44), with Wilson intervals, for both judges.

## 4. Intervals for small n

When fewer than 10 answers apply in a cell, leave the interval empty and set a column `note` to `n < 10`. Keep the value and n.

## 5. C4 error attribution

In `c4_error_attribution_S1.csv`, drop the factual-accuracy column, and add for each category the number of answers that answer, partly answer and abstain (primary judge's label).

## 6. Correct vs not correct

In `correct_vs_incorrect_C4.csv`, keep only metrics 2–8.

## 7. Inspect Evidence faithfulness outliers (no score changes)

Outliers: answers where (a) the two judges' Faithfulness differs by 0.5 or more, or (b) Citation support is 1.0 for a judge but that judge's Faithfulness is 0.5 or lower. For each outlier, save the Ragas extracted statements and their verdicts for both judges (from the cache; re-run with tracing only if the cache does not hold them) to `items/faithfulness_outliers.jsonl`. Write `summary/faithfulness_outlier_report.md` with counts only: number of outliers per condition and per rule, and counts of likely causes (for example meta-statements about the passages, false-premise corrections, list items or markdown split into claims, citation markers treated as claims, verdict parsing). No item IDs or text in this report.

## 8. Main table for the thesis

Write `summary/main_table_S1.md`: one row per metric (two rows for Appropriate abstention, plus the two descriptive C4 rows), grouped by dimension, primary judge only. Columns, in this order: Dimension, Metric, then for each condition a value column and a "95% CI" column, with condition headers "C4: retrieved (main)", "C1: gold only (baseline)", "C2: gold + 4 distractors", "C3: 4 distractors only". Values: rates as `k/n (p%)`, scores as the mean with `(n = …)`; `—` where the metric does not apply. Under the table, a legend: what each condition gives the generator, what `k/n`, `n`, the CI and `—` mean, and that the secondary judge's values are in `metric_by_condition_S1.csv`.

## 9. Small fixes

- Replace "No matched comparison in this pilot sample" with "Robustness is computed per question pair, not per answer" in the full-run summary.
- State in the markdown summary that 10 metrics in 4 dimensions are reported.

## 10. Human-validation sheet (only if not filled in yet)

Check whether the blind human-validation sheet already contains any grades. If it does, do not change it; report that. If it is empty, add two columns for each answer, with an instructions tab, keeping the sheet blind (no judge scores):
- `human_response_relevance` (1–5), with the anchors of metric 3 copied from the definitions file;
- `human_citation_support` (yes / no / not applicable): every statement with a citation is supported by the passages it cites.

## When done

Print the aggregate summary and list the files for me to upload:
- everything in `outputs/metrics_v2/summary/` (including `main_table_S1.md`, `retrieval_loss_S1.csv` and `faithfulness_outlier_report.md`);
- `items/faithfulness_outliers.jsonl` and `items/answer_metrics_wide.csv` (I will upload them without opening them);
- whether the human-validation sheet was changed.
