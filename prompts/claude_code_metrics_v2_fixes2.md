# Metrics v2: small fixes to the reported tables

Use the existing judgements; no new judge calls; do not change any score. Same constraints as before (write only under `outputs/metrics_v2/`, English only, blind sample stays blind).

## 1. `metric_by_condition_S1.csv`

- Bug: in the "Appropriate abstention: false-abstention rate" rows, the C3 cells contain the C3 abstention rate (44 / 42). False abstention does not apply to C3; leave these cells empty with the note `not applicable`.
- When `n_applicable` is 0, set the note to `not applicable`, not `n < 10`.
- Controlled robustness rows: leave the cells empty and give one note: `paired C1 → C2; see robustness_paired_S1.csv`.

## 2. `main_table_S1.md`

- Rates: show the interval in percent, like the value, for example `38–66%`. Means: two decimals for the value and the interval.
- Put the two descriptive C4 rows (overall abstention rate, partial-answer rate) directly after the two "Appropriate abstention" rows.
- Controlled robustness row: instead of `—` in every cell, write in the C2 column `paired C1 → C2: no metric changed significantly (Holm-adjusted p ≥ 0.9)` if that is what `robustness_paired_S1.csv` shows for the primary judge, and `—` elsewhere. Check the statement against the file before writing it.
- Add one line under the legend: `The effect of retrieval (C1 → C4) is reported in retrieval_loss_S1.csv.`

## 3. `correct_vs_incorrect_C4.csv`

Abstentions are excluded from metrics 2, 7 and 8, so the group size differs per metric. Add, for each metric, a column with the number of answers used in each group (for example `metric_2_n`).

## 4. `judge_agreement.csv`

For binary or categorical metrics (answer correctness, appropriate abstention), leave `spearman` empty.

## 5. `faithfulness_outlier_report.md`

Remove the "Likely causes" table: the keyword heuristic does not match the traces. Keep the counts by condition and by rule, and add the line `Causes were classified by manual reading of the traces; see the thesis text.`

## 6. Report

- Regenerate `metrics_v2_summary.md` with these changes.
- Say whether step 10 of `claude_code_metrics_v2_revisions.md` changed the human-validation sheet, and if not, why.
- List the changed files for me to upload.
