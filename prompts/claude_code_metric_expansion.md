# Task: score every answer on the 12 quality-model metrics (metrics_v2)

Context: supervisor feedback (20 Oct). RQ4 must show the quality model applied, not only correctness: one score per metric, each judged with the thesis definition, not averaged. The teacher model (GPT-6 Sol) becomes the primary judge; Qwen3.7 Plus becomes the secondary judge, and agreement between them is reported.

The metric set, definitions, measures, judge inputs and reporting rules are fixed in `prompts/metric_definitions_rq1.md` (12 metrics, one number per metric and condition; two numbers for Appropriate abstention and Factual accuracy). Read it fully before starting. Do not add, drop or redefine metrics or measures without asking me.

Scope: the existing answers of S1 (44 questions × C1–C4 = 176 answers). If S2 answers already exist, score them too and report S2 separately; if not, do S1 only and say so.

## Constraints (unchanged)
- Never modify `data/raw/`, S1 `outputs/final/`, or the final S2 question and answer files. Write only under `outputs/metrics_v2/` (and new files under `prompts/`, `scripts/`, `src/rq4/`, `configs/`, `tests/`).
- API keys stay in `.env`; never print them.
- Everything written to the repository is in English.
- Cache and log every LLM call as before. Temperature 0. Fixed seeds.
- Blind human check: the human-validation sample must stay blind for me. Read the sample key only inside code. Never print item IDs, questions, answers or scores of items in that sample. Item-level outputs go to `outputs/metrics_v2/items/`; print only aggregates in the terminal.
- Reuse existing answers. Do not regenerate answers.

## Step 0: plan (no API calls)
1. Find the final answer files for S1 and, if present, S2.
2. List which existing scores can be reused as secondary-judge (Qwen) results without a new call: RAGChecker keypoint labels, Ragas Faithfulness, abstention labels, S2 behaviour labels. Reuse only when the prompt and inputs are identical to what the new pipeline would send.
3. Print the plan: per metric, the script and prompt that compute it, the judge calls per answer, and the expected number of calls per judge for the full run. Then continue to step 1.

## Step 1: configuration and smoke tests (≤ 5 items)
1. In `configs/models.yaml`, add roles `judge_primary` (same model as `teacher`) and `judge_secondary` (the current judge). Keep `judge` working for the old scripts.
2. Smoke-test with 3 items, both judges:
   - RAGChecker in non-joint mode with the primary judge as checker, for the keypoint labels and for the retriever checks (context precision, claim recall). The checker reads a one-word label with `max_new_tokens=10`, so the model must answer without reasoning. If GPT-6 Sol returns reasoning or an empty label, try the provider's documented way to turn reasoning off. If that does not work, stop and tell me; do not patch the library.
   - Ragas Faithfulness with the primary judge (needs valid structured JSON).
   - One Response relevance call with each judge.
   - One statement decomposition with the primary judge.
3. Record token usage per call. Report pass/fail per test.

## Step 2: implement (test with a mock LLM first)
Write prompts under `prompts/judge_metrics/`, one file per prompt. Each prompt quotes the metric's definition exactly as in `metric_definitions_rq1.md` and contains only the inputs listed under "Judge sees". Output JSON with a one-sentence `reason` for every score or label.

1. **RAGChecker** (both judges): keypoint labels against the answer (metrics 1 and 6); retriever checks of the keypoints against the 10 retrieved chunks (metrics 7 and 8, C4 only).
2. **Ragas Faithfulness** (both judges): metric 2, all non-abstaining answers.
3. **Response relevance** (both judges): rubric 1–5 with the anchors from the definitions file (metric 4). Output `{"score": <int>, "reason": "..."}`.
4. **Statement decomposition** (primary judge only, once per non-abstaining answer): split the answer into statements (sentences or list items); for each statement give its cited passage IDs and its atomic claims. Store it; both judges label the same statements and claims.
5. **Label calls** (both judges, on the stored statements):
   - factual accuracy (3): each claim against the full cleaned text of the question's gold pages → `supported` / `contradicted` / `cannot_be_determined`;
   - citation support (9): one call per cited statement, with the text of its cited passages → whether the passages together entail the statement;
   - citation completeness (10): one call per answer, with the statements and the passages → for each statement `needs_attribution` true/false (it states a fact taken from the passages);
   - instruction fulfilment R2 (5): when applicable, whether the answer says that the passages are not enough;
   - abstention (11): answers / abstains / partly answers.
6. **Program checks**: R1 citation format and valid IDs (5); R3 = Faithfulness 1.0 (5); citation presence per statement (10); cited IDs not among the given passages (9).
7. **Robustness** (12): computed in step 5 from metrics 1–10; no extra calls.
8. Per-answer applicability: follow the "Conditions" column and the notes in the definitions file. Count excluded answers per metric and reason.
9. Tests with a mock LLM: output parsing, applicability exclusions, invalid citation IDs, caching, and that no blind-sample item is printed.

## Step 3: pilot, then stop
1. Choose 12 S1 answers that are not in the human-validation sample: 3 per condition, covering all four question types, including at least one C4 answer labelled incorrect and at least one abstention. Choose them in code with seed 42.
2. Run every metric with both judges.
3. Write `outputs/metrics_v2/pilot/pilot_review.md`: for each of the 12 answers, the question, the passage IDs, the answer, the statement decomposition, then a table with one row per metric: primary value, primary reason, secondary value, secondary reason.
4. Write `outputs/metrics_v2/pilot/pilot_summary.md`: calls and tokens per judge; estimated calls, tokens and cost for the full run (use the prices in `configs/models.yaml` if present, otherwise give tokens only); parse failures; excluded answers per metric; any metric where the two judges disagree on more than 3 of the 12 answers (labels) or differ by 2 or more points on more than 3 answers (scores).
5. **Stop and wait.** I will review the pilot before the full run.

## Step 4: full run (only after I say so)
All S1 answers (and S2 answers if available), every metric, both judges. Resume from the cache if interrupted. Log failures; retry a failed parse once; do not drop answers silently.

## Step 5: analysis
Write CSV files and one markdown summary under `outputs/metrics_v2/summary/`. S1 and S2 are always reported separately. Conditions are ordered C4, C1, C2, C3 (C4 is the main result, C1 the baseline).

1. `metric_by_condition_S1.csv` (and `_S2`): one row per metric (two rows for Appropriate abstention and Factual accuracy), grouped by dimension in the order of the definitions file. For each condition: `n_applicable`, value, and the 95% CI in its own column. Binary outcomes as `k/n (p%)` with a Wilson interval; means with a bootstrap interval (10,000 resamples over questions, seed 42).
2. `metric_by_type_C4_S1.csv`: the same for C4 by question type.
3. `robustness_paired_S1.csv`: for every metric applicable in both conditions, the paired difference C1 → C2 and C1 → C4 with a 95% CI, exact McNemar (binary) or Wilcoxon signed-rank (scores), raw p and Holm-adjusted p across the metrics of that comparison. For S2 also question_source → question_user.
4. `judge_agreement.csv`: per judge-based metric: n pairs, exact agreement, Cohen's kappa (labels), quadratic-weighted kappa, Spearman and agreement within one point (1–5 scores), Spearman (continuous scores), and for each judge the share of answers at the best value. Add a note that kappa is unstable when nearly all answers get the same value.
5. `discrimination.csv`: per metric, judge and condition: share of answers at the best value, share at the worst value, standard deviation.
6. `c4_error_attribution_S1.csv` (counts only; item-level rows go to `items/`): every S1 C4 answer that is not correct, assigned to one category in this order:
   - `retrieval_failure`: at least one decisive keypoint not supported by the retrieved chunks (metric 8 on decisive keypoints);
   - `false_abstention`: all decisive keypoints retrieved, but the answer abstains;
   - `generation_failure`: all decisive keypoints retrieved and the answer does not abstain, but it is not correct;
   for each category, also the mean Faithfulness, Factual accuracy and Citation support of its answers. Use the primary judge; report how many answers change category under the secondary judge.
7. `correct_vs_incorrect_C4.csv`: values of metrics 2–10 for correct vs not correct S1 C4 answers (descriptive; give n per group).
8. `metrics_v2_summary.md`: all tables above in markdown. Under each table, one line explaining the interval in plain words, for example: "95% CI (Wilson): with 44 questions, the true rate is plausibly anywhere in this range." Also list: judges and model IDs, the number of calls, parse failures, excluded answers per metric, reused secondary results, and the difference in correctness between the primary judge and the earlier Qwen results.

Item-level outputs (blind until my grading is done):
- `outputs/metrics_v2/items/answer_metrics_wide.csv`: one row per answer, one column per metric, with suffixes `_primary` and `_secondary`;
- `outputs/metrics_v2/items/judgements.jsonl`: every judge output with its reason;
- `outputs/metrics_v2/items/c4_error_attribution_items.csv`.

## When done
Print the aggregate summary and the list of files for me to upload:
- after step 3: `pilot/pilot_review.md`, `pilot/pilot_summary.md`;
- after steps 4–5: everything in `outputs/metrics_v2/summary/`, plus `items/answer_metrics_wide.csv` (I will upload it without opening it).
