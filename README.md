# RQ4: ENVRI chatbot answer quality experiment

This repository contains the code, prompts, and selected results for an experiment on answers to questions about ENVRI knowledge-base pages. A reviewed 300-question main set is sent to four chatbot configurations. Two judges score each answer against question-specific keypoints and citation evidence. The saved rerun has 1,200 question–model pairs. Raw data, answer files, logs, and caches are not included.

## Layout

| Path | Purpose |
|---|---|
| `scripts/` | Question construction, answer collection, and supporting utilities. |
| `src/rq4/` | Shared model/client code. |
| `prompts/` | Question, scoring, and validation rubrics. |
| `configs/models.yaml` | Provider/model configuration; credentials are read from environment variables. |
| `tests/` | Local checks. |
| `data/` | Raw knowledge-base export, web pages, and processed inputs; review before sharing. |
| `outputs/rawpage_questions/question_set_v3/` | Final question-set files used by answer collection. |
| `outputs/api_answers/rerun_kbfix/` | Resumable answer collection output. |
| `outputs/api_scoring/rerun_kbfix/` | Scoring code, item records, final reports, and human validation. |

## Environment

Use Python 3.12 and Node.js for the workbook builders. The workbook builders additionally import `@oai/artifact-tool`, which is not supplied by `requirements.txt` and is not bundled here. Access to that package is needed to regenerate XLSX files; the Python scoring and report code can still be inspected without it. From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` locally. The variable names are `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `QWEN_API_KEY`, and `ENVRI_EXPERT_URL`. The first three authenticate configured model providers; `ENVRI_EXPERT_URL` points to the Environmental Expert service used for answer collection. The values are private and are not stored in this repository.

## Pipeline and commands

Run commands from the repository root. The commands below document the full workflow; **question generation, answer collection, and scoring make network/API calls**. Existing results can be read without running them. Reproduction requires the reviewed input data, including the knowledge-base export, which is held back from the proposed commit.

1. **Question set.** The saved main set is `outputs/rawpage_questions/question_set_v3/questions_main.jsonl`; the reserve set is `questions_extra.jsonl` in the same directory. The construction sequence was:

   ```sh
   .venv/bin/python scripts/rawpage_questions_question_set_v2.py
   .venv/bin/python scripts/audit_question_set_v2.py
   .venv/bin/python scripts/finalize_question_set_v3.py
   .venv/bin/python scripts/assemble_question_set_v3.py
   node scripts/build_question_set_v3_workbooks.mjs
   ```

   This sequence depends on earlier reviewed question files and manual audit decisions. The final JSONL files, rather than a fresh regeneration, are the fixed inputs for the answer run.

2. **Answer collection.** `expert_api_full_run.py` reads both the main and extra JSONL files by default and resumes into `outputs/api_answers/rerun_kbfix/`. The reported scoring cohort uses the 300 main questions.

   ```sh
   .venv/bin/python scripts/expert_api_full_run.py --sets main,extra
   .venv/bin/python scripts/expert_api_full_run.py --status
   ```

3. **Scoring.** `run_rerun.py` snapshots answer hashes, reuses cached judgements where valid, and scores pending main-set answers with the primary and secondary judges. The `--status` command only reads local progress.

   ```sh
   .venv/bin/python outputs/api_scoring/rerun_kbfix/run_rerun.py --prepare
   .venv/bin/python outputs/api_scoring/rerun_kbfix/run_rerun.py
   .venv/bin/python outputs/api_scoring/rerun_kbfix/run_rerun.py --status
   ```

4. **Finalize.** This checks all 1,200 scored items and builds reports and XLSX workbooks. It uses the saved answers and judgements, with no new judge calls.

   ```sh
   .venv/bin/python outputs/api_scoring/rerun_kbfix/finalize.py
   ```

5. **Error analysis.** Read `outputs/api_scoring/rerun_kbfix/final/error_analysis.md` alongside the final workbook. The saved error analysis is a descriptive review of saved primary-judge labels. There is no standalone regeneration script for this report. It applies the precedence Correct → Abstained → Contradicted → Incomplete and reports counts by model, page type, and question type.

   ```sh
   less outputs/api_scoring/rerun_kbfix/final/error_analysis.md
   ```

6. **Human validation.** A fixed-seed sample selects 40 answers, 10 per model. Build the blind sheet, enter human labels in that sheet, then compute agreement. `compute_agreement.py` reads the filled sheet; the sample preparation does not judge answers.

   ```sh
   .venv/bin/python outputs/api_scoring/rerun_kbfix/human_validation/prepare_sample.py
   node outputs/api_scoring/rerun_kbfix/human_validation/build_blind_sheet.mjs
   .venv/bin/python outputs/api_scoring/rerun_kbfix/human_validation/compute_agreement.py --sheet outputs/api_scoring/rerun_kbfix/human_validation/human_validation_blind.xlsx
   ```

   Human labels must be entered before the last command. A filled 40-answer rerun sample and its agreement report are already saved locally under `human_validation/`. The main `final_results.md` table also carries an earlier 40-answer validation from the old answers; use `human_validation/agreement.md` for agreement on the rerun sample.

## Scores

Both judges use the same definitions, and results are reported separately for each judge.

| Metric | Calculation |
|---|---|
| Answer correctness | Binary: **all decisive keypoints** must be labelled `Entailment` against the answer, and **no keypoint** may be labelled `Contradiction`. A `Neutral` non-decisive point does not make an otherwise correct answer incorrect. The reported rate is the share of answers passing this rule. |
| Completeness | Number of all keypoints (decisive and non-decisive) labelled `Entailment` divided by the number of keypoints. |
| Response relevance | A judge applies a 1–5 rubric to the question and answer. Content about the thing asked counts as on-topic even if factually wrong; citations, disclaimers, and offers of help are ignored. Score 5 has no off-topic units; 4 has one or two; 3 has at least three and about as much on-topic as off-topic content; 2 has more off-topic than on-topic content and at most one on-topic sentence; 1 has no on-topic content. A justified statement that information is unavailable can be relevant. |
| Citation support | Among answer statements carrying a citation, the share entailed by the passages associated with their cited URLs. Cited URLs absent from the knowledge base provide no supporting text. No score is reported if there are no cited statements. |
| Citation completeness | Among statements judged to need attribution, the share carrying at least one cited URL. This measures citation presence; citation support is assessed separately. No score is reported if no statement needs attribution. |
| False abstention | Share of answerable main-set questions labelled `abstains`. `partly_answers` is not an abstention. The two judges' rates use their own abstention labels. |

The current correctness results are judge labels; newly flagged alternative answers still need manual review. See the final report for applicability counts and uncertainty intervals.

## Results

The shared results are in `outputs/api_scoring/rerun_kbfix/final/`: `all_answers_and_scores_v4.xlsx` (item-level workbook), `final_results.md` (main tables), `results_by_question_type.md`, `error_analysis.md`, and `supplementary/false_premise_sensitivity.md`. The filled human review sheet, key, and agreement reports are in `outputs/api_scoring/rerun_kbfix/human_validation/`. Raw answer JSONL files, logs, and caches are excluded.

The knowledge-base export is not included; it is available from the maintainer. Recreating the full experiment requires that export and the other reviewed local inputs described above.
