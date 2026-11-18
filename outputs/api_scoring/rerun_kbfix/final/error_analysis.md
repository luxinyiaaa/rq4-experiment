# Rerun error analysis

This report uses only the KB-fix rerun answers and the primary judge’s saved labels. Correctness is judge-only; no manual alternative-answer decisions are applied. All percentages are descriptive, with no new statistical tests.

Outcome precedence: **Correct → Abstained → Contradicted → Incomplete**. Abstained means the primary label is `abstains`; `partly_answers` proceeds to the keypoint checks. Contradicted means any keypoint is labelled `Contradiction`. Incomplete means no contradiction and at least one decisive keypoint is labelled `Neutral`.

For every source-page citation count below, “cites” means the answer text literally contains the question’s full `source_url` string. This is a text-occurrence measure, not a judgment of citation support.

## Outcome distribution

| Model | n | Correct | Abstained | Contradicted | Incomplete |
|---|---|---|---|---|---|
| EGI gpt-oss-120b | 300 | 171/300 (57.0%) | 44/300 (14.7%) | 43/300 (14.3%) | 42/300 (14.0%) |
| GPT-4.1 | 300 | 57/300 (19.0%) | 94/300 (31.3%) | 47/300 (15.7%) | 102/300 (34.0%) |
| GPT-5.6 Sol | 300 | 246/300 (82.0%) | 2/300 (0.7%) | 16/300 (5.3%) | 36/300 (12.0%) |
| GPT-6 Astra | 300 | 240/300 (80.0%) | 23/300 (7.7%) | 7/300 (2.3%) | 30/300 (10.0%) |
| All models | 1200 | 714/1200 (59.5%) | 163/1200 (13.6%) | 113/1200 (9.4%) | 210/1200 (17.5%) |

## About-science answers containing the source URL

Each percentage uses that model’s outcome count among about-science questions as its denominator.

| Model | Outcome | n | Contains source URL |
|---|---|---|---|
| EGI gpt-oss-120b | Correct | 149 | 118/149 (79.2%) |
| EGI gpt-oss-120b | Abstained | 23 | 1/23 (4.3%) |
| EGI gpt-oss-120b | Contradicted | 32 | 16/32 (50.0%) |
| EGI gpt-oss-120b | Incomplete | 36 | 22/36 (61.1%) |
| GPT-4.1 | Correct | 50 | 37/50 (74.0%) |
| GPT-4.1 | Abstained | 62 | 22/62 (35.5%) |
| GPT-4.1 | Contradicted | 38 | 19/38 (50.0%) |
| GPT-4.1 | Incomplete | 90 | 46/90 (51.1%) |
| GPT-5.6 Sol | Correct | 196 | 171/196 (87.2%) |
| GPT-5.6 Sol | Abstained | 2 | 0/2 (0.0%) |
| GPT-5.6 Sol | Contradicted | 12 | 4/12 (33.3%) |
| GPT-5.6 Sol | Incomplete | 30 | 21/30 (70.0%) |
| GPT-6 Astra | Correct | 196 | 171/196 (87.2%) |
| GPT-6 Astra | Abstained | 9 | 0/9 (0.0%) |
| GPT-6 Astra | Contradicted | 7 | 2/7 (28.6%) |
| GPT-6 Astra | Incomplete | 28 | 15/28 (53.6%) |

## Results by page type

Abstention rate uses the primary judge’s `abstains` label. Source-page citations use the literal URL rule above.

| Model | Page type | n | Correct | Abstained | Contains source URL |
|---|---|---|---|---|---|
| EGI gpt-oss-120b | dataset_record | 60 | 22/60 (36.7%) | 22/60 (36.7%) | 30/60 (50.0%) |
| EGI gpt-oss-120b | about_science | 240 | 149/240 (62.1%) | 23/240 (9.6%) | 157/240 (65.4%) |
| GPT-4.1 | dataset_record | 60 | 7/60 (11.7%) | 32/60 (53.3%) | 16/60 (26.7%) |
| GPT-4.1 | about_science | 240 | 50/240 (20.8%) | 62/240 (25.8%) | 124/240 (51.7%) |
| GPT-5.6 Sol | dataset_record | 60 | 50/60 (83.3%) | 0/60 (0.0%) | 54/60 (90.0%) |
| GPT-5.6 Sol | about_science | 240 | 196/240 (81.7%) | 2/240 (0.8%) | 196/240 (81.7%) |
| GPT-6 Astra | dataset_record | 60 | 44/60 (73.3%) | 14/60 (23.3%) | 44/60 (73.3%) |
| GPT-6 Astra | about_science | 240 | 196/240 (81.7%) | 9/240 (3.8%) | 188/240 (78.3%) |

## EDMED records in thesis Figure 7.1

| Record URL | Question ID | Model | Outcome | Cites record |
|---|---|---|---|---|
| https://edmed.seadatanet.org/report/956/ | DR-047-45b032a472_1 | EGI gpt-oss-120b | Abstained | No |
| https://edmed.seadatanet.org/report/956/ | DR-047-45b032a472_1 | GPT-4.1 | Abstained | No |
| https://edmed.seadatanet.org/report/956/ | DR-047-45b032a472_1 | GPT-5.6 Sol | Correct | Yes |
| https://edmed.seadatanet.org/report/956/ | DR-047-45b032a472_1 | GPT-6 Astra | Abstained | No |
| https://edmed.seadatanet.org/report/956/ | DR-047-45b032a472_2 | EGI gpt-oss-120b | Incomplete | No |
| https://edmed.seadatanet.org/report/956/ | DR-047-45b032a472_2 | GPT-4.1 | Correct | No |
| https://edmed.seadatanet.org/report/956/ | DR-047-45b032a472_2 | GPT-5.6 Sol | Correct | Yes |
| https://edmed.seadatanet.org/report/956/ | DR-047-45b032a472_2 | GPT-6 Astra | Abstained | No |
| https://edmed.seadatanet.org/report/950/ | DR-045-bc6e0c28f4_1 | EGI gpt-oss-120b | Correct | No |
| https://edmed.seadatanet.org/report/950/ | DR-045-bc6e0c28f4_1 | GPT-4.1 | Abstained | No |
| https://edmed.seadatanet.org/report/950/ | DR-045-bc6e0c28f4_1 | GPT-5.6 Sol | Correct | Yes |
| https://edmed.seadatanet.org/report/950/ | DR-045-bc6e0c28f4_1 | GPT-6 Astra | Correct | Yes |
| https://edmed.seadatanet.org/report/950/ | DR-045-bc6e0c28f4_2 | EGI gpt-oss-120b | Incomplete | Yes |
| https://edmed.seadatanet.org/report/950/ | DR-045-bc6e0c28f4_2 | GPT-4.1 | Abstained | No |
| https://edmed.seadatanet.org/report/950/ | DR-045-bc6e0c28f4_2 | GPT-5.6 Sol | Contradicted | No |
| https://edmed.seadatanet.org/report/950/ | DR-045-bc6e0c28f4_2 | GPT-6 Astra | Correct | Yes |

## Correctness by question type

| Question type | Model | n | Correct |
|---|---|---|---|
| list_all | EGI gpt-oss-120b | 96 | 76/96 (79.2%) |
| list_all | GPT-4.1 | 96 | 18/96 (18.8%) |
| list_all | GPT-5.6 Sol | 96 | 83/96 (86.5%) |
| list_all | GPT-6 Astra | 96 | 80/96 (83.3%) |
| list_all | All models | 384 | 257/384 (66.9%) |
| paraphrased | EGI gpt-oss-120b | 115 | 77/115 (67.0%) |
| paraphrased | GPT-4.1 | 115 | 32/115 (27.8%) |
| paraphrased | GPT-5.6 Sol | 115 | 102/115 (88.7%) |
| paraphrased | GPT-6 Astra | 115 | 99/115 (86.1%) |
| paraphrased | All models | 460 | 310/460 (67.4%) |
| false_premise | EGI gpt-oss-120b | 89 | 18/89 (20.2%) |
| false_premise | GPT-4.1 | 89 | 7/89 (7.9%) |
| false_premise | GPT-5.6 Sol | 89 | 61/89 (68.5%) |
| false_premise | GPT-6 Astra | 89 | 61/89 (68.5%) |
| false_premise | All models | 356 | 147/356 (41.3%) |

## Four false-premise outcomes

As in the rerun false-premise supplement: the correcting keypoint labelled `Entailment` counts as correcting the premise; `Contradiction` counts as accepting it. If that keypoint is `Neutral`, an answer labelled `abstains` is abstaining; otherwise it is silent on the premise. These categories use the premise-correction keypoint and are separate from the overall answer outcomes above.

| Model | n | Corrects premise | Accepts premise | Silent on premise | Abstains |
|---|---|---|---|---|---|
| EGI gpt-oss-120b | 89 | 21/89 (23.6%) | 29/89 (32.6%) | 31/89 (34.8%) | 8/89 (9.0%) |
| GPT-4.1 | 89 | 8/89 (9.0%) | 35/89 (39.3%) | 23/89 (25.8%) | 23/89 (25.8%) |
| GPT-5.6 Sol | 89 | 62/89 (69.7%) | 3/89 (3.4%) | 24/89 (27.0%) | 0/89 (0.0%) |
| GPT-6 Astra | 89 | 63/89 (70.8%) | 1/89 (1.1%) | 18/89 (20.2%) | 7/89 (7.9%) |
