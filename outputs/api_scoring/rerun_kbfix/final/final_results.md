# Final API-answer analysis: 300 main-set questions

## Main results (primary judge)

| model | n | answer_correctness | completeness | response_relevance | citation_support | citation_completeness | false_abstention_rate | overall_abstention_rate | partly_answers_rate |
|---|---|---|---|---|---|---|---|---|---|
| egi | 300 | 57.0% [51.3, 62.5] (n=300) | 0.688 [0.642, 0.733] (n=300) | 4.980 [4.957, 4.997] (n=300) | 0.697 [0.645, 0.748] (n=245) | 0.612 [0.567, 0.656] (n=255) | 15.0% [11.4, 19.5] (n=300) | 15.0% [11.4, 19.5] (n=300) | 0.7% [0.2, 2.4] (n=300) |
| openai | 300 | 19.0% [15.0, 23.8] (n=300) | 0.315 [0.271, 0.359] (n=300) | 4.760 [4.693, 4.823] (n=300) | 0.589 [0.530, 0.647] (n=201) | 0.497 [0.442, 0.553] (n=206) | 31.3% [26.3, 36.8] (n=300) | 31.3% [26.3, 36.8] (n=300) | 12.7% [9.4, 16.9] (n=300) |
| openai_gpt56_sol | 300 | 82.0% [77.3, 85.9] (n=300) | 0.897 [0.867, 0.924] (n=300) | 4.977 [4.960, 4.990] (n=300) | 0.925 [0.899, 0.949] (n=298) | 0.872 [0.845, 0.899] (n=298) | 0.7% [0.2, 2.4] (n=300) | 0.7% [0.2, 2.4] (n=300) | 1.0% [0.3, 2.9] (n=300) |
| openai_gpt6_astra | 300 | 80.0% [75.1, 84.1] (n=300) | 0.859 [0.821, 0.893] (n=300) | 4.963 [4.933, 4.987] (n=300) | 0.967 [0.947, 0.985] (n=276) | 0.932 [0.911, 0.950] (n=277) | 7.7% [5.2, 11.2] (n=300) | 7.7% [5.2, 11.2] (n=300) | 1.3% [0.5, 3.4] (n=300) |

Correctness is judge-only pending review of the new flagged alternative answers.

## Main results (secondary judge)

| model | n | answer_correctness | completeness | response_relevance | citation_support | citation_completeness | false_abstention_rate | overall_abstention_rate | partly_answers_rate |
|---|---|---|---|---|---|---|---|---|---|
| egi | 300 | 55.7% [50.0, 61.2] (n=300) | 0.675 [0.627, 0.719] (n=300) | 4.987 [4.960, 5.000] (n=300) | 0.816 [0.770, 0.858] (n=245) | 0.631 [0.584, 0.679] (n=255) | 15.0% [11.4, 19.5] (n=300) | 15.0% [11.4, 19.5] (n=300) | 0.3% [0.1, 1.9] (n=300) |
| openai | 300 | 18.3% [14.4, 23.1] (n=300) | 0.313 [0.270, 0.357] (n=300) | 4.897 [4.847, 4.940] (n=300) | 0.655 [0.599, 0.710] (n=201) | 0.505 [0.449, 0.560] (n=205) | 33.7% [28.6, 39.2] (n=300) | 33.7% [28.6, 39.2] (n=300) | 6.0% [3.8, 9.3] (n=300) |
| openai_gpt56_sol | 300 | 78.7% [73.7, 82.9] (n=300) | 0.877 [0.845, 0.907] (n=300) | 4.997 [4.990, 5.000] (n=300) | 0.943 [0.918, 0.965] (n=298) | 0.883 [0.856, 0.910] (n=296) | 0.7% [0.2, 2.4] (n=300) | 0.7% [0.2, 2.4] (n=300) | 0.3% [0.1, 1.9] (n=300) |
| openai_gpt6_astra | 300 | 79.0% [74.0, 83.2] (n=300) | 0.847 [0.811, 0.884] (n=300) | 4.980 [4.963, 4.993] (n=300) | 0.973 [0.956, 0.987] (n=276) | 0.932 [0.910, 0.951] (n=277) | 7.7% [5.2, 11.2] (n=300) | 7.7% [5.2, 11.2] (n=300) | 2.0% [0.9, 4.3] (n=300) |

Secondary false abstention uses the secondary judge's own abstention labels.

## False-premise correctness (strict)

| model | judge | n | strict_correct |
|---|---|---|---|
| egi | primary | 89 | 18/89 (20.2%) |
| openai | primary | 89 | 7/89 (7.9%) |
| openai_gpt56_sol | primary | 89 | 61/89 (68.5%) |
| openai_gpt6_astra | primary | 89 | 61/89 (68.5%) |
| egi | secondary | 89 | 16/89 (18.0%) |
| openai | secondary | 89 | 5/89 (5.6%) |
| openai_gpt56_sol | secondary | 89 | 59/89 (66.3%) |
| openai_gpt6_astra | secondary | 89 | 60/89 (67.4%) |

Lenient false-premise analysis and four outcomes are in `supplementary/false_premise_sensitivity.md`.

## Human-validation agreement

The 40-answer sample and its human labels are unchanged from the 94-question analysis. This validation was not repeated; agreement refers to the original answers and validates the judges, not the new answers.

| group | judge | metric | n | as_filed_exact | corrected_exact | as_filed_within_one | corrected_within_one | as_filed_kappa | corrected_kappa | as_filed_quadratic_kappa | corrected_quadratic_kappa | as_filed_spearman | corrected_spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | primary | correct | 40 | 0.95 | 0.975 |  |  | 0.898989898989899 | 0.949748743718593 |  |  |  |  |
| overall | primary | abstention | 40 | 0.95 | 0.95 |  |  | 0.9041916167664671 | 0.9041916167664671 |  |  |  |  |
| overall | primary | relevance | 40 | 0.925 | 0.925 | 1.0 | 1.0 |  |  | 0.8648648648648649 | 0.8648648648648649 | 0.9087322899218538 | 0.9087322899218538 |
| overall | primary | abstention_partly_merged | 40 | 0.975 | 0.975 |  |  | 0.9418604651162791 | 0.9418604651162791 |  |  |  |  |
| overall | primary | decisive_keypoint_present | 95 | 0.9789473684210527 | 0.9894736842105263 |  |  | 0.9575134168157424 | 0.9787993751394778 |  |  |  |  |
| overall | primary | decisive_keypoint_1_present | 40 | 0.975 | 0.975 |  |  | 0.949238578680203 | 0.949238578680203 |  |  |  |  |
| overall | primary | decisive_keypoint_2_present | 32 | 0.96875 | 1.0 |  |  | 0.9359999999999999 | 1.0 |  |  |  |  |
| overall | primary | decisive_keypoint_3_present | 15 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| overall | primary | decisive_keypoint_4_present | 7 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| overall | primary | decisive_keypoint_5_present | 1 | 1.0 | 1.0 |  |  |  |  |  |  |  |  |
| overall | secondary | correct | 40 | 0.95 | 0.975 |  |  | 0.898989898989899 | 0.949748743718593 |  |  |  |  |
| overall | secondary | abstention | 40 | 0.95 | 0.95 |  |  | 0.8988621997471555 | 0.8988621997471555 |  |  |  |  |
| overall | secondary | relevance | 40 | 0.85 | 0.85 | 0.975 | 0.975 |  |  | 0.5454545454545454 | 0.5454545454545454 | 0.42477858597569834 | 0.42477858597569834 |
| overall | secondary | abstention_partly_merged | 40 | 0.975 | 0.975 |  |  | 0.9418604651162791 | 0.9418604651162791 |  |  |  |  |
| overall | secondary | decisive_keypoint_present | 95 | 0.9789473684210527 | 0.9894736842105263 |  |  | 0.9575134168157424 | 0.9787993751394778 |  |  |  |  |
| overall | secondary | decisive_keypoint_1_present | 40 | 0.975 | 0.975 |  |  | 0.949238578680203 | 0.949238578680203 |  |  |  |  |
| overall | secondary | decisive_keypoint_2_present | 32 | 0.96875 | 1.0 |  |  | 0.9359999999999999 | 1.0 |  |  |  |  |
| overall | secondary | decisive_keypoint_3_present | 15 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| overall | secondary | decisive_keypoint_4_present | 7 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| overall | secondary | decisive_keypoint_5_present | 1 | 1.0 | 1.0 |  |  |  |  |  |  |  |  |
| egi | primary | correct | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | primary | abstention | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | primary | relevance | 10 | 0.9 | 0.9 | 1.0 | 1.0 |  |  | 0.0 | 0.0 |  |  |
| egi | primary | abstention_partly_merged | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | primary | decisive_keypoint_present | 25 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | primary | decisive_keypoint_1_present | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | primary | decisive_keypoint_2_present | 9 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | primary | decisive_keypoint_3_present | 3 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | primary | decisive_keypoint_4_present | 2 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | primary | decisive_keypoint_5_present | 1 | 1.0 | 1.0 |  |  |  |  |  |  |  |  |
| egi | secondary | correct | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | secondary | abstention | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | secondary | relevance | 10 | 0.9 | 0.9 | 1.0 | 1.0 |  |  | 0.0 | 0.0 |  |  |
| egi | secondary | abstention_partly_merged | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | secondary | decisive_keypoint_present | 25 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | secondary | decisive_keypoint_1_present | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | secondary | decisive_keypoint_2_present | 9 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | secondary | decisive_keypoint_3_present | 3 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | secondary | decisive_keypoint_4_present | 2 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| egi | secondary | decisive_keypoint_5_present | 1 | 1.0 | 1.0 |  |  |  |  |  |  |  |  |
| openai | primary | correct | 10 | 0.9 | 0.9 |  |  | 0.6153846153846154 | 0.6153846153846154 |  |  |  |  |
| openai | primary | abstention | 10 | 0.8 | 0.8 |  |  | 0.6666666666666667 | 0.6666666666666667 |  |  |  |  |
| openai | primary | relevance | 10 | 0.9 | 0.9 | 1.0 | 1.0 |  |  | 0.8571428571428571 | 0.8571428571428571 | 0.9759000729485331 | 0.9759000729485331 |
| openai | primary | abstention_partly_merged | 10 | 0.9 | 0.9 |  |  | 0.8 | 0.8 |  |  |  |  |
| openai | primary | decisive_keypoint_present | 24 | 0.9583333333333334 | 0.9583333333333334 |  |  | 0.8333333333333334 | 0.8333333333333334 |  |  |  |  |
| openai | primary | decisive_keypoint_1_present | 10 | 0.9 | 0.9 |  |  | 0.6153846153846154 | 0.6153846153846154 |  |  |  |  |
| openai | primary | decisive_keypoint_2_present | 8 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai | primary | decisive_keypoint_3_present | 4 | 1.0 | 1.0 |  |  |  |  |  |  |  |  |
| openai | primary | decisive_keypoint_4_present | 2 | 1.0 | 1.0 |  |  |  |  |  |  |  |  |
| openai | secondary | correct | 10 | 0.9 | 0.9 |  |  | 0.6153846153846154 | 0.6153846153846154 |  |  |  |  |
| openai | secondary | abstention | 10 | 0.8 | 0.8 |  |  | 0.6551724137931034 | 0.6551724137931034 |  |  |  |  |
| openai | secondary | relevance | 10 | 0.6 | 0.6 | 0.9 | 0.9 |  |  | 0.0 | 0.0 |  |  |
| openai | secondary | abstention_partly_merged | 10 | 0.9 | 0.9 |  |  | 0.8 | 0.8 |  |  |  |  |
| openai | secondary | decisive_keypoint_present | 24 | 0.9583333333333334 | 0.9583333333333334 |  |  | 0.8333333333333334 | 0.8333333333333334 |  |  |  |  |
| openai | secondary | decisive_keypoint_1_present | 10 | 0.9 | 0.9 |  |  | 0.6153846153846154 | 0.6153846153846154 |  |  |  |  |
| openai | secondary | decisive_keypoint_2_present | 8 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai | secondary | decisive_keypoint_3_present | 4 | 1.0 | 1.0 |  |  |  |  |  |  |  |  |
| openai | secondary | decisive_keypoint_4_present | 2 | 1.0 | 1.0 |  |  |  |  |  |  |  |  |
| openai_gpt56_sol | primary | correct | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | primary | abstention | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | primary | relevance | 10 | 0.9 | 0.9 | 1.0 | 1.0 |  |  | 0.7826086956521738 | 0.7826086956521738 | 0.9999999999999999 | 0.9999999999999999 |
| openai_gpt56_sol | primary | abstention_partly_merged | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | primary | decisive_keypoint_present | 21 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | primary | decisive_keypoint_1_present | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | primary | decisive_keypoint_2_present | 7 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | primary | decisive_keypoint_3_present | 4 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | secondary | correct | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | secondary | abstention | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | secondary | relevance | 10 | 0.9 | 0.9 | 1.0 | 1.0 |  |  | 0.7826086956521738 | 0.7826086956521738 | 0.9999999999999999 | 0.9999999999999999 |
| openai_gpt56_sol | secondary | abstention_partly_merged | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | secondary | decisive_keypoint_present | 21 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | secondary | decisive_keypoint_1_present | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | secondary | decisive_keypoint_2_present | 7 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt56_sol | secondary | decisive_keypoint_3_present | 4 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt6_astra | primary | correct | 10 | 0.9 | 1.0 |  |  | 0.782608695652174 | 1.0 |  |  |  |  |
| openai_gpt6_astra | primary | abstention | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt6_astra | primary | relevance | 10 | 1.0 | 1.0 | 1.0 | 1.0 |  |  |  |  |  |  |
| openai_gpt6_astra | primary | abstention_partly_merged | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt6_astra | primary | decisive_keypoint_present | 25 | 0.96 | 1.0 |  |  | 0.9049429657794676 | 1.0 |  |  |  |  |
| openai_gpt6_astra | primary | decisive_keypoint_1_present | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt6_astra | primary | decisive_keypoint_2_present | 8 | 0.875 | 1.0 |  |  | 0.7142857142857143 | 1.0 |  |  |  |  |
| openai_gpt6_astra | primary | decisive_keypoint_3_present | 4 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt6_astra | primary | decisive_keypoint_4_present | 3 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt6_astra | secondary | correct | 10 | 0.9 | 1.0 |  |  | 0.782608695652174 | 1.0 |  |  |  |  |
| openai_gpt6_astra | secondary | abstention | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt6_astra | secondary | relevance | 10 | 1.0 | 1.0 | 1.0 | 1.0 |  |  |  |  |  |  |
| openai_gpt6_astra | secondary | abstention_partly_merged | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt6_astra | secondary | decisive_keypoint_present | 25 | 0.96 | 1.0 |  |  | 0.9049429657794676 | 1.0 |  |  |  |  |
| openai_gpt6_astra | secondary | decisive_keypoint_1_present | 10 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt6_astra | secondary | decisive_keypoint_2_present | 8 | 0.875 | 1.0 |  |  | 0.7142857142857143 | 1.0 |  |  |  |  |
| openai_gpt6_astra | secondary | decisive_keypoint_3_present | 4 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |
| openai_gpt6_astra | secondary | decisive_keypoint_4_present | 3 | 1.0 | 1.0 |  |  | 1.0 | 1.0 |  |  |  |  |

## Judge agreement

| metric | n_pairs | exact_agreement | cohen_kappa | quadratic_weighted_kappa | spearman | agreement_within_one_point |
|---|---|---|---|---|---|---|
| answer_correctness | 1200 | 0.9525 | 0.9020534410172696 | — | 0.9025345965675028 | — |
| completeness | 1200 | 0.9308333333333333 | — | — | 0.9357772990720311 | — |
| response_relevance | 1200 | 0.9491666666666667 | — | 0.49835826340751543 | 0.5054527674408636 | 0.9925 |
| citation_support | 1020 | 0.8303921568627451 | — | — | 0.6827237918556449 | — |
| citation_completeness | 1033 | 0.8809293320425944 | — | — | 0.9457287578884804 | — |
| appropriate_abstention | 1200 | 0.975 | 0.9140333407360178 | — | — | — |

## Low relevance scores

| experiment | model_or_condition | n_low |
|---|---|---|
| API | egi | 3 |
| API | openai | 10 |
| API | openai_gpt56_sol | 0 |
| API | openai_gpt6_astra | 1 |
| S1 | C1 | 0 |
| S1 | C2 | 0 |
| S1 | C3 | 5 |
| S1 | C4 | 1 |

Either judge scored relevance at 3 or below for 20 answers. The item-level list is in `relevance_low_scores.xlsx`.

## Relevance among non-abstaining answers

| experiment | model_or_condition | judge | n | mean_relevance | 95% CI |
|---|---|---|---|---|---|
| API | egi | primary | 255 | 4.976 | [4.949, 4.996] |
| API | egi | secondary | 255 | 4.984 | [4.953, 5.000] |
| API | openai | primary | 206 | 4.816 | [4.738, 4.883] |
| API | openai | secondary | 206 | 4.927 | [4.869, 4.976] |
| API | openai_gpt56_sol | primary | 298 | 4.980 | [4.963, 4.993] |
| API | openai_gpt56_sol | secondary | 298 | 4.997 | [4.990, 5.000] |
| API | openai_gpt6_astra | primary | 277 | 4.978 | [4.949, 4.996] |
| API | openai_gpt6_astra | secondary | 277 | 4.989 | [4.975, 5.000] |
| S1 | C1 | primary | 44 | 4.977 | [4.932, 5.000] |
| S1 | C1 | secondary | 44 | 5.000 | [5.000, 5.000] |
| S1 | C2 | primary | 44 | 4.977 | [4.932, 5.000] |
| S1 | C2 | secondary | 44 | 4.977 | [4.932, 5.000] |
| S1 | C3 | primary | 2 | 5.000 | — |
| S1 | C3 | secondary | 2 | 5.000 | — |
| S1 | C4 | primary | 32 | 4.938 | [4.844, 5.000] |
| S1 | C4 | secondary | 32 | 5.000 | [5.000, 5.000] |

Non-abstaining means `answers` or `partly_answers` under the primary judge's shared gate. Intervals use 10,000 bootstrap draws (seed 42).
