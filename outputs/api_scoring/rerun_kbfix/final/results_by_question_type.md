# Results by question type

Correctness and false abstention are rates with Wilson 95% intervals. Other cells give mean ± sample SD; citation metrics show their applicable n.

## Primary judge

| question_type | model | n | correctness | false abstention | completeness | relevance | citation support | citation completeness |
|---|---|---|---|---|---|---|---|---|
| list_all | egi | 96 | 79.2% [70.0, 86.1] (n=96) | 8.3% [4.3, 15.6] (n=96) | 0.849 ± 0.323 (n=96) | 4.958 ± 0.287 (n=96) | 0.750 ± 0.389 (n=85) | 0.509 ± 0.378 (n=88) |
| list_all | openai | 96 | 18.8% [12.2, 27.7] (n=96) | 22.9% [15.6, 32.3] (n=96) | 0.386 ± 0.384 (n=96) | 4.729 ± 0.640 (n=96) | 0.660 ± 0.390 (n=70) | 0.487 ± 0.434 (n=74) |
| list_all | openai_gpt56_sol | 96 | 86.5% [78.2, 91.9] (n=96) | 0.0% [0.0, 3.8] (n=96) | 0.943 ± 0.178 (n=96) | 4.979 ± 0.144 (n=96) | 0.948 ± 0.156 (n=96) | 0.860 ± 0.271 (n=96) |
| list_all | openai_gpt6_astra | 96 | 83.3% [74.6, 89.5] (n=96) | 6.2% [2.9, 13.0] (n=96) | 0.898 ± 0.272 (n=96) | 4.948 ± 0.336 (n=96) | 0.971 ± 0.136 (n=90) | 0.974 ± 0.083 (n=90) |
| paraphrased | egi | 115 | 67.0% [57.9, 74.9] (n=115) | 21.7% [15.2, 30.1] (n=115) | 0.698 ± 0.447 (n=115) | 5.000 ± 0.000 (n=115) | 0.680 ± 0.392 (n=86) | 0.684 ± 0.341 (n=90) |
| paraphrased | openai | 115 | 27.8% [20.5, 36.6] (n=115) | 33.0% [25.1, 42.1] (n=115) | 0.343 ± 0.432 (n=115) | 4.757 ± 0.470 (n=115) | 0.578 ± 0.404 (n=76) | 0.564 ± 0.393 (n=77) |
| paraphrased | openai_gpt56_sol | 115 | 88.7% [81.6, 93.3] (n=115) | 1.7% [0.5, 6.1] (n=115) | 0.912 ± 0.258 (n=115) | 4.974 ± 0.160 (n=115) | 0.909 ± 0.250 (n=113) | 0.906 ± 0.204 (n=113) |
| paraphrased | openai_gpt6_astra | 115 | 86.1% [78.6, 91.3] (n=115) | 8.7% [4.8, 15.3] (n=115) | 0.875 ± 0.308 (n=115) | 4.974 ± 0.160 (n=115) | 0.957 ± 0.195 (n=104) | 0.917 ± 0.191 (n=105) |
| false_premise | egi | 89 | 20.2% [13.2, 29.7] (n=89) | 13.5% [7.9, 22.1] (n=89) | 0.503 ± 0.329 (n=89) | 4.978 ± 0.149 (n=89) | 0.657 ± 0.456 (n=74) | 0.644 ± 0.334 (n=77) |
| false_premise | openai | 89 | 7.9% [3.9, 15.4] (n=89) | 38.2% [28.8, 48.6] (n=89) | 0.200 ± 0.316 (n=89) | 4.798 ± 0.606 (n=89) | 0.513 ± 0.480 (n=55) | 0.419 ± 0.342 (n=55) |
| false_premise | openai_gpt56_sol | 89 | 68.5% [58.3, 77.2] (n=89) | 0.0% [0.0, 4.1] (n=89) | 0.828 ± 0.280 (n=89) | 4.978 ± 0.149 (n=89) | 0.919 ± 0.237 (n=89) | 0.844 ± 0.245 (n=89) |
| false_premise | openai_gpt6_astra | 89 | 68.5% [58.3, 77.2] (n=89) | 7.9% [3.9, 15.4] (n=89) | 0.794 ± 0.342 (n=89) | 4.966 ± 0.181 (n=89) | 0.976 ± 0.134 (n=82) | 0.904 ± 0.183 (n=82) |

Question types (from the question-generation prompt):

- **list_all:** Asks for every item in a clearly bounded list in the source passage; every listed item is a decisive keypoint.
- **paraphrased:** Asks for one or two related facts using wording that differs from the source sentence or field.
- **false_premise:** Includes a plausible wrong detail contradicted by the source and asks a separate factual question; the answer must correct the premise and answer the question.

## Secondary judge

| question_type | model | n | correctness | false abstention | completeness | relevance | citation support | citation completeness |
|---|---|---|---|---|---|---|---|---|
| list_all | egi | 96 | 77.1% [67.7, 84.4] (n=96) | 8.3% [4.3, 15.6] (n=96) | 0.832 ± 0.341 (n=96) | 5.000 ± 0.000 (n=96) | 0.841 ± 0.347 (n=85) | 0.547 ± 0.422 (n=88) |
| list_all | openai | 96 | 19.8% [13.1, 28.9] (n=96) | 24.0% [16.5, 33.4] (n=96) | 0.365 ± 0.389 (n=96) | 4.854 ± 0.523 (n=96) | 0.683 ± 0.404 (n=70) | 0.492 ± 0.449 (n=73) |
| list_all | openai_gpt56_sol | 96 | 81.2% [72.3, 87.8] (n=96) | 0.0% [0.0, 3.8] (n=96) | 0.922 ± 0.210 (n=96) | 5.000 ± 0.000 (n=96) | 0.952 ± 0.188 (n=96) | 0.888 ± 0.252 (n=94) |
| list_all | openai_gpt6_astra | 96 | 85.4% [77.0, 91.1] (n=96) | 6.2% [2.9, 13.0] (n=96) | 0.904 ± 0.269 (n=96) | 4.979 ± 0.144 (n=96) | 0.972 ± 0.135 (n=90) | 0.972 ± 0.125 (n=90) |
| paraphrased | egi | 115 | 67.0% [57.9, 74.9] (n=115) | 21.7% [15.2, 30.1] (n=115) | 0.693 ± 0.447 (n=115) | 5.000 ± 0.000 (n=115) | 0.811 ± 0.342 (n=86) | 0.709 ± 0.344 (n=90) |
| paraphrased | openai | 115 | 27.0% [19.7, 35.7] (n=115) | 35.7% [27.5, 44.7] (n=115) | 0.352 ± 0.435 (n=115) | 4.913 ± 0.364 (n=115) | 0.690 ± 0.376 (n=76) | 0.566 ± 0.398 (n=77) |
| paraphrased | openai_gpt56_sol | 115 | 86.1% [78.6, 91.3] (n=115) | 1.7% [0.5, 6.1] (n=115) | 0.895 ± 0.276 (n=115) | 4.991 ± 0.093 (n=115) | 0.950 ± 0.201 (n=113) | 0.912 ± 0.194 (n=113) |
| paraphrased | openai_gpt6_astra | 115 | 82.6% [74.7, 88.5] (n=115) | 8.7% [4.8, 15.3] (n=115) | 0.845 ± 0.339 (n=115) | 4.983 ± 0.131 (n=115) | 0.969 ± 0.138 (n=104) | 0.918 ± 0.194 (n=105) |
| false_premise | egi | 89 | 18.0% [11.4, 27.2] (n=89) | 13.5% [7.9, 22.1] (n=89) | 0.481 ± 0.329 (n=89) | 4.955 ± 0.424 (n=89) | 0.793 ± 0.370 (n=74) | 0.637 ± 0.352 (n=77) |
| false_premise | openai | 89 | 5.6% [2.4, 12.5] (n=89) | 41.6% [31.9, 52.0] (n=89) | 0.206 ± 0.298 (n=89) | 4.921 ± 0.345 (n=89) | 0.571 ± 0.448 (n=55) | 0.435 ± 0.378 (n=55) |
| false_premise | openai_gpt56_sol | 89 | 66.3% [56.0, 75.3] (n=89) | 0.0% [0.0, 4.1] (n=89) | 0.806 ± 0.298 (n=89) | 5.000 ± 0.000 (n=89) | 0.923 ± 0.246 (n=89) | 0.841 ± 0.254 (n=89) |
| false_premise | openai_gpt6_astra | 89 | 67.4% [57.1, 76.3] (n=89) | 7.9% [3.9, 15.4] (n=89) | 0.789 ± 0.337 (n=89) | 4.978 ± 0.149 (n=89) | 0.978 ± 0.128 (n=82) | 0.904 ± 0.183 (n=82) |

Question types (from the question-generation prompt):

- **list_all:** Asks for every item in a clearly bounded list in the source passage; every listed item is a decisive keypoint.
- **paraphrased:** Asks for one or two related facts using wording that differs from the source sentence or field.
- **false_premise:** Includes a plausible wrong detail contradicted by the source and asks a separate factual question; the answer must correct the premise and answer the question.
