# Human–judge agreement for the rerun sample

The 40 human-labelled answers were selected with seed 20261116, 10 per model. Correctness is calculated from human keypoint labels. Decisive-keypoint agreement uses Entailment versus Neutral/Contradiction, pooled across decisive keypoints. Cohen's kappa is unweighted for correctness, keypoints and abstention; relevance uses quadratic weighted kappa, matching the earlier Table 6.7 calculation.

| Group | Judge | Metric | n | Exact agreement | Cohen κ | Within one | Quadratic weighted κ | Spearman ρ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| overall | primary | correctness | 40 | 1.000 | 1.000 | — | — | — |
| overall | primary | decisive_keypoint_present | 96 | 0.969 | 0.936 | — | — | — |
| overall | primary | decisive_keypoint_1_present | 40 | 0.975 | 0.950 | — | — | — |
| overall | primary | decisive_keypoint_2_present | 33 | 0.970 | 0.930 | — | — | — |
| overall | primary | decisive_keypoint_3_present | 11 | 0.909 | 0.814 | — | — | — |
| overall | primary | decisive_keypoint_4_present | 4 | 1.000 | 1.000 | — | — | — |
| overall | primary | decisive_keypoint_5_present | 3 | 1.000 | 1.000 | — | — | — |
| overall | primary | decisive_keypoint_6_present | 3 | 1.000 | 1.000 | — | — | — |
| overall | primary | decisive_keypoint_7_present | 1 | 1.000 | — | — | — | — |
| overall | primary | decisive_keypoint_8_present | 1 | 1.000 | — | — | — | — |
| overall | primary | abstention | 40 | 0.950 | 0.880 | — | — | — |
| overall | primary | abstention_partly_merged | 40 | 0.950 | 0.875 | — | — | — |
| overall | primary | relevance | 40 | 0.925 | — | 1.000 | 0.700 | 0.753 |
| overall | secondary | correctness | 40 | 0.950 | 0.900 | — | — | — |
| overall | secondary | decisive_keypoint_present | 96 | 0.938 | 0.873 | — | — | — |
| overall | secondary | decisive_keypoint_1_present | 40 | 0.900 | 0.800 | — | — | — |
| overall | secondary | decisive_keypoint_2_present | 33 | 0.939 | 0.857 | — | — | — |
| overall | secondary | decisive_keypoint_3_present | 11 | 1.000 | 1.000 | — | — | — |
| overall | secondary | decisive_keypoint_4_present | 4 | 1.000 | 1.000 | — | — | — |
| overall | secondary | decisive_keypoint_5_present | 3 | 1.000 | 1.000 | — | — | — |
| overall | secondary | decisive_keypoint_6_present | 3 | 1.000 | 1.000 | — | — | — |
| overall | secondary | decisive_keypoint_7_present | 1 | 1.000 | — | — | — | — |
| overall | secondary | decisive_keypoint_8_present | 1 | 1.000 | — | — | — | — |
| overall | secondary | abstention | 40 | 0.975 | 0.939 | — | — | — |
| overall | secondary | abstention_partly_merged | 40 | 0.975 | 0.939 | — | — | — |
| overall | secondary | relevance | 40 | 0.950 | — | 1.000 | 0.737 | 0.816 |
| EGI gpt-oss-120b | primary | correctness | 10 | 1.000 | 1.000 | — | — | — |
| EGI gpt-oss-120b | primary | decisive_keypoint_present | 19 | 0.842 | 0.650 | — | — | — |
| EGI gpt-oss-120b | primary | decisive_keypoint_1_present | 10 | 0.900 | 0.800 | — | — | — |
| EGI gpt-oss-120b | primary | decisive_keypoint_2_present | 7 | 0.857 | 0.588 | — | — | — |
| EGI gpt-oss-120b | primary | decisive_keypoint_3_present | 2 | 0.500 | 0.000 | — | — | — |
| EGI gpt-oss-120b | primary | abstention | 10 | 1.000 | 1.000 | — | — | — |
| EGI gpt-oss-120b | primary | abstention_partly_merged | 10 | 1.000 | 1.000 | — | — | — |
| EGI gpt-oss-120b | primary | relevance | 10 | 1.000 | — | 1.000 | — | — |
| EGI gpt-oss-120b | secondary | correctness | 10 | 1.000 | 1.000 | — | — | — |
| EGI gpt-oss-120b | secondary | decisive_keypoint_present | 19 | 0.895 | 0.774 | — | — | — |
| EGI gpt-oss-120b | secondary | decisive_keypoint_1_present | 10 | 0.900 | 0.800 | — | — | — |
| EGI gpt-oss-120b | secondary | decisive_keypoint_2_present | 7 | 0.857 | 0.588 | — | — | — |
| EGI gpt-oss-120b | secondary | decisive_keypoint_3_present | 2 | 1.000 | 1.000 | — | — | — |
| EGI gpt-oss-120b | secondary | abstention | 10 | 1.000 | 1.000 | — | — | — |
| EGI gpt-oss-120b | secondary | abstention_partly_merged | 10 | 1.000 | 1.000 | — | — | — |
| EGI gpt-oss-120b | secondary | relevance | 10 | 1.000 | — | 1.000 | — | — |
| GPT-4.1 | primary | correctness | 10 | 1.000 | 1.000 | — | — | — |
| GPT-4.1 | primary | decisive_keypoint_present | 23 | 1.000 | 1.000 | — | — | — |
| GPT-4.1 | primary | decisive_keypoint_1_present | 10 | 1.000 | 1.000 | — | — | — |
| GPT-4.1 | primary | decisive_keypoint_2_present | 8 | 1.000 | 1.000 | — | — | — |
| GPT-4.1 | primary | decisive_keypoint_3_present | 2 | 1.000 | 1.000 | — | — | — |
| GPT-4.1 | primary | decisive_keypoint_4_present | 1 | 1.000 | — | — | — | — |
| GPT-4.1 | primary | decisive_keypoint_5_present | 1 | 1.000 | — | — | — | — |
| GPT-4.1 | primary | decisive_keypoint_6_present | 1 | 1.000 | — | — | — | — |
| GPT-4.1 | primary | abstention | 10 | 0.800 | 0.565 | — | — | — |
| GPT-4.1 | primary | abstention_partly_merged | 10 | 0.800 | 0.545 | — | — | — |
| GPT-4.1 | primary | relevance | 10 | 0.800 | — | 1.000 | 0.677 | 0.759 |
| GPT-4.1 | secondary | correctness | 10 | 0.900 | 0.000 | — | — | — |
| GPT-4.1 | secondary | decisive_keypoint_present | 23 | 0.870 | 0.496 | — | — | — |
| GPT-4.1 | secondary | decisive_keypoint_1_present | 10 | 0.800 | -0.111 | — | — | — |
| GPT-4.1 | secondary | decisive_keypoint_2_present | 8 | 0.875 | 0.600 | — | — | — |
| GPT-4.1 | secondary | decisive_keypoint_3_present | 2 | 1.000 | 1.000 | — | — | — |
| GPT-4.1 | secondary | decisive_keypoint_4_present | 1 | 1.000 | — | — | — | — |
| GPT-4.1 | secondary | decisive_keypoint_5_present | 1 | 1.000 | — | — | — | — |
| GPT-4.1 | secondary | decisive_keypoint_6_present | 1 | 1.000 | — | — | — | — |
| GPT-4.1 | secondary | abstention | 10 | 0.900 | 0.737 | — | — | — |
| GPT-4.1 | secondary | abstention_partly_merged | 10 | 0.900 | 0.737 | — | — | — |
| GPT-4.1 | secondary | relevance | 10 | 0.800 | — | 1.000 | 0.630 | 0.745 |
| GPT-5.6 Sol | primary | correctness | 10 | 1.000 | 1.000 | — | — | — |
| GPT-5.6 Sol | primary | decisive_keypoint_present | 27 | 1.000 | 1.000 | — | — | — |
| GPT-5.6 Sol | primary | decisive_keypoint_1_present | 10 | 1.000 | 1.000 | — | — | — |
| GPT-5.6 Sol | primary | decisive_keypoint_2_present | 9 | 1.000 | — | — | — | — |
| GPT-5.6 Sol | primary | decisive_keypoint_3_present | 4 | 1.000 | 1.000 | — | — | — |
| GPT-5.6 Sol | primary | decisive_keypoint_4_present | 2 | 1.000 | — | — | — | — |
| GPT-5.6 Sol | primary | decisive_keypoint_5_present | 1 | 1.000 | — | — | — | — |
| GPT-5.6 Sol | primary | decisive_keypoint_6_present | 1 | 1.000 | — | — | — | — |
| GPT-5.6 Sol | primary | abstention | 10 | 1.000 | — | — | — | — |
| GPT-5.6 Sol | primary | abstention_partly_merged | 10 | 1.000 | — | — | — | — |
| GPT-5.6 Sol | primary | relevance | 10 | 0.900 | — | 1.000 | 0.000 | — |
| GPT-5.6 Sol | secondary | correctness | 10 | 0.900 | 0.737 | — | — | — |
| GPT-5.6 Sol | secondary | decisive_keypoint_present | 27 | 0.963 | 0.780 | — | — | — |
| GPT-5.6 Sol | secondary | decisive_keypoint_1_present | 10 | 0.900 | 0.615 | — | — | — |
| GPT-5.6 Sol | secondary | decisive_keypoint_2_present | 9 | 1.000 | — | — | — | — |
| GPT-5.6 Sol | secondary | decisive_keypoint_3_present | 4 | 1.000 | 1.000 | — | — | — |
| GPT-5.6 Sol | secondary | decisive_keypoint_4_present | 2 | 1.000 | — | — | — | — |
| GPT-5.6 Sol | secondary | decisive_keypoint_5_present | 1 | 1.000 | — | — | — | — |
| GPT-5.6 Sol | secondary | decisive_keypoint_6_present | 1 | 1.000 | — | — | — | — |
| GPT-5.6 Sol | secondary | abstention | 10 | 1.000 | — | — | — | — |
| GPT-5.6 Sol | secondary | abstention_partly_merged | 10 | 1.000 | — | — | — | — |
| GPT-5.6 Sol | secondary | relevance | 10 | 1.000 | — | 1.000 | — | — |
| GPT-6 Astra | primary | correctness | 10 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | primary | decisive_keypoint_present | 27 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | primary | decisive_keypoint_1_present | 10 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | primary | decisive_keypoint_2_present | 9 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | primary | decisive_keypoint_3_present | 3 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | primary | decisive_keypoint_4_present | 1 | 1.000 | — | — | — | — |
| GPT-6 Astra | primary | decisive_keypoint_5_present | 1 | 1.000 | — | — | — | — |
| GPT-6 Astra | primary | decisive_keypoint_6_present | 1 | 1.000 | — | — | — | — |
| GPT-6 Astra | primary | decisive_keypoint_7_present | 1 | 1.000 | — | — | — | — |
| GPT-6 Astra | primary | decisive_keypoint_8_present | 1 | 1.000 | — | — | — | — |
| GPT-6 Astra | primary | abstention | 10 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | primary | abstention_partly_merged | 10 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | primary | relevance | 10 | 1.000 | — | 1.000 | 1.000 | 1.000 |
| GPT-6 Astra | secondary | correctness | 10 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | secondary | decisive_keypoint_present | 27 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | secondary | decisive_keypoint_1_present | 10 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | secondary | decisive_keypoint_2_present | 9 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | secondary | decisive_keypoint_3_present | 3 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | secondary | decisive_keypoint_4_present | 1 | 1.000 | — | — | — | — |
| GPT-6 Astra | secondary | decisive_keypoint_5_present | 1 | 1.000 | — | — | — | — |
| GPT-6 Astra | secondary | decisive_keypoint_6_present | 1 | 1.000 | — | — | — | — |
| GPT-6 Astra | secondary | decisive_keypoint_7_present | 1 | 1.000 | — | — | — | — |
| GPT-6 Astra | secondary | decisive_keypoint_8_present | 1 | 1.000 | — | — | — | — |
| GPT-6 Astra | secondary | abstention | 10 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | secondary | abstention_partly_merged | 10 | 1.000 | 1.000 | — | — | — |
| GPT-6 Astra | secondary | relevance | 10 | 1.000 | — | 1.000 | 1.000 | 1.000 |
