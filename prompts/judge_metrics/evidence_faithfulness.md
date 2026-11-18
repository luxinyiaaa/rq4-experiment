# Evidence faithfulness

## Metric definition

> Evidence faithfulness concerns whether the claims in an answer can be inferred from the context supplied for the task. This is agreement with specified evidence; it does not by itself establish that the evidence is factually correct.

## Judge sees

Question, passages, and answer.

## Output

Use Ragas Faithfulness. Store the score and return a JSON record with `score` and a one-sentence `reason` describing the ratio of supported claims to extracted claims. If no claims can be extracted, store an excluded/NaN result and its reason.
