# Instruction fulfilment: R2 evidence insufficiency

## Metric definition

> Instruction fulfilment concerns whether a response meets the explicit and implicit requirements of the instruction, as described by the Comprehension criterion in FLASK. Here the metric denotes observable fulfilment of task requirements, not an inference about the model's internal understanding.

## Judge sees

Generator instruction, passages, and answer.

## Output

When applicable, return JSON `{"met": true|false, "reason": "<one sentence>"}` indicating whether the answer says so when the passages are not enough. This check applies when the answer abstains or partly answers, or when the condition is C3. R1 citation format and valid IDs, and R3 Faithfulness=1.0, are program checks.
