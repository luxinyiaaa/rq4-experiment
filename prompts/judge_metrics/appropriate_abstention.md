# Appropriate abstention

## Metric definition

> Appropriate abstention concerns withholding an answer when the supplied evidence cannot support one. Under clinical uncertainty, justified deferral or seeking further information also contributes to this judgment. Assessment must also consider inappropriate refusals on benign tasks, rather than treating every refusal as desirable.

## Judge sees

Question and answer.

## Output

Return JSON `{"label": "answers|abstains|partly_answers", "reason": "<one sentence>"}`. C3 reports abstention rate; C1 and C2 report false-abstention rate; C4 reports false-abstention rate only when all decisive keypoints were retrieved. Partial answers count as not abstaining and are listed separately.
