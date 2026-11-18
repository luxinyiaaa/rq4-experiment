# Controlled robustness

## Metric definition

> Controlled robustness concerns maintaining answer quality under specified input changes that preserve the intended task. In retrieval-based tasks, this can include irrelevant retrieved material and misleading evidence under explicitly stated conditions. Evaluation uses matched conditions and a stated baseline; a changed answer alone does not establish failure.

## Judge sees

No judge input. Compute from paired per-question metrics 1–10.

## Output

No extra calls. Report paired differences C1 to C2 and C1 to C4 for metrics applicable in both conditions.
