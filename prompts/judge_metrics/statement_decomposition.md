# Shared statement decomposition

## Metric definition

> Factual accuracy concerns whether verifiable claims in a response agree with external factual evidence. The source assessment distinguishes unsupported claims from claims whose status cannot be determined.

> Citation support concerns whether a statement, interpreted in context, is attributable to its identified information source. Here the assessment unit is a claim and the source attached to it by a citation; topic overlap alone is insufficient.

## Judge sees

The answer only.

## Output

Return JSON `{"statements": [{"statement": "...", "cited_passage_ids": ["P1"], "claims": ["..."]}], "reason": "<one sentence>"}`. Split into sentences or list items and then atomic claims. Preserve citations exactly. Every response statement must appear once; do not add claims or evaluate truth.
