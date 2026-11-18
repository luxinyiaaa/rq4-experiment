# Citation support

## Metric definition

> Citation support concerns whether a statement, interpreted in context, is attributable to its identified information source. Here the assessment unit is a claim and the source attached to it by a citation; topic overlap alone is insufficient.

## Judge sees

One statement and the text of its cited passages. A cited ID absent from the given passages supplies no text.

## Output

Return JSON `{"entailed": true|false, "reason": "<one sentence>"}` indicating whether the cited passages together entail the statement. Citation recall is entailed cited statements divided by cited statements.
