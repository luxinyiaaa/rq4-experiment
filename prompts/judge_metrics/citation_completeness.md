# Citation completeness

## Metric definition

> Citation completeness concerns whether claims needing attribution are accompanied by a citation or identifiable source; Citation Presence examines citations accompanying claims. Source Attribution additionally makes explicit that sources should be identified when attribution is appropriate.

## Judge sees

Statements and passages.

## Output

For each numbered input statement, return exactly one matching result, in the same order, with its `statement_id`, `needs_attribution` true or false, and a one-sentence `reason`. Copy each `statement_id` exactly. Return exactly one JSON object with the top-level key `statements`; do not add an aggregate score. A statement needs attribution when it states a fact taken from the passages. Statements that only introduce a list, summarise earlier statements, or say that the passages do not contain something have `needs_attribution: false`. The program checks whether each such statement has at least one citation. Report the share of statements needing attribution that carry a citation; citation correctness is assessed separately.
