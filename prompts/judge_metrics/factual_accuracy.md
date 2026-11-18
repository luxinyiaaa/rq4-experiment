# Factual accuracy

## Metric definition

> Factual accuracy concerns whether verifiable claims in a response agree with external factual evidence. The source assessment distinguishes unsupported claims from claims whose status cannot be determined.

## Judge sees

Atomic claims and the full cleaned text of the answer's gold pages.

## Output

For each claim return JSON with `label` equal to `supported`, `contradicted`, or `cannot_be_determined`, plus a one-sentence `reason`. Report the supported share and the cannot-be-determined share separately.
