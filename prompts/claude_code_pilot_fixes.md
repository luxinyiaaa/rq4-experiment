# Pilot review: fixes, pilot re-run, then full run

I reviewed the pilot. `prompts/metric_definitions_rq1.md` has been updated (replace it with the new version first). Apply the fixes below, re-run the pilot, check the results yourself, and continue to the full run only if every check passes.

## Fixes

1. **One abstention decision for both judges.** Use the primary judge's abstention label (metric 11) to decide, for both judges, which answers count as abstentions for metrics 1–10 and whether R2 of metric 5 applies. The secondary judge still gives its own abstention label, used only for metric 11 and its agreement. Reason: in the pilot, one C4 answer was "partly answers" for the primary and "abstains" for the secondary, so the secondary skipped faithfulness, factual accuracy and the citation metrics for it, and the two judges scored different sets of answers.
2. **Response relevance prompt.** Update the anchors to the new text in the definitions file: pointing out a false premise addresses the question; describing other records or other passage content beyond one short sentence is off-topic, also in an abstention. Reason: the primary judge marked a required false-premise correction as "unnecessary" (W068, C1), and the secondary gave 5 to abstentions with long descriptions of other records.
3. **Citation completeness prompt.** Statements that only introduce a list (for example "The service offers these options:"), summarise earlier statements, or say that the passages do not contain something get `needs_attribution = false`. Reason: in the pilot such statements were counted as missing a citation (W067 C1, W035 C4).
4. **Factual accuracy measure.** Main value = supported / (supported + contradicted); second value = cannot_be_determined / all claims. If an answer has no claim that the gold pages can decide, the main value is not applicable for that answer. Reason: claims that cannot be determined must not count as errors (definition).
5. **Item-level detail.** Store in `items/judgements.jsonl`, with reasons: per-keypoint labels (metrics 1, 6), per-chunk labels (7, 8), per-claim labels (3), per-statement labels (9, 10). In the pilot review, show these labels instead of "Derived from the corresponding judged labels or program check".
6. **Disagreement check.** In the pilot summary, also check continuous metrics: flag a metric when the two judges differ by 0.3 or more on more than 3 of the 12 answers.

## Pilot re-run

Re-run the same 12 pilot answers into `outputs/metrics_v2/pilot_v2/` (keep the first pilot). Unchanged calls come from the cache. Write `pilot_review.md` and `pilot_summary.md` as before. In `pilot_review.md`, for the three C4 answers, list the per-chunk retrieved-evidence labels of both judges side by side (the first pilot had 0.1 vs 0.3 and 0.2 vs 0.6).

## Checks (all must pass)

1. For metrics 1–10, the number of applicable answers is the same for both judges.
2. W068 C1 (false-premise correction): the primary judge's relevance reason does not count the correction as off-topic.
3. No statement that only introduces a list, summarises, or states that the passages lack something is labelled `needs_attribution = true` by either judge.
4. Factual accuracy uses the new formula; claims that cannot be determined are reported separately.
5. Zero parse failures after the single retry.

If any check fails, stop and report which one and why. If all pass, write `pilot_v2/checks.md` with the result of each check and continue with **Step 4 (full run)** and **Step 5 (analysis)** of `prompts/claude_code_metric_expansion.md`, with these fixes applied. If the per-chunk labels show that one judge counts chunks that only mention the topic without containing a keypoint, note it in the summary; do not change the method.

## When done

Print the aggregate summary and list the files for me to upload:
- `pilot_v2/checks.md`, `pilot_v2/pilot_summary.md`;
- everything in `outputs/metrics_v2/summary/`;
- `items/answer_metrics_wide.csv` (I will upload it without opening it).
