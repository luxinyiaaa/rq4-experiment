# Supporting quote check

Decide whether the quote, in the context of its supplied source passage, supports the stated keypoint. Exact textual occurrence alone is insufficient. The quote may support a simple, valid inference from the stated evidence, including interpreting a year within a stated coverage interval or comparing numeric values. Do not accept an inference that requires unstated facts. Mark `supported` false if the quote is unrelated, ambiguous, incomplete, or supports only a different claim. Each source/quote pair is checked independently; multiple pairs may jointly support the keypoint. Derived conclusion keypoints and absence claims are not assessed with this prompt.

Return JSON only:
{"supported":true,"reason":"One concise explanation."}

## User template

Keypoint: {keypoint}
Source quote: {quote}
Source passage:
{passage}
