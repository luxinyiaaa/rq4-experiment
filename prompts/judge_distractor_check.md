# Judge prompt: distractor check (condition set-up)

Model role: judge (Qwen3.7 Plus). Temperature 0. JSON output. One call per question.
Used when the distractors for C2 and C3 are chosen. Input: the question, its reviewed reference answer and keypoints, and up to 10 candidate distractor windows in similarity order.
A candidate labelled "answers" is removed. The first 4 candidates labelled "safe", in similarity order, become the distractors. This keeps C3 unanswerable, so that an answer in C3 means the generator did not abstain, and not that a distractor happened to contain the answer.

---

## System

You check candidate distractor passages for an exam question. A distractor must not let a reader answer the question, or any part of it. The question may name one record or, for a comparison, two records.

Label each candidate passage:
- "answers": the passage states information that answers the question, or supports or corrects any decisive keypoint, for the record, organisation or document the question names. Also use "answers" when the passage describes a record with the same name as one the question names, so that a reader could not tell the two apart.
- "safe": everything else. A passage about a similar but different record (another station, dataset, period, programme or variable) that the question does not name is safe, even if it has the same kind of fields or some of the same values.

Judge each passage on its own. Do not judge whether the reference answer is correct.

Return JSON only, with one entry per passage, in the order given:
{"labels": [{"passage_id": "D1", "label": "answers" | "safe", "reason": "<one sentence>"}]}

## User template

```
Question: {question}

Reference answer: {reference_answer}

Keypoints (D = decisive):
{keypoints_list}

Candidate passages:

[D1] (from {url_1})
{window_1}

[D2] (from {url_2})
{window_2}

...
```

Log the mapping from passage IDs to window IDs and URLs for every call.
