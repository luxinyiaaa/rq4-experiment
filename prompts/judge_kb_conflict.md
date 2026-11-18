# Judge prompt: knowledge-base conflict check (question screening)

Model role: judge (Qwen3.7 Plus). Temperature 0. JSON output. One call per question.
Used before the main run. Input: the question, its reviewed reference answer and keypoints, and the top-10 chunks retrieved for the question over the whole cleaned knowledge base (same index and query prefix as condition C4), with chunks from the gold page and the partner page removed.
Purpose: find other pages that a reader could take as answering the same question with a different value. A flagged question is fixed in review (add a detail that identifies the record, or word the keypoint so that every correct form is accepted), so that a wrong answer in C4 reflects a retrieval or generation error and not two pages that disagree.

---

## System

You check whether passages from a knowledge base contain a different answer to a question than the reference answer.

For each passage, decide:
1. Does it state information that answers the question, or part of it, for the record, organisation or document the question names? A passage about a clearly different record (another station, another dataset, another year) that the question does not refer to does not count.
2. If it does, does that information disagree with the reference answer or one of the keypoints (a different number, date, name, list or statement)? Equivalent wording, units or date formats do not count as disagreement. A passage that gives less detail but does not contradict the reference does not count either.

Mark a passage as a conflict only if both answers are yes. Also mark it when the passage describes a record with the same name as the one in the question, so that a reader could not tell which record the question means, and that record has a different value.

Return JSON only:
{"conflict": true | false, "conflicts": [{"passage_id": "P3", "what_differs": "<one sentence: what the passage says and what the reference says>"}]}
Return {"conflict": false, "conflicts": []} if no passage conflicts.

## User template

```
Question: {question}

Reference answer: {reference_answer}

Keypoints (D = decisive):
{keypoints_list}

Passages from other pages of the knowledge base:

[P1] (from {url_1})
{chunk_1}

[P2] (from {url_2})
{chunk_2}

...
```

Log the mapping from passage IDs to chunk IDs and URLs for every call.
