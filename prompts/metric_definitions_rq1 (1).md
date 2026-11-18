# RQ4 metric set: which quality-model metrics are measured, and how

**Selection.** 10 metrics of the quality model (RQ1, 8 dimensions, 53 metrics) are measured. They cover four dimensions: Content quality, Task fulfilment, Evidence and citation quality, and Reliability.

**Methods.** Each metric is measured with a method from RQ2 (Table 9.5), with the match level from Table 9.5 (Direct, Partial, Proxy). Each metric is reported as one number per condition. Appropriate abstention reports two numbers because its definition requires both directions: refusing when it should and not refusing when it should not.

**Judges.** Primary judge: the teacher model (GPT-6 Sol), which wrote the questions and reference answers. Secondary judge: Qwen3.7 Plus. Every judge-based measure is scored by both; agreement is reported per metric. Each judge call sees only the inputs listed under "Judge sees".

**Applicability per answer.** An answer that does not meet a metric's applicability (for example, Evidence faithfulness for an abstention) is left out of that metric; the number of applicable answers is reported. Whether an answer abstains is decided once, by the primary judge's abstention label, and both judges then score the same set of answers, so that agreement compares the same answers. Agreement on the abstention label itself is reported under metric 9.

**Reporting.** All 10 metrics appear in the main results table. A 95% interval is shown when at least 10 answers apply; with fewer, the value is listed with its n only. A discrimination table shows, per metric, how many answers get the best possible value. The effect of retrieval (C1 → C4) is reported in its own table, "Retrieval loss", next to Controlled robustness (C1 → C2).

---

## Summary

| # | Dimension | Metric | RQ2 method (match) | Measure | Conditions |
|---|---|---|---|---|---|
| 1 | Content quality | Answer correctness | RAGChecker reference-claim check (Partial) | Share correct: all decisive keypoints Entailment, no keypoint Contradiction | C1, C2, C4 |
| 2 | Content quality | Evidence faithfulness | Ragas claim–context support (Direct) | Ragas Faithfulness (0–1) | all; not for abstentions |
| 3 | Task fulfilment | Response relevance | User-intent item (Direct; human item → LLM judge) | Rubric 1–5 | all |
| 4 | Task fulfilment | Completeness | RAGChecker reference-claim recall (Direct) | Share of all keypoints Entailment | C1, C2, C4 |
| 5 | Evidence and citation | Retrieved-evidence relevance | RAGChecker context precision (Partial) | Share of the 10 retrieved chunks that contain at least one keypoint | C4 |
| 6 | Evidence and citation | Retrieved-evidence coverage | RAGChecker retrieved-set claim recall (Direct) | Share of keypoints covered by the retrieved chunks | C4 |
| 7 | Evidence and citation | Citation support | ALCE citation entailment (Direct) | Citation recall: share of cited statements entailed by their cited passages | answers with citations |
| 8 | Evidence and citation | Citation completeness | ALCE supported-citation recall (Partial); LLM-Rubric citation presence (Partial) | Share of statements needing attribution that carry a citation | non-abstaining answers |
| 9 | Reliability | Appropriate abstention | RGB negative-evidence rejection (Partial); refusal audit (Partial) | C3: abstention rate; C1, C2, C4: false-abstention rate | all |
| 10 | Reliability | Controlled robustness | HELM task-preserving perturbation (Direct) | Paired difference per question on metrics 1–4 and 7–8: C1 → C2 (distractors added) | C1 and C2 |

---

## Metric details

Definitions are quoted from Table 9.3 without reference numbers. "Applicability" is the Table 9.3 entry.

### Content quality

**1. Answer correctness**
- Definition: Answer correctness concerns whether the final answer reaches the determinate result required by the task. Where a reference answer is used, equivalent meaning can satisfy the criterion despite different wording, provided that additional information remains correct.
- Applicability: Tasks with a determinate result or defensible reference answer; allow valid equivalent answers.
- Measure: RAGChecker (non-joint mode) labels each keypoint against the answer as Entailment, Neutral or Contradiction. Correct = every decisive keypoint Entailment and no keypoint Contradiction. In S2 the expected behaviour (list all, disambiguate, flag the conflict, state the absence) is part of the decisive keypoints.
- Judge sees: question, keypoints, answer.
- Not in C3: the passages contain no answer.

**2. Evidence faithfulness**
- Definition: Evidence faithfulness concerns whether the claims in an answer can be inferred from the context supplied for the task. This is agreement with specified evidence; it does not by itself establish that the evidence is factually correct.
- Applicability: Tasks in which supporting context or evidence is supplied to the model.
- Measure: Ragas Faithfulness: the answer is split into claims, each claim is judged as inferable or not from the passages shown in that condition (C1 gold window; C2 gold window and four distractors; C3 four distractors; C4 the 10 retrieved chunks); score = supported claims / all claims.
- Judge sees: question, passages, answer.

### Task fulfilment

**3. Response relevance**
- Definition: Response relevance concerns whether an answer addresses the requested question without off-topic elaboration. In dialogue, it also concerns whether the response continues the preceding conversational context appropriately.
- Applicability: Question answering and dialogue; include relevant preceding turns when interpreting the request.
- Measure: rubric 1–5 with a one-sentence reason. 5 addresses the question without off-topic elaboration; 4 one minor off-topic detail; 3 partly addresses it or contains substantial off-topic content; 2 mostly off-topic; 1 does not address it. Pointing out that a detail of the question is wrong (a false premise) addresses the question. A justified statement that the information is not available addresses the question; describing other records or other content of the passages beyond one short sentence is off-topic elaboration, also in an abstention. Single-turn, so only the first sentence of the definition applies.
- Judge sees: question, answer.

**4. Completeness**
- Definition: Completeness concerns whether the response addresses every part of the question. For explanations, it also concerns inclusion of the necessary reference facts or reasoning elements. The required elements are determined by the task and its reference.
- Applicability: Tasks where required content or answer components can be specified.
- Measure: share of all keypoints (decisive and non-decisive) labelled Entailment, from the same RAGChecker run as metric 1.

### Evidence and citation quality

**5. Retrieved-evidence relevance**
- Definition: Retrieved-evidence relevance concerns whether retrieved passages are useful for answering the current question. The assessed object is the retrieved evidence, rather than the generated answer.
- Applicability: Retrieval-based tasks with an identified question and the retrieved material available for inspection.
- Measure: RAGChecker context precision with the keypoints as ground-truth claims: share of the 10 retrieved chunks that contain at least one keypoint.
- Note: with one gold page per question and a fixed top 10, most chunks are expected to be irrelevant; the value partly reflects the choice of k = 10.

**6. Retrieved-evidence coverage**
- Definition: Retrieved-evidence coverage concerns whether the retrieved documents jointly supply evidence for the aspects required by the question. Coverage is assessed over the evidence set, rather than inferred from the relevance of one passage.
- Applicability: Retrieval-based tasks where required information aspects can be specified independently of passage relevance.
- Measure: RAGChecker claim recall: share of keypoints supported by the 10 retrieved chunks together.

**7. Citation support**
- Definition: Citation support concerns whether a statement, interpreted in context, is attributable to its identified information source. Here the assessment unit is a claim and the source attached to it by a citation; topic overlap alone is insufficient.
- Applicability: Responses containing identifiable citations with accessible source content.
- Measure (ALCE citation recall): for each statement with at least one citation, the judge decides whether its cited passages together entail it. A cited ID that is not among the given passages supplies no text (program). Score = entailed cited statements / cited statements.
- Judge sees: statement, text of its cited passages.

**8. Citation completeness**
- Definition: Citation completeness concerns whether claims needing attribution are accompanied by a citation or identifiable source; Citation Presence examines citations accompanying claims. Source Attribution additionally makes explicit that sources should be identified when attribution is appropriate.
- Applicability: Tasks requiring source attribution. Thesis scope: assess attribution within the current response.
- Measure: share of statements needing attribution that carry at least one citation. The judge labels which statements state a fact taken from the passages (`needs_attribution`); citation presence is checked by program. Statements that only introduce a list, summarise earlier statements, or say that the passages do not contain something do not need attribution. Presence only; whether the citation is right is metric 7.
- Judge sees: statements, passages.

### Reliability

**9. Appropriate abstention**
- Definition: Appropriate abstention concerns withholding an answer when the supplied evidence cannot support one. Under clinical uncertainty, justified deferral or seeking further information also contributes to this judgment. Assessment must also consider inappropriate refusals on benign tasks, rather than treating every refusal as desirable.
- Applicability: Tasks with assessable evidence sufficiency, answerability or refusal conditions; include unjustified refusal.
- Measure: the judge labels each answer as answers, abstains or partly answers. As the definition requires both directions:
  - C3 (the passages contain no answer): abstention rate, higher is better;
  - C1 and C2 (the passages contain the answer): false-abstention rate, lower is better;
  - C4: false-abstention rate among questions whose decisive keypoints were all retrieved (metric 6), lower is better. This set of questions is defined once, by the primary judge's coverage labels, and used for both judges, so that both judges are compared on the same questions.
  Partial answers count as not abstaining. Descriptive C4 rows report the overall abstention rate and partial-answer rate. S2 `state_absence` questions: whether the answer states that no such record is available.
- Judge sees: question, answer.

**10. Controlled robustness**
- Definition: Controlled robustness concerns maintaining answer quality under specified input changes that preserve the intended task. In retrieval-based tasks, this can include irrelevant retrieved material and misleading evidence under explicitly stated conditions. Evaluation uses matched conditions and a stated baseline; a changed answer alone does not establish failure.
- Applicability: Matched input conditions. Thesis restriction: changes must preserve intended task requirements.
- Measure: for each of metrics 1–4 and 7–8, the paired difference per question C1 → C2: the same question and gold window, with four distractor windows added and the gold position rotated. S2: question_source → question_user. Binary outcomes: paired difference with a Newcombe interval and an exact McNemar test; scores: mean paired difference with a bootstrap interval and a Wilcoxon signed-rank test; Holm adjustment across the metrics of one comparison. No extra judge calls.

---

## Retrieval loss (C1 → C4)

Not a metric of the quality model, but the main effect studied: the same paired statistics as metric 10, for C1 → C4 (retrieved chunks instead of the gold window). C4 replaces the evidence source rather than perturbing the input while keeping the evidence, so it is reported separately from Controlled robustness.

---

## Shared steps

- **RAGChecker run** (primary and secondary judge): keypoint labels for metrics 1 and 4; retriever checks of the keypoints against the retrieved chunks for metrics 5 and 6.
- **Statement decomposition** (primary judge, once per non-abstaining answer): the answer is split into statements (sentences or list items), each with its cited passage IDs. Both judges then label the same statements (metrics 7 and 8), so the units are identical for agreement.
