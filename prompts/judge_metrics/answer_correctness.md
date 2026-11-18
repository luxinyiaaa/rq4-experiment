# Answer correctness and completeness

## Metric definition

> Answer correctness concerns whether the final answer reaches the determinate result required by the task. Where a reference answer is used, equivalent meaning can satisfy the criterion despite different wording, provided that additional information remains correct.

> Completeness concerns whether the response addresses every part of the question. For explanations, it also concerns inclusion of the necessary reference facts or reasoning elements. The required elements are determined by the task and its reference.

## Inputs

The RAGChecker response-to-answer interface receives the question, keypoints, and answer. Keypoint labels are Entailment, Neutral, or Contradiction. The library's label endpoint must return exactly one label; the scoring record stores that label with a one-sentence reason based on the returned label and keypoint text.

## Decision

Correctness is true only when every decisive keypoint is Entailment and no keypoint is Contradiction. Completeness is the share of all keypoints labelled Entailment.
