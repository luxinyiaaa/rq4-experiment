# Retrieved-evidence relevance

## Metric definition

> Retrieved-evidence relevance concerns whether retrieved passages are useful for answering the current question. The assessed object is the retrieved evidence, rather than the generated answer.

## Judge sees

Question, keypoints, and the ten retrieved chunks.

## Output

Use RAGChecker context precision: the share of retrieved chunks that contain at least one keypoint. RAGChecker's binary label interface must return exactly one Entailment, Neutral, or Contradiction label per keypoint/chunk comparison. Store each label in a JSON record with a one-sentence reason tied to the chunk and keypoint. Note that with one gold page and fixed k=10, most chunks are expected to be irrelevant and the value partly reflects k.
