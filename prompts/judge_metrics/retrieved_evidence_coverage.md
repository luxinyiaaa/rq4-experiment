# Retrieved-evidence coverage

## Metric definition

> Retrieved-evidence coverage concerns whether the retrieved documents jointly supply evidence for the aspects required by the question. Coverage is assessed over the evidence set, rather than inferred from the relevance of one passage.

## Judge sees

Question, keypoints, and the ten retrieved chunks.

## Output

Use RAGChecker claim recall: the share of keypoints supported by the ten retrieved chunks together. Return the RAGChecker one-word label for each keypoint/chunk comparison and store a JSON record with a one-sentence reason for each label.
