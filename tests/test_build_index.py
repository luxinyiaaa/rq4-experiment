"""Tests for index ranking, prefix handling, and distractor exclusions."""
import unittest

import numpy as np

from scripts.build_index import RetrievalIndex, _url_template


class FakeEncoder:
    def __init__(self):
        self.queries = []

    def encode(self, texts, **kwargs):
        self.queries.extend(texts)
        return np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (len(texts), 1))


class BuildIndexTests(unittest.TestCase):
    def test_url_template_is_domain_and_first_path_segment(self):
        self.assertEqual(
            _url_template("https://example.org/data/record/123"),
            "example.org/data",
        )
        self.assertEqual(_url_template("https://example.org/"), "example.org/")

    def test_retrieve_adds_prefix_and_returns_cosine_ranked_chunks(self):
        encoder = FakeEncoder()
        index = RetrievalIndex(
            encoder,
            {"query_prefix": "Represent this sentence for searching relevant passages: "},
            [
                {"chunk_id": "A", "url": "https://a.example/", "text": "first"},
                {"chunk_id": "B", "url": "https://b.example/", "text": "second"},
            ],
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            [], np.empty((0, 2), dtype=np.float32), [],
        )

        result = index.retrieve("What is measured?", k=1)

        self.assertEqual(result[0]["chunk_id"], "B")
        self.assertAlmostEqual(result[0]["similarity"], 1.0)
        self.assertEqual(
            encoder.queries[0],
            "Represent this sentence for searching relevant passages: What is measured?",
        )

    def test_distractor_excludes_gold_page_same_dataset_and_near_duplicate(self):
        encoder = FakeEncoder()
        gold = {
            "window_id": "W001", "url": "https://gold.example/datasets/1",
            "domain": "gold.example", "page_type": "dataset_record",
            "text": "# Gold dataset title\nUnique measurements and observations from this record.",
        }
        candidates = [
            {"window_id": "same-page", "url": gold["url"], "domain": "gold.example",
             "page_type": "dataset_record", "text": "Different page text."},
            {"window_id": "same-dataset", "url": "https://elsewhere.example/datasets/2",
             "domain": "elsewhere.example", "page_type": "dataset_record",
             "text": "# Gold dataset title\nA different passage for this record."},
            {"window_id": "near-duplicate", "url": "https://elsewhere.example/datasets/3",
             "domain": "elsewhere.example", "page_type": "dataset_record",
             "text": gold["text"]},
            {"window_id": "same-template", "url": "https://gold.example/datasets/4",
             "domain": "gold.example", "page_type": "dataset_record",
             "text": "A separate dataset reports salinity observations in the Baltic Sea."},
            {"window_id": "other-site", "url": "https://other.example/records/5",
             "domain": "other.example", "page_type": "dataset_record",
             "text": "A separate page reports a different set of ocean measurements."},
        ]
        index = RetrievalIndex(
            encoder,
            {"query_prefix": "P: "},
            [], np.empty((0, 2), dtype=np.float32),
            candidates,
            np.array([[0.8, 0.6], [1.0, 0.0], [0.99, 0.1], [0.6, 0.8], [0.99, 0.1]], dtype=np.float32),
            [gold],
        )

        result = index.get_distractor_candidates("W001", n=10)

        self.assertEqual([item["window_id"] for item in result], ["same-template", "other-site"])
        self.assertTrue(all(item["url"] != gold["url"] for item in result))


if __name__ == "__main__":
    unittest.main()
