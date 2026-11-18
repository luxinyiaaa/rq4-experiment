from __future__ import annotations

import unittest

from scripts.analyze_metrics_v2 import get_value, newcombe_paired, wilson


class MetricsV2AnalysisTests(unittest.TestCase):
    def test_wilson_interval_for_small_sample(self):
        lo, hi = wilson(8, 12)
        self.assertAlmostEqual(lo, 0.3906, places=3)
        self.assertAlmostEqual(hi, 0.8619, places=3)

    def test_newcombe_method_10_paired_interval(self):
        # a=25, b=8, c=4, d=7 gives a paired difference of -4/44.
        left = [True] * 33 + [False] * 11
        right = [True] * 25 + [False] * 8 + [True] * 4 + [False] * 7
        delta, lower, upper = newcombe_paired(left, right)
        self.assertAlmostEqual(delta, -4 / 44, places=12)
        self.assertLess(lower, delta)
        self.assertGreater(upper, delta)

    def test_applicability_and_metric_value_extraction(self):
        record = {"metrics": {
            "answer_correctness": {"primary": True},
            "evidence_faithfulness": {"primary": {"value": 0.75, "status": "scored"}},
            "appropriate_abstention": {"primary": {"label": "answers", "measure": {"name": "false_abstention_rate", "value": False}}},
        }}
        self.assertIs(get_value(record, "answer_correctness", "primary"), True)
        self.assertEqual(get_value(record, "evidence_faithfulness", "primary"), 0.75)
        self.assertIsNone(get_value(record, "appropriate_abstention", "primary", "abstention_rate"))
        self.assertIs(get_value(record, "appropriate_abstention", "primary", "false_abstention_rate"), False)


if __name__ == "__main__":
    unittest.main()
