"""Tests for paired-proportion confidence intervals."""
import unittest

from scripts.supplement_final_analysis import newcombe_paired


class NewcombeMethod10Tests(unittest.TestCase):
    def test_method_10_known_paired_table(self):
        # Table: both=20, left-only=10, right-only=15, neither=55.
        left = [True] * 30 + [False] * 70
        right = [True] * 20 + [False] * 10 + [True] * 15 + [False] * 55
        estimate, lower, upper = newcombe_paired(left, right)
        self.assertAlmostEqual(estimate, 0.05)
        self.assertAlmostEqual(lower, -0.0472082, places=6)
        self.assertAlmostEqual(upper, 0.1459199, places=6)

    def test_swapping_paired_conditions_reverses_interval(self):
        left = [True] * 30 + [False] * 70
        right = [True] * 20 + [False] * 10 + [True] * 15 + [False] * 55
        estimate, lower, upper = newcombe_paired(left, right)
        reverse_estimate, reverse_lower, reverse_upper = newcombe_paired(right, left)
        self.assertAlmostEqual(reverse_estimate, -estimate)
        self.assertAlmostEqual(reverse_lower, -upper)
        self.assertAlmostEqual(reverse_upper, -lower)


if __name__ == "__main__":
    unittest.main()
