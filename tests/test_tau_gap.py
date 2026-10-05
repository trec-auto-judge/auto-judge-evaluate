import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autojudge_evaluate.evaluation import LeaderboardEvaluator, parse_correlation_method, tau_gap


class TestTauGap(unittest.TestCase):
    def test_fails_too_few_entries(self):
        with self.assertRaises(ValueError):
            tau_gap([1, 2], [10, 20])

    def test_fails_unequal_entries(self):
        with self.assertRaises(ValueError):
            tau_gap([1, 2, 3, 4], [10, 20, 30, 40, 50])

    def test_perfect_agreement(self):
        self.assertAlmostEqual(tau_gap([0.9, 0.5, 0.45, 0.2], [4, 3, 2, 1]), 1.0)

    def test_perfect_reversal(self):
        self.assertAlmostEqual(tau_gap([0.9, 0.5, 0.45, 0.2], [1, 2, 3, 4]), -1.0)

    def test_hand_computed_three_items(self):
        # pred order A, C, B.  i=C: above {A}, gap .5 correct -> 1.
        # i=B: above {A (gap .4, correct), C (gap .1, wrong)} -> .4/.5 = .8.
        # tau_gap = 2 * (1 + .8) / 2 - 1 = 0.8
        self.assertAlmostEqual(tau_gap([0.9, 0.5, 0.4], [3, 1, 2]), 0.8)

    def test_small_gap_swap_costs_less_than_large_gap_swap(self):
        truth = [0.90, 0.50, 0.45, 0.20]
        small = tau_gap(truth, [0.8, 0.5, 0.6, 0.1])  # swaps B/C, truth gap .05
        large = tau_gap(truth, [0.6, 0.8, 0.4, 0.1])  # swaps A/B, truth gap .40
        self.assertAlmostEqual(small, 0.9259259, places=5)
        self.assertAlmostEqual(large, 0.3333333, places=5)

    def test_invariant_to_scaling_truth(self):
        pred = [0.8, 0.5, 0.6, 0.1]
        self.assertAlmostEqual(tau_gap([0.9, 0.5, 0.45, 0.2], pred),
                               tau_gap([90, 50, 45, 20], pred))

    def test_is_asymmetric(self):
        truth = [0.9, 0.5, 0.45, 0.2]
        pred = [0.8, 0.5, 0.6, 0.1]
        self.assertNotAlmostEqual(tau_gap(truth, pred), tau_gap(pred, truth))

    def test_all_tied_prediction_returns_zero(self):
        self.assertEqual(tau_gap([1, 2, 3], [5, 5, 5]), 0.0)

    def test_ties_in_prediction_only_count_items_strictly_above(self):
        # pred ties B and C; neither counts as "above" the other.
        # i=B: above {A} gap .4 correct -> 1; i=C: above {A} gap .5 correct -> 1
        self.assertAlmostEqual(tau_gap([0.9, 0.5, 0.4], [3, 1, 1]), 1.0)

    def test_parses_as_correlation_method(self):
        self.assertEqual(parse_correlation_method("tau_gap"), ("tau_gap", None))
        self.assertEqual(parse_correlation_method("tau_gap@10"), ("tau_gap", 10))


TRUTH = """
run_01 M query-01 0.90
run_02 M query-01 0.50
run_03 M query-01 0.45
run_04 M query-01 0.20
run_01 M all 0.90
run_02 M all 0.50
run_03 M all 0.45
run_04 M all 0.20
"""

JUDGE = """
run_01 J query-01 0.8
run_02 J query-01 0.5
run_03 J query-01 0.6
run_04 J query-01 0.1
run_01 J all 0.8
run_02 J all 0.5
run_03 J all 0.6
run_04 J all 0.1
"""


class TestTauGapInEvaluator(unittest.TestCase):
    def test_evaluator_reports_tau_gap_with_truth_as_reference(self):
        with TemporaryDirectory() as d:
            truth, judge = Path(d) / "truth", Path(d) / "judge"
            truth.write_text(TRUTH)
            judge.write_text(JUDGE)
            te = LeaderboardEvaluator(truth, truth_format="tot", eval_format="tot",
                                      correlation_methods=["kendall", "tau_gap", "tau_gap@3"])
            actual = te.evaluate(judge)[("M", "J")]
            self.assertAlmostEqual(actual["kendall"], 0.6666667, places=5)
            self.assertAlmostEqual(actual["tau_gap"], 0.9259259, places=5)
            # top 3 by truth: run_01, run_02, run_03 -> only the small B/C swap remains
            self.assertAlmostEqual(actual["tau_gap@3"], tau_gap([0.9, 0.5, 0.45], [0.8, 0.5, 0.6]), places=5)


if __name__ == "__main__":
    unittest.main()
