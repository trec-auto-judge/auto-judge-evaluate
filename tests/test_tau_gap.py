import math
import unittest

from autojudge_evaluate.tau_gap import tau_gap


class TestTauGap(unittest.TestCase):
    def test_identical_ordering_is_one(self):
        self.assertAlmostEqual(1.0, tau_gap([4, 3, 2, 1], [4, 3, 2, 1]))

    def test_reversed_ordering_is_minus_one(self):
        self.assertAlmostEqual(-1.0, tau_gap([4, 3, 2, 1], [1, 2, 3, 4]))

    def test_middle_swap_hand_computed(self):
        """Truth 4,3,2,1; prediction orders a, c, b, d.

        Position 2 (c): pair (a,c) gap 2 correct -> 1.
        Position 3 (b): pair (a,b) gap 1 correct, pair (c,b) gap 1 wrong -> 1/2.
        Position 4 (d): all three pairs correct -> 1.
        tau_GAP = 2/3 * (1 + 1/2 + 1) - 1 = 2/3.
        """
        self.assertAlmostEqual(2.0 / 3.0, tau_gap([4, 3, 2, 1], [4, 2, 3, 1]))

    def test_head_swap_costs_more_than_tail_swap(self):
        truth = [4, 3, 2, 1]
        head_swap = tau_gap(truth, [3, 4, 2, 1])
        tail_swap = tau_gap(truth, [4, 3, 1, 2])
        self.assertLess(head_swap, tail_swap)

    def test_is_asymmetric(self):
        """The gaps come from the first argument only.

        truth 10,3,2,1 with the middle swap in the prediction: the wrong pair
        (c,b) has gap 1 against a correct pair (a,b) of gap 7 at position 3,
        so 2/3 * (1 + 7/8 + 1) - 1 = 11/12. With the arguments swapped the
        gaps are 4,2,3,1 and the same wrong pair weighs 1 against 1, so the
        value is 2/3 * (1 + 1/2 + 1) - 1 = 2/3.
        """
        a = [10, 3, 2, 1]
        b = [4, 2, 3, 1]
        self.assertAlmostEqual(11.0 / 12.0, tau_gap(a, b))
        self.assertAlmostEqual(2.0 / 3.0, tau_gap(b, a))

    def test_truth_ties_contribute_nothing(self):
        """All truth scores equal: no gap mass anywhere -> NaN."""
        self.assertTrue(math.isnan(tau_gap([1, 1, 1], [3, 2, 1])))

    def test_predicted_ties_break_by_position(self):
        """Tied predictions keep their input order; here that order agrees with
        the truth, so the coefficient is 1."""
        self.assertAlmostEqual(1.0, tau_gap([3, 2, 1], [5, 5, 5]))

    def test_too_few_items_is_nan(self):
        self.assertTrue(math.isnan(tau_gap([1], [1])))

    def test_unequal_lengths_raise(self):
        with self.assertRaises(ValueError):
            tau_gap([1, 2], [1])


if __name__ == "__main__":
    unittest.main()
