import unittest
from autojudge_evaluate.evaluation import tauap_b

class TestCorrelations(unittest.TestCase):
    def test_fails_too_few_entries(self):
        a = [1, 2]
        b = [10, 20]

        with self.assertRaises(ValueError):
            tauap_b(a, b)

    def test_fails_unequal_entries(self):
        a = [1, 2, 3, 4]
        b = [10, 20, 30, 40, 50]

        with self.assertRaises(ValueError):
            tauap_b(a, b)

    def test_perfect_kendall_correlation_01(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        expected = 1.0

        actual = tauap_b(a, b)
        self.assertAlmostEqual(expected, actual)

        actual = tauap_b(b, a)
        self.assertAlmostEqual(expected, actual)

    def test_perfect_kendall_correlation_02(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected = 1.0

        actual = tauap_b(a, b)
        self.assertAlmostEqual(expected, actual)

    def test_almost_perfect_kendall_correlation_top_switch(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [10, 20, 30, 40, 50, 60, 70, 80, 100, 90]
        expected = 0.7777777777

        actual = tauap_b(a, b)
        self.assertAlmostEqual(expected, actual)

        actual = tauap_b(b, a)
        self.assertAlmostEqual(expected, actual)

    def test_almost_perfect_kendall_correlation_bottom_switch(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [20, 10, 30, 40, 50, 60, 70, 80, 90, 100]
        expected = 0.9753086

        actual = tauap_b(a, b)
        self.assertAlmostEqual(expected, actual)

        actual = tauap_b(b, a)
        self.assertAlmostEqual(expected, actual)

    def test_perfect_reverse_kendall_correlation_01(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [-10, -20, -30, -40, -50, -60, -70, -80, -90, -100]
        expected = -1.0

        actual = tauap_b(a, b)
        self.assertAlmostEqual(expected, actual)

        actual = tauap_b(b, a)
        self.assertAlmostEqual(expected, actual)

    def test_perfect_reverse_kendall_correlation_02(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
        expected = -1.0

        actual = tauap_b(a, b)
        self.assertAlmostEqual(expected, actual)