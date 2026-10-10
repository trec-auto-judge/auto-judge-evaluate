import unittest
from tempfile import TemporaryDirectory
from autojudge_evaluate.evaluation import LeaderboardEvaluator
from pathlib import Path

EXAMPLE_01 = """
run_01 ORACLE query-01 1
run_02 ORACLE query-01 2
run_03 ORACLE query-01 3
run_04 ORACLE query-01 4
run_01 ORACLE all 1
run_02 ORACLE all 2
run_03 ORACLE all 3
run_04 ORACLE all 4
"""

EXAMPLE_02 = """
run_01 M1 query-01 1
run_02 M1 query-01 3
run_03 M1 query-01 2
run_04 M1 query-01 4
run_01 M1 all 1
run_02 M1 all 3
run_03 M1 all 2
run_04 M1 all 4
run_01 M2 query-01 4
run_02 M2 query-01 3
run_03 M2 query-01 2
run_04 M2 query-01 1
run_01 M2 all 4
run_02 M2 all 3
run_03 M2 all 2
run_04 M2 all 1
"""

PER_TOPIC_TRUTH = """
run_01 ORACLE q1 1
run_02 ORACLE q1 2
run_03 ORACLE q1 3
run_04 ORACLE q1 4
run_01 ORACLE q2 1
run_02 ORACLE q2 2
run_03 ORACLE q2 3
run_04 ORACLE q2 4
run_01 ORACLE all 9
run_02 ORACLE all 9
run_03 ORACLE all 9
run_04 ORACLE all 9
"""

PER_TOPIC_EVAL = """
run_01 M1 q1 1
run_02 M1 q1 2
run_03 M1 q1 3
run_04 M1 q1 4
run_01 M1 q2 4
run_02 M1 q2 3
run_03 M1 q2 2
run_04 M1 q2 1
run_01 M1 all 9
run_02 M1 all 9
run_03 M1 all 9
run_04 M1 all 9
"""

PER_TOPIC_EVAL_SHORT = """
run_01 M1 q1 1
run_02 M1 q1 2
run_01 M1 all 9
run_02 M1 all 9
"""


class TestEvaluation(unittest.TestCase):
    def test_evaluate_per_topic(self):
        """Agreement on q1 and reversal on q2, read from the per-topic rows only."""
        with TemporaryDirectory() as d:
            truth = Path(d) / "truth"
            truth.write_text(PER_TOPIC_TRUTH)
            evaluated = Path(d) / "evaluated"
            evaluated.write_text(PER_TOPIC_EVAL)
            te = LeaderboardEvaluator(truth, truth_measures=["ORACLE"], eval_measures=["M1"], truth_format="tot", eval_format="tot", correlation_methods=["tauap_b"])
            actual = te.evaluate_per_topic(evaluated)
            self.assertEqual({('ORACLE', 'M1'): {'q1': {'tauap_b': 1.0}, 'q2': {'tauap_b': -1.0}}}, actual)

    def test_evaluate_per_topic_raises_on_too_few_runs(self):
        """A topic with fewer than 3 aligned runs raises, as evaluate() does."""
        with TemporaryDirectory() as d:
            truth = Path(d) / "truth"
            truth.write_text(PER_TOPIC_TRUTH)
            evaluated = Path(d) / "evaluated"
            evaluated.write_text(PER_TOPIC_EVAL_SHORT)
            te = LeaderboardEvaluator(truth, truth_measures=["ORACLE"], eval_measures=["M1"], truth_format="tot", eval_format="tot", correlation_methods=["tauap_b"], only_shared_runs=True)
            with self.assertRaises(ValueError):
                te.evaluate_per_topic(evaluated)

    def test_correlation_is_perfect_for_identical_leaderboards(self):
        with TemporaryDirectory() as d:
            expected = {('ORACLE', 'ORACLE'): {'kendall': 1.0, 'pearson': 1.0, 'spearman': 1.0, 'tauap_b': 1.0}}
            leaderboard = Path(d) / "leaderboard"
            leaderboard.write_text(EXAMPLE_01)
            te = LeaderboardEvaluator(leaderboard, truth_measures=["ORACLE"], eval_measures=["ORACLE"], truth_format="tot", eval_format="tot", correlation_methods=["kendall", "pearson", "spearman", "tauap_b"])
            actual = te.evaluate(leaderboard)
            self.assertEqual(expected, actual)

    def test_correlation_is_perfect_for_identical_leaderboards_on_multiple_measures_01(self):
        """Truth=M2, Eval=M1,M2 -> correlations for (M2,M1) and (M2,M2)."""
        with TemporaryDirectory() as d:
            expected = {
                ('M2', 'M1'): {'kendall': -0.666666666, 'pearson': -0.8, 'spearman': -0.8, 'tauap_b': -0.666666666},
                ('M2', 'M2'): {'kendall': 1.0, 'pearson': 1.0, 'spearman': 1.0, 'tauap_b': 1.0},
            }
            leaderboard = Path(d) / "leaderboard"
            leaderboard.write_text(EXAMPLE_02)
            te = LeaderboardEvaluator(leaderboard, truth_measures=["M2"], eval_measures=None, truth_format="tot", eval_format="tot", correlation_methods=["kendall", "pearson", "spearman", "tauap_b"])
            actual = te.evaluate(leaderboard)
            self.assertIn(('M2', 'M1'), actual)
            self.assertIn(('M2', 'M2'), actual)

            self.assertEqual(expected[('M2', 'M2')], actual[('M2', 'M2')])
            for m in expected[('M2', 'M1')].keys():
                self.assertAlmostEqual(expected[('M2', 'M1')][m], actual[('M2', 'M1')][m], 5, m)

    def test_correlation_is_perfect_for_identical_leaderboards_on_multiple_measures_02(self):
        """Truth=M1, Eval=M1,M2 -> correlations for (M1,M1) and (M1,M2)."""
        with TemporaryDirectory() as d:
            expected = {
                ('M1', 'M1'): {'kendall': 1.0, 'pearson': 1.0, 'spearman': 1.0, 'tauap_b': 1.0},
                ('M1', 'M2'): {'kendall': -0.666666666, 'pearson': -0.8, 'spearman': -0.8, 'tauap_b': -0.666666666},
            }
            leaderboard = Path(d) / "leaderboard"
            leaderboard.write_text(EXAMPLE_02)
            te = LeaderboardEvaluator(leaderboard, truth_measures=["M1"], eval_measures=None, truth_format="tot", eval_format="tot", correlation_methods=["kendall", "pearson", "spearman", "tauap_b"])
            actual = te.evaluate(leaderboard)
            self.assertIn(('M1', 'M1'), actual)
            self.assertIn(('M1', 'M2'), actual)

            self.assertEqual(expected[('M1', 'M1')], actual[('M1', 'M1')])
            for m in expected[('M1', 'M2')].keys():
                self.assertAlmostEqual(expected[('M1', 'M2')][m], actual[('M1', 'M2')][m], 5, m)

    def test_correlation_on_two_leaderboards_01(self):
        """Truth=ORACLE (l1), Eval=M1,M2 (l2) -> correlations for (ORACLE,M1) and (ORACLE,M2)."""
        with TemporaryDirectory() as d:
            expected = {
                ('ORACLE', 'M1'): {'kendall': 0.666666666, 'pearson': 0.8, 'spearman': 0.8, 'tauap_b': 0.666666666},
                ('ORACLE', 'M2'): {'kendall': -1.0, 'pearson': -1.0, 'spearman': -1.0, 'tauap_b': -1.0},
            }
            l1 = Path(d) / "leaderboard-1"
            l1.write_text(EXAMPLE_01)
            l2 = Path(d) / "leaderboard-2"
            l2.write_text(EXAMPLE_02)

            te = LeaderboardEvaluator(l1, truth_measures=["ORACLE"], eval_measures=None, truth_format="tot", eval_format="tot", correlation_methods=["kendall", "pearson", "spearman", "tauap_b"])
            actual = te.evaluate(l2)
            self.assertIn(('ORACLE', 'M1'), actual)
            self.assertIn(('ORACLE', 'M2'), actual)

            self.assertEqual(expected[('ORACLE', 'M2')], actual[('ORACLE', 'M2')])
            for m in expected[('ORACLE', 'M1')].keys():
                self.assertAlmostEqual(expected[('ORACLE', 'M1')][m], actual[('ORACLE', 'M1')][m], 5, m)

    def test_correlation_on_two_leaderboards_02(self):
        """Truth=M2 (l2), Eval=ORACLE (l1) -> correlation for (M2,ORACLE)."""
        with TemporaryDirectory() as d:
            expected = {
                ('M2', 'ORACLE'): {'kendall': -1.0, 'pearson': -1.0, 'spearman': -1.0, 'tauap_b': -1.0},
            }
            l1 = Path(d) / "leaderboard-1"
            l1.write_text(EXAMPLE_01)
            l2 = Path(d) / "leaderboard-2"
            l2.write_text(EXAMPLE_02)

            te = LeaderboardEvaluator(l2, truth_measures=["M2"], eval_measures=["ORACLE"], truth_format="tot", eval_format="tot", correlation_methods=["kendall", "pearson", "spearman", "tauap_b"])
            actual = te.evaluate(l1)

            self.assertEqual(expected, actual)

    def test_evaluation_throws_exception_for_non_existing_measure(self):
        with TemporaryDirectory() as d:
            leaderboard = Path(d) / "leaderboard"
            leaderboard.write_text(EXAMPLE_02)

            te = LeaderboardEvaluator(leaderboard, truth_measures=["measure-does-not-exist"], eval_measures=["M1"], truth_format="tot", eval_format="tot")
            with self.assertRaises(ValueError):
                te.evaluate(leaderboard)