import unittest
from click.testing import CliRunner
import click
import json
from pathlib import Path
from . import TREC_25_DATA
from autojudge_evaluate import main
from tempfile import TemporaryDirectory

EXAMPLE_LEADERBOARD = str((TREC_25_DATA / "spot-check-dataset" / "trec-leaderboard.txt").absolute())

def evaluate_command(measure, truth=EXAMPLE_LEADERBOARD, inp=EXAMPLE_LEADERBOARD, eval_measure="Measure-01", correlation="kendall"):
    cmd = ["meta-evaluate", "--truth-leaderboard", truth, "--input", inp, "--truth-measure", measure, "--eval-measure", eval_measure or measure, "--truth-format", "tot", "--eval-format", "tot", "--correlation", correlation]
    return run_cmd_on_main(cmd)
    

def run_cmd_on_main(cmd):
    runner = CliRunner()
    ret = runner.invoke(main, cmd)
    return ret, ' '.join(ret.stdout.split())


class TestEvaluationInterface(unittest.TestCase):
    def test_self_comparison_gives_perfect_correlation(self):
        """Sanity check: same leaderboard, same measure -> 1.0"""
        result, stdout = evaluate_command("Measure-01", eval_measure="Measure-01")
        self.assertIsNone(result.exception)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("trec-leaderboard Measure-01 Measure-01 1.0", stdout)

    def test_cross_measure_correlation_is_negative(self):
        """Cross-measure: Measure-01 vs Measure-02 are inversely ranked -> -1.0"""
        result, stdout = evaluate_command("Measure-01", eval_measure="Measure-02")
        self.assertIsNone(result.exception)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("trec-leaderboard Measure-01 Measure-02 -1.0", stdout)

    def test_invalid_output_path_fails_gracefully(self):
        """Test that invalid output path causes failure but still prints results."""
        # Use /proc/invalid/ which cannot be created on Linux (proc is a virtual filesystem)
        target_file = Path("/proc/invalid/results.jsonl")
        cmd = ["meta-evaluate", "--truth-leaderboard", EXAMPLE_LEADERBOARD,
               "--input", EXAMPLE_LEADERBOARD, "--truth-measure", "Measure-01",
               "--eval-measure", "Measure-02", "--truth-format", "tot",
               "--eval-format", "tot", "--correlation", "kendall", "--output", str(target_file)]
        result, stdout = run_cmd_on_main(cmd)

        self.assertIsNotNone(result.exception)
        self.assertEqual(result.exit_code, 1)
        self.assertFalse(target_file.is_file())
        self.assertIn("trec-leaderboard Measure-01 Measure-02 -1.0", stdout)

    def test_missing_truth_measure_raises_error(self):
        """Test that requesting non-existent truth measure raises clear error."""
        result, stdout = evaluate_command("NonExistent", eval_measure="Measure-01")
        self.assertIsNotNone(result.exception)
        self.assertNotEqual(result.exit_code, 0)

    def test_missing_eval_measure_raises_error(self):
        """Test that requesting non-existent eval measure raises clear error."""
        result, stdout = evaluate_command("Measure-01", eval_measure="NonExistent")
        self.assertIsNotNone(result.exception)
        self.assertNotEqual(result.exit_code, 0)

    def test_kendall_at_k_computes_top_k_correlation(self):
        """Test that kendall@10 computes correlation on top-k systems from truth."""
        # With only 3 runs in test data, @10 includes all runs (same as kendall)
        result, stdout = evaluate_command("Measure-01", eval_measure="Measure-01", correlation="kendall@10")
        self.assertIsNone(result.exception)
        self.assertEqual(result.exit_code, 0)
        # Self-comparison should still be 1.0
        self.assertIn("trec-leaderboard Measure-01 Measure-01 1.0", stdout)

    def test_produces_jsonl_output(self):
        """Test that --output with .jsonl extension writes valid JSONL."""
        with TemporaryDirectory() as tmp_dir:
            target_file = (Path(tmp_dir) / "results.jsonl").absolute()
            cmd = ["meta-evaluate", "--truth-leaderboard", EXAMPLE_LEADERBOARD,
                   "--input", EXAMPLE_LEADERBOARD, "--truth-measure", "Measure-01",
                   "--eval-measure", "Measure-02", "--truth-format", "tot",
                   "--eval-format", "tot", "--correlation", "kendall",
                   "--output", target_file]
            result, stdout = run_cmd_on_main(cmd)

            self.assertIsNone(result.exception)
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(target_file.is_file())

            content = target_file.read_text()
            self.assertIn('"TruthMeasure":"Measure-01"', content)
            self.assertIn('"EvalMeasure":"Measure-02"', content)
            self.assertIn('"kendall":-1.0', content)