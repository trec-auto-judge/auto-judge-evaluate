"""Per-topic agreement and judge qrels as eval input (meta-evaluate --eval-qrels)."""

import math
from pathlib import Path

import pytest
from click.testing import CliRunner

from autojudge_evaluate import main
from autojudge_evaluate.eval_results import ALL_TOPIC_ID, QRELS_MEASURE, load_qrels
from autojudge_evaluate.evaluation import LeaderboardEvaluator

# Truth (tot: run measure topic value). In t1 the truth order is r1>r2>r3>r4,
# in t2 it is the reverse; t3 has constant truth (no ordering to agree with).
TRUTH = """\
r1 F1 t1 0.4
r2 F1 t1 0.3
r3 F1 t1 0.2
r4 F1 t1 0.1
r1 F1 t2 0.1
r2 F1 t2 0.2
r3 F1 t2 0.3
r4 F1 t2 0.4
r1 F1 t3 0.5
r2 F1 t3 0.5
r3 F1 t3 0.5
r4 F1 t3 0.5
r1 F1 all 0.9
r2 F1 all 0.5
r3 F1 all 0.4
r4 F1 all 0.1
"""

# Judge qrels on responses: the same grades r1=3 > r2=2 > r3=1 > r4=0 in every
# topic -> agrees with t1 (tau 1), disagrees with t2 (tau -1), t3 skipped.
QRELS = "".join(
    f"{t} 0 {r} {g}\n" for t in ("t1", "t2", "t3") for r, g in (("r1", 3), ("r2", 2), ("r3", 1), ("r4", 0))
)


@pytest.fixture
def files(tmp_path: Path):
    truth = tmp_path / "truth.txt"
    truth.write_text(TRUTH)
    qrels = tmp_path / "judge.qrels"
    qrels.write_text(QRELS)
    return truth, qrels


class _QrelsEvaluator(LeaderboardEvaluator):
    """Evaluator whose inputs are judge qrels."""
    def evaluate(self, eval_file, is_qrels=True):
        return super().evaluate(eval_file, is_qrels=is_qrels)


def _evaluator(truth: Path, methods, on_missing="default") -> LeaderboardEvaluator:
    return _QrelsEvaluator(truth, truth_format="tot", on_missing=on_missing, correlation_methods=methods)


def test_load_qrels_as_per_topic_grades_with_mean_aggregate(files):
    _, qrels = files
    result = load_qrels(qrels)
    assert result.run_ids == {"r1", "r2", "r3", "r4"}
    assert result.get_value("r2", "t1", QRELS_MEASURE) == 2.0
    assert result.get_value("r2", ALL_TOPIC_ID, QRELS_MEASURE) == pytest.approx(2.0)


def test_per_topic_correlation_averages_over_topics(files):
    truth, qrels = files
    actual = _evaluator(truth, ["kendall_per_topic", "spearman_per_topic"]).evaluate(qrels)
    # mean(+1 for t1, -1 for t2); t3 has constant truth and is left out.
    assert actual[("F1", QRELS_MEASURE)]["kendall_per_topic"] == pytest.approx(0.0)
    assert actual[("F1", QRELS_MEASURE)]["spearman_per_topic"] == pytest.approx(0.0)


def test_qrels_aggregate_gives_leaderboard_correlation(files):
    """The mean grade per run ranks runs like the truth's 'all' rows."""
    truth, qrels = files
    actual = _evaluator(truth, ["kendall"]).evaluate(qrels)
    assert actual[("F1", QRELS_MEASURE)]["kendall"] == pytest.approx(1.0)


def test_missing_response_defaults_to_zero_or_is_skipped(tmp_path: Path, files):
    truth, _ = files
    # Judge graded t1 only, and not r1: default scores r1 as 0 (now ranked last,
    # against the truth's first); skip leaves 3 runs in perfect agreement.
    qrels = tmp_path / "partial.qrels"
    qrels.write_text("t1 0 r2 2\nt1 0 r3 1\nt1 0 r4 0\n")
    default = _evaluator(truth, ["kendall_per_topic"]).evaluate(qrels)
    skip = _evaluator(truth, ["kendall_per_topic"], on_missing="skip").evaluate(qrels)
    assert default[("F1", QRELS_MEASURE)]["kendall_per_topic"] < 1.0
    assert skip[("F1", QRELS_MEASURE)]["kendall_per_topic"] == pytest.approx(1.0)


def test_no_qualifying_topic_is_nan(tmp_path: Path, files):
    truth, _ = files
    qrels = tmp_path / "flat.qrels"  # the judge gives every response the same grade
    qrels.write_text("".join(f"{t} 0 {r} 2\n" for t in ("t1", "t2") for r in ("r1", "r2", "r3", "r4")))
    actual = _evaluator(truth, ["kendall_per_topic"]).evaluate(qrels)
    assert math.isnan(actual[("F1", QRELS_MEASURE)]["kendall_per_topic"])


def test_cli_meta_evaluate_with_eval_qrels(files):
    truth, qrels = files
    result = CliRunner().invoke(main, [
        "meta-evaluate", "--truth-leaderboard", str(truth), "--truth-format", "tot",
        "--eval-qrels", str(qrels), "--on-missing", "default",
        "--correlation", "kendall", "--correlation", "kendall_per_topic",
    ])
    assert result.exit_code == 0, result.output
    assert "kendall_per_topic" in result.output


def test_cli_leaderboard_inputs_require_eval_format(files):
    truth, qrels = files
    result = CliRunner().invoke(main, [
        "meta-evaluate", "--truth-leaderboard", str(truth), "--truth-format", "tot", str(truth),
    ])
    assert result.exit_code != 0
    assert "--eval-format" in result.output


def test_cli_mixes_leaderboards_and_qrels(files):
    """Leaderboard files and judge qrels are meta-evaluated in one call."""
    truth, qrels = files
    result = CliRunner().invoke(main, [
        "meta-evaluate", "--truth-leaderboard", str(truth), "--truth-format", "tot",
        "--eval-format", "tot", str(truth), "--eval-qrels", str(qrels),
        "--on-missing", "default", "--correlation", "kendall",
    ])
    assert result.exit_code == 0, result.output
    assert "truth" in result.output and "judge.qrels" in result.output
    assert "QRELS_GRADE" in result.output


def test_whole_system_qrels_used_as_given(tmp_path: Path, files):
    """Qrels grading each system as a whole (topic 'all') are the leaderboard as-is."""
    truth, _ = files
    qrels = tmp_path / "whole.qrels"
    qrels.write_text("all 0 r1 3\nall 0 r2 2\nall 0 r3 1\nall 0 r4 0\n")
    result = load_qrels(qrels)
    assert result.get_value("r2", ALL_TOPIC_ID, QRELS_MEASURE) == 2.0
    actual = _evaluator(truth, ["kendall", "kendall_per_topic"]).evaluate(qrels)
    assert actual[("F1", QRELS_MEASURE)]["kendall"] == pytest.approx(1.0)
    assert math.isnan(actual[("F1", QRELS_MEASURE)]["kendall_per_topic"])  # no per-topic grades
