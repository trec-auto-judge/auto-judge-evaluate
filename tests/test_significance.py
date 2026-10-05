"""Discriminative power via randomised Tukey HSD (significance.py) and its columns."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from autojudge_evaluate import main
from autojudge_evaluate.evaluation import LeaderboardEvaluator
from autojudge_evaluate.significance import DISCPOWER_METHODS, compare, rand_tukey_hsd

# Sakai's Discpower README sample: 5 topics x 3 runs (RunA, RunB, RunC).
# Its documented result: Metric1 finds no significant pair; Metric2 finds only
# RunB-RunC (difference 0.82) at alpha 0.05.
METRIC1 = np.array([[0.1, 0.9, 0.2], [0.8, 0.3, 0.3], [0.2, 0.9, 0.1], [0.3, 0.6, 0.0], [0.4, 0.5, 0.4]])
METRIC2 = np.array([[0.1, 1.0, 0.2], [0.7, 1.0, 0.3], [0.3, 0.9, 0.1], [0.3, 1.0, 0.0], [0.5, 0.9, 0.1]])


# ---------------------------------------------------------------- statistics

def test_matches_sakai_readme_sample():
    _, p1 = rand_tukey_hsd(METRIC1, trials=5000)
    assert (p1[np.triu_indices(3, 1)] >= 0.05).all()
    diff2, p2 = rand_tukey_hsd(METRIC2, trials=5000)
    assert abs(diff2[1, 2]) == pytest.approx(0.82)
    assert p2[1, 2] < 0.05                    # RunB-RunC significant
    assert p2[0, 1] >= 0.05 and p2[0, 2] >= 0.05


def test_p_values_are_symmetric_and_diff_antisymmetric():
    diff, p = rand_tukey_hsd(METRIC2, trials=500)
    assert np.allclose(p, p.T)
    assert np.allclose(diff, -diff.T)


def test_fixed_seed_repeats_exactly():
    assert np.array_equal(rand_tukey_hsd(METRIC2, trials=300)[1], rand_tukey_hsd(METRIC2, trials=300)[1])


def test_compare_hand_built():
    # 3 runs -> pairs (0,1), (0,2), (1,2)
    truth_diff = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], float)
    truth_p = np.array([[1, .01, .01], [.01, 1, .5], [.01, .5, 1]])     # truth: (0,1), (0,2) significant
    judge_diff = np.array([[0, -1, 1], [1, 0, 1], [-1, -1, 0]], float)  # judge flips (0,1)
    judge_p = np.array([[1, .01, .01], [.01, 1, .01], [.01, .01, 1]])   # judge: all significant
    stats = compare(truth_diff, truth_p, judge_diff, judge_p, alpha=0.05)
    assert stats["dp_truth"] == pytest.approx(2 / 3)
    assert stats["dp_judge"] == pytest.approx(1.0)
    assert stats["dp_sig_recall"] == pytest.approx(1 / 2)     # only (0,2) agrees
    assert stats["dp_sig_precision"] == pytest.approx(1 / 3)
    assert stats["dp_reversals"] == 1.0                       # (0,1) both sure, opposite winners


def test_compare_no_significant_pairs_is_nan_ratio():
    d, p = np.zeros((3, 3)), np.ones((3, 3))
    stats = compare(d, p, d, p)
    assert stats["dp_truth"] == 0.0 and stats["dp_judge"] == 0.0
    assert math.isnan(stats["dp_sig_recall"]) and math.isnan(stats["dp_sig_precision"])


# ---------------------------------------------------------------- evaluator

def _tot(measure: str, matrix: np.ndarray, runs) -> str:
    return "".join(
        f"{r} {measure} t{t} {matrix[t, i]}\n" for t in range(matrix.shape[0]) for i, r in enumerate(runs)
    )


@pytest.fixture
def files(tmp_path: Path):
    runs = ["RunA", "RunB", "RunC"]
    truth, judge = tmp_path / "truth", tmp_path / "judge"
    truth.write_text(_tot("M2", METRIC2, runs))
    judge.write_text(_tot("J", METRIC2, runs) + _tot("FLAT", np.full((5, 3), 0.5), runs))
    return truth, judge


def _evaluate(truth: Path, judge: Path, **kwargs):
    te = LeaderboardEvaluator(truth, truth_format="tot", eval_format="tot",
                              truth_drop_aggregate=True, eval_drop_aggregate=True,
                              correlation_methods=DISCPOWER_METHODS, dp_trials=2000, **kwargs)
    return te, te.evaluate(judge)


def test_judge_equal_to_truth_agrees_fully(files):
    truth, judge = files
    _, actual = _evaluate(truth, judge)
    same = actual[("M2", "J")]
    assert same["dp_truth"] == pytest.approx(1 / 3) and same["dp_judge"] == pytest.approx(1 / 3)
    assert same["dp_sig_recall"] == 1.0 and same["dp_sig_precision"] == 1.0
    assert same["dp_reversals"] == 0.0


def test_flat_judge_finds_nothing(files):
    truth, judge = files
    _, actual = _evaluate(truth, judge)
    flat = actual[("M2", "FLAT")]
    assert flat["dp_judge"] == 0.0 and flat["dp_sig_recall"] == 0.0
    assert math.isnan(flat["dp_sig_precision"])


def test_truth_test_is_reused_across_judge_measures(files):
    truth, judge = files
    te, _ = _evaluate(truth, judge)
    assert len(te._dp_truth_cache) == 1  # one truth measure, same runs/topics for J and FLAT


def test_fewer_than_three_shared_runs_is_nan(tmp_path: Path):
    truth, judge = tmp_path / "truth", tmp_path / "judge"
    truth.write_text(_tot("M", METRIC2, ["RunA", "RunB", "RunC"]))
    judge.write_text(_tot("J", METRIC2[:, :2], ["RunA", "RunB"]))  # judge scored only 2 runs
    te = LeaderboardEvaluator(truth, truth_format="tot", eval_format="tot", truth_drop_aggregate=True,
                              eval_drop_aggregate=True, correlation_methods=DISCPOWER_METHODS)
    assert all(math.isnan(v) for v in te.evaluate(judge)[("M", "J")].values())


def test_diagnostics_files(tmp_path: Path, files):
    truth, judge = files
    _evaluate(truth, judge, diagnostics_dir=tmp_path / "diag")
    out = tmp_path / "diag" / "judge" / "M2__J"
    summary = json.loads((out / "discpower.summary.json").read_text())
    assert summary["n_runs"] == 3 and summary["n_topics"] == 5 and summary["trials"] == 2000
    rows = [json.loads(line) for line in (out / "discpower.pairs.jsonl").read_text().splitlines()]
    assert [(r["run_a"], r["run_b"]) for r in rows] == [("RunA", "RunB"), ("RunA", "RunC"), ("RunB", "RunC")]
    assert rows[2]["truth_p"] < 0.05 and rows[2]["truth_diff"] == pytest.approx(0.82)


def test_cli_options(files):
    truth, judge = files
    base = ["meta-evaluate", "--truth-leaderboard", str(truth), "--truth-format", "tot",
            "--eval-format", "tot", "--no-truth-header", "--no-eval-header",
            "--truth-drop-aggregate", "--eval-drop-aggregate", "--correlation", "dp_judge", str(judge)]
    result = CliRunner().invoke(main, base + ["--dp-trials", "500", "--dp-alpha", "0.01"])
    assert result.exit_code == 0, result.output
    assert "dp_judge" in result.output
    for bad in (["--dp-trials", "10"], ["--dp-alpha", "1.5"]):
        assert CliRunner().invoke(main, base + bad).exit_code != 0
