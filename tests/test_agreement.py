"""Bland-Altman agreement (agreement.py) and its meta-evaluate columns."""

import json
import math
from pathlib import Path

import pytest
from click.testing import CliRunner

from autojudge_evaluate import main
from autojudge_evaluate.agreement import bland_altman, parse_range, rescale, scale_of
from autojudge_evaluate.evaluation import LeaderboardEvaluator


# ---------------------------------------------------------------- statistics

def test_bland_altman_hand_computed():
    # diffs = judge - truth = [0.2, 0.3, 0.4]: bias 0.3, sd 0.1
    stats = bland_altman([0.1, 0.2, 0.3], [0.3, 0.5, 0.7])
    assert stats["n"] == 3
    assert stats["bias"] == pytest.approx(0.3)
    assert stats["sd"] == pytest.approx(0.1)
    assert stats["loa_low"] == pytest.approx(0.3 - 1.96 * 0.1)
    assert stats["loa_high"] == pytest.approx(0.3 + 1.96 * 0.1)
    # means = [0.2, 0.35, 0.5] rise with the diffs: slope 0.1 / 0.15
    assert stats["slope"] == pytest.approx(0.1 / 0.15)


def test_identical_scores_have_zero_bias_and_zero_width():
    stats = bland_altman([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
    assert stats["bias"] == 0.0 and stats["loa_low"] == 0.0 and stats["loa_high"] == 0.0


def test_strict_judge_has_negative_bias():
    assert bland_altman([0.6, 0.7, 0.8], [0.2, 0.4, 0.3])["bias"] < 0


def test_fewer_than_two_pairs_is_nan():
    assert math.isnan(bland_altman([0.5], [0.7])["bias"])
    assert math.isnan(bland_altman([], [])["loa_high"])


def test_rescale_uses_the_scale_not_the_observed_range():
    # a lenient judge using only grades 2-3 stays high (0.67-1), not stretched to 0-1
    assert rescale([2, 3], (0, 3)) == pytest.approx([2 / 3, 1.0])


def test_scale_of_uses_only_declared_ranges():
    assert scale_of("F1", {}) is None                                   # nothing assumed
    assert scale_of("F1", {"*": (0.0, 1.0)}) == (0.0, 1.0)              # wildcard
    assert scale_of("GRADE", {"*": (0.0, 1.0), "GRADE": (0.0, 3.0)}) == (0.0, 3.0)  # own entry wins


def test_parse_range():
    assert parse_range("GRADE=0:3") == ("GRADE", (0.0, 3.0))
    assert parse_range("*=0:1") == ("*", (0.0, 1.0))
    for bad in ("GRADE", "GRADE=3", "=0:3", "GRADE=3:0", "GRADE=a:b"):
        with pytest.raises(ValueError):
            parse_range(bad)


# ---------------------------------------------------------------- evaluator

# Truth F1 (0-1) for 4 runs x 2 topics; the judge's SCORE is truth + 0.2 (lenient)
# and its GRADE is a 0-3 grade. run r4 is not graded on t2.
TRUTH = """\
r1 F1 t1 0.8
r2 F1 t1 0.6
r3 F1 t1 0.4
r4 F1 t1 0.2
r1 F1 t2 0.7
r2 F1 t2 0.5
r3 F1 t2 0.3
r4 F1 t2 0.1
"""

JUDGE = """\
r1 SCORE t1 1.0
r2 SCORE t1 0.8
r3 SCORE t1 0.6
r4 SCORE t1 0.4
r1 SCORE t2 0.9
r2 SCORE t2 0.7
r3 SCORE t2 0.5
r1 GRADE t1 3
r2 GRADE t1 3
r3 GRADE t1 2
r4 GRADE t1 2
r1 GRADE t2 3
r2 GRADE t2 2
r3 GRADE t2 2
"""

BA_COLUMNS = ["ba_sys_bias", "ba_sys_loa_low", "ba_sys_loa_high",
              "ba_resp_bias", "ba_resp_loa_low", "ba_resp_loa_high"]


@pytest.fixture
def files(tmp_path: Path):
    truth, judge = tmp_path / "truth", tmp_path / "judge"
    truth.write_text(TRUTH)
    judge.write_text(JUDGE)
    return truth, judge


UNIT = {"*": (0.0, 1.0)}


def _evaluate(truth: Path, judge: Path, **kwargs):
    kwargs.setdefault("ba_truth_ranges", UNIT)
    kwargs.setdefault("ba_judge_ranges", UNIT)
    te = LeaderboardEvaluator(truth, truth_format="tot", eval_format="tot",
                              truth_drop_aggregate=True, eval_drop_aggregate=True,
                              on_missing="default", correlation_methods=BA_COLUMNS, **kwargs)
    return te.evaluate(judge)


def test_lenient_judge_bias_at_response_level(files):
    truth, judge = files
    score = _evaluate(truth, judge)[("F1", "SCORE")]
    # every graded response is truth + 0.2; r4/t2 is not graded and is left out
    assert score["ba_resp_bias"] == pytest.approx(0.2)
    assert score["ba_resp_loa_low"] == pytest.approx(0.2)
    assert score["ba_resp_loa_high"] == pytest.approx(0.2)
    # system level compares run means; r4's mean covers t1 only (0.4 vs truth 0.15)
    assert score["ba_sys_bias"] > 0


def test_missing_scores_are_left_out_not_zero(files):
    """With on_missing='default' a missing response would be scored 0 by the
    correlations; Bland-Altman must not, or a lenient judge would look strict."""
    truth, judge = files
    assert _evaluate(truth, judge)[("F1", "SCORE")]["ba_resp_bias"] > 0


def test_no_declared_range_gives_nan_even_for_values_in_0_1(files):
    truth, judge = files
    for truth_ranges, eval_ranges in (({}, UNIT), (UNIT, {}), ({}, {})):
        score = _evaluate(truth, judge, ba_truth_ranges=truth_ranges, ba_judge_ranges=eval_ranges)
        assert all(math.isnan(score[("F1", "SCORE")][c]) for c in BA_COLUMNS)


def test_each_side_uses_its_own_range(files):
    truth, judge = files
    # '*=0:1' would read grades 2-3 as 2.0-3.0: a meaningless bias above 1
    wrong = _evaluate(truth, judge)[("F1", "GRADE")]
    assert wrong["ba_resp_bias"] > 1
    grade = _evaluate(truth, judge, ba_judge_ranges={"*": (0.0, 1.0), "GRADE": (0.0, 3.0)})[("F1", "GRADE")]
    assert 0 < grade["ba_resp_bias"] < 1  # grades 2-3 on 0-3 sit above truth F1
    assert all(not math.isnan(grade[c]) for c in BA_COLUMNS)


def test_diagnostics_files(tmp_path: Path, files):
    truth, judge = files
    diag = tmp_path / "diag"
    _evaluate(truth, judge, diagnostics_dir=diag)
    out = diag / "judge" / "F1__SCORE"
    summary = json.loads((out / "bland_altman.summary.json").read_text())
    assert summary["truth_scale"] == [0.0, 1.0]
    assert summary["levels"]["resp"]["n"] == 7
    assert summary["levels"]["resp"]["bias"] == pytest.approx(0.2)
    rows = [json.loads(line) for line in (out / "bland_altman.points.jsonl").read_text().splitlines()]
    assert {r["level"] for r in rows} == {"sys", "resp"}
    assert all(r["diff"] == pytest.approx(r["judge"] - r["truth"]) for r in rows)


def test_cli_ba_range(files):
    truth, judge = files
    base = ["meta-evaluate", "--truth-leaderboard", str(truth), "--truth-format", "tot",
            "--eval-format", "tot", "--no-truth-header", "--no-eval-header",
            "--truth-drop-aggregate", "--eval-drop-aggregate", "--correlation", "ba_resp_bias", str(judge)]
    ranges = ["--ba-truth-range", "*=0:1", "--ba-judge-range", "*=0:1", "--ba-judge-range", "GRADE=0:3"]
    result = CliRunner().invoke(main, base + ranges)
    assert result.exit_code == 0, result.output
    assert "ba_resp_bias" in result.output
    for option in ("--ba-truth-range", "--ba-judge-range"):
        result = CliRunner().invoke(main, base + [option, "GRADE=3"])
        assert result.exit_code != 0
        assert option in result.output
