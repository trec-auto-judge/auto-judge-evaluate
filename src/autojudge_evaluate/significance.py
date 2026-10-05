"""Discriminative power via the randomised Tukey HSD test (Sakai's Discpower).

Given a topics x runs score matrix, each trial shuffles the scores within every
topic (row) across runs and records the largest gap between run means,
max(mean) - min(mean). The p-value of a run pair is the fraction of trials whose
largest gap is at least the pair's observed |mean_a - mean_b|. Because every pair
is judged against the largest gap of the whole set, the test accounts for
comparing many pairs at once.

Discriminative power = fraction of run pairs with p < alpha (Sakai, "Metrics,
Statistics, Tests", PROMISE Winter School 2013, LNCS 8173).

For meta-evaluation the test runs on both the truth and the judge, and the
significant pairs are compared:
    sig_recall     truth-significant pairs the judge also finds, same direction
    sig_precision  judge-significant pairs the truth confirms, same direction
    reversals      pairs both find significant, in opposite directions
"""

from typing import Dict, List, Tuple

import numpy as np

# Headline columns in the meta-evaluate output.
DISCPOWER_METHODS: List[str] = ["dp_truth", "dp_judge", "dp_sig_recall", "dp_sig_precision", "dp_reversals"]

DEFAULT_TRIALS = 1000
DEFAULT_ALPHA = 0.05
SEED = 0  # fixed so results repeat exactly


def rand_tukey_hsd(matrix: np.ndarray, trials: int = DEFAULT_TRIALS, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Randomised Tukey HSD on a topics x runs matrix.

    Returns (diff, p): diff[a, b] = mean_a - mean_b, p[a, b] its p-value.
    As in Discpower (2014+), a trial whose largest gap equals the observed gap
    counts against significance (the more conservative choice).
    """
    rng = np.random.default_rng(seed)
    n_topics, n_runs = matrix.shape
    diff = matrix.mean(axis=0)[:, None] - matrix.mean(axis=0)[None, :]
    max_gaps = np.empty(trials)
    for b in range(trials):
        order = np.argsort(rng.random((n_topics, n_runs)), axis=1)
        means = np.take_along_axis(matrix, order, axis=1).mean(axis=0)
        max_gaps[b] = means.max() - means.min()
    # p[a, b] = fraction of trials with max_gap >= |diff[a, b]|
    sorted_gaps = np.sort(max_gaps)
    p = 1.0 - np.searchsorted(sorted_gaps, np.abs(diff), side="left") / trials
    return diff, p


def compare(
    truth_diff: np.ndarray,
    truth_p: np.ndarray,
    judge_diff: np.ndarray,
    judge_p: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, float]:
    """Headline columns from the truth's and the judge's pairwise results
    (same runs in the same order on both sides)."""
    iu = np.triu_indices(truth_diff.shape[0], k=1)
    truth_sig, judge_sig = truth_p[iu] < alpha, judge_p[iu] < alpha
    same_dir = np.sign(truth_diff[iu]) == np.sign(judge_diff[iu])
    agree = truth_sig & judge_sig & same_dir
    n_pairs = len(truth_sig)
    nan = float("nan")
    return {
        "dp_truth": truth_sig.mean() if n_pairs else nan,
        "dp_judge": judge_sig.mean() if n_pairs else nan,
        "dp_sig_recall": agree.sum() / truth_sig.sum() if truth_sig.any() else nan,
        "dp_sig_precision": agree.sum() / judge_sig.sum() if judge_sig.any() else nan,
        "dp_reversals": float((truth_sig & judge_sig & ~same_dir).sum()),
    }
