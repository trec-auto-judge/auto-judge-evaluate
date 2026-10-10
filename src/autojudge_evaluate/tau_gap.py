"""tau_GAP: head-weighted, gap-sensitive rank correlation (Gao and Oard, SIGIR 2015).

Ning Gao and Douglas W. Oard. A Head-Weighted Gap-Sensitive Correlation
Coefficient. SIGIR 2015.

Compares a ground-truth scoring with a predicted ordering of the same items:

    tau_GAP = 2/(N-1) * sum_{i=2..N} [ sum_{j<i} |CG_ji| / sum_{j<i} |G_ji| ] - 1

where i runs over the positions of the PREDICTED list, j < i over the items
ranked above position i, G_ji is the ground-truth score gap of the pair, and
CG_ji equals G_ji when the prediction orders the pair as the truth does and 0
otherwise. Each position contributes the share of the gap mass above it that
the prediction orders correctly, so a swap near the head of the list, or
across a large truth gap, costs more than a swap near the tail across a small
one. Pairs with a zero truth gap (manual ties) contribute to neither sum, and
a position whose pairs are all tied is left out of the average.

The coefficient is asymmetric: the first argument supplies the gaps, the
second only the ordering. In the meta-evaluation the manual truth is the first
argument and the judge's leaderboard the second.

Ported from the prototype used for the ECIR circularity analysis
(trec2026-unh/scratchpad/tau_gap.py); ties in the predicted scores are broken
by position, which with sorted run-id alignment is the prototype's run-id
tie-break.
"""

from __future__ import annotations

from typing import Sequence


def tau_gap(truth: Sequence[float], predicted: Sequence[float]) -> float:
    """tau_GAP of a predicted ordering against ground-truth scores.

    ``truth[k]`` and ``predicted[k]`` are the scores of the same item. The
    predicted list is sorted by decreasing predicted score, ties by position.
    Returns NaN when fewer than 2 items are given or every pair is truth-tied.
    """
    if len(truth) != len(predicted):
        raise ValueError(f"unequal lengths: {len(truth)} != {len(predicted)}")
    n = len(truth)
    if n < 2:
        return float("nan")
    order = sorted(range(n), key=lambda k: (-predicted[k], k))
    total = 0.0
    positions_used = 0
    for i in range(1, n):
        gaps = correct = 0.0
        for j in range(i):
            g = abs(truth[order[j]] - truth[order[i]])
            gaps += g
            if truth[order[j]] > truth[order[i]]:
                correct += g
        if gaps > 0:
            total += correct / gaps
            positions_used += 1
    if positions_used == 0:
        return float("nan")
    return 2.0 / positions_used * total - 1.0
