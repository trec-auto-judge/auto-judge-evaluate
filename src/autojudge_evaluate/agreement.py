"""Bland-Altman agreement between judge and truth scores.

Correlations only compare orderings, so a judge that scores everything too high
(lenient) or too low (strict) can still correlate perfectly. Bland-Altman compares
the values themselves: for each item, diff = judge - truth, and

    bias      = mean(diff)            > 0: judge is lenient, < 0: strict
    limits    = bias +- 1.96 * sd(diff)  (95% limits of agreement)
    slope     = OLS slope of diff on (judge + truth) / 2   (proportional bias)

Both sides are mapped onto 0-1 by their declared scale; see ``rescale``. Used for LLM relevance
labels by Rahmani et al. (CIKM 2025), after Altman & Bland (1983).
"""

import math
from typing import Dict, List, Tuple

# Headline columns in the meta-evaluate output, at two levels:
#   sys:  one point per run (the run's aggregate score)
#   resp: one point per (run, topic), i.e. per response
BLAND_ALTMAN_METHODS: List[str] = [
    f"ba_{level}_{stat}" for level in ("sys", "resp") for stat in ("bias", "loa_low", "loa_high")
]

# Key in a range map that applies to every measure without its own entry.
ANY_MEASURE = "*"


def parse_range(spec: str) -> Tuple[str, Tuple[float, float]]:
    """'GRADE=0:3' -> ('GRADE', (0.0, 3.0)); '*=0:1' -> ('*', (0.0, 1.0))."""
    measure, sep, bounds = spec.partition("=")
    lo, colon, hi = bounds.partition(":")
    if not sep or not colon or not measure:
        raise ValueError(f"Expected MEASURE=LOW:HIGH, got '{spec}'")
    low, high = float(lo), float(hi)
    if high <= low:
        raise ValueError(f"HIGH must be above LOW in '{spec}'")
    return measure, (low, high)


def scale_of(measure: str, ranges: Dict[str, Tuple[float, float]]) -> Tuple[float, float] | None:
    """The measure's declared scale: its own entry, else the '*' entry, else None.

    Nothing is inferred from the values: a measure whose values happen to lie in
    0-1 may still be on a wider scale, so an undeclared scale means "unknown".
    """
    return ranges.get(measure, ranges.get(ANY_MEASURE))


def rescale(values: List[float], scale: Tuple[float, float]) -> List[float]:
    """Map a measure's own scale (not its observed min/max) onto 0-1.

    Using the observed min/max instead would stretch a lenient judge's 2-3 grades
    onto 0-1 and hide exactly the bias this analysis is meant to show.
    """
    low, high = scale
    return [(v - low) / (high - low) for v in values]


def bland_altman(truth: List[float], judge: List[float]) -> Dict[str, float]:
    """Bias, 95% limits of agreement and proportional-bias slope for paired values
    already on a common scale. NaN statistics when fewer than 2 pairs."""
    n = len(truth)
    nan = float("nan")
    if n != len(judge):
        raise ValueError(f"Unequal lengths: {n} != {len(judge)}")
    if n < 2:
        return {"n": n, "bias": nan, "sd": nan, "loa_low": nan, "loa_high": nan, "slope": nan}
    diffs = [j - t for t, j in zip(truth, judge)]
    means = [(t + j) / 2 for t, j in zip(truth, judge)]
    bias = sum(diffs) / n
    sd = math.sqrt(sum((d - bias) ** 2 for d in diffs) / (n - 1))
    mean_x = sum(means) / n
    sxx = sum((x - mean_x) ** 2 for x in means)
    slope = sum((x - mean_x) * (d - bias) for x, d in zip(means, diffs)) / sxx if sxx > 0 else nan
    return {
        "n": n,
        "bias": bias,
        "sd": sd,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "slope": slope,
    }
