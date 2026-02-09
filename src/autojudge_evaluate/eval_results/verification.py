"""
C. Verification - Validation and checking for EvalResult.

All checks are functions that either:
- Return True/False for boolean checks
- Raise VerificationError with details for strict checks
- Return a list of issues for inspection

These are called by Builder.build() to ensure result integrity.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Literal, Set

from .eval_result import EvalEntry, ALL_TOPIC_ID, MeasureDtype


class VerificationError(Exception):
    """Raised when verification fails."""
    pass


OnFail = Literal["error", "warn", "ignore"]


# =============================================================================
# Topic checks
# =============================================================================


def check_same_topics_per_run(
    entries: list[EvalEntry],
    on_fail: OnFail = "error",
) -> dict[str, Set[str]]:
    """
    Check that all runs have the same set of topic_ids.

    Computes the union of all topics across runs, then reports which runs
    are missing topics compared to that union.

    Returns dict of run_id -> topic_ids for runs with incomplete topic sets.
    """
    topics_by_run: dict[str, Set[str]] = defaultdict(set)

    for e in entries:
        if e.topic_id != ALL_TOPIC_ID:
            topics_by_run[e.run_id].add(e.topic_id)

    if not topics_by_run:
        return {}

    # Compute union of all topics across all runs
    all_topics = set()
    for topics in topics_by_run.values():
        all_topics.update(topics)

    # Find runs missing topics (compared to union)
    differing = {}
    for run_id, topics in topics_by_run.items():
        missing = all_topics - topics
        if missing:
            differing[run_id] = topics

    if differing:
        # Build detailed message
        details = []
        for run_id, topics in sorted(differing.items()):
            missing = sorted(all_topics - topics)
            details.append(f"  {run_id}: missing {missing}")
        detail_str = "\n".join(details[:5])  # Limit to first 5
        if len(details) > 5:
            detail_str += f"\n  ... and {len(details) - 5} more"
        _handle_failure(
            on_fail,
            f"{len(differing)} runs have incomplete topic sets (expected {len(all_topics)} topics):\n{detail_str}"
        )

    return differing


# =============================================================================
# Measure checks (spec-aware)
# =============================================================================


def check_measures_match_specs(
    entries: list[EvalEntry],
    per_topic_specs: Dict[str, MeasureDtype],
    aggregate_specs: Dict[str, MeasureDtype],
    on_fail: OnFail = "error",
) -> dict[tuple[str, str], Set[str]]:
    """
    Check that entries have measures matching their specs.

    Per-topic entries should have measures from per_topic_specs.
    Aggregate entries should have measures from aggregate_specs.

    Returns dict of (run_id, topic_id) -> actual measures for entries that differ.
    """
    expected_per_topic = set(per_topic_specs.keys())
    expected_aggregate = set(aggregate_specs.keys())

    # Group measures by (run_id, topic_id)
    measures_by_entry: dict[tuple[str, str], Set[str]] = defaultdict(set)
    for e in entries:
        measures_by_entry[(e.run_id, e.topic_id)].add(e.measure)

    differing = {}
    for (run_id, topic_id), actual_measures in measures_by_entry.items():
        if topic_id == ALL_TOPIC_ID:
            expected = expected_aggregate
        else:
            expected = expected_per_topic

        if actual_measures != expected:
            differing[(run_id, topic_id)] = actual_measures

    if differing:
        # Analyze which measures are problematic per-measure
        # Pass all entries to find where measures ARE present (not just differing)
        per_topic_missing = _analyze_missing_measures(
            differing, measures_by_entry, expected_per_topic, is_aggregate=False
        )
        aggregate_missing = _analyze_missing_measures(
            differing, measures_by_entry, expected_aggregate, is_aggregate=True
        )
        msg = _format_measure_diagnostic(per_topic_missing, aggregate_missing)
        _handle_failure(on_fail, msg)

    return differing


def check_aggregates_match_specs(
    entries: list[EvalEntry],
    aggregate_specs: Dict[str, MeasureDtype],
    on_fail: OnFail = "error",
) -> list[tuple[str, str]]:
    """
    Check that aggregate entries exist for all (run_id, measure) pairs in specs.

    Returns list of missing (run_id, measure) aggregate entries.
    """
    if not aggregate_specs:
        # No aggregate measures expected
        return []

    # Get all run_ids from per-topic entries
    run_ids = {e.run_id for e in entries if e.topic_id != ALL_TOPIC_ID}

    # Expected: every run_id should have aggregate for every measure in specs
    expected: Set[tuple[str, str]] = set()
    for run_id in run_ids:
        for measure in aggregate_specs.keys():
            expected.add((run_id, measure))

    # Existing aggregates
    existing: Set[tuple[str, str]] = set()
    for e in entries:
        if e.topic_id == ALL_TOPIC_ID:
            existing.add((e.run_id, e.measure))

    missing = list(expected - existing)

    if missing:
        # Group by measure to show per-measure breakdown
        by_measure: dict[str, list[str]] = defaultdict(list)
        for run_id, measure in missing:
            by_measure[measure].append(run_id)

        total_runs = len(run_ids)
        lines = ["Missing aggregate entries by measure:"]
        for measure, runs in sorted(by_measure.items()):
            lines.append(f"  - '{measure}': missing for {len(runs)}/{total_runs} runs")

        _handle_failure(on_fail, "\n".join(lines))

    return missing


# =============================================================================
# Legacy checks (kept for compatibility, but not used by default)
# =============================================================================


def check_complete_grid(
    entries: list[EvalEntry],
    on_fail: OnFail = "error",
) -> list[tuple[str, str, str]]:
    """
    Check that all (run_id, topic_id, measure) combinations exist.

    Returns list of missing (run_id, topic_id, measure) tuples.
    """
    # Collect all dimensions
    run_ids: Set[str] = set()
    topic_ids: Set[str] = set()
    measures: Set[str] = set()
    existing: Set[tuple[str, str, str]] = set()

    for e in entries:
        if e.topic_id != ALL_TOPIC_ID:
            run_ids.add(e.run_id)
            topic_ids.add(e.topic_id)
            measures.add(e.measure)
            existing.add((e.run_id, e.topic_id, e.measure))

    # Find missing
    missing = []
    for run_id in run_ids:
        for topic_id in topic_ids:
            for measure in measures:
                if (run_id, topic_id, measure) not in existing:
                    missing.append((run_id, topic_id, measure))

    if missing:
        _handle_failure(
            on_fail,
            f"Missing {len(missing)} entries. First 5: {missing[:5]}"
        )

    return missing


def check_no_extra_aggregates(
    entries: list[EvalEntry],
    on_fail: OnFail = "error",
) -> list[tuple[str, str]]:
    """
    Check that no aggregate rows exist without corresponding per-topic entries.

    Returns list of extra (run_id, measure) aggregate entries.
    """
    # Collect per-topic (run_id, measure) pairs
    per_topic: Set[tuple[str, str]] = set()
    for e in entries:
        if e.topic_id != ALL_TOPIC_ID:
            per_topic.add((e.run_id, e.measure))

    # Find aggregates without per-topic data
    extra = []
    for e in entries:
        if e.topic_id == ALL_TOPIC_ID:
            if (e.run_id, e.measure) not in per_topic:
                extra.append((e.run_id, e.measure))

    if extra:
        _handle_failure(
            on_fail,
            f"{len(extra)} aggregate entries without per-topic data: {extra[:5]}"
        )

    return extra


def check_consistent_dtypes(
    entries: list[EvalEntry],
    measure_dtypes: dict[str, str],
    on_fail: OnFail = "error",
) -> list[tuple[str, str, type]]:
    """
    Check that all values match their inferred dtype.

    Returns list of (measure, value, actual_type) for mismatches.
    """
    mismatches = []

    for e in entries:
        expected_dtype = measure_dtypes.get(e.measure, "str")
        if expected_dtype == "float":
            if not isinstance(e.value, (int, float)):
                mismatches.append((e.measure, str(e.value), type(e.value)))
        else:
            if not isinstance(e.value, str):
                mismatches.append((e.measure, str(e.value), type(e.value)))

    if mismatches:
        _handle_failure(
            on_fail,
            f"{len(mismatches)} dtype mismatches. First 5: {mismatches[:5]}"
        )

    return mismatches


# =============================================================================
# Helper
# =============================================================================


def _analyze_missing_measures(
    differing: dict[tuple[str, str], Set[str]],
    all_entries: dict[tuple[str, str], Set[str]],
    expected: Set[str],
    is_aggregate: bool,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze which measures are missing from which entries.

    Args:
        differing: Entries that don't match spec (used to count missing)
        all_entries: ALL entries (used to find where measures ARE present)
        expected: Set of expected measure names
        is_aggregate: If True, filter to aggregate entries (topic_id == ALL_TOPIC_ID)

    Returns:
        Dict of measure -> {"missing_count": N, "total": M, "present_in_runs": [run_ids...]}
    """
    # Filter differing to aggregate or per-topic entries
    filtered_differing = {
        k: v for k, v in differing.items()
        if (k[1] == ALL_TOPIC_ID) == is_aggregate
    }

    # Filter ALL entries to find where measures ARE present
    filtered_all = {
        k: v for k, v in all_entries.items()
        if (k[1] == ALL_TOPIC_ID) == is_aggregate
    }

    total = len(filtered_all)  # Total entries of this type
    if not filtered_differing:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for measure in expected:
        # Count missing from differing entries
        missing_from = [k for k, actual in filtered_differing.items() if measure not in actual]
        # Find present in ALL entries (not just differing)
        present_in = [k for k, actual in filtered_all.items() if measure in actual]

        if missing_from:
            # Get unique run_ids where this measure IS present (first 5)
            present_run_ids = sorted(set(k[0] for k in present_in))[:5]
            result[measure] = {
                "missing_count": len(missing_from),
                "total": total,
                "present_in_runs": present_run_ids,
            }

    return result


def _format_measure_diagnostic(
    per_topic_missing: Dict[str, Dict[str, Any]],
    aggregate_missing: Dict[str, Dict[str, Any]],
) -> str:
    """Format diagnostic message showing per-measure breakdown."""
    lines = []

    if per_topic_missing:
        lines.append("Per-topic measure mismatches:")
        for measure, info in sorted(per_topic_missing.items()):
            present = info["present_in_runs"]
            present_str = f"only in: {', '.join(present)}" if present else "in none"
            lines.append(
                f"  - '{measure}': missing from {info['missing_count']}/{info['total']} "
                f"entries ({present_str})"
            )

    if aggregate_missing:
        lines.append("Aggregate measure mismatches:")
        for measure, info in sorted(aggregate_missing.items()):
            present = info["present_in_runs"]
            present_str = f"only in: {', '.join(present)}" if present else "in none"
            lines.append(
                f"  - '{measure}': missing from {info['missing_count']}/{info['total']} "
                f"runs ({present_str})"
            )

    return "\n".join(lines)


def _handle_failure(on_fail: OnFail, message: str) -> None:
    """Handle verification failure based on policy."""
    if on_fail == "error":
        raise VerificationError(message)
    elif on_fail == "warn":
        import sys
        print(f"Warning: {message}", file=sys.stderr)
    # "ignore" does nothing
