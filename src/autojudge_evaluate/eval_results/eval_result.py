"""
A. EvalResult - Immutable read-only container for evaluation results.

This is a thin data vessel. All construction, filtering, and modification
goes through EvalResultBuilder. All validation goes through verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Set, Tuple, List, TYPE_CHECKING

# Reserved topic_id for aggregate rows (macro-average across topics)
ALL_TOPIC_ID = "all"

# Measure dtype: float for numeric, str for categorical
MeasureDtype = Literal["float", "str"]


@dataclass(frozen=True)
class EvalEntry:
    """
    Single evaluation cell: (run_id, topic_id, measure) -> value.

    This is the atomic unit of evaluation data, matching file format lines.
    Value is stored as-is (string from file); casting happens in Builder.
    """
    run_id: str
    topic_id: str
    measure: str
    value: float | str


@dataclass(frozen=True)
class MeasureSpecs:
    """
    Specification of measures and their dtypes for per-topic and aggregate entries.

    Per-topic and aggregate entries can have different measures:
    - per_topic: measures that appear in per-topic entries (e.g., COUNT, FRACT, TOP_WORD)
    - aggregate: measures that appear in aggregate entries (e.g., COUNT, FRACT, COUNT_SUM)

    String measures only appear in per_topic, not aggregate.
    """
    per_topic: Dict[str, MeasureDtype]
    aggregate: Dict[str, MeasureDtype]

    @classmethod
    def from_single(cls, dtypes: Dict[str, MeasureDtype]) -> "MeasureSpecs":
        """
        Create specs where aggregate mirrors per_topic (floats only).

        String measures are excluded from aggregate.
        """
        aggregate = {m: d for m, d in dtypes.items() if d == "float"}
        return cls(per_topic=dict(dtypes), aggregate=aggregate)

    @classmethod
    def infer(cls, entries: List["EvalEntry"]) -> "MeasureSpecs":
        """
        Infer specs from entries by separating per-topic from aggregate.

        Examines actual entries to determine which measures exist in
        per-topic vs aggregate rows.
        """
        per_topic_values: Dict[str, List[str]] = {}
        aggregate_values: Dict[str, List[str]] = {}

        for e in entries:
            value_str = str(e.value).strip()
            if e.topic_id == ALL_TOPIC_ID:
                if e.measure not in aggregate_values:
                    aggregate_values[e.measure] = []
                aggregate_values[e.measure].append(value_str)
            else:
                if e.measure not in per_topic_values:
                    per_topic_values[e.measure] = []
                per_topic_values[e.measure].append(value_str)

        per_topic = {m: _infer_dtype(vals) for m, vals in per_topic_values.items()}
        aggregate = {m: _infer_dtype(vals) for m, vals in aggregate_values.items()}

        return cls(per_topic=per_topic, aggregate=aggregate)

    @classmethod
    def empty(cls) -> "MeasureSpecs":
        """Create empty specs."""
        return cls(per_topic={}, aggregate={})

    @property
    def all_measures(self) -> Set[str]:
        """Union of per-topic and aggregate measures."""
        return set(self.per_topic.keys()) | set(self.aggregate.keys())

    def filter(self, measures: Set[str]) -> "MeasureSpecs":
        """Return new specs with only the specified measures."""
        return MeasureSpecs(
            per_topic={m: d for m, d in self.per_topic.items() if m in measures},
            aggregate={m: d for m, d in self.aggregate.items() if m in measures},
        )

    def with_empty_aggregate(self) -> "MeasureSpecs":
        """Return new specs with empty aggregate (for drop_aggregates)."""
        return MeasureSpecs(per_topic=dict(self.per_topic), aggregate={})

    def with_computed_aggregate(self) -> "MeasureSpecs":
        """Return new specs where aggregate mirrors per_topic floats (for recompute)."""
        aggregate = {m: d for m, d in self.per_topic.items() if d == "float"}
        return MeasureSpecs(per_topic=dict(self.per_topic), aggregate=aggregate)


def _infer_dtype(values: List[str]) -> MeasureDtype:
    """Infer dtype from a list of string values."""
    if not values:
        return "float"  # Default for empty

    try:
        for v in values:
            float(v)
        return "float"
    except (ValueError, TypeError):
        return "str"


@dataclass(frozen=True)
class EvalResult:
    """
    Immutable evaluation result container.

    Contains a tuple of entries and specs defining per-topic/aggregate measures.
    Read-only: use EvalResultBuilder to create or modify.
    """
    entries: Tuple[EvalEntry, ...]
    specs: MeasureSpecs

    # =========================================================================
    # Read-only accessors
    # =========================================================================

    @property
    def run_ids(self) -> Set[str]:
        """All unique run_ids in entries."""
        return {e.run_id for e in self.entries}

    @property
    def topic_ids(self) -> Set[str]:
        """All unique topic_ids, excluding ALL_TOPIC_ID."""
        return {e.topic_id for e in self.entries if e.topic_id != ALL_TOPIC_ID}

    @property
    def all_topic_ids(self) -> Set[str]:
        """All unique topic_ids, including ALL_TOPIC_ID if present."""
        return {e.topic_id for e in self.entries}

    @property
    def measures(self) -> Set[str]:
        """All unique measure names."""
        return {e.measure for e in self.entries}

    @property
    def measure_dtypes(self) -> Dict[str, MeasureDtype]:
        """All measure dtypes (union of per_topic and aggregate). For backward compatibility."""
        result = dict(self.specs.per_topic)
        result.update(self.specs.aggregate)
        return result

    @property
    def has_aggregates(self) -> bool:
        """Whether any entries have topic_id == ALL_TOPIC_ID."""
        return any(e.topic_id == ALL_TOPIC_ID for e in self.entries)

    # =========================================================================
    # Grouped accessors
    # =========================================================================

    def entries_by_run(self, run_id: str) -> Tuple[EvalEntry, ...]:
        """Get all entries for a specific run_id."""
        return tuple(e for e in self.entries if e.run_id == run_id)

    def entries_by_topic(self, topic_id: str) -> Tuple[EvalEntry, ...]:
        """Get all entries for a specific topic_id."""
        return tuple(e for e in self.entries if e.topic_id == topic_id)

    def entries_by_measure(self, measure: str) -> Tuple[EvalEntry, ...]:
        """Get all entries for a specific measure."""
        return tuple(e for e in self.entries if e.measure == measure)

    def get_value(self, run_id: str, topic_id: str, measure: str) -> float | str | None:
        """Get specific value, or None if not found."""
        for e in self.entries:
            if e.run_id == run_id and e.topic_id == topic_id and e.measure == measure:
                return e.value
        return None

    def get_aggregate_ranking(self, measure: str) -> Dict[str, float]:
        """
        Get run_id -> value mapping from aggregate (ALL_TOPIC_ID) rows.

        Returns:
            Dict mapping run_id to aggregate value for the measure.

        Raises:
            ValueError: If measure not found in aggregate rows or not numeric.
        """
        ranking: Dict[str, float] = {}
        for e in self.entries:
            if e.topic_id == ALL_TOPIC_ID and e.measure == measure:
                if not isinstance(e.value, (int, float)):
                    raise ValueError(f"Measure '{measure}' has non-numeric aggregate value")
                ranking[e.run_id] = float(e.value)

        if not ranking:
            raise ValueError(f"Measure '{measure}' not found in aggregate rows")

        return ranking

    def top_k_run_ids(self, measure: str, k: int) -> list[str]:
        """
        Get top k run_ids by aggregate measure value (descending).

        Args:
            measure: Measure name to sort by
            k: Number of top run_ids to return

        Returns:
            List of top k run_ids.
        """
        ranking = self.get_aggregate_ranking(measure)
        sorted_runs = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
        return [run_id for run_id, _ in sorted_runs[:k]]

    def filter_runs(self, run_ids: Set[str]) -> "EvalResult":
        """
        Filter entries by run_ids WITHOUT recomputing aggregates.

        Use this for run-based filtering where aggregate values should be preserved.
        The aggregate entries are also filtered to only include the specified run_ids.

        Args:
            run_ids: Keep only these run_ids.

        Returns:
            New EvalResult with filtered entries (including filtered aggregates).
        """
        filtered_entries = tuple(e for e in self.entries if e.run_id in run_ids)
        return EvalResult(entries=filtered_entries, specs=self.specs)

    def filter_topics_and_recompute(
        self,
        topic_ids: Set[str] | None = None,
    ) -> "EvalResult":
        """
        Filter entries by topic_ids and recompute aggregates.

        Use this for topic-based filtering which requires recomputing aggregates
        (since aggregates are means over topics).

        Args:
            topic_ids: Keep only these topic_ids. None = keep all (but still recompute).

        Returns:
            New EvalResult with filtered entries and fresh aggregates.
        """
        from .builder import EvalResultBuilder

        builder = EvalResultBuilder(self.specs)
        for e in self.entries:
            builder.add_entry(e)

        filtered = builder.filter(
            run_ids=None,
            topic_ids=topic_ids,
            drop_aggregates=True,
        )

        return filtered.build(
            compute_aggregates=True,
            verify=True,
            on_missing="ignore",
        )

    def filter_and_recompute(
        self,
        run_ids: Set[str] | None = None,
        topic_ids: Set[str] | None = None,
    ) -> "EvalResult":
        """
        Filter entries and recompute aggregates.

        Creates new EvalResult with only specified run_ids/topic_ids,
        dropping existing aggregates and recomputing them.

        Note: For run-only filtering without recompute, use filter_runs() instead.

        Args:
            run_ids: Keep only these run_ids. None = keep all.
            topic_ids: Keep only these topic_ids. None = keep all.

        Returns:
            New EvalResult with filtered entries and fresh aggregates.
        """
        from .builder import EvalResultBuilder

        builder = EvalResultBuilder(self.specs)
        for e in self.entries:
            builder.add_entry(e)

        filtered = builder.filter(
            run_ids=run_ids,
            topic_ids=topic_ids,
            drop_aggregates=True,
        )

        return filtered.build(
            compute_aggregates=True,
            verify=True,
            on_missing="ignore",
        )

    def check_aggregate_discrepancies(
        self,
        tolerance: float = 0.001,
    ) -> Dict[str, Dict[str, tuple]]:
        """
        Check if file aggregates match recomputed aggregates (simple mean).

        This is useful for:
        - Detecting when upstream data uses weighted averages
        - Finding data errors in aggregate computation
        - Understanding why correlation results differ between preserve/recompute modes

        Args:
            tolerance: Maximum allowed difference between original and recomputed.

        Returns:
            Dict of {run_id: {measure: (original, recomputed, diff)}} for discrepancies.
            Empty dict if all aggregates match.

        Example:
            >>> discrepancies = result.check_aggregate_discrepancies()
            >>> for run_id, measures in discrepancies.items():
            ...     for measure, (orig, recomp, diff) in measures.items():
            ...         print(f"{run_id}/{measure}: {orig} vs {recomp} (diff={diff})")
        """
        recomputed = self.filter_and_recompute()

        discrepancies: Dict[str, Dict[str, tuple]] = {}

        for run_id in self.run_ids:
            run_discrepancies = {}
            for measure in self.measures:
                orig = self.get_value(run_id, ALL_TOPIC_ID, measure)
                recomp = recomputed.get_value(run_id, ALL_TOPIC_ID, measure)

                if orig is None or recomp is None:
                    continue

                # Only compare numeric values
                if not isinstance(orig, (int, float)) or not isinstance(recomp, (int, float)):
                    continue

                diff = abs(orig - recomp)
                if diff > tolerance:
                    run_discrepancies[measure] = (orig, recomp, diff)

            if run_discrepancies:
                discrepancies[run_id] = run_discrepancies

        return discrepancies

    def report_aggregate_discrepancies(self, tolerance: float = 0.001) -> str:
        """
        Generate a human-readable report of aggregate discrepancies.

        Args:
            tolerance: Maximum allowed difference between original and recomputed.

        Returns:
            Multi-line string report, or "No discrepancies found." if clean.
        """
        discrepancies = self.check_aggregate_discrepancies(tolerance)

        if not discrepancies:
            return "No aggregate discrepancies found."

        lines = ["Aggregate discrepancies (file vs recomputed):"]

        # Group by measure for clearer reporting
        by_measure: Dict[str, list] = {}
        for run_id, measures in discrepancies.items():
            for measure, (orig, recomp, diff) in measures.items():
                if measure not in by_measure:
                    by_measure[measure] = []
                by_measure[measure].append((run_id, orig, recomp, diff))

        for measure, runs in sorted(by_measure.items()):
            lines.append(f"\n  {measure}:")
            for run_id, orig, recomp, diff in sorted(runs, key=lambda x: -x[3])[:5]:
                lines.append(f"    {run_id}: {orig:.6f} vs {recomp:.6f} (diff={diff:.6f})")
            if len(runs) > 5:
                lines.append(f"    ... and {len(runs) - 5} more runs")

        return "\n".join(lines)
