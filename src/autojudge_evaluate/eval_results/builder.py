"""
D. Builder - Construction, filtering, and aggregation for EvalResult.

All mutable operations happen here. The build() method is the only place
where spec checking and completeness verification occurs.

Design:
- add() methods accumulate raw entries
- filter() methods mark entries for inclusion/exclusion
- build() casts values, computes aggregates, runs verification, returns immutable EvalResult
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Callable, Dict, List, Literal, Optional, Set, Sequence

from .eval_result import EvalResult, EvalEntry, MeasureSpecs, ALL_TOPIC_ID, MeasureDtype
from . import verification


class EvalResultBuilder:
    """
    Builder for constructing EvalResult instances.

    Usage (direct construction with specs):
        specs = MeasureSpecs.from_single({"ndcg": "float", "map": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", 0.5)
        builder.add("run1", "t2", "ndcg", 0.6)
        result = builder.build()

    For loading from files, use io.load() which infers specs automatically.
    """

    def __init__(self, specs: MeasureSpecs):
        """
        Create builder with given specs.

        Args:
            specs: MeasureSpecs defining per-topic and aggregate measures.
        """
        self._entries: List[_RawEntry] = []
        self._specs = specs

    @property
    def specs(self) -> MeasureSpecs:
        """Current specs."""
        return self._specs

    # =========================================================================
    # Add methods - accumulate raw entries
    # =========================================================================

    def add(
        self,
        run_id: str,
        topic_id: str,
        measure: str,
        value: str | float | int,
    ) -> "EvalResultBuilder":
        """
        Add a single entry.

        Value can be string (from file) or already numeric.
        Casting to final dtype happens in build().
        """
        self._entries.append(_RawEntry(run_id, topic_id, measure, value))
        return self

    def add_entry(self, entry: EvalEntry) -> "EvalResultBuilder":
        """Add an existing EvalEntry."""
        return self.add(entry.run_id, entry.topic_id, entry.measure, entry.value)

    def add_from(self, result: EvalResult) -> "EvalResultBuilder":
        """
        Add all entries from an existing EvalResult.

        Raises:
            ValueError: If result's specs don't match this builder's specs.
        """
        if result.specs != self._specs:
            raise ValueError(
                f"Cannot add_from: specs mismatch. "
                f"Builder has {self._specs}, result has {result.specs}"
            )
        for e in result.entries:
            self.add_entry(e)
        return self

    def add_records(
        self,
        records: Sequence,
        run_id: Callable,
        topic_id: Callable,
        measures: Callable,
    ) -> "EvalResultBuilder":
        """
        Add entries from arbitrary record objects.

        Args:
            records: Iterable of record objects
            run_id: Function to extract run_id from record
            topic_id: Function to extract topic_id from record
            measures: Function returning dict of {measure: value} from record
        """
        for r in records:
            r_run = run_id(r)
            r_topic = topic_id(r)
            for m, v in measures(r).items():
                self.add(r_run, r_topic, m, v)
        return self

    # =========================================================================
    # Filter methods - create new builder with subset
    # =========================================================================

    def filter(
        self,
        run_ids: Optional[Set[str]] = None,
        topic_ids: Optional[Set[str]] = None,
        measures: Optional[Set[str]] = None,
        drop_aggregates: bool = True,
    ) -> "EvalResultBuilder":
        """
        Create new builder with filtered entries.

        Args:
            run_ids: Keep only these run_ids. None = keep all.
            topic_ids: Keep only these topic_ids. None = keep all.
            measures: Keep only these measures. None = keep all.
            drop_aggregates: If True, exclude ALL_TOPIC_ID entries.

        Returns:
            New EvalResultBuilder with filtered entries and shrunk specs.
        """
        # Determine new specs
        if measures is not None:
            new_specs = self._specs.filter(measures)
        elif drop_aggregates:
            new_specs = self._specs.with_empty_aggregate()
        else:
            new_specs = self._specs

        new_builder = EvalResultBuilder(new_specs)

        for e in self._entries:
            # Drop aggregates if requested
            if drop_aggregates and e.topic_id == ALL_TOPIC_ID:
                continue

            # Apply filters
            if run_ids is not None and e.run_id not in run_ids:
                continue
            if topic_ids is not None and e.topic_id not in topic_ids:
                continue
            if measures is not None and e.measure not in measures:
                continue

            new_builder._entries.append(e)

        return new_builder

    # =========================================================================
    # Build - the only place where validation and aggregation happens
    # =========================================================================

    def build(
        self,
        *,
        compute_aggregates: bool,
        verify: bool,
        on_missing: Literal["error", "warn", "ignore"],
    ) -> EvalResult:
        """
        Build immutable EvalResult from accumulated entries.

        This is the single place where:
        1. Value casting happens (string -> float where applicable)
        2. Aggregates are computed (if compute_aggregates=True)
        3. Verification runs (if verify=True)

        Args:
            compute_aggregates: If True, compute ALL_TOPIC_ID rows via macro-average
                               for float measures. Updates specs.aggregate to mirror
                               specs.per_topic (floats only).
            verify: If True, run verification checks.
            on_missing: How to handle incomplete data (missing entries).

        Returns:
            Immutable EvalResult.
        """
        # Determine final specs based on compute_aggregates
        if compute_aggregates:
            final_specs = self._specs.with_computed_aggregate()
        else:
            final_specs = self._specs

        # Cast values and build entries
        casted_entries = _cast_entries(self._entries, final_specs)

        # Separate per-topic and aggregate entries
        per_topic = [e for e in casted_entries if e.topic_id != ALL_TOPIC_ID]
        existing_agg = [e for e in casted_entries if e.topic_id == ALL_TOPIC_ID]

        # Compute aggregates if requested (only for float measures)
        if compute_aggregates:
            agg_entries = _compute_aggregates(per_topic, final_specs)
        else:
            agg_entries = existing_agg

        # Combine entries
        all_entries = per_topic + agg_entries

        # Verify if requested
        if verify:
            _run_verification(all_entries, final_specs, on_missing)

        return EvalResult(
            entries=tuple(all_entries),
            specs=final_specs,
        )


# =============================================================================
# Internal data structures (local to this module)
# =============================================================================


class _RawEntry:
    """Raw entry with uncasted value. Internal to builder."""
    __slots__ = ("run_id", "topic_id", "measure", "value")

    def __init__(self, run_id: str, topic_id: str, measure: str, value):
        self.run_id = run_id
        self.topic_id = topic_id
        self.measure = measure
        self.value = value


# =============================================================================
# Casting (local functions)
# =============================================================================


def _cast_entries(
    entries: List[_RawEntry],
    specs: MeasureSpecs,
) -> List[EvalEntry]:
    """Cast raw entries to typed EvalEntry objects using specs."""
    result = []
    for e in entries:
        # Use appropriate spec based on whether this is aggregate or per-topic
        if e.topic_id == ALL_TOPIC_ID:
            dtype = specs.aggregate.get(e.measure, "str")
        else:
            dtype = specs.per_topic.get(e.measure, "str")

        if dtype == "float":
            value = float(e.value)
        else:
            value = str(e.value)

        result.append(EvalEntry(
            run_id=e.run_id,
            topic_id=e.topic_id,
            measure=e.measure,
            value=value,
        ))
    return result


# =============================================================================
# Aggregation (local functions)
# =============================================================================


def _compute_aggregates(
    per_topic_entries: List[EvalEntry],
    specs: MeasureSpecs,
) -> List[EvalEntry]:
    """
    Compute aggregate (ALL_TOPIC_ID) entries via macro-average.

    Only computes aggregates for measures in specs.aggregate (floats only).
    String measures are NOT aggregated.
    """
    # Group values by (run_id, measure) - only for aggregate measures
    grouped: Dict[tuple[str, str], List[float]] = defaultdict(list)

    for e in per_topic_entries:
        # Only aggregate if measure is in aggregate specs
        if e.measure in specs.aggregate:
            grouped[(e.run_id, e.measure)].append(float(e.value))

    # Compute aggregates
    agg_entries = []
    for (run_id, measure), values in grouped.items():
        agg_value = mean(values) if values else 0.0
        agg_entries.append(EvalEntry(
            run_id=run_id,
            topic_id=ALL_TOPIC_ID,
            measure=measure,
            value=agg_value,
        ))

    return agg_entries


# =============================================================================
# Verification (local function)
# =============================================================================


def _run_verification(
    entries: List[EvalEntry],
    specs: MeasureSpecs,
    on_missing: Literal["error", "warn", "ignore"],
) -> None:
    """Run verification checks on final entries."""
    # Check that all runs have same topics
    verification.check_same_topics_per_run(entries, on_fail=on_missing)

    # Check that all (run_id, topic_id) have measures matching per_topic specs
    verification.check_measures_match_specs(
        entries, specs.per_topic, specs.aggregate, on_fail=on_missing
    )

    # Check aggregates exist for all measures in aggregate specs
    verification.check_aggregates_match_specs(
        entries, specs.aggregate, on_fail=on_missing
    )
