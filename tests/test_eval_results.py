"""Simple tests for the eval_results module."""

import tempfile
from pathlib import Path

import pytest

from autojudge_evaluate.eval_results import (
    EvalResult,
    EvalEntry,
    EvalResultBuilder,
    MeasureSpecs,
    ALL_TOPIC_ID,
    load,
    write,
)


class TestEvalResultBuilder:
    """Tests for EvalResultBuilder."""

    def test_basic_build(self):
        """Build a simple result with two runs, two topics, one measure."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", "0.5")
        builder.add("run1", "t2", "ndcg", "0.7")
        builder.add("run2", "t1", "ndcg", "0.3")
        builder.add("run2", "t2", "ndcg", "0.9")

        result = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        assert result.run_ids == {"run1", "run2"}
        assert result.topic_ids == {"t1", "t2"}
        assert result.measures == {"ndcg"}
        assert result.has_aggregates

    def test_dtype_inference_float(self):
        """Float values should be inferred as float dtype."""
        specs = MeasureSpecs.from_single({"score": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "score", "0.5")
        builder.add("run1", "t2", "score", "0.7")

        result = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        assert result.measure_dtypes["score"] == "float"
        assert result.get_value("run1", "t1", "score") == pytest.approx(0.5)

    def test_dtype_inference_int(self):
        """Integer values should be inferred as float dtype."""
        specs = MeasureSpecs.from_single({"count": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "count", "5")
        builder.add("run1", "t2", "count", "10")

        result = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        assert result.measure_dtypes["count"] == "float"
        assert result.get_value("run1", "t1", "count") == pytest.approx(5.0)

    def test_dtype_string(self):
        """String values use str dtype."""
        specs = MeasureSpecs(per_topic={"label": "str"}, aggregate={})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "label", "good")
        builder.add("run1", "t2", "label", "bad")

        result = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        assert result.measure_dtypes["label"] == "str"
        assert result.get_value("run1", "t1", "label") == "good"
        # No aggregate for string measures
        assert result.get_value("run1", ALL_TOPIC_ID, "label") is None

    def test_aggregate_computation(self):
        """Aggregates should be computed as macro-average."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", "0.4")
        builder.add("run1", "t2", "ndcg", "0.6")

        result = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        # Macro-average: (0.4 + 0.6) / 2 = 0.5
        agg_value = result.get_value("run1", ALL_TOPIC_ID, "ndcg")
        assert agg_value == pytest.approx(0.5)

    def test_get_aggregate_ranking(self):
        """Get ranking from aggregate values."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", "0.4")
        builder.add("run1", "t2", "ndcg", "0.6")
        builder.add("run2", "t1", "ndcg", "0.8")
        builder.add("run2", "t2", "ndcg", "0.9")

        result = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        ranking = result.get_aggregate_ranking("ndcg")
        assert ranking["run1"] == pytest.approx(0.5)  # (0.4 + 0.6) / 2
        assert ranking["run2"] == pytest.approx(0.85)  # (0.8 + 0.9) / 2

    def test_top_k_run_ids(self):
        """Get top k runs by aggregate measure."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", "0.3")
        builder.add("run2", "t1", "ndcg", "0.9")
        builder.add("run3", "t1", "ndcg", "0.6")

        result = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        top_2 = result.top_k_run_ids("ndcg", 2)
        assert top_2 == ["run2", "run3"]

    def test_filter_by_run_ids(self):
        """Filter to subset of runs."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", "0.5")
        builder.add("run2", "t1", "ndcg", "0.7")
        builder.add("run3", "t1", "ndcg", "0.9")

        filtered = builder.filter(run_ids={"run1", "run2"})
        result = filtered.build(compute_aggregates=True, verify=False, on_missing="ignore")

        assert result.run_ids == {"run1", "run2"}

    def test_filter_by_topic_ids(self):
        """Filter to subset of topics."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", "0.5")
        builder.add("run1", "t2", "ndcg", "0.7")
        builder.add("run1", "t3", "ndcg", "0.9")

        filtered = builder.filter(topic_ids={"t1", "t2"})
        result = filtered.build(compute_aggregates=True, verify=False, on_missing="ignore")

        assert result.topic_ids == {"t1", "t2"}


class TestIO:
    """Tests for load/write functions."""

    def test_write_and_load_tot_format(self):
        """Round-trip test for tot format."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", "0.5")
        builder.add("run1", "t2", "ndcg", "0.7")
        original = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.txt"
            write(original, path, format="tot")
            loaded = load(path, format="tot", drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        assert loaded.run_ids == original.run_ids
        assert loaded.topic_ids == original.topic_ids
        assert loaded.get_value("run1", "t1", "ndcg") == pytest.approx(0.5)

    def test_write_and_load_ir_measures_format(self):
        """Round-trip test for ir_measures format."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", "0.5")
        builder.add("run1", "t2", "ndcg", "0.7")
        original = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.txt"
            write(original, path, format="ir_measures")
            loaded = load(path, format="ir_measures", drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        assert loaded.run_ids == original.run_ids
        assert loaded.get_value("run1", "t1", "ndcg") == pytest.approx(0.5)

    def test_write_and_load_jsonl_format(self):
        """Round-trip test for jsonl format."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", "0.5")
        builder.add("run1", "t2", "ndcg", "0.7")
        original = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.jsonl"
            write(original, path, format="jsonl")
            loaded = load(path, format="jsonl", drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        assert loaded.run_ids == original.run_ids
        assert loaded.get_value("run1", "t1", "ndcg") == pytest.approx(0.5)

    def test_load_directory(self):
        """Load multiple files from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write two run files
            (tmppath / "run1").write_text("ndcg\tt1\t0.5\nndcg\tt2\t0.7\n")
            (tmppath / "run2").write_text("ndcg\tt1\t0.8\nndcg\tt2\t0.9\n")

            loaded = load(tmppath, format="trec_eval", drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        assert loaded.run_ids == {"run1", "run2"}
        assert loaded.get_value("run1", "t1", "ndcg") == pytest.approx(0.5)
        assert loaded.get_value("run2", "t1", "ndcg") == pytest.approx(0.8)


class TestEvalResult:
    """Tests for EvalResult accessors."""

    def test_entries_by_run(self):
        """Get entries for a specific run."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", "0.5")
        builder.add("run1", "t2", "ndcg", "0.7")
        builder.add("run2", "t1", "ndcg", "0.9")
        result = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        run1_entries = result.entries_by_run("run1")
        # 2 per-topic + 1 aggregate
        assert len([e for e in run1_entries if e.topic_id != ALL_TOPIC_ID]) == 2

    def test_entries_by_topic(self):
        """Get entries for a specific topic."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", "0.5")
        builder.add("run2", "t1", "ndcg", "0.7")
        result = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        t1_entries = result.entries_by_topic("t1")
        assert len(t1_entries) == 2

    def test_entries_by_measure(self):
        """Get entries for a specific measure."""
        specs = MeasureSpecs.from_single({"ndcg": "float", "map": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "ndcg", "0.5")
        builder.add("run1", "t1", "map", "0.3")
        result = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        ndcg_entries = result.entries_by_measure("ndcg")
        # 1 per-topic + 1 aggregate
        assert len(ndcg_entries) == 2


class TestMeasureSpecs:
    """Tests for MeasureSpecs."""

    def test_from_single_excludes_strings_from_aggregate(self):
        """from_single should exclude string measures from aggregate."""
        dtypes = {"score": "float", "label": "str"}
        specs = MeasureSpecs.from_single(dtypes)

        assert specs.per_topic == {"score": "float", "label": "str"}
        assert specs.aggregate == {"score": "float"}  # no label

    def test_infer_separates_per_topic_and_aggregate(self):
        """infer should create separate specs for per-topic and aggregate."""
        entries = [
            EvalEntry("run1", "t1", "score", "0.5"),
            EvalEntry("run1", "t2", "score", "0.7"),
            EvalEntry("run1", ALL_TOPIC_ID, "score", "0.6"),
            EvalEntry("run1", ALL_TOPIC_ID, "score_sum", "1.2"),
        ]
        specs = MeasureSpecs.infer(entries)

        assert specs.per_topic == {"score": "float"}
        assert specs.aggregate == {"score": "float", "score_sum": "float"}

    def test_with_empty_aggregate(self):
        """with_empty_aggregate should return specs with empty aggregate."""
        specs = MeasureSpecs(
            per_topic={"a": "float", "b": "str"},
            aggregate={"a": "float", "c": "float"},
        )
        new_specs = specs.with_empty_aggregate()

        assert new_specs.per_topic == {"a": "float", "b": "str"}
        assert new_specs.aggregate == {}

    def test_with_computed_aggregate(self):
        """with_computed_aggregate should mirror per_topic floats only."""
        specs = MeasureSpecs(
            per_topic={"a": "float", "b": "str"},
            aggregate={},
        )
        new_specs = specs.with_computed_aggregate()

        assert new_specs.per_topic == {"a": "float", "b": "str"}
        assert new_specs.aggregate == {"a": "float"}  # no b (string)

    def test_filter(self):
        """filter should shrink both per_topic and aggregate."""
        specs = MeasureSpecs(
            per_topic={"a": "float", "b": "float", "c": "str"},
            aggregate={"a": "float", "b": "float"},
        )
        new_specs = specs.filter({"a", "c"})

        assert new_specs.per_topic == {"a": "float", "c": "str"}
        assert new_specs.aggregate == {"a": "float"}


class TestFilterAndRecompute:
    """Tests for EvalResult.filter_and_recompute()."""

    def test_filter_by_topic_ids(self):
        """Filter topics and verify aggregates are recomputed."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run_a", "t1", "ndcg", "0.5")
        builder.add("run_a", "t2", "ndcg", "0.6")
        builder.add("run_a", "t3", "ndcg", "0.7")
        builder.add("run_b", "t1", "ndcg", "0.3")
        builder.add("run_b", "t2", "ndcg", "0.4")
        builder.add("run_b", "t3", "ndcg", "0.5")
        original = builder.build(compute_aggregates=True, verify=True, on_missing="ignore")

        # Original aggregates: run_a = 0.6, run_b = 0.4
        assert original.get_aggregate_ranking("ndcg") == {
            "run_a": pytest.approx(0.6),
            "run_b": pytest.approx(0.4),
        }

        # Filter to topics t1, t2 only
        filtered = original.filter_and_recompute(topic_ids={"t1", "t2"})

        # Filtered aggregates: run_a = 0.55, run_b = 0.35
        ranking = filtered.get_aggregate_ranking("ndcg")
        assert ranking["run_a"] == pytest.approx(0.55)
        assert ranking["run_b"] == pytest.approx(0.35)

    def test_filter_by_run_ids(self):
        """Filter runs and verify only those runs remain."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run_a", "t1", "ndcg", "0.5")
        builder.add("run_a", "t2", "ndcg", "0.7")
        builder.add("run_b", "t1", "ndcg", "0.3")
        builder.add("run_b", "t2", "ndcg", "0.5")
        original = builder.build(compute_aggregates=True, verify=True, on_missing="ignore")

        filtered = original.filter_and_recompute(run_ids={"run_a"})

        assert filtered.run_ids == {"run_a"}
        assert filtered.get_aggregate_ranking("ndcg") == {"run_a": pytest.approx(0.6)}

    def test_filter_both_run_ids_and_topic_ids(self):
        """Filter both runs and topics simultaneously."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run_a", "t1", "ndcg", "0.5")
        builder.add("run_a", "t2", "ndcg", "0.7")
        builder.add("run_b", "t1", "ndcg", "0.3")
        builder.add("run_b", "t2", "ndcg", "0.5")
        original = builder.build(compute_aggregates=True, verify=True, on_missing="ignore")

        filtered = original.filter_and_recompute(run_ids={"run_a"}, topic_ids={"t1"})

        assert filtered.run_ids == {"run_a"}
        assert filtered.topic_ids == {"t1"}
        assert filtered.get_aggregate_ranking("ndcg") == {"run_a": pytest.approx(0.5)}

    def test_filter_none_keeps_all(self):
        """Passing None for both filters keeps all entries."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run_a", "t1", "ndcg", "0.5")
        builder.add("run_b", "t1", "ndcg", "0.3")
        original = builder.build(compute_aggregates=True, verify=True, on_missing="ignore")

        filtered = original.filter_and_recompute()

        assert filtered.run_ids == {"run_a", "run_b"}
        assert filtered.get_aggregate_ranking("ndcg") == {
            "run_a": pytest.approx(0.5),
            "run_b": pytest.approx(0.3),
        }


class TestBuilderAddRecords:
    """Tests for EvalResultBuilder.add_records()."""

    def test_add_records_from_objects(self):
        """add_records extracts values from arbitrary objects."""
        specs = MeasureSpecs.from_single({"score": "float", "label": "str"})

        class Record:
            def __init__(self, run, topic, score, label):
                self.run = run
                self.topic = topic
                self.score = score
                self.label = label

        records = [
            Record("run1", "t1", 0.8, "good"),
            Record("run1", "t2", 0.6, "fair"),
            Record("run2", "t1", 0.9, "excellent"),
        ]

        builder = EvalResultBuilder(specs)
        builder.add_records(
            records,
            run_id=lambda r: r.run,
            topic_id=lambda r: r.topic,
            measures=lambda r: {"score": r.score, "label": r.label},
        )

        result = builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        # Check per-topic entries
        assert result.get_value("run1", "t1", "score") == pytest.approx(0.8)
        assert result.get_value("run1", "t1", "label") == "good"
        assert result.get_value("run2", "t1", "score") == pytest.approx(0.9)

        # Check aggregates (mean for float)
        assert result.get_value("run1", ALL_TOPIC_ID, "score") == pytest.approx(0.7)


class TestAggregateSanityCheck:
    """Tests for detecting when file aggregates differ from recomputed ones.

    This is important for diagnosing issues like:
    - f1 vs f1_macro (might use different aggregation formulas)
    - nugget_coverage vs nugget_coverage_macro
    - Upstream data errors
    """

    def test_aggregates_match_when_simple_mean(self):
        """When file uses simple mean, recomputed should match."""
        specs = MeasureSpecs.from_single({"score": "float"})
        builder = EvalResultBuilder(specs)
        # Add per-topic data
        builder.add("run1", "t1", "score", "0.4")
        builder.add("run1", "t2", "score", "0.6")
        # Add aggregate as simple mean: (0.4 + 0.6) / 2 = 0.5
        builder.add("run1", ALL_TOPIC_ID, "score", "0.5")

        original = builder.build(compute_aggregates=False, verify=False, on_missing="ignore")
        recomputed = original.filter_and_recompute()

        orig_agg = original.get_value("run1", ALL_TOPIC_ID, "score")
        recomp_agg = recomputed.get_value("run1", ALL_TOPIC_ID, "score")

        assert orig_agg == pytest.approx(recomp_agg), \
            f"Aggregates should match: original={orig_agg}, recomputed={recomp_agg}"

    def test_aggregates_differ_when_weighted_mean(self):
        """When file uses weighted mean, recomputed (simple mean) will differ."""
        specs = MeasureSpecs.from_single({"score": "float"})
        builder = EvalResultBuilder(specs)
        # Add per-topic data with different "weights" (simulated)
        # t1 has 1 item, t2 has 3 items (but we only store the average)
        builder.add("run1", "t1", "score", "1.0")  # 1 item with score 1.0
        builder.add("run1", "t2", "score", "0.5")  # 3 items with avg 0.5

        # File aggregate uses weighted mean: (1*1.0 + 3*0.5) / 4 = 2.5/4 = 0.625
        builder.add("run1", ALL_TOPIC_ID, "score", "0.625")

        original = builder.build(compute_aggregates=False, verify=False, on_missing="ignore")
        recomputed = original.filter_and_recompute()

        orig_agg = original.get_value("run1", ALL_TOPIC_ID, "score")
        recomp_agg = recomputed.get_value("run1", ALL_TOPIC_ID, "score")
        # Recomputed uses simple mean: (1.0 + 0.5) / 2 = 0.75

        assert orig_agg == pytest.approx(0.625)
        assert recomp_agg == pytest.approx(0.75)
        assert orig_agg != pytest.approx(recomp_agg), \
            "Aggregates should differ when original uses weighted mean"

    def test_detect_aggregate_discrepancy(self):
        """Utility test showing how to detect aggregate discrepancies."""
        specs = MeasureSpecs.from_single({"f1": "float", "f1_macro": "float"})
        builder = EvalResultBuilder(specs)

        # Simulate data where f1 and f1_macro have different values
        # f1 per-topic: 0.8, 0.6
        # f1_macro in file: 0.7 (simple mean = correct)
        # But imagine f1 aggregate in file was computed differently
        builder.add("run1", "t1", "f1", "0.8")
        builder.add("run1", "t2", "f1", "0.6")
        builder.add("run1", ALL_TOPIC_ID, "f1", "0.65")  # Wrong! Should be 0.7

        builder.add("run1", "t1", "f1_macro", "0.8")
        builder.add("run1", "t2", "f1_macro", "0.6")
        builder.add("run1", ALL_TOPIC_ID, "f1_macro", "0.7")  # Correct

        original = builder.build(compute_aggregates=False, verify=False, on_missing="ignore")
        recomputed = original.filter_and_recompute()

        # Check discrepancies
        discrepancies = []
        for measure in ["f1", "f1_macro"]:
            orig = original.get_value("run1", ALL_TOPIC_ID, measure)
            recomp = recomputed.get_value("run1", ALL_TOPIC_ID, measure)
            if orig is not None and recomp is not None:
                if abs(orig - recomp) > 0.001:
                    discrepancies.append((measure, orig, recomp))

        assert len(discrepancies) == 1
        assert discrepancies[0][0] == "f1"
        assert discrepancies[0][1] == pytest.approx(0.65)  # original (wrong)
        assert discrepancies[0][2] == pytest.approx(0.7)   # recomputed (correct)


class TestBuilderMerge:
    """Tests for merging EvalResults via add_from."""

    def test_add_from_merges_results(self):
        """add_from adds entries from an existing EvalResult."""
        specs = MeasureSpecs.from_single({"ndcg": "float"})

        # Build first result
        b1 = EvalResultBuilder(specs)
        b1.add("run1", "t1", "ndcg", 0.8)
        r1 = b1.build(compute_aggregates=True, verify=False, on_missing="ignore")

        # Build second result
        b2 = EvalResultBuilder(specs)
        b2.add("run2", "t1", "ndcg", 0.6)
        r2 = b2.build(compute_aggregates=True, verify=False, on_missing="ignore")

        # Merge via new builder
        merged_builder = EvalResultBuilder(specs)
        merged_builder.add_from(r1)
        merged_builder.add_from(r2)
        merged = merged_builder.build(compute_aggregates=True, verify=False, on_missing="ignore")

        # Check merged has both runs
        assert merged.run_ids == {"run1", "run2"}
        assert merged.get_value("run1", "t1", "ndcg") == pytest.approx(0.8)
        assert merged.get_value("run2", "t1", "ndcg") == pytest.approx(0.6)
