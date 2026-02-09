"""Extensive IO tests for EvalResults module."""

import tempfile
from pathlib import Path
from statistics import mean

import pytest

from autojudge_evaluate.eval_results import (
    EvalResultBuilder,
    MeasureSpecs,
    ALL_TOPIC_ID,
    load,
    write,
)
from autojudge_evaluate.eval_results.io import write_by_run


# =============================================================================
# Test fixtures - the larger dataset
# =============================================================================

TOPICS = ["topic1", "topic2", "topic4", "topic5"]
RUNS = ["apple", "orange", "rose", "plum"]
MEASURES = ["COUNT", "FRACT", "TOP_WORD"]

# Test data: run -> topic -> {measure: value}
# COUNT is int, FRACT is float, TOP_WORD is string
TEST_DATA = {
    "apple": {
        "topic1": {"COUNT": 5, "FRACT": 0.75, "TOP_WORD": "banana"},
        "topic2": {"COUNT": 12, "FRACT": 0.33, "TOP_WORD": "cherry"},
        "topic4": {"COUNT": 8, "FRACT": 0.91, "TOP_WORD": "date"},
        "topic5": {"COUNT": 3, "FRACT": 0.12, "TOP_WORD": "elderberry"},
    },
    "orange": {
        "topic1": {"COUNT": 7, "FRACT": 0.88, "TOP_WORD": "fig"},
        "topic2": {"COUNT": 2, "FRACT": 0.45, "TOP_WORD": "grape"},
        "topic4": {"COUNT": 15, "FRACT": 0.67, "TOP_WORD": "honeydew"},
        "topic5": {"COUNT": 9, "FRACT": 0.29, "TOP_WORD": "imbe"},
    },
    "rose": {
        "topic1": {"COUNT": 11, "FRACT": 0.52, "TOP_WORD": "jackfruit"},
        "topic2": {"COUNT": 4, "FRACT": 0.81, "TOP_WORD": "kiwi"},
        "topic4": {"COUNT": 6, "FRACT": 0.19, "TOP_WORD": "lemon"},
        "topic5": {"COUNT": 14, "FRACT": 0.73, "TOP_WORD": "mango"},
    },
    "plum": {
        "topic1": {"COUNT": 1, "FRACT": 0.64, "TOP_WORD": "nectarine"},
        "topic2": {"COUNT": 10, "FRACT": 0.37, "TOP_WORD": "olive"},
        "topic4": {"COUNT": 13, "FRACT": 0.95, "TOP_WORD": "papaya"},
        "topic5": {"COUNT": 0, "FRACT": 0.08, "TOP_WORD": "quince"},
    },
}


def build_test_dataset(include_manual_aggregates: bool = False) -> EvalResultBuilder:
    """Build the test dataset as an EvalResultBuilder."""
    # Define specs: per-topic has all measures, aggregate only has floats
    per_topic = {"COUNT": "float", "FRACT": "float", "TOP_WORD": "str"}

    if include_manual_aggregates:
        # Aggregate-only measures: COUNT_SUM, FRACT_macro
        aggregate = {"COUNT_SUM": "float", "FRACT_macro": "float"}
    else:
        # Will be computed from per_topic floats when build(compute_aggregates=True)
        aggregate = {}

    specs = MeasureSpecs(per_topic=per_topic, aggregate=aggregate)
    builder = EvalResultBuilder(specs)

    for run_id, topics in TEST_DATA.items():
        for topic_id, measures in topics.items():
            for measure, value in measures.items():
                builder.add(run_id, topic_id, measure, value)

        # Add manual aggregates if requested
        if include_manual_aggregates:
            # COUNT_SUM: sum of COUNT across topics
            count_sum = sum(topics[t]["COUNT"] for t in TOPICS)
            builder.add(run_id, ALL_TOPIC_ID, "COUNT_SUM", count_sum)

            # FRACT_macro: mean of FRACT across topics (same as auto-aggregate)
            fract_macro = mean(topics[t]["FRACT"] for t in TOPICS)
            builder.add(run_id, ALL_TOPIC_ID, "FRACT_macro", fract_macro)

    return builder


def get_expected_auto_aggregates() -> dict:
    """Calculate expected auto-aggregates (macro-average for float measures only)."""
    expected = {}
    for run_id, topics in TEST_DATA.items():
        expected[run_id] = {
            "COUNT": mean(topics[t]["COUNT"] for t in TOPICS),
            "FRACT": mean(topics[t]["FRACT"] for t in TOPICS),
            # TOP_WORD (str) is NOT aggregated - strings don't have aggregates
        }
    return expected


# =============================================================================
# Test: Basic round-trip for all formats
# =============================================================================


class TestFormatRoundTrips:
    """Test write/load round-trips for all supported formats."""

    @pytest.mark.parametrize("format", ["tot", "ir_measures", "jsonl"])
    def test_single_file_roundtrip(self, format):
        """Round-trip test for single file in each format."""
        builder = build_test_dataset()
        original = builder.build(compute_aggregates=True, verify=True, on_missing="error")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"result.{format}"
            write(original, path, format=format)
            loaded = load(path, format=format, drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        # Check run_ids and topic_ids
        assert loaded.run_ids == set(RUNS)
        assert loaded.topic_ids == set(TOPICS)

        # Check per-topic values
        for run_id, topics in TEST_DATA.items():
            for topic_id, measures in topics.items():
                assert loaded.get_value(run_id, topic_id, "COUNT") == pytest.approx(float(measures["COUNT"]))
                assert loaded.get_value(run_id, topic_id, "FRACT") == pytest.approx(measures["FRACT"])
                # TOP_WORD is string dtype, check it's preserved
                assert loaded.get_value(run_id, topic_id, "TOP_WORD") == measures["TOP_WORD"]

    def test_trec_eval_directory_roundtrip(self):
        """Round-trip test for trec_eval format (requires directory for multiple runs)."""
        builder = build_test_dataset()
        original = builder.build(compute_aggregates=True, verify=True, on_missing="error")

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "output"
            write_by_run(original, outdir, format="trec_eval")
            loaded = load(outdir, format="trec_eval", drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        assert loaded.run_ids == set(RUNS)
        assert loaded.topic_ids == set(TOPICS)

        # Check some values
        assert loaded.get_value("apple", "topic1", "COUNT") == pytest.approx(5.0)
        assert loaded.get_value("orange", "topic4", "FRACT") == pytest.approx(0.67)


class TestDirectoryLoading:
    """Test loading from directories with multiple files."""

    @pytest.mark.parametrize("format", ["tot", "ir_measures"])
    def test_load_directory_multiple_runs(self, format):
        """Load multiple run files from a directory."""
        builder = build_test_dataset()
        original = builder.build(compute_aggregates=True, verify=True, on_missing="error")

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "runs"
            write_by_run(original, outdir, format=format)
            loaded = load(outdir, format=format, drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        assert loaded.run_ids == set(RUNS)
        assert loaded.topic_ids == set(TOPICS)

    def test_trec_eval_run_id_from_filename(self):
        """trec_eval format gets run_id from filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write files with run names as filenames
            (tmppath / "apple").write_text("COUNT\ttopic1\t5\nFRACT\ttopic1\t0.75\n")
            (tmppath / "orange").write_text("COUNT\ttopic1\t7\nFRACT\ttopic1\t0.88\n")

            loaded = load(tmppath, format="trec_eval", drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        assert loaded.run_ids == {"apple", "orange"}
        assert loaded.get_value("apple", "topic1", "COUNT") == pytest.approx(5.0)
        assert loaded.get_value("orange", "topic1", "FRACT") == pytest.approx(0.88)


class TestAggregates:
    """Test aggregate computation and preservation."""

    def test_auto_aggregates_computed(self):
        """Auto-aggregates should be computed as macro-average for float measures only."""
        builder = build_test_dataset()
        result = builder.build(compute_aggregates=True, verify=True, on_missing="error")

        expected = get_expected_auto_aggregates()

        for run_id, expected_agg in expected.items():
            assert result.get_value(run_id, ALL_TOPIC_ID, "COUNT") == pytest.approx(expected_agg["COUNT"])
            assert result.get_value(run_id, ALL_TOPIC_ID, "FRACT") == pytest.approx(expected_agg["FRACT"])
            # TOP_WORD (str) should NOT have an aggregate
            assert result.get_value(run_id, ALL_TOPIC_ID, "TOP_WORD") is None

    def test_manual_aggregates_preserved_without_recompute(self):
        """Manual aggregates with custom measure names should be preserved when not recomputing."""
        builder = build_test_dataset(include_manual_aggregates=True)
        # Use compute_aggregates=False to preserve aggregate-only measures
        result = builder.build(compute_aggregates=False, verify=True, on_missing="error")

        # Check manual aggregates
        for run_id, topics in TEST_DATA.items():
            expected_sum = sum(topics[t]["COUNT"] for t in TOPICS)
            expected_macro = mean(topics[t]["FRACT"] for t in TOPICS)

            assert result.get_value(run_id, ALL_TOPIC_ID, "COUNT_SUM") == pytest.approx(expected_sum)
            assert result.get_value(run_id, ALL_TOPIC_ID, "FRACT_macro") == pytest.approx(expected_macro)

    def test_aggregate_only_measures_dropped_with_recompute(self):
        """Aggregate-only measures should be dropped when compute_aggregates=True."""
        builder = build_test_dataset(include_manual_aggregates=True)
        # With compute_aggregates=True, aggregate-only measures are dropped
        result = builder.build(compute_aggregates=True, verify=True, on_missing="error")

        # COUNT_SUM and FRACT_macro only have aggregate entries, so they get dropped
        for run_id in RUNS:
            assert result.get_value(run_id, ALL_TOPIC_ID, "COUNT_SUM") is None
            assert result.get_value(run_id, ALL_TOPIC_ID, "FRACT_macro") is None

        # But regular aggregates (for measures with per-topic entries) ARE computed
        expected = get_expected_auto_aggregates()
        for run_id in RUNS:
            assert result.get_value(run_id, ALL_TOPIC_ID, "COUNT") == pytest.approx(expected[run_id]["COUNT"])
            assert result.get_value(run_id, ALL_TOPIC_ID, "FRACT") == pytest.approx(expected[run_id]["FRACT"])

    def test_manual_aggregates_roundtrip(self):
        """Manual aggregates should survive write/load round-trip."""
        builder = build_test_dataset(include_manual_aggregates=True)
        # Use compute_aggregates=False to preserve aggregate-only measures
        original = builder.build(compute_aggregates=False, verify=True, on_missing="error")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.jsonl"
            write(original, path, format="jsonl")
            loaded = load(path, format="jsonl", drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        # Check manual aggregates survived
        for run_id in RUNS:
            orig_sum = original.get_value(run_id, ALL_TOPIC_ID, "COUNT_SUM")
            loaded_sum = loaded.get_value(run_id, ALL_TOPIC_ID, "COUNT_SUM")
            assert loaded_sum == pytest.approx(orig_sum)

    def test_drop_and_recompute_aggregates(self):
        """Loading with drop_aggregates=True should recompute fresh aggregates."""
        builder = build_test_dataset(include_manual_aggregates=True)
        original = builder.build(compute_aggregates=True, verify=True, on_missing="error")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.jsonl"
            write(original, path, format="jsonl")
            loaded = load(path, format="jsonl", drop_aggregates=True, recompute_aggregates=True, verify=False, on_missing="ignore")

        # Auto-aggregates should be recomputed
        expected = get_expected_auto_aggregates()
        for run_id in RUNS:
            assert loaded.get_value(run_id, ALL_TOPIC_ID, "COUNT") == pytest.approx(expected[run_id]["COUNT"])

        # Manual aggregates (COUNT_SUM, FRACT_macro) should NOT exist
        # because they were dropped and not recomputed
        assert loaded.get_value(run_id, ALL_TOPIC_ID, "COUNT_SUM") is None
        assert loaded.get_value(run_id, ALL_TOPIC_ID, "FRACT_macro") is None


class TestDtypeInference:
    """Test dtype inference for different value types."""

    def test_int_values_inferred_as_float(self):
        """Integer values should be inferred as float dtype."""
        builder = build_test_dataset()
        result = builder.build(compute_aggregates=False, verify=False, on_missing="ignore")

        assert result.measure_dtypes["COUNT"] == "float"

    def test_float_values_inferred_as_float(self):
        """Float values should be inferred as float dtype."""
        builder = build_test_dataset()
        result = builder.build(compute_aggregates=False, verify=False, on_missing="ignore")

        assert result.measure_dtypes["FRACT"] == "float"

    def test_string_values_inferred_as_str(self):
        """String values should be inferred as str dtype."""
        builder = build_test_dataset()
        result = builder.build(compute_aggregates=False, verify=False, on_missing="ignore")

        assert result.measure_dtypes["TOP_WORD"] == "str"

    def test_dtype_preserved_after_roundtrip(self):
        """Dtypes should be correctly inferred after round-trip."""
        builder = build_test_dataset()
        original = builder.build(compute_aggregates=False, verify=False, on_missing="ignore")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.jsonl"
            write(original, path, format="jsonl")
            loaded = load(path, format="jsonl", drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        assert loaded.measure_dtypes["COUNT"] == "float"
        assert loaded.measure_dtypes["FRACT"] == "float"
        assert loaded.measure_dtypes["TOP_WORD"] == "str"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_directory_raises_error(self):
        """Loading from empty directory should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No files found"):
                load(Path(tmpdir), format="tot", drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

    def test_zero_values_handled(self):
        """Zero values should be handled correctly."""
        specs = MeasureSpecs.from_single({"score": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "score", 0)
        builder.add("run1", "t2", "score", 0.0)
        builder.add("run1", "t3", "score", "0")

        result = builder.build(compute_aggregates=False, verify=False, on_missing="ignore")

        assert result.get_value("run1", "t1", "score") == pytest.approx(0.0)
        assert result.get_value("run1", "t2", "score") == pytest.approx(0.0)
        assert result.get_value("run1", "t3", "score") == pytest.approx(0.0)

    def test_negative_values_handled(self):
        """Negative values should be handled correctly."""
        specs = MeasureSpecs.from_single({"score": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "score", -0.5)
        builder.add("run1", "t2", "score", -10)

        result = builder.build(compute_aggregates=False, verify=False, on_missing="ignore")

        assert result.get_value("run1", "t1", "score") == pytest.approx(-0.5)
        assert result.get_value("run1", "t2", "score") == pytest.approx(-10.0)

    def test_whitespace_in_values_handled(self):
        """Values with whitespace should be trimmed."""
        specs = MeasureSpecs.from_single({"score": "float"})
        builder = EvalResultBuilder(specs)
        builder.add("run1", "t1", "score", " 0.5 ")

        result = builder.build(compute_aggregates=False, verify=False, on_missing="ignore")

        assert result.get_value("run1", "t1", "score") == pytest.approx(0.5)


class TestHeaderHandling:
    """Test loading files with headers."""

    def test_load_with_header_tot(self):
        """Load tot format file with header row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.txt"
            # Write file with header
            content = "run_id\tmeasure\ttopic_id\tvalue\n"
            content += "apple\tCOUNT\ttopic1\t5\n"
            content += "apple\tFRACT\ttopic1\t0.75\n"
            path.write_text(content)

            loaded = load(path, format="tot", has_header=True, drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        assert loaded.run_ids == {"apple"}
        assert loaded.get_value("apple", "topic1", "COUNT") == pytest.approx(5.0)

    def test_load_with_header_ir_measures(self):
        """Load ir_measures format file with header row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.txt"
            content = "run_id\ttopic_id\tmeasure\tvalue\n"
            content += "orange\ttopic2\tFRACT\t0.45\n"
            content += "orange\ttopic2\tCOUNT\t2\n"
            path.write_text(content)

            loaded = load(path, format="ir_measures", has_header=True, drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        assert loaded.run_ids == {"orange"}
        assert loaded.get_value("orange", "topic2", "FRACT") == pytest.approx(0.45)

    def test_load_without_header_when_present_causes_bad_data(self):
        """Loading with has_header=False when header exists should cause issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.txt"
            content = "run_id\tmeasure\ttopic_id\tvalue\n"
            content += "apple\tCOUNT\ttopic1\t5\n"
            path.write_text(content)

            # Load without skipping header - first row becomes data
            loaded = load(path, format="tot", has_header=False, drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        # The header row is parsed as data: run_id="run_id", measure="measure", etc.
        assert "run_id" in loaded.run_ids  # Header became a run_id

    def test_load_directory_with_header(self):
        """Load directory of files, each with header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write files with headers
            (tmppath / "run1.txt").write_text("measure\ttopic\tvalue\nCOUNT\tt1\t5\n")
            (tmppath / "run2.txt").write_text("measure\ttopic\tvalue\nCOUNT\tt1\t7\n")

            loaded = load(tmppath, format="trec_eval", has_header=True, drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        assert loaded.run_ids == {"run1.txt", "run2.txt"}


class TestAllFormatsComprehensive:
    """Comprehensive test across all formats with full dataset."""

    @pytest.mark.parametrize("format", ["tot", "ir_measures", "jsonl"])
    def test_roundtrip_with_computed_aggregates(self, format):
        """Round-trip with compute_aggregates=True: auto-aggregates exist, aggregate-only dropped."""
        builder = build_test_dataset(include_manual_aggregates=True)
        original = builder.build(compute_aggregates=True, verify=True, on_missing="error")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"full.{format}"
            write(original, path, format=format)
            loaded = load(path, format=format, drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        # Verify all per-topic entries
        for run_id, topics in TEST_DATA.items():
            for topic_id, measures in topics.items():
                assert loaded.get_value(run_id, topic_id, "COUNT") == pytest.approx(float(measures["COUNT"]))
                assert loaded.get_value(run_id, topic_id, "FRACT") == pytest.approx(measures["FRACT"])
                assert loaded.get_value(run_id, topic_id, "TOP_WORD") == measures["TOP_WORD"]

        # Auto-aggregates should exist
        expected = get_expected_auto_aggregates()
        for run_id in RUNS:
            assert loaded.get_value(run_id, ALL_TOPIC_ID, "COUNT") == pytest.approx(expected[run_id]["COUNT"])
            assert loaded.get_value(run_id, ALL_TOPIC_ID, "FRACT") == pytest.approx(expected[run_id]["FRACT"])

        # Aggregate-only measures (COUNT_SUM, FRACT_macro) should be dropped
        for run_id in RUNS:
            assert loaded.get_value(run_id, ALL_TOPIC_ID, "COUNT_SUM") is None
            assert loaded.get_value(run_id, ALL_TOPIC_ID, "FRACT_macro") is None

    @pytest.mark.parametrize("format", ["tot", "ir_measures", "jsonl"])
    def test_roundtrip_without_computed_aggregates(self, format):
        """Round-trip with compute_aggregates=False: manual aggregates preserved, no auto-aggregates."""
        builder = build_test_dataset(include_manual_aggregates=True)
        original = builder.build(compute_aggregates=False, verify=True, on_missing="error")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"full.{format}"
            write(original, path, format=format)
            loaded = load(path, format=format, drop_aggregates=False, recompute_aggregates=False, verify=False, on_missing="ignore")

        # Verify all per-topic entries
        for run_id, topics in TEST_DATA.items():
            for topic_id, measures in topics.items():
                assert loaded.get_value(run_id, topic_id, "COUNT") == pytest.approx(float(measures["COUNT"]))
                assert loaded.get_value(run_id, topic_id, "FRACT") == pytest.approx(measures["FRACT"])
                assert loaded.get_value(run_id, topic_id, "TOP_WORD") == measures["TOP_WORD"]

        # Manual aggregates should be preserved
        for run_id in RUNS:
            assert loaded.get_value(run_id, ALL_TOPIC_ID, "COUNT_SUM") is not None
            assert loaded.get_value(run_id, ALL_TOPIC_ID, "FRACT_macro") is not None

        # Auto-aggregates for COUNT/FRACT should NOT exist (not computed)
        for run_id in RUNS:
            assert loaded.get_value(run_id, ALL_TOPIC_ID, "COUNT") is None
            assert loaded.get_value(run_id, ALL_TOPIC_ID, "FRACT") is None
