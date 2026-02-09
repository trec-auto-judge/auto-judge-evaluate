"""
B. IO - Load and write EvalResult in various formats.

Supported formats:
- trec_eval: measure topic_id value (3 cols, run_id from filename)
- tot: run_id measure topic_id value (4 cols)
- ir_measures: run_id topic_id measure value (4 cols)
- ranking: topic_id Q0 run_id rank value measure (6 cols, TREC ranking format)
- jsonl: JSON lines with run_id, topic_id, measure, value fields

Handles both single files and directories (one file per run).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal, List

from .eval_result import EvalResult, EvalEntry, MeasureSpecs, ALL_TOPIC_ID
from .builder import EvalResultBuilder

Format = Literal["trec_eval", "tot", "ir_measures", "ranking", "jsonl"]


def load(
    path: Path,
    format: Format,
    has_header: bool = False,
    drop_aggregates: bool = False,
    *,
    recompute_aggregates: bool,
    verify: bool,
    on_missing: Literal["error", "warn", "ignore"],
) -> EvalResult:
    """
    Load EvalResult from file or directory.

    Args:
        path: Path to file or directory of files
        format: File format (trec_eval, tot, ir_measures, jsonl)
        has_header: If True, skip first line of each file
        drop_aggregates: If True, discard existing aggregate rows
        recompute_aggregates: If True, compute fresh aggregates in build()

    Returns:
        EvalResult with entries loaded and optionally aggregated.

    If path is a directory, all files are loaded and merged.
    For trec_eval format, each filename becomes the run_id.
    """
    if path.is_dir():
        return _load_directory(path, format, has_header, drop_aggregates, recompute_aggregates, verify, on_missing)
    else:
        return _load_file(path, format, has_header, drop_aggregates, recompute_aggregates, verify, on_missing)


def _load_directory(
    directory: Path,
    format: Format,
    has_header: bool,
    drop_aggregates: bool,
    recompute_aggregates: bool,
    verify: bool,
    on_missing: Literal["error", "warn", "ignore"],
) -> EvalResult:
    """Load and merge all files from a directory."""
    files = sorted([
        f for f in directory.iterdir()
        if f.is_file() and not f.name.startswith(".")
    ])
    if not files:
        raise ValueError(f"No files found in directory: {directory}")

    print(f"Loading {len(files)} files from {directory}", file=sys.stderr)

    # Parse all entries first
    all_entries: List[EvalEntry] = []
    for f in files:
        entries = _parse_file(f, format, has_header)
        for e in entries:
            if drop_aggregates and e.topic_id == ALL_TOPIC_ID:
                continue
            all_entries.append(e)

    # Infer specs from parsed entries
    specs = MeasureSpecs.infer(all_entries)

    # Build with inferred specs
    builder = EvalResultBuilder(specs)
    for e in all_entries:
        builder.add(e.run_id, e.topic_id, e.measure, e.value)

    return builder.build(
        compute_aggregates=recompute_aggregates,
        verify=verify,
        on_missing=on_missing,
    )


def _load_file(
    path: Path,
    format: Format,
    has_header: bool,
    drop_aggregates: bool,
    recompute_aggregates: bool,
    verify: bool,
    on_missing: Literal["error", "warn", "ignore"],
) -> EvalResult:
    """Load a single file."""
    entries = _parse_file(path, format, has_header)

    # Filter if needed
    if drop_aggregates:
        entries = [e for e in entries if e.topic_id != ALL_TOPIC_ID]

    # Infer specs from entries
    specs = MeasureSpecs.infer(entries)

    # Build with inferred specs
    builder = EvalResultBuilder(specs)
    for e in entries:
        builder.add(e.run_id, e.topic_id, e.measure, e.value)

    return builder.build(
        compute_aggregates=recompute_aggregates,
        verify=verify,
        on_missing=on_missing,
    )


def _parse_file(path: Path, format: Format, has_header: bool) -> list[EvalEntry]:
    """
    Parse file into raw entries (no casting, no aggregation).

    Returns list of EvalEntry with string values (casting happens in Builder).
    """
    text = path.read_text(encoding="utf-8")
    lines = text.strip().split("\n")

    if has_header and lines:
        lines = lines[1:]

    entries: list[EvalEntry] = []

    for line in lines:
        if not line.strip():
            continue

        entry = _parse_line(line, format, path.name)
        if entry:
            entries.append(entry)

    return entries


def _parse_line(line: str, format: Format, filename: str) -> EvalEntry | None:
    """Parse a single line into an EvalEntry."""
    if format == "jsonl":
        return _parse_jsonl_line(line)

    parts = line.split()

    if format == "trec_eval":
        if len(parts) != 3:
            raise ValueError(
                f"trec_eval format expects 3 columns, got {len(parts)}: {line!r}"
            )
        measure, topic_id, value = parts
        run_id = filename  # run_id from filename

    elif format == "tot":
        if len(parts) != 4:
            raise ValueError(
                f"tot format expects 4 columns, got {len(parts)}: {line!r}"
            )
        run_id, measure, topic_id, value = parts

    elif format == "ir_measures":
        if len(parts) != 4:
            raise ValueError(
                f"ir_measures format expects 4 columns, got {len(parts)}: {line!r}"
            )
        run_id, topic_id, measure, value = parts

    elif format == "ranking":
        if len(parts) != 6:
            raise ValueError(
                f"ranking format expects 6 columns, got {len(parts)}: {line!r}"
            )
        topic_id, _q0, run_id, _rank, value, measure = parts

    else:
        raise ValueError(f"Unknown format: {format!r}")

    return EvalEntry(run_id=run_id, topic_id=topic_id, measure=measure, value=value)


def _parse_jsonl_line(line: str) -> EvalEntry | None:
    """Parse a JSONL line."""
    obj = json.loads(line)
    return EvalEntry(
        run_id=obj["run_id"],
        topic_id=obj["topic_id"],
        measure=obj["measure"],
        value=obj["value"],
    )


# =============================================================================
# Write
# =============================================================================


def write(result: EvalResult, path: Path, format: Format) -> None:
    """
    Write EvalResult to file.

    Args:
        result: EvalResult to write
        path: Output file path
        format: Output format (trec_eval, tot, ir_measures, jsonl)

    Note: trec_eval format writes all entries with run_id in the data,
    not using filename. Use write_by_run() for one-file-per-run output.
    """
    lines = []

    for e in result.entries:
        line = _format_entry(e, format)
        lines.append(line)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {path}")


def write_by_run(result: EvalResult, directory: Path, format: Format) -> None:
    """
    Write EvalResult to directory with one file per run_id.

    Useful for trec_eval format where run_id comes from filename.
    """
    directory.mkdir(parents=True, exist_ok=True)

    for run_id in result.run_ids:
        entries = result.entries_by_run(run_id)
        lines = [_format_entry(e, format) for e in entries]

        out_path = directory / run_id
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(result.run_ids)} files to {directory}")


def _format_entry(entry: EvalEntry, format: Format) -> str:
    """Format a single entry as a line."""
    value_str = str(entry.value)

    if format == "trec_eval":
        # Note: run_id not included in trec_eval format
        return f"{entry.measure}\t{entry.topic_id}\t{value_str}"

    elif format == "tot":
        return f"{entry.run_id}\t{entry.measure}\t{entry.topic_id}\t{value_str}"

    elif format == "ir_measures":
        return f"{entry.run_id}\t{entry.topic_id}\t{entry.measure}\t{value_str}"

    elif format == "ranking":
        # rank is always 0 since EvalResult doesn't track ranking
        return f"{entry.topic_id}\tQ0\t{entry.run_id}\t0\t{value_str}\t{entry.measure}"

    elif format == "jsonl":
        obj = {
            "run_id": entry.run_id,
            "topic_id": entry.topic_id,
            "measure": entry.measure,
            "value": entry.value,
        }
        return json.dumps(obj)

    else:
        raise ValueError(f"Unknown format: {format!r}")
