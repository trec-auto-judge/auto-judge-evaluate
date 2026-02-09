"""Leaderboard statistics command - compute statistics from leaderboard files."""

from math import sqrt
import click
import glob
from pathlib import Path
import pandas as pd
from statistics import mean, stdev
from typing import List, Optional

from autojudge_base.click_plus import (
    detect_header_interactive,
    LEADERBOARD_FORMATS,
    LEADERBOARD_FORMAT_HELP,
)
from autojudge_evaluate.eval_results import load as load_eval_result, EvalResult, ALL_TOPIC_ID


@click.option(
    "--eval-format",
    type=click.Choice(LEADERBOARD_FORMATS),
    required=True,
    help="Format of the input leaderboard file(s):\n" + LEADERBOARD_FORMAT_HELP,
)
@click.option(
    "--eval-header/--no-eval-header",
    default=False,
    help="Input file(s) have header row to skip.",
)
@click.option(
    "--eval-measure",
    type=str,
    multiple=True,
    help="Measure(s) to include. Repeatable. If omitted, uses all.",
)
@click.option(
    "--input", "-i",
    type=str,
    multiple=True,
    help="Input leaderboard file(s) or glob pattern (e.g., --input '*.txt').",
)
@click.option(
    "--output",
    type=Path,
    required=False,
    help="Output file path. Format determined by extension: .jsonl for JSON Lines, .csv for CSV.",
)
@click.option(
    "--sort/--no-sort",
    default=False,
    help="Sort run_ids by mean (descending) within each judge and measure.",
)
@click.argument("input_files", nargs=-1, type=str)
def leaderboard(
    eval_format: str,
    eval_header: bool,
    eval_measure: tuple,
    input: tuple,
    output: Optional[Path],
    sort: bool,
    input_files: tuple,
) -> int:
    """Compute statistics from leaderboard files."""
    # Combine --input options and positional arguments, expand globs
    all_inputs: List[Path] = []
    for pattern in list(input) + list(input_files):
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            all_inputs.extend(Path(m) for m in matches)
        else:
            # No glob match - treat as literal path
            all_inputs.append(Path(pattern))

    if not all_inputs:
        raise click.ClickException("No input files specified. Use --input or positional arguments.")

    # Detect header interactively if not explicitly specified
    has_header = eval_header
    if all_inputs and not eval_header:
        has_header = detect_header_interactive(
            all_inputs[0], eval_format, eval_header, "eval"
        )

    # Filter measures if specified
    measure_filter = set(eval_measure) if eval_measure else None

    df_rows = []

    for input_path in all_inputs:
        er = load_eval_result(
            input_path,
            format=eval_format,
            has_header=has_header,
            drop_aggregates=False,
            recompute_aggregates=False,
            verify=False,
            on_missing="ignore",
        )

        # Get measures to process
        measures = sorted(er.measures)
        if measure_filter:
            measures = [m for m in measures if m in measure_filter]

        # Get unique run_ids
        run_ids = sorted(er.run_ids)

        # Group values by (run_id, measure), compute mean across topics
        # run_id is the IR system being judged, topic_id varies
        for run_id in run_ids:
            for m in measures:
                # Collect values across topics for this (run_id, measure)
                values = [
                    float(e.value)
                    for e in er.entries
                    if e.run_id == run_id
                    and e.measure == m
                    and e.topic_id != ALL_TOPIC_ID
                    and isinstance(e.value, (int, float))
                ]

                if not values:
                    continue

                row = {
                    "Judge": input_path.name.replace(".txt", ""),
                    "RunID": run_id,
                    "Measure": m,
                    "Topics": len(values),
                    "Mean": mean(values),
                    "Stderr": stdev(values)/sqrt(len(values)) if len(values) > 1 else 0.0,
                    "Stdev": stdev(values) if len(values) > 1 else 0.0,
                    "Min": min(values),
                    "Max": max(values),
                }
                df_rows.append(row)

    df = pd.DataFrame(df_rows)

    if sort:
        df = df.sort_values(
            by=["Judge", "Measure", "Mean"],
            ascending=[True, True, False]
        )

    print(df.to_string(index=False))

    if output:
        if output.name.endswith(".jsonl"):
            df.to_json(output, lines=True, orient="records")
        elif output.name.endswith(".csv"):
            df.to_csv(output, index=False)
        else:
            raise ValueError(f"Unknown output format: {output}")

    return 0