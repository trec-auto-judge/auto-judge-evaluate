import click
import glob
from pathlib import Path
import pandas as pd
from typing import List, Set

from tira.io_utils import to_prototext

from autojudge_base.click_plus import (
    detect_header_interactive,
    LEADERBOARD_FORMATS,
    LEADERBOARD_FORMAT_HELP,
)
from autojudge_evaluate.evaluation import LeaderboardEvaluator, CorrelationMethodType
from autojudge_evaluate.eval_results import load as load_eval_result, EvalResult


def persist_output(df: pd.DataFrame, output: Path, out_format: str = "jsonl") -> None:
    # Use explicit format, or infer from extension
    if out_format == "jsonl" or output.name.endswith(".jsonl"):
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output, lines=True, orient="records")
    elif out_format == "table" or output.name.endswith(".txt"):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(df.to_string(index=False))
    elif output.name.endswith(".prototext"):
        ret = {k: v for k, v in df.iloc[0].to_dict().items()}
        ret = to_prototext([ret])
        output.write_text(ret)
    else:
        raise ValueError(f"Can not handle output file format {output}")

@click.option(
    "--truth-leaderboard",
    type=Path,
    required=True,
    help="The ground truth leaderboard file or directory.",
)
@click.option(
    "--truth-measure",
    type=str,
    multiple=True,
    help="Measure(s) from truth leaderboard to use. Repeatable. If omitted, uses all.",
)
@click.option(
    "--eval-measure",
    type=str,
    multiple=True,
    help="Measure(s) from eval leaderboard to use. Repeatable. If omitted, uses all.",
)
@click.option(
    "--truth-format",
    type=click.Choice(LEADERBOARD_FORMATS),
    required=True,
    help="Format of the ground truth leaderboard file:\n" + LEADERBOARD_FORMAT_HELP,
)
@click.option(
    "--eval-format",
    type=click.Choice(LEADERBOARD_FORMATS),
    required=True,
    help="Format of the input leaderboard file(s):\n" + LEADERBOARD_FORMAT_HELP,
)
@click.option(
    "--truth-header/--no-truth-header",
    default=False,
    help="Truth leaderboard has header row to skip.",
)
@click.option(
    "--eval-header/--no-eval-header",
    default=False,
    help="Eval leaderboard(s) have header row to skip.",
)
@click.option(
    "--truth-drop-aggregate/--no-truth-drop-aggregate",
    default=False,
    help="Drop pre-existing aggregate rows from truth and recompute from per-topic data.",
)
@click.option(
    "--eval-drop-aggregate/--no-eval-drop-aggregate",
    default=False,
    help="Drop pre-existing aggregate rows from eval and recompute from per-topic data.",
)
@click.option(
    "--on-missing",
    type=click.Choice(["error", "warn", "skip", "default"]),
    default="error",
    help="How to handle run_id mismatches between truth and eval leaderboards: \n"
         "error: raise an error \n"
         "warn: print warning, use common systems only \n"
         "skip: silently use common systems only \n"
         "default: use 0.0 for missing values",
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
    help="Output file path. Format determined by extension: .jsonl for JSON Lines, .prototext for Prototext.",
)
@click.option(
    "--aggregate",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help="Should only aggregates scores be reported.",
)
@click.option(
    "--correlation",
    type=CorrelationMethodType(),
    multiple=True,
    help="Correlation method(s) to compute (e.g., kendall, kendall@15). Repeatable. If omitted, computes all.",
)
@click.option(
    "--topic",
    "topic_id",
    type=str,
    multiple=True,
    help="Topic ID(s) to use for evaluation. Repeatable. If omitted, uses truth's topics.",
)
@click.option(
    "--topic-ids-from-eval",
    is_flag=True,
    default=False,
    help="Derive topic IDs from the union of topics in eval leaderboards (ignores --topic).",
)
@click.option(
    "--only-shared-topics/--all-topics",
    help="Filter to topics present in both truth and eval (recomputes aggregates). "
         "Default (--all-topics) uses provided aggregates without topic filtering.",
)
@click.option(
    "--run",
    "run_id",
    type=str,
    multiple=True,
    help="Run ID(s) to include in evaluation. Repeatable. If omitted, uses all runs.",
)
@click.option(
    "--only-shared-runs/--all-runs",
    default=False,
    help="Filter to runs present in both truth and eval (no recompute). "
         "Default (--all-runs) includes all runs, handling mismatches via --on-missing.",
)
@click.option(
    "--out-format",
    type=click.Choice(["table", "jsonl"]),
    default="jsonl",
    help="Output file format: jsonl (default) or table. Only affects --output file.",
)
@click.option(
    "--silent",
    is_flag=True,
    default=False,
    help="Suppress table output to stdout.",
)
@click.argument("input_files", nargs=-1, type=str)
def meta_evaluate(
    truth_leaderboard: Path,
    truth_measure: tuple,
    eval_measure: tuple,
    truth_format: str,
    truth_header: bool,
    eval_format: str,
    eval_header: bool,
    truth_drop_aggregate: bool,
    eval_drop_aggregate: bool,
    on_missing: str,
    input: tuple,
    output: Path,
    aggregate: bool,
    correlation: tuple,
    topic_id: tuple,
    topic_ids_from_eval: bool,
    only_shared_topics: bool,
    run_id: tuple,
    only_shared_runs: bool,
    out_format: str,
    silent: bool,
    input_files: tuple,
) -> int:
    """Compute correlation between predicted leaderboards and ground-truth leaderboard."""
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

    # Detect headers interactively if not explicitly specified
    truth_has_header = detect_header_interactive(
        truth_leaderboard, truth_format, truth_header, "truth"
    )

    # For eval files, check the first one and apply to all
    eval_has_header = eval_header
    if all_inputs and not eval_header:
        eval_has_header = detect_header_interactive(
            all_inputs[0], eval_format, eval_header, "eval"
        )

    # Convert tuples to lists/sets (empty tuple means "all" / None)
    truth_measures = list(truth_measure) if truth_measure else None
    eval_measures = list(eval_measure) if eval_measure else None
    correlation_methods = list(correlation) if correlation else None

    # Determine topic IDs to use
    # Load truth first to get its topics (needed for --all-topics and --only-shared-topics)
    truth_result_for_topics = load_eval_result(
        truth_leaderboard,
        format=truth_format,
        has_header=truth_has_header,
        drop_aggregates=truth_drop_aggregate,
        recompute_aggregates=False,
        verify=False,
        on_missing="ignore",
    )
    truth_topic_ids = set(truth_result_for_topics.topic_ids)

    if topic_id:
        # Explicit --topic-id flags
        topic_ids_set = set(topic_id)
    elif only_shared_topics or topic_ids_from_eval:
        # Need to load eval files to get their topics
        eval_topics_union: Set[str] = set()
        for eval_path in all_inputs:
            er = load_eval_result(
                eval_path,
                format=eval_format,
                has_header=eval_has_header,
                drop_aggregates=eval_drop_aggregate,
                recompute_aggregates=False,
                verify=False,
                on_missing="ignore",
            )
            eval_topics_union.update(er.topic_ids)

        if only_shared_topics:
            # Intersection of truth and eval topics
            topic_ids_set = truth_topic_ids & eval_topics_union
            click.echo(f"Using {len(topic_ids_set)} shared topics (intersection of truth and eval)", err=True)
        else:
            # topic_ids_from_eval: use union of eval topics
            topic_ids_set = eval_topics_union
            click.echo(f"Derived {len(topic_ids_set)} topic IDs from eval results", err=True)
    else:
        # --all-topics (default): use original aggregates without topic filtering
        # Pass None to signal preserve mode (no filtering, no recomputation)
        topic_ids_set = None

    # Convert run_id tuple to set (or None if empty)
    run_ids_set = set(run_id) if run_id else None

    te = LeaderboardEvaluator(
        truth_leaderboard,
        truth_measures=truth_measures,
        eval_measures=eval_measures,
        truth_format=truth_format,
        truth_has_header=truth_has_header,
        truth_drop_aggregate=truth_drop_aggregate,
        eval_format=eval_format,
        eval_has_header=eval_has_header,
        eval_drop_aggregate=eval_drop_aggregate,
        on_missing=on_missing,
        correlation_methods=correlation_methods,
        topic_ids=topic_ids_set,
        run_ids=run_ids_set,
        only_shared_runs=only_shared_runs,
    )

    # Print diagnostic info
    truth_runs = len(te.truth_result.run_ids)
    truth_topics = len(te.truth_result.topic_ids)
    topic_info = f"{len(topic_ids_set)} topic(s)" if topic_ids_set else "all topics (no filtering)"
    click.echo(
        f"Truth leaderboard: {truth_runs} run(s), {truth_topics} topic(s). "
        f"Using {topic_info} for evaluation.",
        err=True
    )

    # Collect eval leaderboard stats (union across all input files)
    eval_run_ids: Set[str] = set()
    eval_topic_ids: Set[str] = set()
    for eval_path in all_inputs:
        er = load_eval_result(
            eval_path,
            format=eval_format,
            has_header=eval_has_header,
            drop_aggregates=eval_drop_aggregate,
            recompute_aggregates=False,
            verify=False,
            on_missing="ignore",
        )
        eval_run_ids.update(er.run_ids)
        eval_topic_ids.update(er.topic_ids)
    click.echo(
        f"Eval leaderboards: {len(all_inputs)} file(s), {len(eval_run_ids)} run(s), {len(eval_topic_ids)} topic(s).",
        err=True
    )

    # Report topic overlap
    truth_topic_set = set(te.truth_result.topic_ids)
    common_topics = truth_topic_set & eval_topic_ids
    truth_only = truth_topic_set - eval_topic_ids
    eval_only = eval_topic_ids - truth_topic_set
    click.echo(
        f"Topic overlap: {len(common_topics)} common, {len(truth_only)} truth-only, {len(eval_only)} eval-only.",
        err=True
    )

    # Report run overlap
    truth_run_set = set(te.truth_result.run_ids)
    common_runs = truth_run_set & eval_run_ids
    truth_only_runs = truth_run_set - eval_run_ids
    eval_only_runs = eval_run_ids - truth_run_set
    click.echo(
        f"Run overlap: {len(common_runs)} common, {len(truth_only_runs)} truth-only, {len(eval_only_runs)} eval-only.",
        err=True
    )

    # Report run filtering
    run_info_parts = []
    if run_ids_set:
        run_info_parts.append(f"explicit: {len(run_ids_set)}")
    if only_shared_runs:
        run_info_parts.append("shared-only")
    if run_info_parts:
        click.echo(f"Run filtering: {', '.join(run_info_parts)}", err=True)

    # Note about @k correlation methods
    if correlation_methods:
        top_k_methods = [m for m in correlation_methods if "@" in m]
        if top_k_methods:
            click.echo(
                f"Note: correlation@k methods ({top_k_methods}) filter to common runs, then select top k.",
                err=True
            )

    df = []

    for c in all_inputs:
        result = te.evaluate(c)

        for (truth_m, eval_m), metrics in result.items():
            tmp = {
                "Judge": c.name.replace(".txt", ""),
                "TruthMeasure": truth_m,
                "EvalMeasure": eval_m,
            }
            for k, v in metrics.items():
                tmp[k] = v
            df.append(tmp)

    df = pd.DataFrame(df)

    if aggregate:
        df_aggr = {"Judges": len(df)}
        for k in df.columns:
            if k in ("Judge", "TruthMeasure", "EvalMeasure"):
                continue
            df_aggr[k] = df[k].mean()
        df = pd.DataFrame([df_aggr])

    if not silent:
        print(df.to_string(index=False))

    if output:
        persist_output(df, output, out_format)

    return 0