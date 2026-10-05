import click
import glob
from pathlib import Path
import pandas as pd
from typing import List, Set, Tuple

from tira.io_utils import to_prototext

from autojudge_base.click_plus import (
    detect_header_interactive,
    LEADERBOARD_FORMATS,
    LEADERBOARD_FORMAT_HELP,
)
from autojudge_base.leaderboard import check_format_mismatch
from autojudge_evaluate.evaluation import EvalResultFormat, LeaderboardEvaluator, CorrelationMethodType, OnMissing
from autojudge_evaluate.eval_results import load as load_eval_result, load_qrels, EvalResult, QRELS_MISSING_CHOICES


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
    default=None,
    help="Format of the input leaderboard file(s) (required when leaderboard files are given):\n" + LEADERBOARD_FORMAT_HELP,
)
@click.option(
    "--eval-qrels",
    type=str,
    multiple=True,
    help="Judge qrels file or glob ('topic 0 run_id grade', or 'all 0 run_id grade' for "
         "whole-system grades), evaluated as a leaderboard next to the leaderboard inputs. Repeatable.",
)
@click.option(
    "--qrels-missing",
    type=click.Choice(QRELS_MISSING_CHOICES),
    default="graded-only",
    help="For --eval-qrels: how a (run, topic) the judge did not grade is treated. "
         "graded-only (default): left out, a run's mean is over its graded topics only. "
         "zero: counted as grade 0, a run's mean is over all topics in the qrels file.",
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
    default="default",
    help="How to handle run_id mismatches between truth and eval leaderboards: \n"
         "default (the default): use 0.0 for missing values, keeping all runs \n"
         "error: raise an error \n"
         "warn: print warning, use common systems only \n"
         "skip: silently use common systems only",
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
    help="Correlation method(s) to compute (e.g., kendall, kendall@15), or coverage (fraction of truth "
         "runs the judge scored). Repeatable. If omitted, computes all.",
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
@click.option(
    "--diagnostics-dir",
    type=Path,
    default=None,
    help="If set, dump per-(truth_measure, eval_measure, method) ranking JSONL files "
         "under this directory for debugging correlation results. "
         "Layout: <dir>/<eval_label>/<truth_m>__<eval_m>/<method>.jsonl",
)
@click.argument("input_files", nargs=-1, type=str)
def meta_evaluate(
    truth_leaderboard: Path,
    truth_measure: tuple,
    eval_measure: tuple,
    truth_format: EvalResultFormat,
    truth_header: bool,
    eval_format: EvalResultFormat | None,
    eval_qrels: tuple,
    qrels_missing: str,
    eval_header: bool,
    truth_drop_aggregate: bool,
    eval_drop_aggregate: bool,
    on_missing: OnMissing,
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
    diagnostics_dir: Path | None,
    input_files: tuple,
) -> int:
    """Compute correlation between predicted leaderboards and ground-truth leaderboard."""
    def expand(patterns) -> List[Path]:
        """Expand globs; a pattern without matches is taken as a literal path."""
        paths: List[Path] = []
        for pattern in patterns:
            matches = sorted(glob.glob(pattern, recursive=True))
            if matches:
                paths.extend(Path(m) for m in matches)
            else:
                paths.append(Path(pattern))
        return paths

    # Leaderboards come from --input and positional arguments, judge qrels from
    # --eval-qrels; each input is (path, is_qrels).
    leaderboard_inputs = expand(list(input) + list(input_files))
    all_inputs: List[Tuple[Path, bool]] = (
        [(p, False) for p in leaderboard_inputs] + [(p, True) for p in expand(eval_qrels)]
    )

    if not all_inputs:
        raise click.ClickException("No input files specified. Use --input, positional arguments or --eval-qrels.")
    if leaderboard_inputs and eval_format is None:
        raise click.UsageError("Specify --eval-format for the leaderboard inputs (qrels go via --eval-qrels).")

    def load_eval_input(path: Path, is_qrels: bool) -> EvalResult:
        """Load one eval input without filtering (for topic/run statistics)."""
        if is_qrels:
            return load_qrels(path, missing=qrels_missing)
        return load_eval_result(
            path,
            format=eval_format,
            has_header=eval_has_header,
            drop_aggregates=eval_drop_aggregate,
            recompute_aggregates=False,
            verify=False,
            on_missing="ignore",
        )

    # Detect headers interactively if not explicitly specified
    truth_has_header = detect_header_interactive(
        truth_leaderboard, truth_format, truth_header, "truth"
    )

    # For eval files, check the first one and apply to all
    eval_has_header = eval_header
    if leaderboard_inputs and not eval_header:
        eval_has_header = detect_header_interactive(
            leaderboard_inputs[0], eval_format, eval_header, "eval"
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
        for eval_path, is_qrels in all_inputs:
            er = load_eval_input(eval_path, is_qrels)
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
        diagnostics_dir=diagnostics_dir,
        qrels_missing=qrels_missing,
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
    for eval_path, is_qrels in all_inputs:
        er = load_eval_input(eval_path, is_qrels)
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

    # Pre-hoc format check: warn if eval files appear to be wrong format
    for eval_path in leaderboard_inputs:
        warning = check_format_mismatch(
            eval_path,
            specified_format=eval_format,
            known_topics=truth_topic_ids,
            has_header=eval_has_header,
        )
        if warning:
            click.echo(warning, err=True)

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

    for c, is_qrels in all_inputs:
        result = te.evaluate(c, is_qrels=is_qrels)

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