"""CLI command for evaluating predicted qrels against truth qrels."""

from pathlib import Path
from typing import Optional

import click
import pandas as pd

from autojudge_base.qrels import Qrels


def _load_qrels(nugget_docs: Optional[Path], qrels_file: Optional[Path], label: str) -> Qrels:
    """Load Qrels from either a nugget-docs directory or a qrels file.

    Exactly one of the two must be provided.
    """
    if nugget_docs and qrels_file:
        raise click.UsageError(f"Specify only one of --{label}-nugget-docs or --{label}-qrels, not both.")
    if not nugget_docs and not qrels_file:
        raise click.UsageError(f"Specify either --{label}-nugget-docs or --{label}-qrels.")

    if nugget_docs:
        from autojudge_evaluate.nugget_doc_eval import (
            read_nugget_docs_collaborator,
            nugget_docs_to_qrels,
        )
        topics = read_nugget_docs_collaborator(nugget_docs)
        return nugget_docs_to_qrels(topics)
    else:
        from autojudge_base.qrels.qrels import read_qrel_file
        return read_qrel_file(qrels_file)


@click.command()
@click.option("--truth-nugget-docs", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Truth nugget-docs directory (collaborator format)")
@click.option("--truth-qrels", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Truth qrels file (TREC format)")
@click.option("--predict-nugget-docs", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Predicted nugget-docs directory (collaborator format)")
@click.option("--predict-qrels", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Predicted qrels file (TREC format)")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Output file (.txt or .jsonl)")
@click.option(
    "--on-missing",
    type=click.Choice(["error", "warn", "default", "skip"]),
    default="default",
    help="How to handle topics present in only one side: "
         "error=raise, warn=warn+skip, default=treat as empty set, skip=silently skip",
)
@click.option("--truth-max-grade", type=int, default=1,
              help="Grade scale upper bound for truth qrels (e.g. 3 for 0-3). Grades are clipped.")
@click.option("--predict-max-grade", type=int, default=1,
              help="Grade scale upper bound for predicted qrels. Grades are clipped.")
@click.option("--truth-relevance-threshold", type=int, default=1,
              help="Threshold for truth side binary metrics (grade >= threshold = relevant).")
@click.option("--predict-relevance-threshold", type=int, default=1,
              help="Threshold for predicted side binary metrics (grade >= threshold = relevant).")
@click.option("--default-grade", type=int, default=0,
              help="Grade assigned to doc_ids present in only one side.")
def qrel_evaluate(
    truth_nugget_docs: Optional[Path],
    truth_qrels: Optional[Path],
    predict_nugget_docs: Optional[Path],
    predict_qrels: Optional[Path],
    output: Optional[Path],
    on_missing: str,
    truth_max_grade: int,
    predict_max_grade: int,
    truth_relevance_threshold: int,
    predict_relevance_threshold: int,
    default_grade: int,
):
    """Evaluate predicted qrels against truth qrels.

    For each side (truth and predicted), provide either a nugget-docs directory
    or a qrels file:

    \b
      --truth-nugget-docs DIR   or  --truth-qrels FILE
      --predict-nugget-docs DIR or  --predict-qrels FILE

    Computes per-topic set overlap (precision, recall, F1) and inter-annotator
    agreement metrics (Jaccard, Cohen's Kappa, Krippendorff's Alpha, ARI).
    """
    from autojudge_evaluate.nugget_doc_eval import set_overlap, grade_agreement

    truth = _load_qrels(truth_nugget_docs, truth_qrels, "truth")
    predicted = _load_qrels(predict_nugget_docs, predict_qrels, "predict")

    overlap_results = set_overlap(
        truth, predicted, on_missing=on_missing,
        truth_relevance_threshold=truth_relevance_threshold,
        predict_relevance_threshold=predict_relevance_threshold,
    )
    agreement_results = grade_agreement(
        truth, predicted,
        on_missing=on_missing,
        default_grade=default_grade,
        truth_max_grade=truth_max_grade,
        predict_max_grade=predict_max_grade,
        truth_relevance_threshold=truth_relevance_threshold,
        predict_relevance_threshold=predict_relevance_threshold,
    )

    if not overlap_results and not agreement_results:
        click.echo("No overlapping topics found.")
        return

    # Build set-overlap rows keyed by topic_id
    overlap_by_topic = {r.topic_id: r for r in overlap_results}
    agreement_by_topic = {r.topic_id: r for r in agreement_results}
    all_topic_ids = sorted(set(overlap_by_topic.keys()) | set(agreement_by_topic.keys()))

    # Determine kappa threshold columns (cross-product of truth x predicted thresholds)
    kappa_keys = [(tt, pt) for tt in range(1, truth_max_grade + 1) for pt in range(1, predict_max_grade + 1)]
    # Column naming: single column when 1x1, otherwise "Kappa@{tt}v{pt}"
    is_single_kappa = (truth_max_grade == 1 and predict_max_grade == 1)

    def _kappa_col(tt: int, pt: int) -> str:
        if is_single_kappa:
            return "Kappa@1"
        return f"Kappa@{tt}v{pt}"

    rows = []
    for topic_id in all_topic_ids:
        row = {"Topic": topic_id}

        # Set overlap columns
        ov = overlap_by_topic.get(topic_id)
        if ov:
            row["Precision"] = round(ov.precision, 4)
            row["Recall"] = round(ov.recall, 4)
            row["F1"] = round(ov.f1, 4)
            row["TP"] = ov.tp
            row["FP"] = ov.fp
            row["FN"] = ov.fn

        # Agreement columns
        ag = agreement_by_topic.get(topic_id)
        if ag:
            row["Jaccard"] = round(ag.jaccard, 4) if ag.jaccard == ag.jaccard else ag.jaccard
            for tt, pt in kappa_keys:
                val = ag.kappa_binary.get((tt, pt), float("nan"))
                row[_kappa_col(tt, pt)] = round(val, 4) if val == val else val
            row["Kappa_w"] = round(ag.kappa_weighted, 4) if ag.kappa_weighted == ag.kappa_weighted else ag.kappa_weighted
            row["Kripp_a"] = round(ag.kripp_alpha, 4) if ag.kripp_alpha == ag.kripp_alpha else ag.kripp_alpha
            row["ARI"] = round(ag.ari, 4) if ag.ari == ag.ari else ag.ari

        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute MEAN row (numeric columns only)
    mean_row = {"Topic": "MEAN"}
    for col in df.columns:
        if col == "Topic":
            continue
        if col in ("TP", "FP", "FN"):
            mean_row[col] = ""
        elif pd.api.types.is_numeric_dtype(df[col]):
            mean_row[col] = round(df[col].mean(), 4)
        else:
            mean_row[col] = ""
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    click.echo(df.to_string(index=False))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.name.endswith(".jsonl"):
            df.to_json(output, lines=True, orient="records")
        else:
            output.write_text(df.to_string(index=False))
