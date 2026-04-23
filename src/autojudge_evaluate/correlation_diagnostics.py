"""Diagnostics for `LeaderboardEvaluator.evaluate()`.

Per (truth_m, eval_m, method) triple:
  - `diagnose_correlation()` classifies the case (issue or None) and, when
    `diagnostics_dir` is provided, also writes one JSONL file (one row per
    run, self-contained, tagged with the denormalized issue).
  - `evaluate()` collects the returned issues across all triples and emits
    one end-of-run summary via `format_issues_warning()`.

Layout when dumping:
    <diagnostics_dir>/<eval_label>/<truth_m>__<eval_m>/<method>.jsonl
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel


CorrelationIssue = Literal[
    "measure_defaulted",
    "too_few",
    "all_tied",
    "length_mismatch",
]


class RankingExtractionDiagnostic(BaseModel):
    """Per-side diagnostic from `LeaderboardEvaluator.extract_ranking`.

    Starts minimal. Room to grow toward the vocabulary in
    `eval_results/verification.py` (e.g. `check_aggregates_match_specs`):
      - `measure_missing: bool`    — not in eval_result.measures at all
      - `aggregate_missing: bool`  — per-topic rows exist, no aggregate row
      - `runs_with_measure: list[str]` — richer enrichment
    """

    defaulted: bool = False


class CorrelationDiagnosticRow(BaseModel):
    """One run in the diagnostic dump for (eval_label, truth_m, eval_m, method)."""

    run: str
    truth_score: Optional[float]
    eval_score: Optional[float]
    in_top_k: bool
    kept: bool
    issue: Optional[CorrelationIssue]


def _compute_issue(
    kept_systems: set[str],
    truth_ranking: Dict[str, float],
    eval_ranking: Dict[str, float],
    on_missing_default: bool,
    truth_diag: RankingExtractionDiagnostic,
    eval_diag: RankingExtractionDiagnostic,
) -> Optional[CorrelationIssue]:
    """Derive the verification tag for this (truth_m, eval_m, method) triple.

    Inline stand-in for the shelved `correlation_verification` module
    (see docs/evaluate-diagnostic.md). Swap this for
    `check_correlation_inputs(a, b, on_fail="ignore")` when that lands.

    Precedence: measure_defaulted > too_few > all_tied. A defaulted measure
    synthesizes all-zero rankings, which would also trip all_tied — report
    the root cause (defaulting) instead.
    """
    if truth_diag.defaulted or eval_diag.defaulted:
        return "measure_defaulted"
    if len(kept_systems) < 3:
        return "too_few"
    truth_vals = [truth_ranking[r] for r in kept_systems]
    eval_vals = [
        (eval_ranking[r] if r in eval_ranking else 0.0)
        for r in kept_systems
    ]
    if len(set(truth_vals)) == 1 and len(set(eval_vals)) == 1:
        return "all_tied"
    return None


def format_issues_warning(
    issues: List[Tuple[str, str, str, CorrelationIssue]],
    eval_label: str,
    diagnostics_dir_was_set: bool,
) -> str:
    """Short recap for the end-of-evaluate() warning.

    `issues` is a list of (truth_m, eval_m, method, issue) tuples.
    """
    by_type: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for tm, em, method, issue in issues:
        by_type[issue].append((tm, em, method))

    lines = [f"[{eval_label}] correlation issues in {len(issues)} (measure, method) combination(s):"]
    for issue, triples in sorted(by_type.items()):
        example = triples[0]
        lines.append(f"  {issue}: {len(triples)} (e.g., {example})")
    if not diagnostics_dir_was_set:
        lines.append("Rerun with --diagnostics-dir <path> for per-run details.")
    return "\n".join(lines)


def _safe_path_component(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def diagnose_correlation(
    truth_m: str,
    eval_m: str,
    method: str,
    truth_ranking_pre_topk: Dict[str, float],
    eval_ranking_pre_topk: Dict[str, float],
    top_run_ids: Optional[set[str]],
    truth_ranking: Dict[str, float],
    eval_ranking: Dict[str, float],
    on_missing: str,
    truth_diag: RankingExtractionDiagnostic,
    eval_diag: RankingExtractionDiagnostic,
    diagnostics_dir: Optional[Path] = None,
    eval_label: Optional[str] = None,
) -> Optional[CorrelationIssue]:
    """Classify this (truth_m, eval_m, method) triple; optionally dump details.

    Returns the CorrelationIssue (or None). If `diagnostics_dir` is provided,
    also writes one JSONL file (one row per run in the pre-topk union).
    `eval_label` is required when `diagnostics_dir` is set.
    """
    truth_systems = set(truth_ranking.keys())
    eval_systems = set(eval_ranking.keys())
    on_missing_default = on_missing == "default"
    kept_systems = truth_systems if on_missing_default else truth_systems & eval_systems

    issue = _compute_issue(
        kept_systems, truth_ranking, eval_ranking, on_missing_default,
        truth_diag, eval_diag,
    )

    if diagnostics_dir is not None:
        if eval_label is None:
            raise ValueError("eval_label required when diagnostics_dir is set")
        out_dir = (
            diagnostics_dir
            / _safe_path_component(eval_label)
            / f"{_safe_path_component(truth_m)}__{_safe_path_component(eval_m)}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{_safe_path_component(method)}.jsonl"

        all_runs = set(truth_ranking_pre_topk.keys()) | set(eval_ranking_pre_topk.keys())
        with out_path.open("w", encoding="utf-8") as f:
            for run in sorted(all_runs):
                row = CorrelationDiagnosticRow(
                    run=run,
                    truth_score=truth_ranking_pre_topk.get(run),
                    eval_score=eval_ranking_pre_topk.get(run),
                    in_top_k=(top_run_ids is None) or (run in top_run_ids),
                    kept=run in kept_systems,
                    issue=issue,
                )
                f.write(row.model_dump_json())
                f.write("\n")

    return issue