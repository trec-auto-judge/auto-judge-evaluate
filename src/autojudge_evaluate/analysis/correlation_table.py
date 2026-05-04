"""
Correlation analysis tables from meta-evaluate JSONL output.

Analyzes how well different AutoJudge configurations correlate with ground truth
across datasets (ragtime, rag, dragun).

Usage:
    python -m autojudge_evaluate.analysis.correlation_table \\
        --dataset ragtime:ragtime.jsonl \\
        --dataset rag:rag.jsonl \\
        --dataset dragun:dragun.jsonl
"""

import sys
import click
import yaml
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path
from typing import Optional, Sequence


# Meta columns (not metrics) - category columns get added dynamically
BASE_META_COLS = ["Judge", "Dataset", "TruthMeasure", "EvalMeasure"]


def get_meta_columns(df: pd.DataFrame) -> list[str]:
    """Get all non-metric columns present in df."""
    meta = []
    for c in df.columns:
        if c in BASE_META_COLS:
            meta.append(c)
        elif not pd.api.types.is_numeric_dtype(df[c]) and c not in BASE_META_COLS:
            # String columns added by judges YAML (categories)
            meta.append(c)
    return meta


def get_metric_columns(df: pd.DataFrame) -> list[str]:
    """Extract metric columns from DataFrame (everything except meta columns)."""
    meta = get_meta_columns(df)
    return [c for c in df.columns if c not in meta]


def _escape(s: str, fmt: str) -> str:
    """Escape special characters for the target output format."""
    if fmt == "latex":
        return _latex_escape(s)
    # Markdown: escape underscores so they aren't read as emphasis
    return str(s).replace("_", "\\_")


def _escape_df(df: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """Escape string cell values and column names in *df* for *fmt*."""
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: _escape(str(v), fmt) if pd.notna(v) else v)
    out.columns = [_escape(str(c), fmt) for c in out.columns]
    return out


def _heading(text: str, fmt: str, level: int = 4) -> str:
    """Format a section heading: markdown ``####`` or plain text for latex."""
    escaped = _escape(text, fmt)
    if fmt == "latex":
        return f"\n\n{escaped}\n"
    return f"{'#' * level} {escaped}\n"


class JudgesConfig:
    """Parsed judges YAML: judge mappings + optional style overrides."""

    def __init__(self, judges: dict[str, dict],
                 colors: dict[str, str] | None = None,
                 hatches: dict[str, str] | None = None):
        self.judges = judges
        self.colors = colors or {}
        self.hatches = hatches or {}


def load_judges_yaml(path: Path) -> JudgesConfig:
    """Load judges YAML file.

    Returns:
        JudgesConfig with judge mappings and optional style overrides.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    judges = data.get("judges", data)
    styles = data.get("styles", {})
    return JudgesConfig(
        judges=judges,
        colors=styles.get("colors"),
        hatches=styles.get("hatches"),
    )


def apply_judges(df: pd.DataFrame, judges: dict[str, dict], all_judges: bool = False) -> pd.DataFrame:
    """Apply judge mappings: rename judges, add category columns, sort.

    Category columns (everything except 'name') are added and used as sort keys.
    Unless all_judges=True, drops judges not defined in the YAML.
    """
    df = df.copy()

    # Filter to only defined judges (unless --all-judges)
    if not all_judges:
        df = df[df["Judge"].isin(judges.keys())]

    # Discover category dimensions from YAML (all keys except 'name')
    all_categories = set()
    for info in judges.values():
        all_categories.update(k for k in info if k != "name")
    category_cols = sorted(all_categories)

    # Add category columns
    for cat in category_cols:
        df[cat] = df["Judge"].map(lambda j, c=cat: judges.get(j, {}).get(c, ""))

    # Sort by YAML definition order (before renaming, so we match on original keys)
    judge_order = {j: i for i, j in enumerate(judges.keys())}
    df["_sort_key"] = df["Judge"].map(lambda j: judge_order.get(j, len(judge_order)))
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)

    # Rename judges
    name_map = {j: info["name"] for j, info in judges.items() if "name" in info}
    df["Judge"] = df["Judge"].map(lambda j: name_map.get(j, j))

    return df


def category_comparison(
    df: pd.DataFrame,
    judges: dict[str, dict],
    metric_cols: list[str],
    fmt: str = "github",
    same_threshold: float = 0.01,
) -> str:
    """Compare average correlation by category dimension.

    For each category dimension (e.g., 'graded'), groups judges by their
    category value (e.g., 'docs' vs 'response') and shows average metrics.
    """
    # Discover categories
    all_categories = set()
    for info in judges.values():
        all_categories.update(k for k in info if k != "name")

    sections = []

    # Group by (Dataset, TruthMeasure, EvalMeasure) then by category
    group_keys = [c for c in ["Dataset", "TruthMeasure", "EvalMeasure"] if c in df.columns]

    for cat in sorted(all_categories):
        if cat not in df.columns:
            continue

        sections.append(_heading(f"Category: {cat}", fmt, level=2))

        for group_vals, group_df in df.groupby(group_keys, sort=True, observed=True):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            header_parts = [f"{k}={v}" for k, v in zip(group_keys, group_vals)]
            sections.append(_heading(', '.join(header_parts), fmt))

            # Average metrics by category value
            avail_metrics = [m for m in metric_cols if m in group_df.columns]
            avg_df = group_df.groupby(cat, observed=True)[avail_metrics].max().reset_index()
            avg_df = avg_df.rename(columns={cat: cat.title()})

            sections.append(format_table(
                avg_df, fmt, highlight_max=True,
                metric_cols=avail_metrics, same_threshold=same_threshold,
            ))
            sections.append("")

    return "\n".join(sections)


def category_wilcoxon(
    df: pd.DataFrame,
    judges: dict[str, dict],
    metric_cols: list[str],
    fmt: str = "github",
) -> str:
    """Wilcoxon signed-rank test comparing category values across matched conditions.

    For each category dimension (e.g., 'graded'), pairs judges that differ only
    in that dimension, merges on condition columns (Dataset, TruthMeasure,
    EvalMeasure), and runs Wilcoxon signed-rank tests on each metric.
    """
    from itertools import combinations
    from scipy.stats import wilcoxon

    # Discover category dimensions from judges config
    all_categories = sorted(
        {k for info in judges.values() for k in info if k != "name"}
    )
    cat_cols = [c for c in all_categories if c in df.columns]
    if not cat_cols:
        return ""

    condition_cols = [c for c in ["Dataset", "TruthMeasure", "EvalMeasure"] if c in df.columns]
    rows = []

    for tested_cat in cat_cols:
        other_cats = [c for c in cat_cols if c != tested_cat]
        # Group judges by their "other category" values to find matched sets
        judge_cat_vals = {}
        for j, info in judges.items():
            name = info.get("name", j)
            key = tuple(info.get(c, "") for c in other_cats) if other_cats else ()
            judge_cat_vals.setdefault(key, []).append((name, info.get(tested_cat, "")))

        # For each matched group, collect paired observations per (cat_a, cat_b) pair
        pair_diffs: dict[tuple[str, str], dict[str, list[float]]] = {}

        for _, members in judge_cat_vals.items():
            # Get distinct category values in this group
            val_to_judges = {}
            for name, val in members:
                val_to_judges.setdefault(val, []).append(name)

            vals = sorted(val_to_judges.keys())
            if len(vals) < 2:
                continue

            for va, vb in combinations(vals, 2):
                pair_key = (va, vb)
                if pair_key not in pair_diffs:
                    pair_diffs[pair_key] = {m: [] for m in metric_cols}

                for ja in val_to_judges[va]:
                    for jb in val_to_judges[vb]:
                        da = df[df["Judge"] == ja]
                        db = df[df["Judge"] == jb]
                        merged = da.merge(db, on=condition_cols, suffixes=("_a", "_b"))
                        for m in metric_cols:
                            ca, cb = f"{m}_a", f"{m}_b"
                            if ca in merged.columns and cb in merged.columns:
                                diffs = (merged[ca] - merged[cb]).dropna()
                                pair_diffs[pair_key][m].extend(diffs.tolist())

        for (va, vb), metrics_diffs in sorted(pair_diffs.items()):
            for m, diffs in metrics_diffs.items():
                diffs_arr = np.array(diffs)
                # Need at least 6 non-zero differences for a meaningful test
                nonzero = diffs_arr[diffs_arr != 0]
                if len(nonzero) < 6:
                    continue
                _, p = wilcoxon(diffs_arr)
                med = float(np.median(diffs_arr))
                winner = va if med > 0 else vb if med < 0 else "tie"
                rows.append({
                    "Category": tested_cat,
                    "Comparison": f"{va} vs {vb}",
                    "Metric": m,
                    "N": len(diffs_arr),
                    "median_diff": f"{med:+.4f}",
                    "p-value": f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}",
                    "winner": winner,
                    "sig": "*" if p < 0.05 else "",
                })

    if not rows:
        print("Wilcoxon: no matched pairs found")
        return "\n"

    result_df = _escape_df(pd.DataFrame(rows), fmt)
    if fmt == "latex":
        return "\n" + _heading("Category Wilcoxon Test", fmt=fmt, level=1) + result_df.to_latex(index=False, escape=False) + "\n"
    return result_df.to_markdown(index=False) + "\n"


def load_dataset(path: Path, label: str) -> pd.DataFrame:
    """Load correlation results from JSONL file (pandas-based, original behavior)."""
    import json
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                preview = stripped[:120] + ("..." if len(stripped) > 120 else "")
                raise ValueError(
                    f"{path}:{lineno}: invalid JSON ({e.msg} at col {e.colno}): {preview!r}"
                ) from None
    if not rows:
        raise ValueError(f"{path}: no records loaded (file empty or all lines blank)")
    df = pd.DataFrame(rows)
    df["Dataset"] = label

    # Detect duplicate (Judge, TruthMeasure, EvalMeasure) rows — usually means
    # the producing script appended (>>) instead of overwriting (>) the file.
    dup_keys = [c for c in ("Judge", "TruthMeasure", "EvalMeasure") if c in df.columns]
    if dup_keys:
        dups = df.groupby(dup_keys).size()
        dups = dups[dups > 1]
        if not dups.empty:
            sample = dups.head(5).to_dict()
            print(
                f"WARNING: {path}: {len(dups)} duplicate {tuple(dup_keys)} row(s); "
                f"first {min(5, len(dups))}: {sample}",
                file=sys.stderr,
            )

    return df

    
    # with path.open(encoding="utf-8") as f:
    #     df = pd.read_json(f, lines=True)
    #     df["Dataset"] = label
    #     return df


def load_datasets(dataset_specs: Sequence[str]) -> pd.DataFrame:
    """Load multiple datasets from 'label:path' specs.

    Args:
        dataset_specs: List of 'label:path' strings, or just 'path' (uses stem as label)
    """
    dfs = []
    for spec in dataset_specs:
        if ":" in spec:
            label, path_str = spec.split(":", 1)
        else:
            path_str = spec
            label = Path(path_str).stem
        dfs.append(load_dataset(Path(path_str), label))
    return pd.concat(dfs, ignore_index=True)


def _make_ordered_concat(df: pd.DataFrame, columns: list[str], sep: str = " / ") -> pd.Series:
    """Concatenate columns into a single string column, preserving categorical order.

    If the component columns are ordered categoricals, the result is an ordered
    categorical whose categories follow the cross-product order of the inputs
    (first column varies slowest).  Otherwise falls back to first-appearance order.
    """
    result = df[columns[0]].astype(str)
    for col in columns[1:]:
        result = result + sep + df[col].astype(str)

    # Build ordered categories from cross-product of component categoricals
    cat_lists = []
    for col in columns:
        s = df[col]
        if hasattr(s, "cat") and s.cat.ordered:
            cat_lists.append(list(s.cat.categories))
        else:
            cat_lists.append(list(dict.fromkeys(s)))
    ordered = [sep.join(str(v) for v in combo) for combo in product(*cat_lists)]
    # Only keep categories actually present
    present = set(result)
    ordered = [c for c in ordered if c in present]
    return pd.Categorical(result, categories=ordered, ordered=True)


def _dataset_label(spec: str) -> str:
    """Extract dataset label from a 'label:path' or 'path' spec."""
    if ":" in spec:
        return spec.split(":", 1)[0]
    return Path(spec).stem


def _set_categorical_order(df: pd.DataFrame, column: str, cli_values: Sequence[str]) -> None:
    """Set column to ordered Categorical preserving CLI order.

    If cli_values is empty/None, uses the order of first appearance in df.
    For "Dataset", extracts labels from 'label:path' specs.
    """
    if column not in df.columns:
        return
    if cli_values:
        if column == "Dataset":
            ordered = list(dict.fromkeys(_dataset_label(s) for s in cli_values))
        else:
            ordered = list(dict.fromkeys(cli_values))
    else:
        # Preserve first-appearance order from the dataframe
        ordered = list(dict.fromkeys(df[column]))
    # Include any values present in df but not in ordered (append at end)
    for val in df[column].unique():
        if val not in ordered:
            ordered.append(val)
    df[column] = pd.Categorical(df[column], categories=ordered, ordered=True)


def aggregate_by_judge(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Aggregate correlation metrics across datasets per judge.

    Groups by (Judge, TruthMeasure, EvalMeasure), averages metrics.
    """
    if metrics is None:
        metrics = get_metric_columns(df)

    # Filter to metrics that exist
    available = [m for m in metrics if m in df.columns]

    group_cols = ["Judge", "TruthMeasure", "EvalMeasure"]
    return df.groupby(group_cols, observed=True)[available].mean().reset_index()



def bold_max(
    df: pd.DataFrame,
    metric_cols: list[str],
    fmt: str = "github",
    same_threshold: float = 0.05,
) -> pd.DataFrame:
    """Return a copy with string-formatted values, max and near-max bolded.

    Values within `same_threshold` of the column max are considered
    statistically indistinguishable and also bolded.

    For github/pipe markdown: **0.8182**
    For latex: \\textbf{0.8182}
    """
    out = df.copy()
    for col in metric_cols:
        if col not in out.columns:
            continue
        max_val = out[col].max()
        cutoff = max_val - same_threshold
        if fmt in ("latex", "latex_raw", "latex_booktabs"):
            out[col] = out[col].apply(
                lambda v, c=cutoff: f"\\textbf{{{v:.4f}}}" if v >= c else f"{v:.4f}"
            )
        else:
            out[col] = out[col].apply(
                lambda v, c=cutoff: f"**{v:.4f}**" if v >= c else f"{v:.4f}"
            )
    return out


# -- LaTeX rendering -----------------------------------------------------------

def _latex_escape(s: str) -> str:
    """Escape LaTeX special characters in text."""
    s = str(s)
    s = s.replace("_", "\\_")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("#", "\\#")
    return s


def _render_latex_booktabs(
    df: pd.DataFrame,
    metric_cols: list[str],
    same_threshold: float = 0.02,
    highlight_max: bool = True,
    float_format: str = ".4f",
) -> str:
    """Render DataFrame as LaTeX booktabs table with proper escaping.

    Auto-detects hierarchical columns (containing ' / ') and renders
    multi-row headers with \\cmidrule(lr).
    """
    value_cols = [c for c in metric_cols if c in df.columns]
    row_id_cols = [c for c in df.columns if c not in value_cols]

    # Format numeric cells (with optional bold for max)
    formatted = df.copy()
    for col in value_cols:
        if highlight_max:
            max_val = formatted[col].max()
            cutoff = max_val - same_threshold
            formatted[col] = formatted[col].apply(
                lambda v, c=cutoff: f"\\textbf{{{v:.4f}}}" if v >= c else f"{v:.4f}"
            )
        else:
            formatted[col] = formatted[col].apply(
                lambda v: f"{v:{float_format}}" if isinstance(v, (int, float)) else str(v)
            )

    hierarchical = any(" / " in str(c) for c in value_cols)

    n_total = len(df.columns)
    lines = []
    lines.append("\\begin{tabular}{" + "l" * n_total + "}")
    lines.append("\\toprule")

    if hierarchical:
        parsed = [col.split(" / ") for col in value_cols]
        n_levels = max(len(p) for p in parsed)
        for p in parsed:
            while len(p) < n_levels:
                p.append("")

        # Multi-row headers (all levels except last)
        for level in range(n_levels - 1):
            groups = []
            i = 0
            while i < len(parsed):
                val = parsed[i][level]
                span = 1
                while i + span < len(parsed) and parsed[i + span][level] == val:
                    span += 1
                col_start = len(row_id_cols) + i + 1  # 1-indexed
                groups.append((val, span, col_start))
                i += span

            parts = [""] * len(row_id_cols)
            for val, span, _ in groups:
                parts.append(f"\\multicolumn{{{span}}}{{c}}{{{_latex_escape(val)}}}")
            lines.append(" & ".join(parts))
            lines.append("\\tabularnewline")

            cmidrules = []
            for val, span, start in groups:
                if val:
                    cmidrules.append(f"\\cmidrule(lr){{{start}-{start + span - 1}}}")
            if cmidrules:
                lines.append("".join(cmidrules))

        # Last level header row
        last_parts = [_latex_escape(c) for c in row_id_cols]
        for p in parsed:
            last_parts.append(_latex_escape(p[-1]))
        lines.append(" & ".join(last_parts) + "\\tabularnewline")
    else:
        header = [_latex_escape(c) for c in row_id_cols + value_cols]
        lines.append(" & ".join(header) + "\\tabularnewline")

    lines.append("\\midrule")

    # Data rows
    for _, row in formatted.iterrows():
        parts = [_latex_escape(str(row[c])) for c in row_id_cols]
        for col in value_cols:
            parts.append(str(row[col]))  # Already formatted; don't escape \textbf
        lines.append(" & ".join(parts) + " \\tabularnewline")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


# -- Generic table formatting --------------------------------------------------

def format_table(
    df: pd.DataFrame,
    fmt: str = "github",
    float_format: str = ".4f",
    highlight_max: bool = False,
    metric_cols: Optional[list[str]] = None,
    same_threshold: float = 0.02,
) -> str:
    """Format DataFrame as table string using pandas to_markdown."""
    if fmt == "latex":
        return _render_latex_booktabs(
            df, metric_cols or [], same_threshold=same_threshold,
            highlight_max=highlight_max, float_format=float_format,
        )
    df = _escape_df(df, fmt)
    if highlight_max and metric_cols:
        # Remap metric_cols to escaped names
        esc_metrics = [_escape(c, fmt) for c in metric_cols]
        df = bold_max(df, esc_metrics, fmt, same_threshold)
        # Already string-formatted, don't apply floatfmt
        return df.to_markdown(index=False, tablefmt=fmt)
    return df.to_markdown(index=False, tablefmt=fmt, floatfmt=float_format)


def correlation_consistency(
    df: pd.DataFrame,
    metric_cols: list[str],
    fmt: str = "github",
    same_threshold: float = 0.01,
    summary: bool = False,
) -> str:
    """One table per (Dataset, TruthMeasure, EvalMeasure), or one wide table if summary.

    Rows: Judge
    Columns: correlation metrics
    Highest value per column bolded.
    """
    if summary:
        # One wide table: pivot Dataset/TruthMeasure/EvalMeasure into column prefixes
        df = df.copy()
        if "Dataset" in df.columns:
            df["_group"] = _make_ordered_concat(df, ["Dataset", "TruthMeasure", "EvalMeasure"])
        else:
            df["_group"] = _make_ordered_concat(df, ["TruthMeasure", "EvalMeasure"])

        meta = get_meta_columns(df)
        row_id_cols = [c for c in meta if c not in ["Dataset", "TruthMeasure", "EvalMeasure", "_group"]]

        # Pivot each metric, then merge
        merged = None
        for metric in metric_cols:
            if metric not in df.columns:
                continue
            piv = df.pivot_table(
                index=row_id_cols, columns="_group", values=metric, aggfunc="first",
                observed=True,
            )
            # Prefix columns with metric name
            piv.columns = [f"{g} / {metric}" for g in piv.columns]
            if merged is None:
                merged = piv
            else:
                merged = merged.join(piv)

        if merged is None:
            return ""
        merged = merged.reset_index()
        merged.columns.name = None

        # Preserve judge order
        judge_order = {j: i for i, j in enumerate(df["Judge"].unique())}
        merged["_sort"] = merged["Judge"].map(lambda j: judge_order.get(j, len(judge_order)))
        merged = merged.sort_values("_sort").drop(columns=["_sort"])

        value_cols = [c for c in merged.columns if c not in row_id_cols]
        return format_table(merged, fmt, highlight_max=True, metric_cols=value_cols, same_threshold=same_threshold)

    sections = []
    group_keys = [c for c in ["Dataset", "TruthMeasure", "EvalMeasure"] if c in df.columns]

    for group_vals, group_df in df.groupby(group_keys, sort=True, observed=True):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        header_parts = [f"{k}={v}" for k, v in zip(group_keys, group_vals)]
        sections.append(_heading(', '.join(header_parts), fmt))

        # Select row-header cols (categories + Judge) + metric columns
        meta = get_meta_columns(group_df)
        row_cols = [c for c in meta if c not in group_keys]
        table_cols = row_cols + [c for c in metric_cols if c in group_df.columns]
        out_df = group_df[table_cols].copy()

        sections.append(format_table(out_df, fmt, highlight_max=True, metric_cols=metric_cols, same_threshold=same_threshold))
        sections.append("")

    return "\n".join(sections)


def measures_as_columns(
    df: pd.DataFrame,
    metric_cols: list[str],
    fmt: str = "github",
    same_threshold: float = 0.01,
    summary: bool = False,
) -> str:
    """One table per (Dataset, correlation metric), or one wide table if summary.

    Rows: Judge
    Columns: TruthMeasure/EvalMeasure combinations (prefixed with Dataset if summary)
    Highest value per column bolded.
    """
    sections = []
    dataset_keys = ["Dataset"] if "Dataset" in df.columns else []

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        if summary:
            # One wide table: Dataset / TruthMeasure / EvalMeasure as columns
            sections.append(_heading(f"Correlation={metric}", fmt))
            sections.append(_pivot_measures_table(df, metric, fmt, same_threshold, include_dataset=True))
            sections.append("")
        elif dataset_keys:
            for dataset, ds_df in df.groupby("Dataset", sort=True, observed=True):
                sections.append(_heading(f"Dataset={dataset}, Correlation={metric}", fmt))
                sections.append(_pivot_measures_table(ds_df, metric, fmt, same_threshold))
                sections.append("")
        else:
            sections.append(_heading(f"Correlation={metric}", fmt))
            sections.append(_pivot_measures_table(df, metric, fmt, same_threshold))
            sections.append("")

    return "\n".join(sections)


def _pivot_measures_table(
    df: pd.DataFrame,
    metric: str,
    fmt: str,
    same_threshold: float,
    include_dataset: bool = False,
) -> str:
    """Pivot a single metric into columns of TruthMeasure/EvalMeasure."""
    # Create composite column name
    df = df.copy()
    if include_dataset and "Dataset" in df.columns:
        df["_measure_col"] = _make_ordered_concat(df, ["Dataset", "TruthMeasure", "EvalMeasure"])
    else:
        df["_measure_col"] = _make_ordered_concat(df, ["TruthMeasure", "EvalMeasure"])

    # Get row identity columns (Judge + any category cols)
    meta = get_meta_columns(df)
    row_id_cols = [c for c in meta if c not in ["Dataset", "TruthMeasure", "EvalMeasure", "_measure_col"]]

    # Pivot — use observed=True to respect categorical order
    pivoted = df.pivot_table(
        index=row_id_cols,
        columns="_measure_col",
        values=metric,
        aggfunc="first",
        observed=True,
    ).reset_index()

    # Flatten MultiIndex columns if needed
    pivoted.columns.name = None

    # Preserve judge order from the original df
    judge_order = {j: i for i, j in enumerate(df["Judge"].unique())}
    pivoted["_sort"] = pivoted["Judge"].map(lambda j: judge_order.get(j, len(judge_order)))
    pivoted = pivoted.sort_values("_sort").drop(columns=["_sort"])

    measure_cols = [c for c in pivoted.columns if c not in row_id_cols]
    return format_table(pivoted, fmt, highlight_max=True, metric_cols=measure_cols, same_threshold=same_threshold)


# -- Plotting ------------------------------------------------------------------

# Fallback palettes when no judges YAML is used
_FALLBACK_COLORS = ["0.30", "0.55", "0.78", "0.92", "0.42", "0.65"]
_FALLBACK_HATCHES = ["", "//", "\\\\", "xx", "..", "++"]


def _get_category_columns(df: pd.DataFrame) -> list[str]:
    """Get category columns (string cols that aren't base meta cols)."""
    return [c for c in df.columns if c not in BASE_META_COLS and not pd.api.types.is_numeric_dtype(df[c])]


def _build_style_map(
    df: pd.DataFrame,
    color_map: dict[str, str] | None = None,
    hatch_map: dict[str, str] | None = None,
) -> dict[str, tuple[str, str]]:
    """Map each judge to (color, hatch) based on category values.

    Color is looked up from the first category value that matches color_map.
    Hatch patterns are combined from all category values matching hatch_map.
    Falls back to per-judge sequential assignment if no categories or no maps.
    """
    cat_cols = _get_category_columns(df)

    if not cat_cols or (not color_map and not hatch_map):
        return {
            judge: (_FALLBACK_COLORS[i % len(_FALLBACK_COLORS)],
                    _FALLBACK_HATCHES[i % len(_FALLBACK_HATCHES)])
            for i, judge in enumerate(df["Judge"].unique())
        }

    _colors = color_map or {}
    _hatches = hatch_map or {}

    style_map = {}
    for _, row in df.iterrows():
        judge = row["Judge"]
        if judge in style_map:
            continue

        # Find color: first category value present in color_map
        color = "0.5"
        for cat in cat_cols:
            val = row[cat]
            if val in _colors:
                color = _colors[val]
                break

        # Combine hatches: all category values present in hatch_map
        hatch = ""
        for cat in cat_cols:
            val = row[cat]
            if val in _hatches:
                hatch += _hatches[val]

        style_map[judge] = (color, hatch)

    return style_map


def _is_dark(color: str) -> bool:
    """Check if a color is dark based on perceived luminance."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    # Perceived luminance (ITU-R BT.601)
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.5


def _setup_plot_style():
    """Configure matplotlib for Google Sheets-like clean style."""
    matplotlib.use("pdf")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 6,
        "axes.linewidth": 0,
        "axes.axisbelow": True,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.edgecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.color": "0.0",
        "ytick.color": "0.0",
        "text.color": "0.0",
        "axes.labelcolor": "0.0",
        "figure.dpi": 300,
        "figure.facecolor": "white",
    })


def plot_grouped_bar(
    df: pd.DataFrame,
    group_cols: list[str],
    series_col: str,
    title: str,
    out_path: Path,
    legend_cols: int,
    ylabel: str = "Correlation",
    color_map: dict[str, str] | None = None,
    hatch_map: dict[str, str] | None = None,
):
    """Grouped bar chart in Google Sheets style with category-based colors.

    Args:
        df: DataFrame with series_col and group_cols as numeric columns
        group_cols: Column names for x-axis groups (bar clusters)
        series_col: Column with series labels (e.g., Judge)
        title: Plot title
        out_path: Output PDF path
        ylabel: Y-axis label
        color_map: Category value -> fill color (from judges.yml styles)
        hatch_map: Category value -> hatch pattern (from judges.yml styles)
    """
    _setup_plot_style()
    style_map = _build_style_map(df, color_map=color_map, hatch_map=hatch_map)

    n_series = len(df)
    n_groups = len(group_cols)
    if n_series == 0 or n_groups == 0:
        print(f"Skipping '{out_path.name}': empty DataFrame (n_series={n_series}, n_groups={n_groups})", file=sys.stderr)
        return

    x = np.arange(n_groups)
    total_width = 0.90
    bar_width = total_width / n_series

    # fig, ax = plt.subplots(figsize=(max(5, n_groups * 1.5), 3.2))
    # fig, ax = plt.subplots(figsize=(max(5, n_groups * 1.5), 1.5))
    
    bar_weight = 0.2 + 0.2 * n_series
    
    fig, ax = plt.subplots(figsize=(max(2, n_groups * bar_weight), 1.5))

    for i, (_, row) in enumerate(df.iterrows()):
        judge = row[series_col]
        color, hatch = style_map.get(judge, ("#999999", ""))
        values = [row[c] for c in group_cols]
        offset = (i - n_series / 2 + 0.5) * bar_width
        edge = ("#E0E0E0" if _is_dark(color) else "#232323" ) if hatch else color
        bars = ax.bar(
            x + offset, values, bar_width * 0.75,
            color=color,
            hatch=hatch,
            edgecolor=edge,
            linewidth=0.3,
            label=judge,
        )
        for bar, v in zip(bars, values):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            label_y = max(v, 0.0)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6,
                    color="0.0", rotation=90)

    # Y-axis: light horizontal grid lines, ticks at 0.25 intervals
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.yaxis.grid(True, color="0.85", linewidth=0.5)
    ax.xaxis.grid(False)
    # ax.set_ylim(0, 1.15)
    ax.set_ylim(0, 1.05)

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(group_cols, rotation=0, ha="center", fontsize=6, color="0.0")
    # ax.set_xlim(-0.5, len(df.columns) - 0.5)
    # ax.set_xlim(-0.1, len(df.columns) - 0.1)
    half_cluster = total_width / 2
    ax.set_xlim(-half_cluster, n_groups - 1 + half_cluster)    
    
    # Labels and title
    ax.set_ylabel(ylabel, fontsize=6, color="0.0")
    # ax.set_title(title, fontsize=6, color="0.3", loc="left", pad=10)

    # Legend above plot, wrapping into multiple rows via ncol
    # ax.legend(
    #     loc="lower left", bbox_to_anchor=(0, 1.02), ncol=min(n_series, 10),
    #     borderaxespad=0, handlelength=1.2, handleheight=0.8,
    #     columnspacing=1.0,
    # )
    # Deduplicate legend entries with same label
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # keeps last handle per label
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="lower left", bbox_to_anchor=(0, 1.02), ncol=min(len(by_label), legend_cols),
        borderaxespad=0, handlelength=1.2, handleheight=0.8,
        columnspacing=1.0,
    )


    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _slugify(s: str) -> str:
    """Convert string to filesystem-safe slug."""
    return s.replace("/", "_").replace(" ", "_").replace("=", "").replace(",", "")


def plot_correlation_consistency(
    df: pd.DataFrame,
    metric_cols: list[str],
    plot_dir: Path,
    legend_cols: int,
    color_map: dict[str, str] | None = None,
    hatch_map: dict[str, str] | None = None,
    all_datasets: bool = False,
):
    """One plot per (Dataset, TruthMeasure, EvalMeasure), or per (TruthMeasure, EvalMeasure) if all_datasets."""
    if all_datasets and "Dataset" in df.columns:
        group_keys = [c for c in ["TruthMeasure", "EvalMeasure"] if c in df.columns]
    else:
        group_keys = [c for c in ["Dataset", "TruthMeasure", "EvalMeasure"] if c in df.columns]

    for group_vals, group_df in df.groupby(group_keys, sort=True, observed=True):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        label_parts = [f"{k}={v}" for k, v in zip(group_keys, group_vals)]
        title = ", ".join(label_parts)
        slug = _slugify("_".join(str(v) for v in group_vals))

        if all_datasets and "Dataset" in group_df.columns:
            # Prefix metric columns with dataset name
            datasets = sorted(group_df["Dataset"].unique())
            merged = None
            meta = get_meta_columns(group_df)
            row_id_cols = [c for c in meta if c not in ["Dataset", "TruthMeasure", "EvalMeasure"]]

            for ds in datasets:
                ds_df = group_df[group_df["Dataset"] == ds][row_id_cols + metric_cols].copy()
                ds_df = ds_df.rename(columns={m: f"{ds}\n{m}" for m in metric_cols if m in ds_df.columns})
                if merged is None:
                    merged = ds_df
                else:
                    merged = merged.merge(ds_df, on=row_id_cols, how="outer")

            if merged is None:
                continue
            avail = [c for c in merged.columns if c not in row_id_cols]
            slug = _slugify("all_" + "_".join(str(v) for v in group_vals))
            plot_grouped_bar(merged, avail, "Judge",  title, plot_dir / f"{slug}.pdf", legend_cols,
                             color_map=color_map, hatch_map=hatch_map)
        else:
            avail = [c for c in metric_cols if c in group_df.columns]
            plot_grouped_bar(group_df, avail, "Judge", title, plot_dir / f"{slug}.pdf", legend_cols,
                             color_map=color_map, hatch_map=hatch_map)


def plot_measures_as_columns(
    df: pd.DataFrame,
    metric_cols: list[str],
    plot_dir: Path,
    legend_cols:int,
    color_map: dict[str, str] | None = None,
    hatch_map: dict[str, str] | None = None,
    all_datasets: bool = False,
):
    """One plot per (Dataset, correlation metric), or per metric if all_datasets."""
    dataset_keys = ["Dataset"] if "Dataset" in df.columns else []

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        def _plot_one(sub_df: pd.DataFrame, title: str, slug: str, include_dataset: bool = False):
            if sub_df.empty or metric not in sub_df.columns or sub_df[metric].dropna().empty:
                print(f"Skipping plot '{slug}': no data for metric={metric}", file=sys.stderr)
                return
            sub_df = sub_df.copy()
            if include_dataset and "Dataset" in sub_df.columns:
                sub_df["_measure_col"] = _make_ordered_concat(sub_df, ["Dataset", "TruthMeasure", "EvalMeasure"], sep="\n")
            else:
                sub_df["_measure_col"] = _make_ordered_concat(sub_df, ["TruthMeasure", "EvalMeasure"], sep="\n")
            meta = get_meta_columns(sub_df)
            row_id_cols = [c for c in meta if c not in ["Dataset", "TruthMeasure", "EvalMeasure", "_measure_col"]]

            pivoted = sub_df.pivot_table(
                index=row_id_cols, columns="_measure_col", values=metric, aggfunc="first",
                observed=True,
            ).reset_index()
            pivoted.columns.name = None

            judge_order = {j: i for i, j in enumerate(sub_df["Judge"].unique())}
            pivoted["_sort"] = pivoted["Judge"].map(lambda j: judge_order.get(j, len(judge_order)))
            pivoted = pivoted.sort_values("_sort").drop(columns=["_sort"])

            measure_cols = [c for c in pivoted.columns if c not in row_id_cols]
            plot_grouped_bar(pivoted, measure_cols, "Judge", title, plot_dir / f"{slug}.pdf", legend_cols, ylabel=metric,
                             color_map=color_map, hatch_map=hatch_map)

        if all_datasets and dataset_keys:
            title = f"Correlation={metric}"
            slug = _slugify(f"all_{metric}")
            _plot_one(df, title, slug, include_dataset=True)
        elif dataset_keys:
            for dataset, ds_df in df.groupby("Dataset", sort=True, observed=True):
                title = f"Dataset={dataset}, Correlation={metric}"
                slug = _slugify(f"{dataset}_{metric}")
                _plot_one(ds_df, title, slug)
        else:
            title = f"Correlation={metric}"
            slug = _slugify(metric)
            _plot_one(df, title, slug)


@click.command()
@click.option(
    "--dataset", "-d",
    "datasets",
    type=str,
    multiple=True,
    help="Dataset as 'label:path' or just 'path'. Repeatable.",
)
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--format", "-f",
    type=click.Choice(["github", "latex", "tsv", "plain", "html", "pipe"]),
    default="github",
    help="Table format.",
)
@click.option(
    "--judges", "-j",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="YAML file mapping judge names to nice names and categories.",
)
@click.option(
    "--all-judges/--no-all-judges",
    default=False,
    help="Include judges not defined in --judges YAML (default: only defined judges).",
)
@click.option(
    "--columns",
    type=click.Choice(["correlations", "measures"]),
    default="correlations",
    help="What to show as columns: 'correlations' (one table per dataset/truth/eval) "
         "or 'measures' (one table per dataset/correlation metric).",
)
@click.option(
    "--aggregate/--no-aggregate",
    default=False,
    help="Average metrics across datasets (per judge) before tabulating.",
)
@click.option(
    "--summary/--no-summary",
    default=False,
    help="Combine all datasets into one table and one plot (dataset name prefixed to columns).",
)
@click.option(
    "--correlation",
    type=str,
    multiple=True,
    help="Correlation metric columns to include (e.g., kendall, spearman@5). Repeatable. Default: all.",
)
@click.option(
    "--truth-measure", "-t",
    type=str,
    multiple=True,
    help="Filter to specific TruthMeasure(s). Repeatable.",
)
@click.option(
    "--eval-measure", "-e",
    "eval_measures",
    type=str,
    multiple=True,
    help="Filter to specific EvalMeasure(s) (e.g., AVG_GRADE). Repeatable.",
)
@click.option(
    "--same",
    type=float,
    default=0.01,
    help="Threshold for 'statistically same': values within this of the max are also bolded.",
)
@click.option(
    "--legend-cols",
    type=int,
    default=6,
    help="Columns in the legens.",
)
@click.option(
    "--plot-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for PDF plots (one per table).",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output file. If omitted, prints to stdout.",
)
def main(
    datasets: tuple,
    files: tuple,
    format: str,
    judges: Optional[Path],
    all_judges: bool,
    columns: str,
    aggregate: bool,
    summary: bool,
    correlation: tuple,
    truth_measure: tuple,
    eval_measures: tuple,
    same: float,
    legend_cols: int,
    plot_dir: Optional[Path],
    output: Optional[Path],
):
    """Generate correlation consistency tables from meta-evaluate JSONL output.

    Produces one table per (Dataset, TruthMeasure, EvalMeasure) combination,
    with judges as rows, correlation metrics as columns, and the best value
    per column highlighted in bold.

    Examples:
        # Single file
        correlation_table results.jsonl

        # Multiple labeled datasets
        correlation_table -d ragtime:rt.jsonl -d rag:r.jsonl

        # Filter by measures
        correlation_table file.jsonl -t nugget_coverage -e AVG_GRADE

        # Select specific correlation metrics
        correlation_table file.jsonl --correlation kendall --correlation spearman

        # Aggregate across datasets, then tabulate
        correlation_table -d rt:rt.jsonl -d rag:rag.jsonl --aggregate
    """
    # Combine --dataset specs and positional files
    all_specs = list(datasets) + list(files)
    if not all_specs:
        click.echo("No input files specified.", err=True)
        return 1

    # Load data
    df = load_datasets(all_specs)

    # Load and apply judges YAML
    judges_config = None
    color_map = None
    hatch_map = None
    if judges:
        judges_config = load_judges_yaml(judges)
        df = apply_judges(df, judges_config.judges, all_judges=all_judges)
        color_map = judges_config.colors or None
        hatch_map = judges_config.hatches or None

    # Filter rows by TruthMeasure if specified
    if truth_measure:
        df = df[df["TruthMeasure"].isin(truth_measure)]

    # Filter rows by EvalMeasure if specified
    if eval_measures:
        df = df[df["EvalMeasure"].isin(eval_measures)]

    # Preserve CLI order for datasets, measures via ordered categoricals
    _set_categorical_order(df, "Dataset", all_specs)
    _set_categorical_order(df, "TruthMeasure", truth_measure)
    _set_categorical_order(df, "EvalMeasure", eval_measures)

    # Select correlation metric columns
    if correlation:
        metric_cols = list(correlation)
    else:
        metric_cols = get_metric_columns(df)

    # Optionally aggregate across datasets first
    if aggregate:
        df = aggregate_by_judge(df, metrics=metric_cols)

    # Produce tables based on --columns mode
    if columns == "measures":
        result = measures_as_columns(df, metric_cols, format, same_threshold=same, summary=summary)
        if plot_dir:
            plot_measures_as_columns(df, metric_cols, plot_dir, legend_cols,
                                     color_map=color_map, hatch_map=hatch_map,
                                     all_datasets=summary)
    else:
        result = correlation_consistency(df, metric_cols, format, same_threshold=same, summary=summary)
        if plot_dir:
            plot_correlation_consistency(df, metric_cols, plot_dir, legend_cols,
                                         color_map=color_map, hatch_map=hatch_map,
                                         all_datasets=summary)

    if plot_dir:
        click.echo(f"Plots written to {plot_dir}/", err=True)

    # Append category comparison if judges YAML provides categories
    if judges_config:
        sep = "\n\n" if format == "latex" else "\n\n---\n\n"
        # result += sep + _heading("Category Comparison", format, level=1)
        # result += category_comparison(df, judges_config.judges, metric_cols, format, same_threshold=same)
        result += category_wilcoxon(df, judges_config.judges, metric_cols, fmt=format)

    # Output
    if output:
        output.write_text(result)
        click.echo(f"Wrote to {output}", err=True)
    else:
        click.echo(result)

    return 0


if __name__ == "__main__":
    main()
