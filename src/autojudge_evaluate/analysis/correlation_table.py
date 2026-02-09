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

import click
import yaml
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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

        sections.append(f"## Category: {cat}\n")

        for group_vals, group_df in df.groupby(group_keys, sort=True):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            header_parts = [f"{k}={v}" for k, v in zip(group_keys, group_vals)]
            sections.append(f"#### {', '.join(header_parts)}\n")

            # Average metrics by category value
            avail_metrics = [m for m in metric_cols if m in group_df.columns]
            avg_df = group_df.groupby(cat)[avail_metrics].max().reset_index()
            avg_df = avg_df.rename(columns={cat: cat.title()})

            sections.append(format_table(
                avg_df, fmt, highlight_max=True,
                metric_cols=avail_metrics, same_threshold=same_threshold,
            ))
            sections.append("")

    return "\n".join(sections)


def load_dataset(path: Path, label: str) -> pd.DataFrame:
    """Load correlation results from JSONL file."""
    df = pd.read_json(path, lines=True)
    df["Dataset"] = label
    return df


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
    return df.groupby(group_cols)[available].mean().reset_index()



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
    if highlight_max and metric_cols:
        df = bold_max(df, metric_cols, fmt, same_threshold)
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
            df["_group"] = df["Dataset"] + " / " + df["TruthMeasure"] + " / " + df["EvalMeasure"]
        else:
            df["_group"] = df["TruthMeasure"] + " / " + df["EvalMeasure"]

        meta = get_meta_columns(df)
        row_id_cols = [c for c in meta if c not in ["Dataset", "TruthMeasure", "EvalMeasure", "_group"]]

        # Pivot each metric, then merge
        merged = None
        for metric in metric_cols:
            if metric not in df.columns:
                continue
            piv = df.pivot_table(
                index=row_id_cols, columns="_group", values=metric, aggfunc="first",
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

    for group_vals, group_df in df.groupby(group_keys, sort=True):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        header_parts = [f"{k}={v}" for k, v in zip(group_keys, group_vals)]
        sections.append(f"#### {', '.join(header_parts)}\n")

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
            sections.append(f"#### Correlation={metric}\n")
            sections.append(_pivot_measures_table(df, metric, fmt, same_threshold, include_dataset=True))
            sections.append("")
        elif dataset_keys:
            for dataset, ds_df in df.groupby("Dataset", sort=True):
                sections.append(f"#### Dataset={dataset}, Correlation={metric}\n")
                sections.append(_pivot_measures_table(ds_df, metric, fmt, same_threshold))
                sections.append("")
        else:
            sections.append(f"#### Correlation={metric}\n")
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
        df["_measure_col"] = df["Dataset"] + " / " + df["TruthMeasure"] + " / " + df["EvalMeasure"]
    else:
        df["_measure_col"] = df["TruthMeasure"] + " / " + df["EvalMeasure"]

    # Get row identity columns (Judge + any category cols)
    meta = get_meta_columns(df)
    row_id_cols = [c for c in meta if c not in ["Dataset", "TruthMeasure", "EvalMeasure", "_measure_col"]]

    # Pivot
    pivoted = df.pivot_table(
        index=row_id_cols,
        columns="_measure_col",
        values=metric,
        aggfunc="first",
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
        "font.size": 9,
        "axes.linewidth": 0,
        "axes.axisbelow": True,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.edgecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.color": "0.4",
        "ytick.color": "0.4",
        "text.color": "0.3",
        "axes.labelcolor": "0.3",
        "figure.dpi": 300,
        "figure.facecolor": "white",
    })


def plot_grouped_bar(
    df: pd.DataFrame,
    group_cols: list[str],
    series_col: str,
    title: str,
    out_path: Path,
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

    x = np.arange(n_groups)
    total_width = 0.75
    bar_width = total_width / n_series

    fig, ax = plt.subplots(figsize=(max(5, n_groups * 1.5), 3.2))

    for i, (_, row) in enumerate(df.iterrows()):
        judge = row[series_col]
        color, hatch = style_map.get(judge, ("#999999", ""))
        values = [row[c] for c in group_cols]
        offset = (i - n_series / 2 + 0.5) * bar_width
        edge = ("white" if _is_dark(color) else "0.3") if hatch else color
        ax.bar(
            x + offset, values, bar_width * 0.75,
            color=color,
            hatch=hatch,
            edgecolor=edge,
            linewidth=0.3,
            label=judge,
        )

    # Y-axis: light horizontal grid lines, ticks at 0.25 intervals
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.yaxis.grid(True, color="0.85", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_ylim(0, 1.05)

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(group_cols, rotation=0, ha="center", fontsize=8, color="0.4")

    # Labels and title
    ax.set_ylabel(ylabel, fontsize=9, color="0.4")
    ax.set_title(title, fontsize=10, color="0.3", loc="left", pad=10)

    # Legend outside right, no frame
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
        handlelength=1.2, handleheight=0.8,
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
    color_map: dict[str, str] | None = None,
    hatch_map: dict[str, str] | None = None,
):
    """One plot per (Dataset, TruthMeasure, EvalMeasure)."""
    group_keys = [c for c in ["Dataset", "TruthMeasure", "EvalMeasure"] if c in df.columns]

    for group_vals, group_df in df.groupby(group_keys, sort=True):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        label_parts = [f"{k}={v}" for k, v in zip(group_keys, group_vals)]
        title = ", ".join(label_parts)
        slug = _slugify("_".join(str(v) for v in group_vals))

        avail = [c for c in metric_cols if c in group_df.columns]
        plot_grouped_bar(group_df, avail, "Judge", title, plot_dir / f"{slug}.pdf",
                         color_map=color_map, hatch_map=hatch_map)


def plot_measures_as_columns(
    df: pd.DataFrame,
    metric_cols: list[str],
    plot_dir: Path,
    color_map: dict[str, str] | None = None,
    hatch_map: dict[str, str] | None = None,
):
    """One plot per (Dataset, correlation metric)."""
    dataset_keys = ["Dataset"] if "Dataset" in df.columns else []

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        def _plot_one(sub_df: pd.DataFrame, title: str, slug: str):
            sub_df = sub_df.copy()
            sub_df["_measure_col"] = sub_df["TruthMeasure"] + " / " + sub_df["EvalMeasure"]
            meta = get_meta_columns(sub_df)
            row_id_cols = [c for c in meta if c not in ["Dataset", "TruthMeasure", "EvalMeasure", "_measure_col"]]

            pivoted = sub_df.pivot_table(
                index=row_id_cols, columns="_measure_col", values=metric, aggfunc="first",
            ).reset_index()
            pivoted.columns.name = None

            judge_order = {j: i for i, j in enumerate(sub_df["Judge"].unique())}
            pivoted["_sort"] = pivoted["Judge"].map(lambda j: judge_order.get(j, len(judge_order)))
            pivoted = pivoted.sort_values("_sort").drop(columns=["_sort"])

            measure_cols = [c for c in pivoted.columns if c not in row_id_cols]
            plot_grouped_bar(pivoted, measure_cols, "Judge", title, plot_dir / f"{slug}.pdf", ylabel=metric,
                             color_map=color_map, hatch_map=hatch_map)

        if dataset_keys:
            for dataset, ds_df in df.groupby("Dataset", sort=True):
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
    help="Produce one summary table across all datasets/measures.",
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
            plot_measures_as_columns(df, metric_cols, plot_dir,
                                     color_map=color_map, hatch_map=hatch_map)
    else:
        result = correlation_consistency(df, metric_cols, format, same_threshold=same, summary=summary)
        if plot_dir:
            plot_correlation_consistency(df, metric_cols, plot_dir,
                                         color_map=color_map, hatch_map=hatch_map)

    if plot_dir:
        click.echo(f"Plots written to {plot_dir}/", err=True)

    # Append category comparison if judges YAML provides categories
    if judges_config:
        result += "\n\n---\n\n# Category Comparison\n\n"
        result += category_comparison(df, judges_config.judges, metric_cols, format, same_threshold=same)

    # Output
    if output:
        output.write_text(result)
        click.echo(f"Wrote to {output}", err=True)
    else:
        click.echo(result)

    return 0


if __name__ == "__main__":
    main()
