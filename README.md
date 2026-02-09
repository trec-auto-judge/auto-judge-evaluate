# autojudge-evaluate

Evaluation tools for the TREC AutoJudge framework. Computes leaderboard correlations, inter-annotator agreement on qrels, leaderboard statistics, and format conversion for evaluation result files.

## Installation

```bash
uv pip install autojudge-evaluate
```

## CLI Commands

All commands are available via `auto-judge-evaluate <command>`.

---

### `meta-evaluate` — Leaderboard correlation

Correlate predicted leaderboards against a ground-truth leaderboard.

```bash
auto-judge-evaluate meta-evaluate \
    --truth-leaderboard truth.eval.jsonl --truth-format jsonl \
    --eval-format tot -i results/*eval.txt \
    --correlation kendall --correlation spearman --correlation tauap_b \
    --truth-measure nugget_coverage --truth-measure f1 \
    --on-missing default \
    --output correlations.jsonl
```

**Key options:**

| Option | Description |
|--------|-------------|
| `--truth-leaderboard FILE` | Ground-truth leaderboard file (required) |
| `--truth-format FMT` | Format: `trec_eval`, `tot`, `ir_measures`, `ranking`, `jsonl` |
| `--eval-format FMT` | Format of input leaderboard files |
| `-i FILE` / positional | Input leaderboard file(s), supports globs. Repeatable |
| `--correlation METHOD` | Correlation method. Repeatable. Supports `kendall`, `pearson`, `spearman`, `tauap_b`, and top-k variants like `kendall@15` |
| `--truth-measure NAME` | Truth measure(s) to correlate against. Repeatable. Omit for all |
| `--eval-measure NAME` | Eval measure(s) to include. Repeatable. Omit for all |
| `--on-missing MODE` | Handle run mismatches: `error`, `warn`, `skip`, `default` (fill 0.0) |
| `--only-shared-topics` | Intersect topics across truth and eval (default: `--all-topics`) |
| `--only-shared-runs` | Intersect runs across truth and eval (default: `--all-runs`) |
| `--truth-drop-aggregate` | Recompute aggregates from per-topic data |
| `--output FILE` | Output `.jsonl` or `.txt` |
| `--out-format FMT` | `jsonl` (default) or `table` |
| `--aggregate` | Report only mean across all judges |

**Output:** One row per (Judge, TruthMeasure, EvalMeasure) with correlation values as columns.

---

### `qrel-evaluate` — Inter-annotator agreement on qrels

Compare predicted relevance judgments (qrels) against truth qrels. Computes set overlap (precision, recall, F1) and agreement metrics (Cohen's Kappa, Krippendorff's Alpha, Jaccard, ARI).

```bash
auto-judge-evaluate qrel-evaluate \
    --truth-qrels official.qrels \
    --predict-qrels predicted.qrels
```



**Key options:**

| Option | Description |
|--------|-------------|
| `--truth-qrels FILE` | Truth qrels in TREC format |
| `--truth-nugget-docs DIR` | Alternative: truth as nugget-docs directory |
| `--predict-qrels FILE` | Predicted qrels in TREC format |
| `--predict-nugget-docs DIR` | Alternative: predicted as nugget-docs directory |
| `--truth-max-grade N` | Grade scale upper bound for truth (default: 1 = binary) |
| `--predict-max-grade N` | Grade scale upper bound for predicted (default: 1) |
| `--truth-relevance-threshold N` | Binary threshold for truth side (default: 1) |
| `--predict-relevance-threshold N` | Binary threshold for predicted side (default: 1) |
| `--on-missing MODE` | Handle topics in only one side: `error`, `warn`, `default`, `skip` |
| `--output FILE` | Output `.jsonl` or `.txt` |

**Output:** Per-topic table with Precision, Recall, F1, Jaccard, Kappa, Krippendorff's Alpha, ARI, plus a MEAN row.

---

### `leaderboard` — Leaderboard statistics

Compute per-run statistics (mean, stderr, stdev, min, max) from leaderboard files.

```bash
auto-judge-evaluate leaderboard \
    --eval-format tot -i results/*eval.txt --sort
```

**Key options:**

| Option | Description |
|--------|-------------|
| `--eval-format FMT` | Input format (required) |
| `-i FILE` / positional | Input file(s), supports globs. Repeatable |
| `--eval-measure NAME` | Filter to specific measures. Repeatable |
| `--sort` | Sort runs by mean score (descending) |
| `--output FILE` | Output `.jsonl` or `.csv` |

**Output:** One row per (Judge, RunID, Measure) with Topics, Mean, Stderr, Stdev, Min, Max.

---

## Analysis Module

Post-hoc analysis, tables, plots of `meta-evaluate` output. Produces correlation tables and bar plots with judge categorization.

```bash
python -m autojudge_evaluate.analysis.correlation_table \
    -d ragtime:ragtime-correlations.jsonl \
    -d rag:rag-correlations.jsonl \
    -d dragun:dragun-correlations.jsonl \
    --judges judges.yml \
    --correlation kendall \
    --truth-measure nugget_coverage \
    --format latex \
    --plot-dir plots/
```

**Judge configuration** (`judges.yml`) maps cryptic filenames to display names and categories, with optional plot styling:

```yaml
styles:
  colors:
    pointwise: "#4A90D9"
    pairwise:  "#D94A4A"
  hatches:
    gpt-4o:    ""
    llama-3:   "//"

judges:
  my-judge-A.eval:
    name: System A
    method: pointwise     # category column
    model: gpt-4o         # category column
  my-judge-B.eval:
    name: System B
    method: pairwise
    model: llama-3
```

- **`styles.colors`**: maps category values to fill colors (any matplotlib color string)
- **`styles.hatches`**: maps category values to hatch patterns (`//`, `..`, `xx`, `\\`, etc). Values combine across categories.
- Color is picked from the first matching category value; hatches are combined from all matches.
- Without a `styles:` section, bars use a sequential grayscale fallback.
- Judges not in the YAML are excluded unless `--all-judges` is passed.

**Key options:** `--format` (github, latex, tsv, plain, html, pipe), `--columns` (correlations or measures), `--summary` (add mean/max rows), `--aggregate` (aggregate across datasets), `--same THRESHOLD` (highlight near-equal values).


---

### `eval-result` — Format conversion and verification

Clean and convert evaluation result files. 

```bash
# Convert tot to jsonl
auto-judge-evaluate eval-result data.txt -if tot -of jsonl -o data.jsonl

# Filter to specific runs and topics
auto-judge-evaluate eval-result data.txt -if tot -of jsonl -o filtered.jsonl \
    --filter-runs system_A --filter-runs system_B \
    --filter-topics topic_1
```

**Key options:**

| Option | Description |
|--------|-------------|
| `-if FMT` | Input format: `trec_eval`, `tot`, `ir_measures`, `ranking`, `jsonl` |
| `-of FMT` | Output format (defaults to input format) |
| `-o FILE` | Output file. Omit for roundtrip test to temp file |
| `--filter-runs ID` | Keep only these runs. Repeatable |
| `--filter-topics ID` | Keep only these topics. Repeatable |
| `--filter-measures NAME` | Keep only these measures. Repeatable |
| `--compare-aggregates` | Compare file aggregates vs recomputed from per-topic data |
| `--drop-aggregates` | Drop existing aggregate rows |
| `--recompute-aggregates` | Recompute from per-topic data (implies `--drop-aggregates`) |
| `--roundtrip` / `--no-roundtrip` | Enable/disable roundtrip verification (default: on) |

**Supported formats:**

| Format | Columns |
|--------|---------|
| `trec_eval` | measure topic value (3 cols, run_id from filename) |
| `tot` | run measure topic value (4 cols) |
| `ir_measures` | run topic measure value (4 cols) |
| `ranking` | topic Q0 doc_id rank score run (6 cols) |
| `jsonl` | JSON lines with run_id, topic_id, measure, value |