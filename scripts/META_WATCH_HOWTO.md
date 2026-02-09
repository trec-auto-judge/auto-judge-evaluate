# Meta-Watch: Evaluation Watchdog

A simple watchdog script that monitors for new evaluation files, runs meta-evaluate correlations, and publishes results.

> **Note:** This tool will be replaced with TIRA integration in the future.

## Overview

`meta-watch.sh` uses `rsync` to detect new files, runs `auto-judge-evaluate meta-evaluate` for multiple datasets, and syncs results back to a destination. It's designed to work with the `watch` command for periodic polling.

## Usage

```bash
./meta-watch.sh <TRUTH_DIR> <RSYNC_SRC> <RSYNC_DEST>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `$1` - TRUTH_DIR | Directory containing official leaderboards (truth data) |
| `$2` - RSYNC_SRC | Remote or local path where auto-judge results are collected |
| `$3` - RSYNC_DEST | Remote or local path where correlation results should be written |

## Example

```bash
# Run once
./meta-watch.sh /data/truth server:/auto-judge/in server:/auto-judge/out

# Run periodically with watch (every 60 seconds)
watch -n 60 ./meta-watch.sh /data/truth server:/auto-judge/in server:/auto-judge/out
```

### Expected Directory Structure

**TRUTH_DIR** should contain official evaluation files, such as:
```
/data/truth/
  ragtime-export/eval/ragtime.repgen.official.eval.jsonl
  rag-export/eval/rag.generation.official.eval.jsonl
  rag-export/eval/rag.auggen.official.eval.jsonl
  dragun-export/eval/dragun.repgen.official.eval.jsonl
```
Other formats, such as `ir_measures` or `trec_eval` can be configured in the script.

**RSYNC_SRC** receives incoming evaluation files:
```
server:/auto-judge/in/
  ragtime/*eval.txt
  rag/*eval.txt
  dragun/*eval.txt
```

Evaluation files are expected to be in `tot` format, but other formats (such as `ir_measures`) can be configured in the script.


**RSYNC_DEST** receives correlation results:
```
server:/auto-judge/out/
  ragtime/correlations-<timestamp>.jsonl
  rag/correlations-<timestamp>.jsonl
  rag-auggen/correlations-<timestamp>.jsonl
  dragun/correlations-<timestamp>.jsonl
  log/correlation-<timestamp>.log
```

The output format is jsonl produced by pandas. It is designed to be read with `pandas` or `duckdb`.

It is possible to configure the script to produce a table/TSV output.


## How It Works

1. **Change Detection**: Uses `rsync -Laurai` to sync incoming files and count changes
2. **Conditional Execution**: Only runs meta-evaluate if new files were detected
3. **Multi-Dataset**: Evaluates ragtime, rag, rag-auggen, and dragun datasets
4. **Result Publishing**: Syncs correlation results back to the destination

## Under the Hood

- **rsync**: Handles file synchronization and change detection
- **auto-judge-evaluate meta-evaluate**: Computes correlations between predicted and ground-truth leaderboards
- **watch**: External tool for periodic execution (not part of the script)

## Correlation Methods

The script computes multiple correlation metrics:
- `kendall`, `pearson`, `spearman`
- `tauap_b`
- `kendall@15`, `spearman@15`
- `kendall@5`, `spearman@5`

## Configuration

Edit the script to modify:
- `WATCH_PARENT` / `OUT_PARENT`: Local staging directories (default: `./eval-watch/in` and `./eval-watch/out`)
- `COMMON_OPTS`: Shared meta-evaluate options
- `CORR_OPTS`: Correlation methods to compute
- `VAR_OPTS`: Per-run variable options (topics, runs filtering)

## Troubleshooting

**No changes detected**: The script only runs if rsync detects new files. Check that files are being written to RSYNC_SRC.

**Permission errors**: Ensure rsync can access both source and destination paths.

**Logs**: Check `$OUT_PARENT/log/correlation-<timestamp>.log` for detailed output.
