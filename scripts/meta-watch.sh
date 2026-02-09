#!/bin/bash
#
# meta-watch.sh - Sync, meta-evaluate, and publish leaderboard correlations
#

set -euo pipefail


# === CONFIGURE THESE ===
TRUTH="$1" # auto-judge/truth-eval" # Change this to directory with official leaderboards

# Rsync source and destination (parent directories)
RSYNC_SRC="$2" # "server:/auto-judge/in"  # Change this where auto-judge results are collected
RSYNC_DEST="$3" # "server:/auto-judge/out/"  # Change this to where correlation result should get written to


# Local directories
WATCH_PARENT="./eval-watch/in"
OUT_PARENT="./eval-watch/out"

mkdir -p "$WATCH_PARENT" "$OUT_PARENT"

# Shared options
# COMMON_OPTS=" --only-shared-runs   --truth-drop-aggregate --out-format jsonl  --only-shared-topics  --on-missing default"
COMMON_OPTS="   --truth-drop-aggregate --out-format jsonl   --on-missing default"



CORR_OPTS=" --correlation kendall  --correlation pearson --correlation spearman  --correlation kendall --correlation tauap_b --correlation kendall@15   --correlation spearman@15 --correlation kendall@5   --correlation spearman@5"
# === END CONFIG ===

mkdir -p "$WATCH_PARENT" "$OUT_PARENT"
mkdir -p "$OUT_PARENT/ragtime" "$OUT_PARENT/rag" "$OUT_PARENT/rag-auggen" "$OUT_PARENT/dragun"

TS=$(date +%F_%H-%M-%S)

# Sync incoming, check for new files
CHANGE_COUNT=$(rsync -Laurai "$RSYNC_SRC/" "$WATCH_PARENT/" 2>/dev/null | wc -l)
echo "change count $CHANGE_COUNT"

if [ "$CHANGE_COUNT" -gt 0 ]; then
    LOG="$OUT_PARENT/log/correlation-$TS.log"
    mkdir -p "$OUT_PARENT/log"

    echo "Logging to $LOG"

    set -x

    VAR_OPTS="--all-topics --only-shared-runs"

    auto-judge-evaluate meta-evaluate  --truth-leaderboard $TRUTH/ragtime-export/eval/ragtime.repgen.official.eval.jsonl --truth-format jsonl  \
    --eval-format tot  -i $WATCH_PARENT/ragtime/*eval.txt \
    $CORR_OPTS $COMMON_OPTS $VAR_OPTS \
    --truth-measure nugget_coverage --truth-measure nugget_coverage_weighted  --truth-measure correct_nuggets  --truth-measure correct_nuggets_weighted \
    --truth-measure f1 --truth-measure f1_weighted \
    --truth-measure supporting_citations --truth-measure relevant_citations --truth-measure sentence_support --truth-measure correctly_cited_sentences  \
    --truth-measure character_count --truth-measure sentences \
    --output $OUT_PARENT/ragtime/correlations-$TS.jsonl  >> $LOG 2>&1


    auto-judge-evaluate meta-evaluate  --truth-leaderboard $TRUTH/rag-export/eval/rag.generation.official.eval.jsonl --truth-format jsonl  \
    --eval-format tot  -i $WATCH_PARENT/rag/*eval.txt \
    $CORR_OPTS $COMMON_OPTS  $VAR_OPTS \
    --output $OUT_PARENT/rag/correlations-$TS.jsonl >> $LOG  2>&1

    auto-judge-evaluate meta-evaluate  --truth-leaderboard $TRUTH/rag-export/eval/rag.auggen.official.eval.jsonl --truth-format jsonl  \
    --eval-format tot  -i $WATCH_PARENT/rag/*eval.txt \
    $CORR_OPTS $COMMON_OPTS $VAR_OPTS  \
    --output $OUT_PARENT/rag-auggen/correlations-$TS.jsonl   >> $LOG 2>&1

    auto-judge-evaluate meta-evaluate  --truth-leaderboard $TRUTH/dragun-export/eval/dragun.repgen.official.eval.jsonl --truth-format jsonl  \
    --eval-format tot  -i $WATCH_PARENT/dragun/*eval.txt \
    $CORR_OPTS $COMMON_OPTS  $VAR_OPTS \
    --output $OUT_PARENT/dragun/correlations-$TS.jsonl  >> $LOG 2>&1


    # Publish results
    rsync -Laura "$OUT_PARENT/" "$RSYNC_DEST/" # 2>/dev/null || true
# else echo "no change"
fi
