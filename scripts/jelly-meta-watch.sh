#!/bin/bash

TRUTH="/home/dietz/trec-auto-judge/datacleaned2"

RSYNC_SRC="jelly:/mnt/cherries/share/auto-judge-evaluation/in/"
RSYNC_DEST="jelly:/mnt/cherries/share/auto-judge-evaluation/out/"

bash ./scripts/meta-watch.sh $TRUTH $RSYNC_SRC $RSYNC_DEST
