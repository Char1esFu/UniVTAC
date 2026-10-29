#!/usr/bin/env bash
# Stage one Dream-Tac checkpoint (and the files it needs) where serve_local.sh
# looks for it.
#
# The source is any directory with the training-side layout, i.e. what
# $WORK/Dream-Tac looked like on the training machine:
#
#   $SRC/runs/univtac/cosmos_policy_univtac/cosmos_v2_finetune_<TASK>/
#       cosmos_predict2_2b_480p_univtac/{config.yaml, checkpoints/<ITER>/model/}
#   $SRC/cache/univtac/{dataset_statistics_univtac_<TASK>.json, t5_embeddings.pkl}
#
# Defaults to the backup in ~/cluster_backup/models/Dream-Tac. A newly trained
# model is staged the same way once its run directory is copied there (or point
# DREAMTAC_SRC at wherever it landed).
#
# <TASK> is the training task set exactly as Dream-Tac spells it: a single task
# (insert_hole) or several joined by "__" in sorted order (insert_hole__lift_can).
#
# Only `model/` is copied: it is all inference reads (3.7 GB); optim/,
# scheduler/ and trainer/ exist to resume training. There is no "best"
# checkpoint -- validation is disabled in every Dream-Tac experiment -- so the
# default ITER is the last one; stage others alongside to compare with
# tools/client_check.py.
#
# Not copied, needed once per machine: ~/proj/Dream-Tac (the source) and
# $HF_HOME (~/models/hf) with Cosmos-Predict2-2B-Video2World's tokenizer.pth and
# model-480p-16fps.pt. serve_local.sh checks for all of them.
#
# Usage:  DREAMTAC_TASK=insert_hole DREAMTAC_ITERS="iter_000004000" bash tools/copy_checkpoint.sh [dest]
set -euo pipefail

SRC="${DREAMTAC_SRC:-$HOME/cluster_backup/models/Dream-Tac}"
TASK="${DREAMTAC_TASK:-insert_hole}"
ITERS="${DREAMTAC_ITERS:-iter_000004000}"
DEST="${1:-${DREAMTAC_LOCAL_ROOT:-$HOME/models/dream-tac}}"

RUN="$SRC/runs/univtac/cosmos_policy_univtac/cosmos_v2_finetune_$TASK/cosmos_predict2_2b_480p_univtac"
CACHE="$SRC/cache/univtac"
STATS="$CACHE/dataset_statistics_univtac_$TASK.json"

for f in "$STATS" "$CACHE/t5_embeddings.pkl"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 3; }
done
for iter in $ITERS; do
    [ -e "$RUN/checkpoints/$iter/model/.metadata" ] || {
        echo "missing: $RUN/checkpoints/$iter/model/.metadata" >&2
        echo "available: $(ls "$RUN/checkpoints" 2>/dev/null | grep '^iter_' | tr '\n' ' ')" >&2
        exit 3
    }
done

LOCAL_RUN="$DEST/runs/$TASK"
LOCAL_CACHE="$DEST/cache/univtac"
mkdir -p "$LOCAL_RUN/checkpoints" "$LOCAL_CACHE"

echo "source      : $SRC"
echo "destination : $DEST"

rsync -ah --info=progress2 "$STATS" "$CACHE/t5_embeddings.pkl" "$LOCAL_CACHE/"
rsync -ah "$RUN/config.yaml" "$LOCAL_RUN/" || true   # reference only

for iter in $ITERS; do
    echo "--- $iter (model/ only) ---"
    mkdir -p "$LOCAL_RUN/checkpoints/$iter"
    rsync -ah --info=progress2 "$RUN/checkpoints/$iter/model/" "$LOCAL_RUN/checkpoints/$iter/model/"
done

cat <<EOF

Done. Serve it with:

  DREAMTAC_TASK=$TASK DREAMTAC_ITER=${ITERS%% *} bash policy/DreamTac/tools/serve_local.sh
EOF
