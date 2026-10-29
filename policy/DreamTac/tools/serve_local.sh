#!/usr/bin/env bash
# Run the Dream-Tac inference server on this workstation instead of the cluster.
#
# Reads the files tools/copy_checkpoint.sh pulled into ~/models, and needs one
# more thing that copy_checkpoint.sh does not fetch: the Cosmos base weights
# model-480p-16fps.pt under $HF_HOME. The experiment config resolves that
# hf:// URI at import time, so without it the server dies before loading our
# checkpoint even though it never reads those weights. The repo is gated; the
# blob was copied from the cluster's $WORK/hf.
#
# Serves on 0.0.0.0:5001, the same port tools/tunnel.sh forwards to, so stop
# any tunnel before starting this.
#
# Usage:  bash tools/serve_local.sh
#         DREAMTAC_TASK=insert_HDMI DREAMTAC_ITER=iter_000000400 bash tools/serve_local.sh
set -euo pipefail

TASK="${DREAMTAC_TASK:-insert_hole}"
ITER="${DREAMTAC_ITER:-iter_000004000}"
PORT=5001  # hard-coded in franka_server.py SERVICE_CONFIG
REPO="${DREAMTAC_REPO:-$HOME/proj/Dream-Tac}"
ROOT="${DREAMTAC_LOCAL_ROOT:-$HOME/models/dream-tac}"

export HF_HOME="${HF_HOME:-$HOME/models/hf}"
export HF_HUB_OFFLINE=1
export UNIVTAC_TASKS="$TASK"
export UNIVTAC_CACHE_DIR="$ROOT/cache/univtac"
export COSMOS_TACTILE_SELF_ATTN_BACKEND=flashbias_sdpa
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

export FRANKA_COSMOS_CONFIG=cosmos_predict2_2b_480p_univtac
export FRANKA_COSMOS_CONFIG_FILE=cosmos_policy/config/config.py
export FRANKA_COSMOS_CKPT="$ROOT/runs/$TASK/checkpoints/$ITER/model"
export FRANKA_DATASET_STATS_PATH="$UNIVTAC_CACHE_DIR/dataset_statistics_univtac_$TASK.json"
export FRANKA_T5_EMBEDDINGS_PATH="$UNIVTAC_CACHE_DIR/t5_embeddings.pkl"
export FRANKA_CHUNK_SIZE=20
export FRANKA_NUM_DENOISING_STEPS=5
export FRANKA_USE_TACTILE=1

BASE="$HF_HOME/hub/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots"
for f in "$FRANKA_COSMOS_CKPT/.metadata" "$FRANKA_DATASET_STATS_PATH" "$FRANKA_T5_EMBEDDINGS_PATH" \
         "$BASE"/*/model-480p-16fps.pt "$BASE"/*/tokenizer/tokenizer.pth; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 3; }
done

if ss -ltnH "sport = :$PORT" | grep -q .; then
    echo "port $PORT is already in use (a tools/tunnel.sh forward?):" >&2
    ss -ltnpH "sport = :$PORT" >&2
    exit 3
fi

cd "$REPO"
# transformer_engine dlopens libnvrtc and friends; point it at the venv's CUDA libs.
for d in "$REPO"/cosmos_policy/.venv/lib/python3.10/site-packages/nvidia/*/lib; do
    [ -d "$d" ] && LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export LD_LIBRARY_PATH

echo "serving $TASK/$ITER on :$PORT  (checkpoint $FRANKA_COSMOS_CKPT)"
exec uv run --frozen --no-sync --project cosmos_policy --extra cu128 --group franka \
    python -m cosmos_policy.experiments.robot.franka.franka_server
