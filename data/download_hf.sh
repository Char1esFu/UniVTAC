#!/bin/bash
# Download the Isaac Sim 4.5 UniVTAC release from HuggingFace (byml/UniVTAC)
# and lay it out exactly where the training / eval code reads it:
#
#   isaac45/<task>/hdf5/*.hdf5, metadata.json -> data/<task>/clean/
#   checkpoints/encoder.pth                    -> encoder/checkpoints/resnet18/20251128-125750/best.pth
#   checkpoints/<task>/<variant>/              -> policy/ACT/act_ckpt/act-<task>/clean-100/<variant>/
#
# Files are moved, not linked. isaac51/ and contact/ are never downloaded.
# Anything already in place is skipped, so the script is safe to re-run.
#
# Usage: bash data/download_hf.sh <local|cluster> [task ...]   (default: all 8 tasks)
#   local   : everything goes into this repo (data/, encoder/checkpoints, policy/ACT/act_ckpt)
#   cluster : large files go to $WORK/UniVTAC (or $UNIVTAC_STORAGE_DIR) and are symlinked into the repo
set -e

MODE="$1"
case "$MODE" in
    local|cluster) shift ;;
    *) echo "Usage: bash data/download_hf.sh <local|cluster> [task ...]" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
ALL_TASKS="grasp_classify insert_HDMI insert_hole insert_tube lift_bottle lift_can pull_out_key put_bottle_in_shelf"
TASKS="${*:-$ALL_TASKS}"

if [ "$MODE" = "cluster" ]; then
    # All large files live on the cluster work filesystem; the repo only holds
    # symlinks into it. Override with UNIVTAC_STORAGE_DIR if needed.
    STORAGE_ROOT="${UNIVTAC_STORAGE_DIR:-${WORK:-/home/atuin/g107ea/g107ea11}/UniVTAC}"
    STORAGE_DIR="$STORAGE_ROOT/data"
    ENCODER_CKPT_STORE="$STORAGE_ROOT/encoder_checkpoints"
    POLICY_CKPT_STORE="$STORAGE_ROOT/act_ckpt"
    mkdir -p "$STORAGE_DIR" "$ENCODER_CKPT_STORE" "$POLICY_CKPT_STORE"

    # Keep the HF download cache off the home quota as well.
    export HF_HOME="${HF_HOME:-$(dirname "$STORAGE_ROOT")/.cache/huggingface}"

    # Checkpoint dirs are whole-directory symlinks so future training outputs
    # also land on the work filesystem.
    ln -sfn "$ENCODER_CKPT_STORE" "$REPO_ROOT/encoder/checkpoints"
    ln -sfn "$POLICY_CKPT_STORE" "$REPO_ROOT/policy/ACT/act_ckpt"

    ENCODER_CKPT_DIR="$ENCODER_CKPT_STORE/resnet18/20251128-125750"
else
    # Local mode: everything stays in the repo.
    STORAGE_DIR="$DATA_DIR"
    ENCODER_CKPT_DIR="$REPO_ROOT/encoder/checkpoints/resnet18/20251128-125750"
    POLICY_CKPT_STORE="$REPO_ROOT/policy/ACT/act_ckpt"
fi

# --- Decide what is still missing ---
PATTERNS=()
for task in $TASKS; do
    case " $ALL_TASKS " in *" $task "*) ;; *) echo "Unknown task: $task" >&2; exit 2 ;; esac
    if [ -f "$STORAGE_DIR/$task/clean/metadata.json" ]; then
        echo "    skip data $task (already in $STORAGE_DIR/$task/clean)"
    else
        PATTERNS+=("isaac45/$task/hdf5/*.hdf5" "isaac45/$task/metadata.json")
    fi
    if [ -f "$POLICY_CKPT_STORE/act-$task/clean-100/univtac/policy_last.ckpt" ]; then
        echo "    skip ACT checkpoints $task"
    else
        PATTERNS+=("checkpoints/$task/*")
    fi
done
if [ -f "$ENCODER_CKPT_DIR/best.pth" ]; then
    echo "    skip encoder checkpoint"
else
    PATTERNS+=("checkpoints/encoder.pth")
fi

if [ ${#PATTERNS[@]} -eq 0 ]; then
    echo "Nothing to download."
else
    # Install huggingface_hub if not available
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        python3 -m pip install "huggingface_hub"
    fi

    echo "==> Downloading Isaac Sim 4.5 data from HuggingFace to $STORAGE_DIR ..."
    printf '    %s\n' "${PATTERNS[@]}"
    STORAGE_DIR="$STORAGE_DIR" python3 - "${PATTERNS[@]}" <<'EOF'
import os, sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="byml/UniVTAC",
    repo_type="dataset",
    local_dir=os.environ["STORAGE_DIR"],
    allow_patterns=sys.argv[1:],
)
EOF
fi

# --- Task data: isaac45/<task>/{hdf5/*.hdf5,metadata.json} -> <task>/clean/ ---
for task in $TASKS; do
    src="$STORAGE_DIR/isaac45/$task"
    [ -d "$src" ] || continue
    dest="$STORAGE_DIR/$task/clean"
    mkdir -p "$dest"
    find "$src/hdf5" -maxdepth 1 -name '*.hdf5' -exec mv -t "$dest" {} +
    mv "$src/metadata.json" "$dest/"
    echo "    $dest ($(find "$dest" -maxdepth 1 -name '*.hdf5' | wc -l) episodes)"
done
rm -rf "$STORAGE_DIR/isaac45" "$STORAGE_DIR/.cache/huggingface/download/isaac45"

# --- Encoder checkpoint ---
# checkpoints/encoder.pth -> encoder/checkpoints/resnet18/20251128-125750/best.pth
if [ -f "$STORAGE_DIR/checkpoints/encoder.pth" ]; then
    mkdir -p "$ENCODER_CKPT_DIR"
    mv "$STORAGE_DIR/checkpoints/encoder.pth" "$ENCODER_CKPT_DIR/best.pth"
    echo "    $ENCODER_CKPT_DIR/best.pth"
fi

# --- Policy checkpoints ---
# checkpoints/{task}/{variant}/ -> policy/ACT/act_ckpt/act-{task}/clean-100/{variant}/
# Use with: EP_NUM=100 TRAIN_CONFIG=univtac  (or vision_only)
for variant_dir in "$STORAGE_DIR/checkpoints"/*/*/; do
    [ -d "$variant_dir" ] || continue
    task_name=$(basename "$(dirname "$variant_dir")")
    dest="$POLICY_CKPT_STORE/act-$task_name/clean-100/$(basename "$variant_dir")"
    mkdir -p "$dest"
    mv "$variant_dir"* "$dest/"
    echo "    $dest"
done
rm -rf "$STORAGE_DIR/checkpoints" "$STORAGE_DIR/.cache/huggingface/download/checkpoints"

# --- Link dataset into the repo (cluster only) ---
# Each data/<task> is symlinked from the work filesystem so nothing large lands
# in home. In local mode $STORAGE_DIR is already $DATA_DIR, so nothing to link.
if [ "$MODE" = "cluster" ]; then
    for task in $TASKS; do
        [ -d "$STORAGE_DIR/$task" ] || continue
        ln -sfn "$STORAGE_DIR/$task" "$DATA_DIR/$task"
        echo "    $DATA_DIR/$task -> $STORAGE_DIR/$task"
    done
fi

echo ""
echo "Done. Train / process with task_config 'clean', e.g.:"
echo "  cd policy/ACT && python process_data.py <task> clean 100"
echo "To evaluate the released ACT checkpoints:"
echo "  EP_NUM=100 TRAIN_CONFIG=univtac bash eval_policy.sh <task> clean ACT/deploy_policy <gpu>"
