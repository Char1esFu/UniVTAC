#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"

# Cluster mode stores all large files on the work filesystem and symlinks them
# into the repo; local mode keeps everything inside the repo. Auto-detected from
# $WORK, override explicitly with UNIVTAC_CLUSTER=1 / UNIVTAC_CLUSTER=0.
CLUSTER="${UNIVTAC_CLUSTER:-$([ -n "$WORK" ] && echo 1 || echo 0)}"

if [ "$CLUSTER" = "1" ]; then
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
    # Local mode: everything stays in the repo (original upstream layout).
    STORAGE_DIR="$DATA_DIR"
    ENCODER_CKPT_DIR="$REPO_ROOT/encoder/checkpoints/resnet18/20251128-125750"
    POLICY_CKPT_STORE="$REPO_ROOT/policy/ACT/act_ckpt"
fi

# Install huggingface_hub if not available
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 -m pip install "huggingface_hub"
fi

echo "==> Downloading UniVTAC dataset from HuggingFace (~131 GB) to $STORAGE_DIR ..."
STORAGE_DIR="$STORAGE_DIR" python3 - <<'EOF'
from huggingface_hub import snapshot_download
import os

data_dir = os.environ["STORAGE_DIR"]
snapshot_download(
    repo_id="byml/UniVTAC",
    repo_type="dataset",
    local_dir=data_dir,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.gitattributes"],
)
print(f"Dataset downloaded to {data_dir}")
EOF

# --- Encoder checkpoint ---
# checkpoints/encoder.pth -> encoder/checkpoints/resnet18/20251128-125750/best.pth
echo "==> Setting up encoder checkpoint..."
mkdir -p "$ENCODER_CKPT_DIR"
mv "$STORAGE_DIR/checkpoints/encoder.pth" "$ENCODER_CKPT_DIR/best.pth"
echo "    $ENCODER_CKPT_DIR/best.pth"

# --- Policy checkpoints ---
# checkpoints/{task}/{variant}/ -> policy/ACT/act_ckpt/act-{task}/clean-100/{variant}/
# Use with: EP_NUM=100 TRAIN_CONFIG=univtac  (or vision_only)
echo "==> Setting up policy checkpoints..."
for task_dir in "$STORAGE_DIR/checkpoints"/*/; do
    task_name=$(basename "$task_dir")
    for variant_dir in "$task_dir"*/; do
        [ -d "$variant_dir" ] || continue
        variant=$(basename "$variant_dir")
        dest="$POLICY_CKPT_STORE/act-$task_name/clean-100/$variant"
        mkdir -p "$dest"
        mv "$variant_dir"* "$dest/"
        echo "    $dest"
    done
done

# Remove the emptied checkpoint source now that everything has been moved
# into the paths used by evaluation.
echo "==> Cleaning downloaded checkpoint source..."
rm -rf "$STORAGE_DIR/checkpoints"
echo "    Removed $STORAGE_DIR/checkpoints"

# --- Link dataset into the repo (cluster only) ---
# Everything remaining in $STORAGE_DIR is symlinked into data/ so code that
# expects data/... paths keeps working without copying anything into home.
# In local mode $STORAGE_DIR is already $DATA_DIR, so nothing to link.
if [ "$CLUSTER" = "1" ]; then
    echo "==> Linking dataset into $DATA_DIR ..."
    for item in "$STORAGE_DIR"/* "$STORAGE_DIR"/.[!.]*; do
        [ -e "$item" ] || continue
        name=$(basename "$item")
        ln -sfn "$item" "$DATA_DIR/$name"
        echo "    $DATA_DIR/$name -> $item"
    done
fi

echo ""
echo "Done. To evaluate with downloaded checkpoints:"
echo "  EP_NUM=100 TRAIN_CONFIG=univtac bash eval_policy.sh <task> clean ACT/deploy_policy <gpu>"
