#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
ENCODER_CKPT_DIR="$REPO_ROOT/encoder/checkpoints/resnet18/20251128-125750"
POLICY_CKPT_BASE="$REPO_ROOT/policy/ACT/act_ckpt"

# Install huggingface_hub if not available
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 -m pip install "huggingface_hub"
fi

echo "==> Downloading UniVTAC dataset from HuggingFace (~131 GB)..."
python3 - <<'EOF'
from huggingface_hub import snapshot_download
import os, sys

data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
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
# data/checkpoints/encoder.pth -> encoder/checkpoints/resnet18/20251128-125750/best.pth
echo "==> Setting up encoder checkpoint..."
mkdir -p "$ENCODER_CKPT_DIR"
cp "$DATA_DIR/checkpoints/encoder.pth" "$ENCODER_CKPT_DIR/best.pth"
echo "    $ENCODER_CKPT_DIR/best.pth"

# --- Policy checkpoints ---
# data/checkpoints/{task}/{variant}/ -> policy/ACT/act_ckpt/act-{task}/clean-100/{variant}/
# Use with: EP_NUM=100 TRAIN_CONFIG=univtac  (or vision_only)
echo "==> Setting up policy checkpoints..."
for task_dir in "$DATA_DIR/checkpoints"/*/; do
    task_name=$(basename "$task_dir")
    for variant_dir in "$task_dir"*/; do
        [ -d "$variant_dir" ] || continue
        variant=$(basename "$variant_dir")
        dest="$POLICY_CKPT_BASE/act-$task_name/clean-100/$variant"
        mkdir -p "$dest"
        cp -r "$variant_dir"* "$dest/"
        echo "    $dest"
    done
done

# Remove the downloaded checkpoint source after all checkpoint files have been
# copied into the paths used by evaluation.
echo "==> Cleaning downloaded checkpoint source..."
rm -rf "$DATA_DIR/checkpoints"
echo "    Removed $DATA_DIR/checkpoints"

echo ""
echo "Done. To evaluate with downloaded checkpoints:"
echo "  EP_NUM=100 TRAIN_CONFIG=univtac bash eval_policy.sh <task> clean ACT/deploy_policy <gpu>"
