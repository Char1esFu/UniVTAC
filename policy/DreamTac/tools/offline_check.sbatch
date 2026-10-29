#!/bin/bash -l
#SBATCH --job-name=dreamtac_offline_check
#SBATCH --account=g107ea
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --constraint=a100_80
#SBATCH --time=00:40:00
#SBATCH --output=slurm/offline_check_%j.out
#SBATCH --error=slurm/offline_check_%j.out

# Offline sanity check for the UniVTAC checkpoint: runs the model on a held-out
# episode, compares the predicted chunk with ground truth, and reports peak
# CUDA memory. Environment mirrors slurm/train_univtac.sbatch.
set -euo pipefail

TASK="${DREAMTAC_TASK:-insert_hole}"
ITER="${DREAMTAC_ITER:-iter_000004000}"
EPISODE="${DREAMTAC_EPISODE:-7}"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

# train_univtac.sbatch gets `module` and `ws_find` for free because it is
# submitted from an interactive login shell and Slurm exports that environment.
# A job submitted over a non-interactive ssh inherits neither, so take them from
# the source: the `-l` shebang runs the profile chain (zz-rrze-local.sh is what
# autoinits modules), and these fall back to the real paths if it does not.
# /etc/profile.d/modules.sh is intentionally blank here and is *not* the hook.
if ! command -v module >/dev/null 2>&1; then
    set +u; eval "$(tclsh /apps/modules/modulecmd.tcl bash autoinit)"; set -u
fi
command -v module >/dev/null 2>&1 || { echo "cannot initialise environment modules" >&2; exit 3; }

WS_FIND="$(command -v ws_find || echo /apps/hpc-workspace/current/bin/ws_find)"
if [ -x "$WS_FIND" ]; then
    WS="$("$WS_FIND" "${WS_NAME:-wam-envs}")/dream-tac"
else
    WS="${DREAMTAC_WS:-/anvme/workspace/g107ea11-wam-envs/dream-tac}"
fi
[ -d "$WS/envs/dreamtac" ] || { echo "uv environment missing: $WS/envs/dreamtac" >&2; exit 3; }

module load cuda/12.8.1 cmake/3.31.9
cd "${SLURM_SUBMIT_DIR}"

export UNIVTAC_DATA_DIR="$WORK/UniVTAC/data"
export UNIVTAC_CACHE_DIR="$WORK/Dream-Tac/cache/univtac"
export UNIVTAC_TASKS="$TASK"
export IMAGINAIRE_OUTPUT_ROOT="$WORK/Dream-Tac/runs/univtac"
export COSMOS_TACTILE_SELF_ATTN_BACKEND=flashbias_sdpa
export HF_HOME="$WORK/hf"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

export FRANKA_COSMOS_CONFIG=cosmos_predict2_2b_480p_univtac
export FRANKA_COSMOS_CONFIG_FILE=cosmos_policy/config/config.py
export FRANKA_COSMOS_CKPT="$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy_univtac/cosmos_v2_finetune_$TASK/cosmos_predict2_2b_480p_univtac/checkpoints/$ITER"
export FRANKA_DATASET_STATS_PATH="$UNIVTAC_CACHE_DIR/dataset_statistics_univtac_$TASK.json"
export FRANKA_T5_EMBEDDINGS_PATH="$UNIVTAC_CACHE_DIR/t5_embeddings.pkl"

export UV_PROJECT_ENVIRONMENT="$WS/envs/dreamtac"
export UV_CACHE_DIR="$WS/cache"
export UV_PYTHON_INSTALL_DIR="$WS/python"
export UV_LINK_MODE=hardlink

# transformer_engine resolves libnvrtc through `ldconfig -p` at import; the CUDA
# libraries shipped in the venv are not on the loader path by default, and TE
# dies with CalledProcessError without this.
NVIDIA_SITE_PACKAGES="$UV_PROJECT_ENVIRONMENT/lib/python3.10/site-packages/nvidia"
for NVIDIA_LIB_DIR in "$NVIDIA_SITE_PACKAGES"/*/lib; do
    [ -d "$NVIDIA_LIB_DIR" ] && LD_LIBRARY_PATH="$NVIDIA_LIB_DIR:${LD_LIBRARY_PATH:-}"
done
for NVIDIA_INCLUDE_DIR in "$NVIDIA_SITE_PACKAGES"/*/include; do
    [ -d "$NVIDIA_INCLUDE_DIR" ] && CPATH="$NVIDIA_INCLUDE_DIR:${CPATH:-}"
done
export LD_LIBRARY_PATH CPATH

echo "task       : $TASK"
echo "checkpoint : $FRANKA_COSMOS_CKPT"
echo "episode    : $UNIVTAC_DATA_DIR/$TASK/clean/$EPISODE.hdf5"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$UV_BIN" run --frozen --no-sync --project cosmos_policy --extra cu128 --group franka --python 3.10 \
    python offline_check.py \
    --hdf5 "$UNIVTAC_DATA_DIR/$TASK/clean/$EPISODE.hdf5" \
    --steps 4 --stride 30 --denoising-steps 5 \
    --out "slurm/offline_check_${TASK}_${ITER}.json"
