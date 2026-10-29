#!/bin/bash -l
#SBATCH --job-name=dreamtac_server
#SBATCH --account=g107ea
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/server_%j.out
#SBATCH --error=slurm/server_%j.out

# Dream-Tac inference server for a UniVTAC checkpoint. Listens on the compute
# node's :5001; reach it from the workstation with
#   ssh -N -L 5001:<node>:5001 alex
# The environment prologue matches slurm/offline_check.sbatch -- see the comment
# there for why `module` and `ws_find` are resolved explicitly.
set -euo pipefail

TASK="${DREAMTAC_TASK:-insert_hole}"
ITER="${DREAMTAC_ITER:-iter_000004000}"
PORT="${DREAMTAC_PORT:-5001}"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

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
export FRANKA_CHUNK_SIZE=20
export FRANKA_NUM_DENOISING_STEPS=5
export FRANKA_USE_TACTILE=1

export UV_PROJECT_ENVIRONMENT="$WS/envs/dreamtac"
export UV_CACHE_DIR="$WS/cache"
export UV_PYTHON_INSTALL_DIR="$WS/python"
export UV_LINK_MODE=hardlink

NVIDIA_SITE_PACKAGES="$UV_PROJECT_ENVIRONMENT/lib/python3.10/site-packages/nvidia"
for NVIDIA_LIB_DIR in "$NVIDIA_SITE_PACKAGES"/*/lib; do
    [ -d "$NVIDIA_LIB_DIR" ] && LD_LIBRARY_PATH="$NVIDIA_LIB_DIR:${LD_LIBRARY_PATH:-}"
done
export LD_LIBRARY_PATH

echo "#####################################################################"
echo "# DREAMTAC SERVER"
echo "#   node       : $(hostname)"
echo "#   port       : $PORT"
echo "#   tunnel     : ssh -N -L ${PORT}:$(hostname):${PORT} alex"
echo "#   task       : $TASK"
echo "#   checkpoint : $FRANKA_COSMOS_CKPT"
echo "#####################################################################"

"$UV_BIN" run --frozen --no-sync --project cosmos_policy --extra cu128 --group franka --python 3.10 \
    python -m cosmos_policy.experiments.robot.franka.franka_server
