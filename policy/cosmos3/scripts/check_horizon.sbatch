#!/bin/bash -l
# Run scripts/check_horizon.py (VAE reconstruction across video strides, then the patched
# policy for several video_stride / video_latent_frames / chunk_size settings) on one A40.
#   sbatch policy/cosmos3/scripts/check_horizon.sbatch [put_bottle_in_shelf]
#SBATCH --job-name=cosmos3-check
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=cosmos3-check-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-base cuda/12.8.1
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
TASK_NAME=${1:-put_bottle_in_shelf}
WS=${WS:-$(ws_find univtac)}
REPO=${REPO:-$HOME/Projects/UniVTAC}
export HF_HOME=${HF_HOME:-$WORK/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$WS/smolvla/hf_datasets}
REPO_ID=local/${TASK_NAME}-clean-100_vitac
ROOT=$WS/smolvla/data/$REPO_ID

cd "$WS/cosmos3"
echo "[check] lerobot $(git -C lerobot log --oneline -1) univtac=$(git -C "$REPO" rev-parse --short HEAD)"
.venv/bin/python "$REPO/policy/cosmos3/scripts/check_horizon.py" vae --root "$ROOT" --repo-id "$REPO_ID"
.venv/bin/python "$REPO/policy/cosmos3/scripts/check_horizon.py" policy --root "$ROOT" --repo-id "$REPO_ID"
