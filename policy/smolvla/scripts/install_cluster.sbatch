#!/bin/bash -l
# Build policy/smolvla/.venv on an NHR alex A40 node and smoke-test training.
#   sbatch policy/smolvla/scripts/install_cluster.sbatch           # install + 20-step smoke test
#   SMOKE=0 sbatch policy/smolvla/scripts/install_cluster.sbatch   # install only
#SBATCH --job-name=smolvla-install
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=smolvla-install-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-base cuda/12.8.1 cmake/3.31.9
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export PATH=$HOME/.local/bin:$PATH

REPO=${REPO:-$HOME/Projects/UniVTAC}
SMOKE=${SMOKE:-1}
# uv cache on node-local disk: $WORK is close to its file-count quota.
export UV_CACHE_DIR=${TMPDIR:-/tmp}/uv-cache UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-$WORK/hf}

cd "$REPO/policy/smolvla"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

[ -x .venv/bin/python ] || uv venv .venv --python 3.12
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[training]" --torch-backend=cu128
.venv/bin/python -c "
import torch, numpy, transformers, peft, lerobot
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('numpy', numpy.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)
print('lerobot', lerobot.__file__)
assert torch.cuda.is_available()"

[ "$SMOKE" = 1 ] || exit 0

# Link $WORK/UniVTAC data/checkpoints into the repo (downloads only what is missing).
PATH=$PWD/.venv/bin:$PATH bash "$REPO/data/download_hf.sh" cluster grasp_classify

export PYTHONPATH=$PWD/src
SMOKE_DIR=${TMPDIR:-/tmp}/smolvla-smoke
REPO_ID=local/grasp_classify-clean-3_vitac
.venv/bin/python scripts/process_data.py grasp_classify clean 3 --repo-id $REPO_ID --out $SMOKE_DIR/data
REPO_ID=$REPO_ID DATASET_ROOT=$SMOKE_DIR/data OUTPUT_DIR=$SMOKE_DIR/outputs TASK_NAME=grasp_classify \
    bash scripts/train.sh --steps=20 --save_freq=20 --log_freq=5 --batch_size=4
ls $SMOKE_DIR/outputs/checkpoints/last/pretrained_model
echo "SMOKE TEST PASSED"
