#!/bin/bash -l
# Build the Cosmos3 environment on an NHR alex A40 node: upstream lerobot (which
# has the cosmos3 policy, our vendored lerobot 0.5.1 does not) pinned to
# LEROBOT_REV, and its own .venv. Both live on the anvme workspace, not in the
# repo: $HOME and $WORK are close to their file-count quotas.
#   sbatch policy/cosmos3/scripts/install_cluster.sbatch
#SBATCH --job-name=cosmos3-install
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=cosmos3-install-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-base cuda/12.8.1 cmake/3.31.9
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export PATH=$HOME/.local/bin:$PATH

WS=${WS:-$(ws_find univtac)}
ROOT=${COSMOS3_ROOT:-$WS/cosmos3}
LEROBOT_REV=${LEROBOT_REV:-fe1c545}  # huggingface/lerobot main, 2026-11-13
export UV_CACHE_DIR=${TMPDIR:-/tmp}/uv-cache UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-$WORK/hf}

mkdir -p "$ROOT"
cd "$ROOT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

if [ ! -d lerobot/.git ]; then
    git clone --filter=blob:none https://github.com/huggingface/lerobot.git lerobot
fi
git -C lerobot fetch -q origin
# UniVTAC patches on top of LEROBOT_REV, on the local branch univtac (patches/*.patch).
REPO=${REPO:-$HOME/Projects/UniVTAC}
git -C lerobot switch -q --detach "$LEROBOT_REV"
git -C lerobot switch -q -C univtac
git -C lerobot -c user.name=univtac -c user.email=univtac@localhost am -q "$REPO"/policy/cosmos3/patches/*.patch
echo "[install] lerobot $(git -C lerobot log --oneline "$LEROBOT_REV"..HEAD | tr '\n' ';') on $LEROBOT_REV"

[ -x .venv/bin/python ] || uv venv .venv --python 3.12
VIRTUAL_ENV=$PWD/.venv uv pip install -e "./lerobot[cosmos3,training,peft]" --torch-backend=cu128
.venv/bin/python -c "
import torch, transformers, diffusers, lerobot
from lerobot.policies.cosmos3.modeling_cosmos3 import Cosmos3Policy
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('transformers', transformers.__version__, 'diffusers', diffusers.__version__)
print('lerobot', lerobot.__file__)
assert torch.cuda.is_available()"

# Base checkpoint (about 9 GB, bfloat16), weights and tokenizer.
.venv/bin/hf download lerobot/Cosmos3-Edge
echo "INSTALL DONE"
