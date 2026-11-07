#!/bin/bash -l
# Process data (if needed) and train SmolVLA + tactile on one NHR alex A40.
#   TASK_NAME=put_bottle_in_shelf sbatch policy/smolvla/scripts/train_cluster.sbatch
# The LeRobot dataset and the outputs go to the anvme workspace. Image encoding
# in process_data is parallelised over the 16 cores of the A40 allocation.
#SBATCH --job-name=smolvla-train
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=smolvla-train-%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-base cuda/12.8.1 cmake/3.31.9
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

REPO=${REPO:-$HOME/Projects/UniVTAC}
TASK_NAME=${TASK_NAME:?set TASK_NAME, e.g. TASK_NAME=put_bottle_in_shelf}
TASK_CONFIG=${TASK_CONFIG:-clean}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-100}
WS=${WS:-$(ws_find univtac)}
export HF_HOME=${HF_HOME:-$WORK/hf}

SETTINGS=${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}
export REPO_ID=local/${SETTINGS}_vitac
export DATASET_ROOT=$WS/smolvla/data/$REPO_ID
export OUTPUT_DIR=$WS/smolvla/outputs/smolvla_vitac_${SETTINGS}_vitac
export TASK_NAME TASK_CONFIG EXPERT_DATA_NUM

cd "$REPO/policy/smolvla"
export PYTHONPATH=$PWD/src
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "[train] task=$TASK_NAME dataset=$DATASET_ROOT output=$OUTPUT_DIR"

if [ -f "$DATASET_ROOT/meta/info.json" ]; then
    echo "[train] dataset exists, skipping process_data"
else
    .venv/bin/python scripts/process_data.py "$TASK_NAME" "$TASK_CONFIG" "$EXPERT_DATA_NUM" \
        --repo-id "$REPO_ID" --out "$DATASET_ROOT" \
        --image-writer-processes 4 --image-writer-threads 4
fi

bash scripts/train.sh --save_freq=10000 "$@"
