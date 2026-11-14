#!/bin/bash -l
# Process data (if needed) and train SmolVLA + tactile on one NHR alex A40.
#   sbatch policy/smolvla/scripts/train_cluster.sbatch put_bottle_in_shelf [--steps=... other trainer args]
# --export=NONE drops the submitting shell's environment, so pass the task as the
# first argument (or use `sbatch --export=NONE,TASK_NAME=...,WANDB_MODE=...`).
# The LeRobot dataset and the outputs go to the anvme workspace. Image encoding
# in process_data is parallelised over the 16 cores of the A40 allocation.
#
# wandb: online through the NHR proxy (needs `wandb login`, i.e. ~/.netrc).
# WANDB_MODE=offline keeps the run under $OUTPUT_DIR/wandb for a later
# `wandb sync -e $WANDB_ENTITY -p $WANDB_PROJECT <run dir>` from the login node.
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
if [ $# -gt 0 ] && [[ $1 != --* ]]; then TASK_NAME=$1; shift; fi
TASK_NAME=${TASK_NAME:?usage: sbatch train_cluster.sbatch <task> [trainer args]}
TASK_CONFIG=${TASK_CONFIG:-clean}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-100}
NUM_WORKERS=${NUM_WORKERS:-12}
WS=${WS:-$(ws_find univtac)}
# Model downloads (SmolVLM2) stay on $WORK; the dataset arrow cache goes to NVMe.
export HF_HOME=${HF_HOME:-$WORK/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$WS/smolvla/hf_datasets}

WANDB_MODE=${WANDB_MODE:-online}
WANDB_ENTITY=${WANDB_ENTITY:-pvc_tudarmstadt}
WANDB_PROJECT=${WANDB_PROJECT:-smolvla_univtac}

SETTINGS=${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}
export REPO_ID=local/${SETTINGS}_vitac
export DATASET_ROOT=$WS/smolvla/data/$REPO_ID
export OUTPUT_DIR=${OUTPUT_DIR:-$WS/smolvla/outputs/smolvla_vitac_${SETTINGS}_vitac}
export TASK_NAME TASK_CONFIG EXPERT_DATA_NUM

cd "$REPO/policy/smolvla"
export PYTHONPATH=$PWD/src
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "[train] task=$TASK_NAME dataset=$DATASET_ROOT output=$OUTPUT_DIR git=$(git rev-parse --short HEAD)"

if [ -f "$DATASET_ROOT/meta/info.json" ]; then
    echo "[train] dataset exists, skipping process_data"
else
    .venv/bin/python scripts/process_data.py "$TASK_NAME" "$TASK_CONFIG" "$EXPERT_DATA_NUM" \
        --repo-id "$REPO_ID" --out "$DATASET_ROOT" \
        --image-writer-processes 4 --image-writer-threads 4
fi

bash scripts/train.sh --save_freq=10000 --num_workers="$NUM_WORKERS" \
    --wandb.enable=true --wandb.mode="$WANDB_MODE" \
    --wandb.entity="$WANDB_ENTITY" --wandb.project="$WANDB_PROJECT" \
    --wandb.disable_artifact=true \
    "$@"
