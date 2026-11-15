#!/bin/bash -l
# Fine-tune Cosmos3 Edge (pretrained world action model, lerobot `cosmos3_edge`, patched by
# policy/cosmos3/patches) on one NHR alex A40, on the LeRobot datasets of the SmolVLA pipeline.
#   sbatch policy/cosmos3/scripts/train_cluster.sbatch put_bottle_in_shelf [trainer args]
# Needs scripts/install_cluster.sbatch first. Everything lives on the anvme workspace.
#
# Generation stream: [video latents | actions | tactile tokens], denoised jointly.
#   Video: head | wrist canvas, VIDEO_LATENT_FRAMES future latent frames of 4 frames each,
#     VIDEO_STRIDE dataset frames apart ("-" = the clip ends with the action chunk).
#   Actions: CHUNK_SIZE actions, one per dataset frame (30 fps).
#   Tactile (TACTILE=none|frozen|trainable): one UniVTAC ResNet18 token per stream and frame, the current
#     frame clean, TACTILE_STEPS future frames TACTILE_STRIDE dataset frames apart denoised (2 streams x
#     (1 + steps) tokens). trainable: the encoder trains, the targets come from its EMA.
#
# SMOKE=1 trains SMOKE_STEPS steps for every SMOKE_RUNS entry
# "<batch>:<chunk>:<full|lora>:<video stride>:<video latent frames>:<tactile>" and logs gpu_mem_gb, updt_s.
#
# Measured (put_bottle_in_shelf, A40, bf16 autocast, gradient checkpointing, batch 16, full fine-tuning,
# no tactile): 39 GB; 3.3 s/step with 5 latent frames, 2.3 s/step with 3. Batch 32 exceeds the 60 GB
# host RAM of the job.
#SBATCH --job-name=cosmos3-train
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=cosmos3-train-%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-base cuda/12.8.1 cmake/3.31.9
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

if [ $# -gt 0 ] && [[ $1 != --* ]]; then TASK_NAME=$1; shift; fi
TASK_NAME=${TASK_NAME:?usage: sbatch train_cluster.sbatch <task> [trainer args]}
TASK_CONFIG=${TASK_CONFIG:-clean}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
REPO=${REPO:-$HOME/Projects/UniVTAC}
WS=${WS:-$(ws_find univtac)}
ROOT=${COSMOS3_ROOT:-$WS/cosmos3}
# Checkpoints and caches on NVMe ($WORK is NFS and can stall at a few MB/s).
CKPT=${CKPT:-$ROOT/ckpt/Cosmos3-Edge-hub}  # hf download lerobot/Cosmos3-Edge --local-dir ...
TACTILE_CKPT=${TACTILE_CKPT:-$ROOT/ckpt/univtac_resnet18_20251128-125750.pth}
export HF_HOME=${HF_HOME:-$WS/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$WS/smolvla/hf_datasets}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH_SIZE=${BATCH_SIZE:-16}
CHUNK_SIZE=${CHUNK_SIZE:-50}
MODE=${MODE:-full}
VIDEO_STRIDE=${VIDEO_STRIDE:-5}
VIDEO_LATENT_FRAMES=${VIDEO_LATENT_FRAMES:-2}
TACTILE=${TACTILE:-frozen}
TACTILE_STRIDE=${TACTILE_STRIDE:-5}
TACTILE_STEPS=${TACTILE_STEPS:-6}
STEPS=${STEPS:-30000}
SMOKE=${SMOKE:-0}
SMOKE_STEPS=${SMOKE_STEPS:-30}
SMOKE_RUNS=${SMOKE_RUNS:-"16:50:full:5:2:none 16:50:full:5:2:frozen 16:50:full:5:2:trainable"}
TRACKIO_PROJECT=${TRACKIO_PROJECT:-cosmos3_univtac}  # Trackio database under $HF_HOME/trackio
# The trainer logs to Trackio only; a background scripts/sync_wandb.py mirrors the run to wandb every
# WANDB_SYNC_S seconds (WANDB_SYNC=0 disables it; needs ~/.netrc from `wandb login`). Runs started without
# it are backfilled by running sync_wandb.py once on the login node.
WANDB_SYNC=${WANDB_SYNC:-1}
WANDB_SYNC_S=${WANDB_SYNC_S:-300}
export WANDB_ENTITY=${WANDB_ENTITY:-cong_fu}
export WANDB_PROJECT=${WANDB_PROJECT:-cosmos3_univtac}

SETTINGS=${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}
REPO_ID=local/${SETTINGS}_vitac
DATASET_ROOT=${DATASET_ROOT:-$WS/smolvla/data/$REPO_ID}
[ -f "$DATASET_ROOT/meta/info.json" ] || {
    echo "missing $DATASET_ROOT: build it with policy/smolvla/scripts/train_cluster.sbatch $TASK_NAME"; exit 1; }

cd "$ROOT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "[cosmos3] task=$TASK_NAME dataset=$DATASET_ROOT lerobot=$(git -C lerobot log --oneline -1 | cut -c1-60)" \
    "univtac=$(git -C "$REPO" rev-parse --short HEAD)"

# Writes the trainer config of one run to $1: chunk size $2, video stride $3, video latent frames $4,
# tactile mode $5 (none|frozen|trainable).
write_config() {
    local out=$1 chunk=$2 stride=$3 latent=$4 tactile=$5
    {
        echo "policy:"
        echo "  type: cosmos3_edge"
        echo "  pretrained_path: $CKPT"
        echo "  tokenizer_name: $CKPT/tokenizer"
        echo "  load_action_modules: false"
        echo "  gradient_checkpointing: true"
        echo "  push_to_hub: false"
        echo "  chunk_size: $chunk"
        echo "  n_action_steps: $chunk"
        echo "  video_stride: $stride"
        [ "$latent" = - ] || echo "  video_latent_frames: $latent"
        echo "  view_description: \"The left half shows the third-person head camera; the right half shows the wrist-mounted camera.\""
        echo "  input_features:"
        echo "    observation.images.head: {type: VISUAL, shape: [3, 256, 256]}"
        echo "    observation.images.wrist: {type: VISUAL, shape: [3, 256, 256]}"
        if [ "$tactile" != none ]; then
            echo "  tactile_keys: [observation.images.tactile_ll_marker, observation.images.tactile_lr_marker]"
            echo "  tactile_future_stride: $TACTILE_STRIDE"
            echo "  tactile_future_steps: $TACTILE_STEPS"
            echo "  tactile_encoder_path: $TACTILE_CKPT"
            echo "  tactile_encoder_trainable: $([ "$tactile" = trainable ] && echo true || echo false)"
        fi
    } > "$out"
}

# Runs lerobot-train: $1 batch size, $2 chunk size, $3 full|lora, $4 video stride, $5 video latent frames,
# $6 tactile mode, rest trainer args.
train() {
    local bs=$1 chunk=$2 mode=$3 stride=$4 latent=$5 tactile=$6; shift 6
    local cfg=${TMPDIR:-/tmp}/cosmos3_c${chunk}_s${stride}_k${latent}_${tactile}.yml
    write_config "$cfg" "$chunk" "$stride" "$latent" "$tactile"
    local peft_args=()
    if [ "$mode" = lora ]; then
        peft_args=(--peft.method_type=LORA --peft.r=64)
    fi
    .venv/bin/lerobot-train \
        --config_path="$cfg" \
        --dataset.repo_id="$REPO_ID" --dataset.root="$DATASET_ROOT" \
        --accelerator.mixed_precision=bf16 --tf32=false \
        --batch_size="$bs" --num_workers="$NUM_WORKERS" \
        ${peft_args[@]+"${peft_args[@]}"} "$@"
}

if [ "$SMOKE" = 1 ]; then
    failed=0
    for run in $SMOKE_RUNS; do
        IFS=: read -r bs c m s k t <<< "$run"
        echo "===== [smoke] batch=$bs chunk=$c mode=$m video_stride=$s video_latent_frames=$k tactile=$t"
        out=${TMPDIR:-/tmp}/cosmos3-smoke-b${bs}-c${c}-$m-s$s-k$k-$t
        rm -rf "$out"
        train "$bs" "$c" "$m" "$s" "$k" "$t" --output_dir="$out" --job_name=smoke \
            --steps="$SMOKE_STEPS" --log_freq=5 --save_checkpoint=false "$@" \
            && echo "===== [smoke] OK $run" || { echo "===== [smoke] FAILED $run"; failed=1; }
    done
    exit $failed  # non-zero when a run failed, for sbatch --dependency=afterok
fi

RUN_TAG=${RUN_TAG:-${SETTINGS}_edge_c${CHUNK_SIZE}_s${VIDEO_STRIDE}_k${VIDEO_LATENT_FRAMES}_tac-${TACTILE}-s${TACTILE_STRIDE}x${TACTILE_STEPS}_${MODE}}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/cosmos3_${RUN_TAG}}

# Mirrors the Trackio run "cosmos3_$RUN_TAG-<id>" to wandb in the background, with a final sync on exit.
if [ "$WANDB_SYNC" = 1 ]; then
    SYNC=("${UV:-$HOME/.local/bin/uv}" run --no-project --with wandb python "$REPO/policy/cosmos3/scripts/sync_wandb.py"
        --project "$TRACKIO_PROJECT" --run-filter "cosmos3_${RUN_TAG}-")
    "${SYNC[@]}" --loop "$WANDB_SYNC_S" &
    SYNC_PID=$!
    trap 'kill $SYNC_PID 2>/dev/null; "${SYNC[@]}" || true' EXIT
fi

# RESUME=1 continues the run in OUTPUT_DIR (same env as the first job) from its last checkpoint, e.g. after
# the 24 h limit: sbatch --dependency=afterany:<job> --export=NONE,RESUME=1,... train_cluster.sbatch <task>.
# A run that already reached its steps returns right away.
if [ "${RESUME:-0}" = 1 ]; then
    LAST=$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json
    [ -f "$LAST" ] || { echo "[resume] no checkpoint in $OUTPUT_DIR"; exit 1; }
    STEP=$(cat "$OUTPUT_DIR/checkpoints/last/training_state/training_step.json" 2>/dev/null || echo "?")
    echo "[resume] $OUTPUT_DIR from $STEP"
    .venv/bin/lerobot-train --config_path="$LAST" --resume=true "$@"
    exit
fi
train "$BATCH_SIZE" "$CHUNK_SIZE" "$MODE" "$VIDEO_STRIDE" "$VIDEO_LATENT_FRAMES" "$TACTILE" \
    --output_dir="$OUTPUT_DIR" --job_name="cosmos3_${RUN_TAG}" \
    --steps="$STEPS" --policy.scheduler_decay_steps="$STEPS" \
    --ema.type=power --ema.sigma_rel=0.1 --save_freq=5000 \
    --trackio.enable=true --trackio.project="$TRACKIO_PROJECT" "$@"
