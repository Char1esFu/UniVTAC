# SmolVLA + tactile

SmolVLA runs in its own virtualenv, `policy/smolvla/.venv` (Python 3.12),
separate from the IsaacLab conda env. Training uses it directly, and the
IsaacLab eval starts `.venv/bin/python` as a subprocess, so the path matters.

## Install the .venv

Requires [uv](https://docs.astral.sh/uv/). Dependency ranges in
`pyproject.toml` are copied from upstream lerobot
(`policy/smolvla_vision/lerobot/pyproject.toml`).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # only if uv is missing

cd policy/smolvla
uv venv .venv --python 3.12
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[training]"
```

The default torch wheel is built for CUDA 13.0 and needs NVIDIA driver >= 580
(check with `nvidia-smi`). On older drivers, pick a matching build, e.g.
`--torch-backend=cu128` (or `cu126`) on the install command.

Check the install on a GPU node:

```bash
.venv/bin/python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), lerobot.__file__)"
```

`lerobot.__file__` must point into `policy/smolvla/src/lerobot`.

The first training run downloads SmolVLM2 (~1 GB) from HuggingFace; set
`HF_HOME` to a filesystem with enough quota.

### NHR cluster (alex)

`scripts/install_cluster.sbatch` builds the .venv on an A40 node with
`python/3.12-base`, `cuda/12.8.1` and the proxy, using `--torch-backend=cu128`:

```bash
sbatch policy/smolvla/scripts/install_cluster.sbatch
```

## Train

The data comes from `bash data/download_hf.sh <local|cluster>`. Hyperparameters
are in `scripts/train_config.yml`.

```bash
cd policy/smolvla
export PYTHONPATH=$PWD/src
TASK=lift_bottle

# 1) Raw HDF5 -> LeRobot dataset: data/local/${TASK}-clean-100_vitac
.venv/bin/python scripts/process_data.py $TASK clean 100 \
    --image-writer-processes 4 --image-writer-threads 4

# 2) Train: outputs/smolvla_vitac_${TASK}-clean-100_vitac
TASK_NAME=$TASK bash scripts/train.sh --save_freq=3000 --wandb.enable=true \
    2>&1 | tee train_${TASK}.log
```

On the NHR cluster, `scripts/train_cluster.sbatch` runs both steps on one A40
(processing is skipped when the dataset exists) and logs to wandb online
(`WANDB_MODE`, `WANDB_ENTITY`, `WANDB_PROJECT` override the defaults; the
script uses `--export=NONE`, so set them with `sbatch --export=NONE,WANDB_MODE=...`):

```bash
sbatch policy/smolvla/scripts/train_cluster.sbatch put_bottle_in_shelf
```

Slurm copies the script at submit time, so pull before `sbatch`. The datasets
arrow cache (`HF_DATASETS_CACHE`) must not be on NFS (`$WORK`): cold reads
there take about 0.7 s per sample and starve the GPU.

Tasks: `grasp_classify`, `insert_HDMI`, `insert_hole`, `insert_tube`,
`lift_bottle`, `lift_can`, `pull_out_key`, `put_bottle_in_shelf`.

Data processing:

- Each frame of every stream (head, wrist and four tactile streams) is written
  as a PNG, about 240k files per task. `--videos` stores MP4s instead, which
  saves space but makes training decode the videos.
- The image writer runs `processes × threads` workers. Beyond about 16 to 32
  workers, the single main process that feeds them is the limit, so adding
  more does not help.

Training:

- Extra arguments are passed to the lerobot trainer, for example
  `--steps=50000 --batch_size=16`. For multiple GPUs, set `NUM_GPUS=4`.
- The output directory is created at the first checkpoint (`--save_freq`).
  Without wandb, the loss is only printed to the terminal, hence the `tee`.
- If the output directory already exists, a fresh run fails with
  `FileExistsError`. Delete the directory, or set another `OUTPUT_DIR`.
- `--wandb.enable=true` needs a wandb login (`.venv/bin/wandb login`, which
  writes `~/.netrc`). Without a login, add `--wandb.mode=offline` and upload
  later with `wandb sync`.
- In the log, `data_s` should stay well below `updt_s` (about 0.09 s/step on
  an RTX 4090 at batch size 8). If it does not, data loading is the
  bottleneck.

## Eval

Run from the repo root in the IsaacLab env. `deploy.yml` holds the deploy
parameters for every task; the checkpoint is picked from the environment:

```bash
TRAIN_CONFIG=fp32_scratch EP_NUM=100 \
    python scripts/eval_policy.py lift_bottle clean smolvla/deploy --headless
```

- The checkpoint is
  `outputs/smolvla_vitac_{task}-{task_config}-{EP_NUM}_{TRAIN_CONFIG}/checkpoints/last/pretrained_model`,
  i.e. the suffix after `{EP_NUM}_` in the training `RUN_TAG`. Defaults:
  `TRAIN_CONFIG=vitac` (what `scripts/train.sh` writes without `RUN_TAG`),
  `EP_NUM=100`. A `ckpt_dir` in the deploy args overrides this.
- Results go to `eval_result/smolvla/{task}/deploy/{run tag}/{time}/`, where
  the run tag is `TRAIN_CONFIG` plus the execution mode (below), e.g.
  `fp32_scratch-rtc10`.

### Action execution (RTC10 by default)

The policy predicts a 50-action chunk; how those chunks are turned into one
action per env step is set in `deploy.yml` and changes results a lot, without
retraining:

| Mode | `smolvla_replan_every` | `smolvla_rtc` | Run tag suffix |
|---|---|---|---|
| Temporal aggregation | `0` | – | none |
| Plain receding horizon | `n` | `false` | `-replan{n}` |
| **RTC (default)** | `10` | `true` | `-rtc10` |

- **Temporal aggregation** (the upstream UniVTAC setting, copied from ACT):
  predict a chunk every step and execute the exponentially weighted average
  (`smolvla_temporal_agg_k`) of all predictions for the current step. ACT is
  deterministic at inference (z = 0), so averaging is harmless there. SmolVLA
  samples each chunk from a flow-matching distribution; averaging samples
  from different modes yields actions that belong to none of them.
- **Plain receding horizon**: execute the first `n` actions of a chunk, then
  predict a new one (openpi's pi0/pi0.5 deployment). With no smoothing the
  arm jumps at every chunk switch: the eval sets joint positions directly
  (`take_action(..., force=True)` teleports via `set_dof_positions`), so
  nothing between policy and joints filters the jump.
- **RTC** (Real-Time Chunking, Black et al. 2025, lerobot's `RTCProcessor`),
  synchronous with inference delay 0: as in the receding horizon, `n`
  actions are executed per chunk, but the new chunk's flow-matching sampling
  is guided towards the part of the previous chunk that was not executed
  (weights decay linearly from 1 at the switch to 0 at the end of the
  overlap; `smolvla_rtc_schedule`, guidance capped at
  `smolvla_rtc_max_guidance_weight` = 10). Consecutive chunks therefore join
  up without averaging. On dataset frames it cuts the jump at the switch
  from 1.3x to 0.1x a typical in-chunk step (n = 10).

Measured on lift_bottle, `fp32_scratch` checkpoint, the same 20 scenes
(2026-11-14): temporal aggregation 0/20, RTC with n = 10 9/20, RTC with
n = 25 8/20 (RTC vs temporal aggregation: paired McNemar p < 0.01; n = 10 vs
25 not distinguishable). Results recorded before this change (e.g. 3/50 on
lift_bottle) used temporal aggregation and are not comparable.

Environment variables override the yml for a single run:
`SMOLVLA_REPLAN_EVERY=0` (temporal aggregation), `SMOLVLA_REPLAN_EVERY=25`,
`SMOLVLA_RTC=0/1`. `smolvla_n_action_steps` has no effect here: the deploy
calls `predict_action_chunk`, not lerobot's `select_action` queue.

## World action model (branch `smolvla-wam`)

`--policy.wam_enabled=true` gives the action expert, besides the action chunk,
an observation window of past latents and the latents of several future frames
to denoise jointly with the actions:

```
suffix: [ window: o_{t-(H-1)s} .. o_t (clean) | future: o_{t+k} .. (noisy) | actions x 50 ]
```

- Vision latents: frozen SigLIP + connector tokens of `wam_vision_keys` (head
  and wrist), the 8x8 grid average-pooled to `wam_vision_pool` (2x2) per camera.
- Tactile latents: UniVTAC backbone features of `tactile_keys`, from the
  same encoder as the prefix's tactile tokens, so the modality is either frozen
  or fine-tuned as a whole (`vitac_freeze_backbone`):
  - frozen: prefix, window and targets use the frozen pretrained backbone;
    window and targets are read from the precomputed latents;
  - fine-tuned (default, as in the baselines): prefix and window use the
    fine-tuned backbone, the window is encoded online (the tactile streams are
    then loaded at the window and future offsets), and the targets come from
    an EMA copy of it (`wam_tactile_ema_decay`). A target encoder that the
    prediction loss itself trains could collapse to a constant.
- Window: `wam_history` frames (default 2, 1 = current only),
  `wam_history_stride` frames apart, clamped at the episode start.
- Future: `wam_{vision,tactile}_future_steps` frames, `..._future_stride`
  apart (default 6 frames, 5 apart: t+5 ... t+30). Futures past the
  episode end are left out of the loss.
- Each modality has its own flow time; its loss is averaged over its valid
  frames and weighted (`wam_vision_loss_weight`, `wam_tactile_loss_weight`).
  With probability `wam_marginal_prob` all future latents are pure noise.
- `wam_inference_mode`: `joint` (default) denoises everything together;
  `marginal` keeps the future latents at pure noise.
- `wam_obs_attend_actions=true` also lets the future tokens attend to the actions.

Precompute the latents once per task (about 7 GB), then train:

```bash
.venv/bin/python scripts/precompute_wam_latents.py lift_bottle clean 100
TASK_NAME=lift_bottle RUN_TAG=lift_bottle-clean-100_wam bash scripts/train.sh \
    --batch_size=32 --steps=15000 --policy.wam_enabled=true \
    --policy.wam_latent_dir=$PWD/data/local/lift_bottle-clean-100_vitac/wam_latents
```

Inference encodes the current frame online and keeps the window in the policy,
one frame per `predict_action_chunk` call. Not yet compatible with the RTC10
default: `wam_enabled` rejects RTC, and with replanning every 10 actions the
window would have to be updated every env step instead of every call.
