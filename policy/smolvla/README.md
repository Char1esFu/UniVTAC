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
