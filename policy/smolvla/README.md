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
.venv/bin/python scripts/process_data.py lift_bottle clean 100   # -> data/local/lift_bottle-clean-100_vitac
TASK_NAME=lift_bottle bash scripts/train.sh                     # -> outputs/smolvla_vitac_lift_bottle-clean-100_vitac
```

Extra arguments are passed to the lerobot trainer, for example
`--steps=50000 --save_freq=5000`. For multiple GPUs, set `NUM_GPUS=4`.
