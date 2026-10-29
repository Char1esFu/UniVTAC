"""Offline sanity check for a UniVTAC Dream-Tac checkpoint.

Runs *outside* Isaac Sim, in the Dream-Tac environment. Today that means **on
the cluster**, in the existing uv workspace env -- the UniVTAC conda env on the
workstation cannot import `cosmos_policy` (no transformer_engine, and its torch
is 2.7 against the 2.6 the project pins). It feeds real frames from a held-out
UniVTAC episode straight through `get_action` and reports:

  * the predicted action chunk vs. the ground-truth actions from the HDF5,
  * peak CUDA memory -- the number the paper never states, and the one that
    decides whether the model can share a 24 GB card with Isaac Sim,
  * per-query latency.

Do this before touching the simulator: it isolates "are the weights, config,
normalisation and image pipeline right?" from "is the closed loop right?".

Usage (cluster):

    export UNIVTAC_TASKS=insert_hole
    export UNIVTAC_DATA_DIR=$WORK/UniVTAC/data
    export UNIVTAC_CACHE_DIR=$WORK/Dream-Tac/cache/univtac
    export FRANKA_COSMOS_CONFIG=cosmos_predict2_2b_480p_univtac
    export FRANKA_COSMOS_CKPT=$WORK/Dream-Tac/runs/univtac/cosmos_policy_univtac/\
cosmos_v2_finetune_insert_hole/cosmos_predict2_2b_480p_univtac/checkpoints/iter_000004000
    export FRANKA_DATASET_STATS_PATH=$UNIVTAC_CACHE_DIR/dataset_statistics_univtac_insert_hole.json
    export FRANKA_T5_EMBEDDINGS_PATH=$UNIVTAC_CACHE_DIR/t5_embeddings.pkl

    cd ~/Projects/Dream-Tac
    uv run --frozen --no-sync --project cosmos_policy --extra cu128 --group franka --python 3.10 \
        python policy_dreamtac_offline_check.py \
        --hdf5 $UNIVTAC_DATA_DIR/insert_hole/clean/7.hdf5 --steps 3

(the file needs to sit inside the Dream-Tac repo root, or on PYTHONPATH, so that
`import cosmos_policy` resolves.)
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import torch

from cosmos_policy.datasets.univtac_dataset import (
    _find_tactile_keys,
    _numeric_episode,
)
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)

COSMOS_IMAGE_SIZE = 224


def build_cfg(args):
    return SimpleNamespace(
        suite="franka",
        config=os.environ.get("FRANKA_COSMOS_CONFIG", "cosmos_predict2_2b_480p_univtac"),
        ckpt_path=os.environ["FRANKA_COSMOS_CKPT"],
        config_file=os.environ.get("FRANKA_COSMOS_CONFIG_FILE", "cosmos_policy/config/config.py"),
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        use_wrist_image=True,
        num_wrist_images=1,
        use_third_person_image=True,
        num_third_person_images=1,
        use_tactile=True,
        use_jpeg_compression=True,
        trained_with_image_aug=True,
        flip_images=False,
        use_variance_scale=False,
        chunk_size=args.chunk_size,
        num_denoising_steps_action=args.denoising_steps,
    )


def decode(raw):
    encoded = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw.reshape(-1)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image.shape[:2] != (COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE):
        image = cv2.resize(image, (COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, help="one held-out UniVTAC episode")
    parser.add_argument("--steps", type=int, default=3, help="how many timesteps to query")
    parser.add_argument("--stride", type=int, default=20, help="timesteps between queries")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--denoising-steps", type=int, default=5)
    parser.add_argument("--instruction", default=None, help="defaults to the task folder name")
    parser.add_argument("--out", default="dreamtac_offline_check.json")
    args = parser.parse_args()

    task = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.hdf5))))
    instruction = args.instruction or task.replace("_", " ")

    cfg = build_cfg(args)
    print(f"checkpoint : {cfg.ckpt_path}")
    print(f"instruction: {instruction!r}")

    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model, _ = get_model(cfg)
    print(f"model loaded in {time.perf_counter() - load_start:.1f}s, state_t={model.config.state_t}")
    assert model.config.state_t == 12, (
        f"expected state_t=12 (tactile), got {model.config.state_t}; wrong checkpoint or config"
    )
    weights_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak CUDA memory after load: {weights_gb:.2f} GB")

    dataset_stats = load_dataset_stats(os.environ["FRANKA_DATASET_STATS_PATH"])
    init_t5_text_embeddings_cache(os.environ["FRANKA_T5_EMBEDDINGS_PATH"])

    proprio_all, actions_all = _numeric_episode(args.hdf5)
    records = []
    with h5py.File(args.hdf5, "r") as root:
        left_key, right_key = _find_tactile_keys(root)
        num_steps = min(
            len(proprio_all), len(actions_all), len(root["observation/head/rgb"]), len(root[left_key])
        )

        for index in range(args.steps):
            t = args.start + index * args.stride
            if t >= num_steps:
                break
            obs = {
                "primary_image": decode(root["observation/head/rgb"][t]),
                "wrist_image": decode(root["observation/wrist/rgb"][t]),
                "tactile_left_image": decode(root[left_key][t]),
                "tactile_right_image": decode(root[right_key][t]),
                "proprio": proprio_all[t],
            }

            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = get_action(
                cfg, model, dataset_stats, obs, instruction,
                seed=0, randomize_seed=False,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
            elapsed = time.perf_counter() - started
            peak_gb = torch.cuda.max_memory_allocated() / 1e9

            predicted = np.asarray(result["actions"], dtype=np.float64).reshape(-1, 7)
            horizon = min(len(predicted), num_steps - t)
            truth = actions_all[t : t + horizon]
            error = np.abs(predicted[:horizon] - truth)

            print(
                f"\nt={t:4d}  {elapsed:5.2f}s  peak {peak_gb:5.2f} GB  "
                f"({1.0 / elapsed:.2f} query/s, {horizon / elapsed:.1f} action/s if replayed open-loop)"
            )
            print(f"  pos MAE  {error[:, :3].mean():.5f} m    (chunk mean over {horizon} steps)")
            print(f"  rot MAE  {error[:, 3:6].mean():.5f} rad")
            print(f"  grip MAE {error[:, 6].mean():.5f} m")
            print(f"  first predicted: {np.round(predicted[0], 4).tolist()}")
            print(f"  first truth    : {np.round(truth[0], 4).tolist()}")

            records.append(
                {
                    "t": t,
                    "elapsed": elapsed,
                    "peak_gb": peak_gb,
                    "proprio": proprio_all[t].tolist(),
                    "predicted": predicted.tolist(),
                    "truth": truth.tolist(),
                }
            )

    summary = {
        "checkpoint": cfg.ckpt_path,
        "instruction": instruction,
        "denoising_steps": cfg.num_denoising_steps_action,
        "weights_peak_gb": weights_gb,
        "inference_peak_gb": max((r["peak_gb"] for r in records), default=weights_gb),
        "records": records,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\npeak CUDA memory overall: {summary['inference_peak_gb']:.2f} GB")
    print(f"wrote {args.out}")
    print(
        "\nA 4090 has 24 GB. Subtract the number above from that to see what is left "
        "for Isaac Sim + UIPC before deciding local vs. tunnelled server."
    )


if __name__ == "__main__":
    main()
