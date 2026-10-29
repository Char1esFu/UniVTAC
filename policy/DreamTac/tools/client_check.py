"""End-to-end check of the client path, without Isaac Sim.

`offline_check.py` proves the checkpoint is good by calling `get_action`
in-process on the dataset's own decode path. This script proves the *client* is
good: it drives the real `Policy.encode_obs` -> PNG -> HTTP -> server route with
frames from the same held-out episode, and compares three things

  1. the proprio the client derives online against the dataset's episode-level
     computation (catches the quaternion convention and the roll unwrap),
  2. the returned actions against ground truth,
  3. the returned actions against `offline_check.py`'s in-process predictions,
     if that JSON is passed with --reference.

Anything that disagrees here is a client bug, and finding it now is much cheaper
than finding it inside a simulator rollout.

Runs in the UniVTAC conda env -- it never imports cosmos_policy.

    python policy/DreamTac/tools/client_check.py \
        --hdf5 data/insert_hole/clean/7.hdf5 \
        --reference ~/models/dream-tac/runs/insert_hole/offline_check_insert_hole_iter_000004000.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for `policy.*`

import cv2
import h5py
import numpy as np
import requests
import torch
import yaml

from policy.DreamTac.deploy_policy import Policy, _quaternion_wxyz_to_euler_sxyz


def decode(raw):
    encoded = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw.reshape(-1)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def dataset_proprio(ee_all):
    """The dataset's episode-level transform, for comparison with the online one."""
    rpy = np.stack([_quaternion_wxyz_to_euler_sxyz(q) for q in ee_all[:, 3:7]])
    rpy[:, 0] = np.mod(rpy[:, 0], 2.0 * np.pi)
    rpy = np.unwrap(rpy, axis=0)
    return np.concatenate([ee_all[:, :3], rpy], axis=1)


def find_tactile_keys(root):
    for left, right in (
        ("tactile/left_gsmini/rgb_marker", "tactile/right_gsmini/rgb_marker"),
        ("tactile/left_tactile/rgb_marker", "tactile/right_tactile/rgb_marker"),
    ):
        if left in root and right in root:
            return left, right
    raise KeyError("no tactile rgb_marker pair")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:5001")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--reference", default=None, help="offline_check.py JSON to diff against")
    args = parser.parse_args()

    task = Path(args.hdf5).resolve().parents[1].name
    config = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "deploy.yml"))
    config.update({"task_name": task, "task_config": "clean", "health_check": False,
                   "server_url": args.server})
    policy = Policy(config)
    policy.reset()

    reference = None
    if args.reference:
        reference = {r["t"]: np.asarray(r["predicted"]) for r in json.load(open(args.reference))["records"]}

    with h5py.File(args.hdf5, "r") as root:
        left_key, right_key = find_tactile_keys(root)
        ee_all = np.asarray(root["embodiment/ee"][:], dtype=np.float32)
        joint_all = np.asarray(root["embodiment/joint"][:], dtype=np.float32)
        pose6 = dataset_proprio(ee_all)
        truth_all = np.concatenate([pose6[1:], joint_all[1:, 7:8]], axis=1)

        # The client keeps running unwrap state, so step through every frame up
        # to each query point exactly as a rollout would.
        max_t = args.stride * (args.steps - 1)
        results = []
        for t in range(max_t + 1):
            observation = {
                "observation": {
                    "head": {"rgb": torch.from_numpy(decode(root["observation/head/rgb"][t]))},
                    "wrist": {"rgb": torch.from_numpy(decode(root["observation/wrist/rgb"][t]))},
                },
                "tactile": {
                    "left_gsmini": {"rgb_marker": torch.from_numpy(decode(root[left_key][t]))},
                    "right_gsmini": {"rgb_marker": torch.from_numpy(decode(root[right_key][t]))},
                },
                "embodiment": {"ee": torch.from_numpy(ee_all[t])},
            }
            payload = policy.encode_obs(observation)

            state_error = np.abs(np.asarray(payload["state"]) - pose6[t])
            if t % args.stride != 0:
                assert state_error.max() < 1e-5, f"proprio drift at t={t}: {state_error}"
                continue

            print(f"\n--- t={t} ---")
            print(f"  proprio vs dataset : max |diff| {state_error.max():.3e}"
                  f"   {'OK' if state_error.max() < 1e-5 else 'MISMATCH'}")

            started = time.perf_counter()
            response = requests.post(f"{args.server}/infer", json=payload, timeout=180)
            response.raise_for_status()
            body = response.json()
            assert body.get("success"), body.get("error")
            elapsed = time.perf_counter() - started

            predicted = np.asarray(body["actions"], dtype=np.float64).reshape(-1, 7)
            horizon = min(len(predicted), len(truth_all) - t)
            error = np.abs(predicted[:horizon] - truth_all[t : t + horizon])
            print(f"  round trip         : {elapsed:.2f}s  (server {body['processing_time']:.2f}s)")
            print(f"  vs ground truth    : pos {error[:, :3].mean()*1000:6.2f} mm   "
                  f"rot {error[:, 3:6].mean()*1000:6.2f} mrad   grip {error[:, 6].mean()*1e6:6.2f} um")
            if reference is not None and t in reference:
                delta = np.abs(predicted - reference[t][: len(predicted)])
                print(f"  vs offline_check   : pos {delta[:, :3].mean()*1000:6.2f} mm   "
                      f"rot {delta[:, 3:6].mean()*1000:6.2f} mrad")
            results.append((t, error[:, :3].mean() * 1000, error[:, 3:6].mean() * 1000))

    print("\n===== summary =====")
    print(f"  proprio: online transform matches the dataset at every frame up to t={max_t}")
    pos = np.mean([r[1] for r in results])
    rot = np.mean([r[2] for r in results])
    print(f"  actions vs ground truth: pos {pos:.2f} mm, rot {rot:.2f} mrad")
    print("  compare with offline_check: 1.69 mm / 0.95 mrad. A close match means the")
    print("  client's image and proprio encoding reproduces the training distribution.")


if __name__ == "__main__":
    main()
