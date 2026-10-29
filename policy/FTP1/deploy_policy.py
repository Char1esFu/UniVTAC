"""FTP-1 client policy for the UniVTAC Isaac Sim evaluation loop.

The FTP-1 model (PaliGemma 2B + action/tactile experts) runs in `ftp1_server.py`
inside the ftp1-policy uv env; this file only turns UniVTAC observations into
server requests and the returned action chunks into qpos targets.

Execution mirrors ftp1-policy/UniVTAC/scripts/eval_ftp1.py with the flags used by
its eval_ftp1.sh (the published UniVTAC numbers):

- prompt: the training instruction from parse_data_univtac.py, not task.instruction
- cameras: head always; wrist only for the tasks trained with it (lift_can, insert_tube)
- tactile: [left, right] rgb_marker pads stacked as one GelSightMini group
- actions: re-infer every step, temporal ensemble over the first `chunk_first_n`
  executable steps of each chunk (chunk[0] is a current-time placeholder and skipped),
  weights exp(-K * i) from oldest to newest
"""

import io
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy

import cv2
import numpy as np
import requests
import torch

IMAGE_SIZE = 224
# chunk[0] is the current-timestamp placeholder; the first executable action is chunk[1].
CHUNK_INDEX_OFFSET = 1

# Training prompts, copied from data_processing/parse_data_module/parse_data_univtac.py.
TRAINING_INSTRUCTIONS = {
    "grasp_classify": "use grasped tool for tactile sensing to move to target surface.",
    "insert_HDMI": "insert the HDMI to the fixed slot.",
    "insert_hole": "insert the stick to the hole.",
    "insert_tube": "insert the tube to the fixed slot.",
    "lift_can": "grasp the can and lifts it vertically without slippage.",
    "lift_bottle": "grasp the bottle and lift it vertically, keeping its final base within 5 cm of the wall.",
    "pull_out_key": "pull out the key.",
    "put_bottle_in_shelf": "grasp the bottle, then position it into the shelf cavity.",
}
# Tasks whose zarr included the wrist camera (camera_type == "all" in ftp1-policy's
# data_processing/parse_data_module/task_settings.json). Do not read this repo's
# policy/task_settings.json: it was changed to "all" for every task.
WRIST_CAMERA_TASKS = {"lift_can", "insert_tube"}


def _to_numpy_uint8(img) -> np.ndarray:
    arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 255.0)
            if arr.max() <= 1.0:
                arr = arr * 255.0
        arr = arr.astype(np.uint8)
    return arr


def _model_image(img) -> np.ndarray:
    arr = _to_numpy_uint8(img)
    if arr.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        arr = cv2.resize(arr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(arr)


def _get_tactile_rgb_marker(observation, side):
    tactile = observation["tactile"]
    candidates = (f"{side}_tactile", f"{side}_gsmini")
    for name in candidates:
        if name in tactile:
            return tactile[name]["rgb_marker"]
    raise KeyError(
        f"Missing tactile observation for {side}. Expected one of {candidates}, "
        f"got keys: {list(tactile.keys())}"
    )


def _get_qpos8(observation) -> np.ndarray:
    qpos = observation["embodiment"]["joint"][:8]
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    return np.asarray(qpos, dtype=np.float32).reshape(8)


def _resolve_abs_qpos(action8: np.ndarray, qpos8_base: np.ndarray, action_rep: str) -> np.ndarray:
    """Turn a model-space 8D action into an absolute qpos target."""
    if action_rep == "absolute":
        return action8.copy()
    if action_rep == "relative":
        return qpos8_base + action8
    # mix: arm joints are deltas, gripper is absolute
    resolved = action8.copy()
    resolved[:7] = qpos8_base[:7] + action8[:7]
    return resolved


def _checkpoint_tag(checkpoint_dir) -> str:
    # .../FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1/19999 -> FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1-19999
    path = Path(checkpoint_dir)
    return f"{path.parent.name}-{path.name}" if path.name.isdigit() else path.name


def get_run_tag(deploy_config) -> str:
    """Name eval_result/ runs after the weights; eval_policy.py's TRAIN_CONFIG default does not apply."""
    if deploy_config.get("checkpoint_dir"):
        return _checkpoint_tag(deploy_config["checkpoint_dir"])
    return "ftp1_univtac_finetune"


class Policy(BasePolicy):
    def __init__(self, args):
        self.task_name = args["task_name"]
        self.server_url = os.environ.get("FTP1_SERVER", args.get("server_url", "http://127.0.0.1:5002")).rstrip("/")
        self.request_timeout = float(args.get("request_timeout", 60.0))
        self.chunk_first_n = int(args.get("chunk_first_n", 20))
        self.ensemble_K = float(args.get("ensemble_K", 0.01))

        use_wrist = args.get("use_wrist")
        self.use_wrist = self.task_name in WRIST_CAMERA_TASKS if use_wrist is None else bool(use_wrist)
        self.prompt = args.get("instruction") or TRAINING_INSTRUCTIONS.get(self.task_name)
        if self.prompt is None:
            raise ValueError(f"No training instruction for task '{self.task_name}'; set `instruction` in deploy.yml")

        # Loading a checkpoint takes a while (8 GB of weights), so allow for it here.
        resp = requests.post(
            f"{self.server_url}/load",
            json={
                "task": self.task_name,
                "checkpoint_dir": args.get("checkpoint_dir"),
                "domain_name": args.get("domain_name"),
            },
            timeout=float(args.get("load_timeout", 600.0)),
        )
        if resp.status_code != 200:
            raise RuntimeError(f"FTP-1 server failed to load a model for '{self.task_name}': {resp.text}")
        info = resp.json()
        self.action_rep = info["action_rep"]
        # eval_policy.py logs this dict after constructing the policy, so record what is
        # actually used instead of the yml's nulls.
        args.update(
            instruction=self.prompt,
            use_wrist=self.use_wrist,
            checkpoint_dir=info["checkpoint_dir"],
            domain_name=info["domain_name"],
        )
        print(
            f"[FTP1] {info['checkpoint_dir']} (domain={info['domain_name']}, action_rep={self.action_rep}, "
            f"horizon={info['action_horizon']}) | wrist={self.use_wrist} | prompt={self.prompt!r}",
            flush=True,
        )
        self.reset()

    def reset(self):
        # (chunk8, exec step it was inferred at, qpos8 at that step)
        self._chunk_history = []
        self._exec_step = 0

    def _infer_chunk(self, observation, qpos8: np.ndarray) -> np.ndarray:
        payload = {
            "qpos8": qpos8,
            "head": _model_image(observation["observation"]["head"]["rgb"]),
            "tactile": np.stack(
                [
                    _model_image(_get_tactile_rgb_marker(observation, "left")),
                    _model_image(_get_tactile_rgb_marker(observation, "right")),
                ],
                axis=0,
            ),
            "prompt": np.array(self.prompt),
        }
        if self.use_wrist:
            payload["wrist"] = _model_image(observation["observation"]["wrist"]["rgb"])
        buf = io.BytesIO()
        np.savez(buf, **payload)
        resp = requests.post(f"{self.server_url}/infer", data=buf.getvalue(), timeout=self.request_timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"FTP-1 inference failed: {resp.text}")
        with np.load(io.BytesIO(resp.content), allow_pickle=False) as npz:
            return npz["chunk8"]

    def get_action(self, observation) -> np.ndarray:
        qpos8 = _get_qpos8(observation)
        self._chunk_history.append((self._infer_chunk(observation, qpos8), self._exec_step, qpos8))

        # The chunk inferred at step s predicts step e at index (e - s) + offset.
        max_idx = CHUNK_INDEX_OFFSET + self.chunk_first_n if self.chunk_first_n > 0 else None
        live, candidates = [], []
        for chunk8, s, qpos8_base in self._chunk_history:
            idx = self._exec_step - s + CHUNK_INDEX_OFFSET
            if idx >= chunk8.shape[0] or (max_idx is not None and idx >= max_idx):
                continue
            live.append((chunk8, s, qpos8_base))
            candidates.append(_resolve_abs_qpos(chunk8[idx], qpos8_base, self.action_rep))
        self._chunk_history = live

        weights = np.exp(-self.ensemble_K * np.arange(len(candidates), dtype=np.float32))
        weights /= weights.sum()
        self._exec_step += 1
        return (np.stack(candidates, axis=0) * weights[:, None]).sum(axis=0)

    def eval(self, task, observation):
        action8 = self.get_action(observation)
        task.take_action(torch.from_numpy(action8).to(task.device).float(), action_type="qpos")
