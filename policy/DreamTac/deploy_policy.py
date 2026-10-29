"""Dream-Tac client policy for the UniVTAC Isaac Sim evaluation loop.

The 2B diffusion model runs in a separate process (`franka_server.py` from the
Dream-Tac repo, local or tunnelled from the cluster); everything here is the
thin client: turn a UniVTAC observation into that server's request, turn the
returned absolute EE poses into joint targets, and drive them through the same
`task.take_action(..., 'qpos')` path the ACT baseline uses.

Observation mapping (server key <- UniVTAC observation):

    cam_front      <- observation/head/rgb          (server: primary_image)
    cam_high       <- observation/wrist/rgb         (server: wrist_image)
    tactile_left   <- tactile/left_gsmini/rgb_marker
    tactile_right  <- tactile/right_gsmini/rgb_marker
    state          <- embodiment/ee, xyz + quat(wxyz) -> xyz + Euler(sxyz)

Action: 7-D, [x, y, z, roll, pitch, yaw, gripper], absolute pose in the robot
base frame plus a finger joint position in metres -- already un-normalised by
the server.
"""

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy

import cv2
import numpy as np
import requests
import torch
import transforms3d as t3d
from PIL import Image

from .ik_servo import IKServo

COSMOS_IMAGE_SIZE = 224


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


def _quaternion_wxyz_to_euler_sxyz(quaternion: np.ndarray) -> np.ndarray:
    """Copied from UniVTACDataset so the policy sees bit-identical proprio.

    transforms3d's quat2euler('sxyz') agrees, but the training pipeline used
    this explicit form and it costs nothing to keep them literally the same.
    """
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.clip(np.linalg.norm(quaternion), 1e-12, None)
    w, x, y, z = quaternion

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _as_uint8_hwc(image) -> np.ndarray:
    array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:  # rgba -> rgb
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _to_training_view(image) -> np.ndarray:
    """Reproduce the exact array the model was trained on.

    Collection wrote frames with `cv2.imencode(".jpg", rgb)` (envs/utils/data.py)
    and the Dream-Tac adapter read them back with `cv2.imdecode(..., IMREAD_COLOR)`
    then `cv2.resize(..., INTER_AREA)` (univtac_dataset.py). cv2 assumes BGR on
    both ends, so the two channel swaps cancel and the round trip preserves the
    simulator's native RGB order -- but the JPEG quantisation happens with R and
    B exchanged, so it is not reproducible by encoding through PIL. Run the same
    cv2 calls instead of approximating them.
    """
    array = _as_uint8_hwc(image)
    ok, encoded = cv2.imencode(".jpg", array)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode an observation frame")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded.shape[:2] != (COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE):
        decoded = cv2.resize(
            decoded, (COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), interpolation=cv2.INTER_AREA
        )
    return decoded


def _png_base64(array: np.ndarray) -> str:
    """PNG, not JPEG: the server JPEGs again internally, and a second lossy pass
    on top of the one _to_training_view already applied would drift away from
    the training distribution rather than towards it."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _smooth_chunk(actions: np.ndarray) -> np.ndarray:
    """[1, 2, 1] / 4 along the chunk, ends replicated.

    The iter_000004000 checkpoint emits a period-2 zigzag on top of the right
    trajectory: in the held-out offline check every one of 19 step-to-step
    differences flips sign on every axis, 1-1.7 mm in x, so the predicted
    chunk moves 3-4 mm per step where the demos move 0.1-0.6 mm. Its phase is
    fixed to (-1)^(t+d) across queries, i.e. to the parity of the latent column
    each value lands in: the 20x7 chunk is tiled 5 latent rows (140 = 5*28) at a
    time, so element (t, d) always sits in column (7t + d) mod 28, and the
    2x2-patch DiT leaves an even/odd-column bias the read-out average cannot
    remove. This kernel has exactly zero gain at period 2 and passes the slow
    trajectory nearly untouched (offline: step 0.4-1.3 mm, MAE unchanged).
    """
    padded = np.concatenate([actions[:1], actions, actions[-1:]], axis=0)
    return (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0


def _blend_chunk_start(actions: np.ndarray, state: np.ndarray, steps: int) -> np.ndarray:
    """Ramp out the gap between the current pose and the chunk's first pose.

    Each chunk carries its own near-constant pose offset (offline: ~2.5 mm
    across the whole chunk, ~8 mm during insertion), and consecutive chunks'
    offsets differ, so executing them back to back jumps the arm at every
    chunk boundary. Adding (state - actions[0]) to the leading actions,
    decaying linearly to zero over `steps` actions, turns that jump into a ramp while
    leaving every later target -- and so where the chunk ends up -- unchanged.
    Pose dims only; the gripper column is left alone.
    """
    if steps <= 0:
        return actions
    actions = actions.copy()
    gap = state[:6] - actions[0, :6]
    # (t + 1): the first action already covers 1/steps of the gap rather than
    # commanding the current pose and idling for a step.
    weights = np.clip(1.0 - (np.arange(len(actions)) + 1) / steps, 0.0, 1.0)
    actions[:, :6] += weights[:, None] * gap
    return actions


def _server_url(args) -> str:
    url = os.environ.get("DREAMTAC_SERVER") or args.get("server_url", "http://127.0.0.1:5001")
    return url.rstrip("/")


def _parse_ckpt_path(ckpt_path: str):
    """`(run, iteration)` from a checkpoint path, or None if unrecognised.

    `run` is the training task set as Dream-Tac spells it: `insert_hole`,
    `insert_hole__lift_can` for a multi-task run, `all` for every task.
    Handles the local layout (runs/<run>/checkpoints/<iter>/model) and the
    training one (cosmos_v2_finetune_<run>/<config>/checkpoints/<iter>).
    """
    parts = Path(ckpt_path.rstrip("/")).parts
    if parts and parts[-1] == "model":
        parts = parts[:-1]
    if "checkpoints" not in parts:
        return None
    idx = len(parts) - 1 - parts[::-1].index("checkpoints")
    if idx < 1 or idx + 1 >= len(parts):
        return None
    run = parts[idx - 1]
    if run.startswith("cosmos_predict2") and idx >= 2:
        run = parts[idx - 2].removeprefix("cosmos_v2_finetune_")
    return run, parts[idx + 1]


def _server_info(args) -> dict:
    response = requests.get(f"{_server_url(args)}/info", timeout=10.0)
    response.raise_for_status()
    return response.json()


def get_run_tag(args) -> str:
    """Name the eval_result sub-directory after the checkpoint the server holds.

    The weights live in the server process, so TRAIN_CONFIG / CKPT_NAME (what
    the eval scripts use for ACT) say nothing about them. Ask /info instead:
    e.g. `insert_hole-iter_000004000`.
    """
    try:
        parsed = _parse_ckpt_path(_server_info(args)["ckpt_path"])
    except Exception:
        # Policy.__init__ health-checks right after this and fails with a
        # proper message; do not pre-empt it here.
        return "unknown_ckpt"
    return "-".join(parsed) if parsed else "unknown_ckpt"


class Policy(BasePolicy):
    def __init__(self, args):
        self.task_name = args["task_name"]
        self.args = args

        self.server_url = _server_url(args)
        self.timeout = float(args.get("request_timeout", 120.0))

        # The T5 cache is keyed by the UniVTAC folder name with underscores
        # turned into spaces; task.instruction is "Empty" here and has no
        # embedding, so it must not be used.
        self.instruction = args.get("instruction") or self.task_name.replace("_", " ")

        self.actions_per_chunk = args.get("actions_per_chunk")
        self.smooth_actions = bool(args.get("smooth_actions", True))
        self.blend_steps = int(args.get("blend_steps", 0) or 0)
        self.sim_steps_per_action = int(args.get("sim_steps_per_action", 2))
        self.save_frame_every = int(args.get("save_frame_every", 10))
        self.log_actions = bool(args.get("log_actions", False))
        self.ik_args = dict(args.get("ik") or {})

        self._session = requests.Session()
        self._ik = None            # built lazily: needs the live robot manager
        self._pending = []         # actions left in the current chunk
        self._prev_rpy = None      # running unwrap state, per episode
        self._action_count = 0
        self._chunk_log = None

        if args.get("health_check", True):
            self._check_server()

        print(
            f"[DreamTac] server={self.server_url} instruction={self.instruction!r} "
            f"sim_steps_per_action={self.sim_steps_per_action} "
            f"actions_per_chunk={self.actions_per_chunk or 'all'}"
        )

    # -- server ---------------------------------------------------------------

    def _check_server(self):
        try:
            response = self._session.get(f"{self.server_url}/health", timeout=10.0)
            response.raise_for_status()
            info = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Dream-Tac inference server unreachable at {self.server_url}: {exc}\n"
                f"Start it first:  DREAMTAC_TASK={self.task_name} "
                f"bash policy/DreamTac/tools/serve_local.sh"
            ) from exc
        if not info.get("use_tactile", False):
            print(
                "[DreamTac] WARNING: server reports use_tactile=False, but the UniVTAC "
                "checkpoint was trained with state_t=12 (tactile). Check FRANKA_USE_TACTILE."
            )
        self._check_server_task()

    def _check_server_task(self):
        """Refuse to evaluate one task with a checkpoint trained on another.

        One server serves one checkpoint with one task's normalisation stats,
        and nothing else ties it to the task being simulated: a server left
        running from a previous task would otherwise be evaluated silently.
        """
        ckpt_path = _server_info(self.args).get("ckpt_path", "")
        parsed = _parse_ckpt_path(ckpt_path)
        if parsed is None:
            print(f"[DreamTac] WARNING: cannot tell which task {ckpt_path!r} was trained on")
            return
        run, iteration = parsed
        if run != "all" and self.task_name not in run.split("__"):
            raise RuntimeError(
                f"Server holds a checkpoint trained on {run!r} ({iteration}), but the task "
                f"under evaluation is {self.task_name!r}. Restart the server with "
                f"DREAMTAC_TASK={self.task_name}."
            )
        print(f"[DreamTac] checkpoint: {run}/{iteration}")

    def _request(self, payload):
        response = self._session.post(f"{self.server_url}/infer", json=payload, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        if not body.get("success"):
            raise RuntimeError(f"Inference server error: {body.get('error')}")
        return body

    # -- observation ----------------------------------------------------------

    def _unwrap_rpy(self, rpy: np.ndarray) -> np.ndarray:
        """Online equivalent of the np.unwrap() the dataset applies per episode.

        np.unwrap is causal -- each sample is corrected relative to the already
        corrected previous one -- so keeping a single previous value reproduces
        it exactly. reset() clears the state at every episode boundary.
        """
        if self._prev_rpy is not None:
            rpy = self._prev_rpy + np.mod(rpy - self._prev_rpy + np.pi, 2.0 * np.pi) - np.pi
        self._prev_rpy = rpy.copy()
        return rpy

    def encode_obs(self, observation):
        ee = observation["embodiment"].get("ee")
        if ee is None:
            raise KeyError(
                "observation['embodiment']['ee'] is missing -- add 'ee' to "
                "observations.embodiment in the task config."
            )
        ee = ee.detach().cpu().numpy().astype(np.float64).reshape(-1)

        rpy = _quaternion_wxyz_to_euler_sxyz(ee[3:7])
        # Read roll on [0, 2*pi): the gripper points down throughout, so roll
        # sits exactly on arctan2's (-pi, pi] cut and would otherwise flip sign
        # between neighbouring frames. Same trick as the dataset.
        rpy[0] = np.mod(rpy[0], 2.0 * np.pi)
        rpy = self._unwrap_rpy(rpy)

        return {
            "images": {
                "cam_front": _png_base64(_to_training_view(observation["observation"]["head"]["rgb"])),
                "cam_high": _png_base64(_to_training_view(observation["observation"]["wrist"]["rgb"])),
                "tactile_left": _png_base64(
                    _to_training_view(_get_tactile_rgb_marker(observation, "left"))
                ),
                "tactile_right": _png_base64(
                    _to_training_view(_get_tactile_rgb_marker(observation, "right"))
                ),
            },
            "state": np.concatenate([ee[:3], rpy]).astype(np.float32).tolist(),
            "instruction": self.instruction,
        }

    # -- action ---------------------------------------------------------------

    def _ensure_ik(self, task):
        if self._ik is None:
            self._ik = IKServo(task._robot_manager, **self.ik_args)
        return self._ik

    def _execute(self, task, action):
        robot = task._robot_manager
        ik = self._ensure_ik(task)

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        quaternion = t3d.euler.euler2quat(*action[3:6])  # sxyz -> wxyz
        seed = ik.current_arm_qpos()

        arm_qpos = ik.solve(action[:3], quaternion, seed_qpos=seed)
        if arm_qpos is None:
            # Unreachable or self-colliding target. Holding the current
            # configuration keeps the episode alive; unlike take_action('ee'),
            # nothing here latches plan_success and kills the remaining steps.
            print(f"[DreamTac] IK failed at action {self._action_count}, holding pose")
            arm_qpos = seed

        gripper = float(np.clip(action[6], 0.0, robot.gripper_max_qpos))
        command = torch.cat([arm_qpos.to(task.device), torch.tensor([gripper], device=task.device)]).float()

        exec_success, eval_success = task.take_action(command, action_type="qpos")
        # take_action advances one sim step and counts one action. Data was
        # recorded every save_frequency (=2) steps, so hold the target for the
        # remaining steps to match the training timebase without inflating
        # take_action_cnt against step_lim.
        for _ in range(max(self.sim_steps_per_action - 1, 0)):
            if task.eval_success or task.take_action_cnt >= task.cfg.step_lim:
                break
            task._step()
        self._action_count += 1
        return exec_success, eval_success

    def eval(self, task, observation):
        if not self._pending:
            payload = self.encode_obs(observation)
            started = time.perf_counter()
            body = self._request(payload)
            elapsed = time.perf_counter() - started

            raw = np.asarray(body["actions"], dtype=np.float64).reshape(len(body["actions"]), -1)
            executed = _smooth_chunk(raw) if self.smooth_actions else raw
            executed = _blend_chunk_start(executed, np.asarray(payload["state"], dtype=np.float64), self.blend_steps)
            actions = list(executed)
            limit = self.actions_per_chunk or len(actions)
            self._pending = actions[:limit]
            print(
                f"[DreamTac] chunk @ action {self._action_count}: {len(actions)} actions "
                f"({limit} queued) in {elapsed:.2f}s "
                f"(server {body.get('processing_time', float('nan')):.2f}s)"
            )
            if self.log_actions:
                self._log_chunk(task, raw, actions, payload["state"], elapsed)

        if self.save_frame_every and self._action_count % self.save_frame_every == 0:
            self._save_frame(task, observation)

        self._execute(task, self._pending.pop(0))

    # -- lifecycle ------------------------------------------------------------

    def reset(self):
        self._pending = []
        self._prev_rpy = None
        self._action_count = 0
        if self._ik is not None and self._ik.num_solves:
            print(
                f"[DreamTac] previous episode IK: {self._ik.num_failures}/{self._ik.num_solves} "
                f"failures ({self._ik.failure_rate * 100:.1f}%)"
            )
            self._ik.num_solves = self._ik.num_failures = 0

    def close(self):
        self._session.close()

    # -- debugging ------------------------------------------------------------

    def _log_chunk(self, task, raw, actions, state, elapsed):
        if self._chunk_log is None:
            path = Path(task.save_root) / "chunks.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._chunk_log = open(path, "a")
        self._chunk_log.write(
            json.dumps(
                {
                    "action_count": self._action_count,
                    "state": state,
                    "raw_actions": raw.tolist(),
                    "actions": [a.tolist() for a in actions],
                    "elapsed": elapsed,
                }
            )
            + "\n"
        )
        self._chunk_log.flush()

    def _save_frame(self, task, observation):
        from PIL import ImageDraw, ImageFont

        image = Image.fromarray(task.get_frame_shot(observation).cpu().numpy())
        draw = ImageDraw.Draw(image)
        draw.text(
            (image.width - 100, image.height - 60),
            f"{task.take_action_cnt:03d}",
            fill=(255, 0, 0),
            font=ImageFont.load_default(),
        )
        image.save(f"DreamTac_{self.task_name}.png")
