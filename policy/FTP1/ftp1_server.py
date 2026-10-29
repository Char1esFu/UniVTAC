"""FTP-1 inference server for the UniVTAC evaluation loop.

FTP-1 needs transformers==4.53.2 with the openpi `transformers_replace` patch,
which cannot share a Python env with this repo's Isaac Sim stack. So the model
runs here, in the ftp1-policy uv env, and `deploy_policy.py` is a thin HTTP
client living inside the Isaac Sim process (same split as DreamTac).

Run (from anywhere; the ftp1-policy checkout provides `openpi`):

    uv run --project ~/proj/ftp1-policy python policy/FTP1/ftp1_server.py \
        --ckpt_root policy/FTP1/checkpoints/ftp1_univtac_finetune

Protocol (localhost, one model resident at a time):

    POST /load   JSON {"task": str, "checkpoint_dir": str|null, "domain_name": str|null}
                 -> JSON {"checkpoint_dir", "domain_name", "action_rep", "action_horizon", "use_tactile_input"}
    POST /infer  npz {qpos8 (8,), head (224,224,3) u8, [wrist (224,224,3) u8],
                      [tactile (2,224,224,3) u8], prompt (0-d str)}
                 -> npz {chunk8 (H, 8) f32}: FTP-1 chunk mapped to UniVTAC 7 arm + 1 gripper,
                    still in the checkpoint's action_joint_rep (the client resolves it to qpos).
    GET  /health -> JSON of the currently loaded model (or {"loaded": false})

Mapping and preprocessing mirror ftp1-policy/UniVTAC/scripts/eval_ftp1.py, which
produced the published UniVTAC numbers.
"""

from __future__ import annotations

import os

# JAX is only imported for openpi's Observation dataclass; keep it off the GPU so it
# does not preallocate memory that Isaac Sim needs.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import io
import json
import logging
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from pathlib import Path

import numpy as np
import torch

from openpi.models_pytorch.ftp1_model_config import FTP1_RESERVED_ACTION_DIM
from openpi.models_pytorch.ftp1_model_config import FTP1_SINGLE_ARM_ACTION_REP_DIM
from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper

logger = logging.getLogger("ftp1_server")

# UniVTAC task -> checkpoint folder suffix in the ftp1_univtac_finetune release.
TASK_TO_RELEASE_NAME = {
    "insert_hole": "insert_hole",
    "insert_tube": "insert_tube",
    "lift_bottle": "lift_bottle",
    "lift_can": "lift_can",
    "pull_out_key": "pull_out_key",
    "put_bottle_in_shelf": "put_bottle",
}
# parse_data_univtac.py writes the gripper into slot 28 of the 32-slot hand vector.
GRIPPER_SLOT_IDX = 28
TACTILE_KEY = "right_tactile_gripper"
TACTILE_SENSOR = "GelSightMini"


def _infer_layout(action_dim: int) -> str:
    single = FTP1_SINGLE_ARM_ACTION_REP_DIM
    dual = 2 * single + 9
    if action_dim == single:
        return "single48"
    if action_dim in {dual, dual + FTP1_RESERVED_ACTION_DIM}:
        return "dual105"
    raise ValueError(f"Unsupported FTP1 action_dim={action_dim} for UniVTAC (expected single48/dual105 layout)")


def _canonicalize_action_joint_rep(rep: str | None) -> str | None:
    normalized = str(rep).strip().lower() if rep is not None else ""
    if normalized in {"abs", "absolute"}:
        return "absolute"
    if normalized in {"relative", "mix"}:
        return normalized
    return None


def _resolve_release_checkpoint(ckpt_root: Path, task: str) -> Path:
    if task not in TASK_TO_RELEASE_NAME:
        raise ValueError(
            f"No released FTP-1 checkpoint for task '{task}' (available: {sorted(TASK_TO_RELEASE_NAME)}); "
            "set checkpoint_dir/domain_name in deploy.yml to use your own."
        )
    run_dir = ckpt_root / f"FTP1_UniVTAC_{TASK_TO_RELEASE_NAME[task]}_expert_gsmall_ftp1"
    steps = sorted((int(d.name) for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()), reverse=True)
    if not steps:
        raise FileNotFoundError(f"No checkpoint step directory under {run_dir}")
    return run_dir / str(steps[0])


def _resolve_domain_name(checkpoint_dir: Path) -> str:
    # Per-domain norm stats live in normalization/<domain>/; the release has exactly one.
    domains = [d.name for d in (checkpoint_dir / "normalization").iterdir() if d.is_dir()]
    if len(domains) != 1:
        raise ValueError(f"Cannot infer domain_name from {checkpoint_dir}/normalization (found {domains}); set it explicitly")
    return domains[0]


class FTP1Runner:
    def __init__(self, ckpt_root: Path, device: str, num_inference_steps: int):
        self.ckpt_root = ckpt_root
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.wrapper: FTP1InferenceWrapper | None = None
        self.info: dict = {"loaded": False}

    def load(self, task: str, checkpoint_dir: str | None, domain_name: str | None) -> dict:
        ckpt = Path(checkpoint_dir) if checkpoint_dir else _resolve_release_checkpoint(self.ckpt_root, task)
        domain = domain_name or _resolve_domain_name(ckpt)
        if self.info.get("checkpoint_dir") == str(ckpt) and self.info.get("domain_name") == domain:
            return self.info

        train_config = json.loads((ckpt / "train_config.json").read_text())
        action_rep = _canonicalize_action_joint_rep(train_config.get("action_joint_rep"))
        if action_rep is None:
            raise ValueError(f"Unsupported action_joint_rep in {ckpt}/train_config.json")

        self.wrapper = None  # release the previous model before allocating the next one
        logger.info("Loading FTP-1 checkpoint %s (domain=%s)", ckpt, domain)
        self.wrapper = FTP1InferenceWrapper(
            checkpoint_dir=str(ckpt),
            domain_name=domain,
            device=self.device,
            num_inference_steps=self.num_inference_steps,
        )
        # The loader stages the whole state dict on the GPU; hand that cache back so
        # Isaac Sim can use it (~8 GB on a 24 GB card).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                "GPU memory: %.1f GB allocated, %.1f GB reserved",
                torch.cuda.memory_allocated() / 1e9,
                torch.cuda.memory_reserved() / 1e9,
            )
        action_dim = int(self.wrapper.get_action_dim())
        self.info = {
            "loaded": True,
            "task": task,
            "checkpoint_dir": str(ckpt),
            "domain_name": domain,
            "action_rep": action_rep,
            "action_dim": action_dim,
            "layout": _infer_layout(action_dim),
            "action_horizon": int(self.wrapper.get_action_horizon()),
            "use_tactile_input": bool(getattr(self.wrapper.model_config, "use_tactile_input", True)),
        }
        logger.info("Loaded: %s", self.info)
        return self.info

    def infer(self, req: dict[str, np.ndarray]) -> np.ndarray:
        if self.wrapper is None:
            raise RuntimeError("No model loaded; POST /load first")
        action_dim = self.info["action_dim"]

        # State: absolute qpos, arm in [9:16] of the right-arm block, gripper in hand slot 28.
        qpos8 = np.asarray(req["qpos8"], dtype=np.float32).reshape(8)
        state = np.zeros((1, action_dim), dtype=np.float32)
        state[0, 9:16] = qpos8[:7]
        state[0, 16 + GRIPPER_SLOT_IDX] = qpos8[7]

        images = {"camera_ego_rgb_0": req["head"]}
        if "wrist" in req:
            images["right_wrist_camera_rgb_0"] = req["wrist"]

        tactile_kwargs = {}
        if self.info["use_tactile_input"]:
            if "tactile" not in req:
                raise ValueError("Checkpoint expects tactile input but the request has none")
            tactile_kwargs = {
                "tactiles": {TACTILE_KEY: req["tactile"][None].astype(np.float32)},  # (1, 2, H, W, 3)
                "tactile_function_areas": {TACTILE_KEY: [0, 1]},
                "tactile_sensors": {TACTILE_KEY: TACTILE_SENSOR},
            }

        chunk = self.wrapper.infer(images=images, state=state, prompt=str(req["prompt"]), **tactile_kwargs)
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != action_dim:
            raise ValueError(f"Unexpected FTP-1 output shape {chunk.shape}, expected (H, {action_dim})")
        return np.concatenate([chunk[:, 9:16], chunk[:, 16 + GRIPPER_SLOT_IDX : 17 + GRIPPER_SLOT_IDX]], axis=1)


def _make_handler(runner: FTP1Runner):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def _body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, runner.info)
            else:
                self._send_json(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):
            try:
                if self.path == "/load":
                    req = json.loads(self._body())
                    info = runner.load(req["task"], req.get("checkpoint_dir"), req.get("domain_name"))
                    self._send_json(200, info)
                elif self.path == "/infer":
                    with np.load(io.BytesIO(self._body()), allow_pickle=False) as npz:
                        req = {k: npz[k] for k in npz.files}
                    buf = io.BytesIO()
                    np.savez(buf, chunk8=runner.infer(req))
                    self._send(200, buf.getvalue(), "application/octet-stream")
                else:
                    self._send_json(404, {"error": f"unknown path {self.path}"})
            except Exception as e:
                logger.exception("Request to %s failed", self.path)
                self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, format, *args):  # per-step requests would flood the console
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ckpt_root",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "ftp1_univtac_finetune",
        help="Root of the ftp1_univtac_finetune release (used when the client sends no checkpoint_dir)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--preload_task", default=None, help="Load this task's checkpoint before serving")
    args = parser.parse_args()

    # openpi installs a WARNING-level root handler on import; replace it so INFO lines show.
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(name)s] %(message)s", force=True)
    runner = FTP1Runner(args.ckpt_root, args.device, args.num_inference_steps)
    if args.preload_task:
        runner.load(args.preload_task, None, None)
    server = HTTPServer((args.host, args.port), _make_handler(runner))
    logger.info("FTP-1 server listening on http://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
