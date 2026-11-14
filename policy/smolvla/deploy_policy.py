"""SmolVLA deployment policy for the UniVTAC evaluation harness.

This is the deploy-time bridge between UniVTAC's :class:`BaseTask` observation
dictionaries and the visuo-tactile SmolVLA policy living in
``UniVTAC/policy/smolvla/src/lerobot``. The class follows the same contract
as :class:`policy.ACT.deploy_policy.Policy`:

* ``__init__(args)``  -- load checkpoint + processors, read deploy.yml-ish args.
* ``encode_obs(obs)`` -- convert UniVTAC observation -> SmolVLA batch dict.
* ``eval(task, obs)`` -- run a single env step (encode -> chunk inference -> take_action).
* ``reset()``         -- drop temporal aggregation state and re-arm language.

SmolVLA differs from ACT in two ways the deploy code has to handle:

1.  It needs a language instruction; we tokenize once per episode via the
    standard SmolVLA pre-processor pipeline.
2.  It returns *chunks* of actions. At deploy time we re-run chunk inference
    every step and aggregate overlapping predictions with exponential weights,
    matching the ACT-style temporal aggregation used elsewhere in UniVTAC.

A LeRobot v3.0 dataset stats file (``meta/stats.json``) is optional but
*strongly* recommended at deploy time; without it the normalizer falls
back to identity which usually destroys performance.
"""

from __future__ import annotations

import json
import os
from collections import deque
import base64
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Make the locally vendored lerobot importable.
_SMOLVLA_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SMOLVLA_DIR.parent))  # for the relative _base_policy import

from .._base_policy import BasePolicy  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_chw_float(img, image_scale: str = "raw"):
    """UniVTAC images are HWC uint8 0-255 torch tensors. Convert to CHW float
    while preserving the scale used by training unless explicitly requested."""
    import torch

    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)
    if img.ndim != 3:
        raise ValueError(f"expected HWC tensor, got shape {tuple(img.shape)}")
    if img.dtype != torch.float32 and img.dtype != torch.float64:
        img = img.float()
    if image_scale == "float01" and img.max() > 1.5:
        img = img / 255.0
    elif image_scale != "raw":
        raise ValueError(f"unknown image_scale: {image_scale}")
    # HWC -> CHW
    return img.permute(2, 0, 1).contiguous()


def _resize_chw(img, size: tuple[int, int]):
    """Resize a CHW tensor with bilinear interpolation."""
    if tuple(img.shape[-2:]) == tuple(size):
        return img
    from torchvision import transforms

    return transforms.functional.resize(img, list(size), antialias=True)


def _load_stats(stats_path: Path | None) -> dict | None:
    """Load LeRobot-style dataset stats JSON into the dict-of-tensors format
    expected by :func:`make_smolvla_pre_post_processors`."""
    import torch

    if stats_path is None or not stats_path.exists():
        return None
    with open(stats_path, "r") as f:
        raw = json.load(f)

    def _walk(node):
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return torch.as_tensor(node, dtype=torch.float32)
        return node

    return _walk(raw)


def _get_policy_config(model):
    """Return the underlying SmolVLA config for plain or PEFT-wrapped models."""
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "input_features"):
        return config
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()
        config = getattr(base_model, "config", None)
        if config is not None and hasattr(config, "input_features"):
            return config
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        config = getattr(model.base_model.model, "config", None)
        if config is not None and hasattr(config, "input_features"):
            return config
    raise AttributeError("could not locate SmolVLA config on loaded model")


# ---------------------------------------------------------------------------
# Local model implementation used by the FastAPI service
# ---------------------------------------------------------------------------


def _instruction_for(instruction: str | None, task_name: str, use_env_instruction: bool = False) -> str:
    """Language prompt for the policy.

    scripts/process_data.py labels every frame with _task_string_from_name
    (e.g. lift_bottle -> "lift the bottle") unless policy/instructions/ exists,
    so by default the env's instruction is ignored: eval_policy.py passes
    "Empty" and parallel_eval_policy.py a sentence from instructions/*.json
    that the model never saw. Set smolvla_use_env_instruction: true for a
    checkpoint trained on those instructions.
    """
    if use_env_instruction and instruction and instruction != "Empty":
        return instruction
    words = task_name.replace("-", "_").split("_")
    if len(words) >= 2:
        return f"{words[0]} the {' '.join(words[1:])}"
    return task_name.replace("_", " ")


def _train_config_name(args: dict) -> str:
    """scripts/train.sh writes outputs/smolvla_vitac_{task}-{config}-{num}_{RUN_TAG suffix},
    "vitac" by default. Select another run with TRAIN_CONFIG (e.g. fp32_scratch)."""
    return args.get("train_config_name") or os.environ.get("TRAIN_CONFIG") or "vitac"


def _replan_every(args: dict) -> int:
    """0: predict a chunk every step and average overlapping predictions (temporal
    aggregation). n > 0: execute the first n actions of a chunk, then predict again.
    SMOLVLA_REPLAN_EVERY overrides smolvla_replan_every from the deploy yml."""
    return int(os.environ.get("SMOLVLA_REPLAN_EVERY") or args.get("smolvla_replan_every") or 0)


def _use_rtc(args: dict) -> bool:
    """Real-Time Chunking guidance between consecutive chunks (needs replan_every > 0).
    SMOLVLA_RTC=1/0 overrides smolvla_rtc from the deploy yml."""
    env = os.environ.get("SMOLVLA_RTC")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    return bool(args.get("smolvla_rtc", False))


def get_run_tag(args: dict) -> str:
    """Name of the eval result dir: the checkpoint run, plus the execution mode if not the default."""
    n = _replan_every(args)
    if n <= 0:
        return _train_config_name(args)
    return _train_config_name(args) + (f"-rtc{n}" if _use_rtc(args) else f"-replan{n}")


def _resolve_ckpt_dir(args: dict) -> Path:
    if args.get("ckpt_dir"):
        return Path(args["ckpt_dir"])
    ep_num = args.get("expert_data_num") or os.environ.get("EP_NUM") or "100"
    run = f"smolvla_vitac_{args['task_name']}-{args['task_config']}-{ep_num}_{_train_config_name(args)}"
    return _SMOLVLA_DIR / "outputs" / run / "checkpoints" / "last" / "pretrained_model"


class _SmolVLALocalPolicy(BasePolicy):
    """SmolVLA model runner. This class is instantiated only in .venv service."""

    # Mapping from SmolVLA batch keys -> ((root, camera, key), is_tactile)
    # ``root`` and ``camera`` follow the keys exposed by UniVTAC's
    # ``BaseTask._get_observations``.
    _DEFAULT_IMAGE_MAP = {
        "observation.images.head":           ("observation", "head", "rgb",         False),
        "observation.images.wrist":          ("observation", "wrist", "rgb",        False),
        "observation.images.tactile_ll":     ("tactile",     "left_tactile",  "rgb",        True),
        "observation.images.tactile_lr":     ("tactile",     "right_tactile", "rgb",        True),
        # Marker-overlay variants -- only used if the checkpoint was trained with them.
        "observation.images.tactile_ll_marker": ("tactile", "left_tactile",  "rgb_marker", True),
        "observation.images.tactile_lr_marker": ("tactile", "right_tactile", "rgb_marker", True),
    }

    def __init__(self, args: dict):
        import torch

        sys.path.insert(0, str(_SMOLVLA_DIR / "src"))
        from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
        from lerobot.configs import PreTrainedConfig
        from lerobot.processor import (
            PolicyProcessorPipeline,
            policy_action_to_transition,
            transition_to_policy_action,
        )
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import (
            make_smolvla_pre_post_processors,
        )
        from lerobot.utils.random_utils import set_seed
        from lerobot.utils.constants import (
            POLICY_POSTPROCESSOR_DEFAULT_NAME,
            POLICY_PREPROCESSOR_DEFAULT_NAME,
        )

        # ------------------------------------------------------------------ env
        self.task_name = args["task_name"]
        self.use_env_instruction = bool(args.get("smolvla_use_env_instruction", False))

        ckpt_dir = _resolve_ckpt_dir(args)
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"SmolVLA checkpoint not found: {ckpt_dir} (set TRAIN_CONFIG / EP_NUM, or ckpt_dir)"
            )

        # ------------------------------------------------------------------ task settings
        with open(_SMOLVLA_DIR.parent / "task_settings.json", "r") as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
        # task_settings.json's camera_type is the ACT recipe (head-only for most
        # tasks). SmolVLA feeds every camera the checkpoint was trained on
        # (see _image_map); dropping one leaves it out of distribution.
        self.camera_type = args.get("smolvla_camera_type", "all")
        print(f"[smolvla] camera_type='{self.camera_type}' task='{self.task_name}'")

        # ------------------------------------------------------------------ deploy config
        with open(_SMOLVLA_DIR / "deploy.yml", "r") as f:
            cfg = yaml.safe_load(f) or {}

        # Allow caller / env to override the on-disk yaml.
        cfg.update({k: v for k, v in args.items() if k.startswith("smolvla_")})

        self.device = torch.device(
            args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.image_size = cfg.get("smolvla_image_size")
        self.tactile_size = cfg.get("smolvla_tactile_size")
        self.image_scale = str(cfg.get("smolvla_image_scale", "raw"))
        self.n_action_steps = int(cfg.get("smolvla_n_action_steps", 0))  # 0 -> use config default
        self.use_marker_variants = cfg.get("smolvla_use_marker_variants", "auto")
        self.temporal_agg_max_timesteps = int(cfg.get("smolvla_temporal_agg_max_timesteps", 1000))
        self.temporal_agg_action_dim = int(cfg.get("smolvla_temporal_agg_action_dim", 9))
        self.temporal_agg_k = float(cfg.get("smolvla_temporal_agg_k", 0.01))
        self.replan_every = _replan_every(cfg)
        self.use_rtc = _use_rtc(cfg)
        if self.use_rtc and self.replan_every <= 0:
            raise ValueError("smolvla_rtc needs smolvla_replan_every > 0")
        self.rtc_max_guidance_weight = float(cfg.get("smolvla_rtc_max_guidance_weight", 10.0))
        self.rtc_schedule = str(cfg.get("smolvla_rtc_schedule", "LINEAR")).upper()
        self._action_queue: deque = deque()
        self._prev_chunk = None  # last chunk in model (normalized) space, for RTC
        print(f"[smolvla] execution: " + (
            (f"first {self.replan_every} actions per chunk"
             + (f", RTC guidance (max weight {self.rtc_max_guidance_weight}, {self.rtc_schedule})" if self.use_rtc else ""))
            if self.replan_every > 0
            else f"temporal aggregation every step (k={self.temporal_agg_k})"))

        # ------------------------------------------------------------------ model + processors
        print(f"[smolvla] loading policy from {ckpt_dir}")
        model_file = ckpt_dir / SAFETENSORS_SINGLE_FILE
        adapter_file = ckpt_dir / "adapter_model.safetensors"
        if model_file.exists():
            self.model = SmolVLAPolicy.from_pretrained(str(ckpt_dir)).to(self.device).eval()
        elif adapter_file.exists():
            from peft import PeftConfig, PeftModel

            print(f"[smolvla] loading PEFT adapter from {ckpt_dir}")
            policy_config = PreTrainedConfig.from_pretrained(str(ckpt_dir))
            policy_config.device = str(self.device)
            peft_config = PeftConfig.from_pretrained(str(ckpt_dir))

            if peft_config.base_model_name_or_path:
                self.model = SmolVLAPolicy.from_pretrained(
                    peft_config.base_model_name_or_path,
                    config=policy_config,
                )
            else:
                train_config_path = ckpt_dir / "train_config.json"
                if train_config_path.exists():
                    with open(train_config_path, "r") as f:
                        seed = json.load(f).get("seed")
                    if seed is not None:
                        print(f"[smolvla] setting seed {seed} before reconstructing PEFT base model")
                        set_seed(int(seed))
                else:
                    print("[smolvla] warning: PEFT adapter has no base_model_name_or_path and no train_config.json")
                self.model = SmolVLAPolicy(policy_config)

            self.model = PeftModel.from_pretrained(
                self.model,
                str(ckpt_dir),
                config=peft_config,
                is_trainable=False,
            ).to(self.device).eval()
        else:
            raise FileNotFoundError(
                f"Neither {SAFETENSORS_SINGLE_FILE} nor adapter_model.safetensors found in {ckpt_dir}"
            )

        self.model_config = _get_policy_config(self.model)
        if self.image_size is None:
            self.image_size = self._feature_hw(default=(256, 256), tactile=False)
        if self.tactile_size is None:
            self.tactile_size = self._feature_hw(default=self.image_size, tactile=True)
        self.image_size = tuple(self.image_size)
        self.tactile_size = tuple(self.tactile_size)

        if self.n_action_steps > 0:
            # Cap how many actions of the predicted chunk we actually execute.
            self.model_config.n_action_steps = min(
                self.n_action_steps, self.model_config.chunk_size
            )

        if self.use_rtc:
            from lerobot.configs.types import RTCAttentionSchedule
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            self.model_config.rtc_config = RTCConfig(
                enabled=True,
                prefix_attention_schedule=RTCAttentionSchedule[self.rtc_schedule],
                max_guidance_weight=self.rtc_max_guidance_weight,
                execution_horizon=self.model_config.chunk_size - self.replan_every,
            )
            self.model.init_rtc_processor()

        preprocessor_config = ckpt_dir / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
        postprocessor_config = ckpt_dir / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
        if preprocessor_config.exists() and postprocessor_config.exists():
            self.pre = PolicyProcessorPipeline.from_pretrained(
                ckpt_dir,
                config_filename=preprocessor_config.name,
                overrides={"device_processor": {"device": str(self.device)}},
            )
            self.post = PolicyProcessorPipeline.from_pretrained(
                ckpt_dir,
                config_filename=postprocessor_config.name,
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            )
        else:
            stats_path = args.get("stats_path")
            if stats_path is None:
                stats_path = self._infer_dataset_stats_path(ckpt_dir)
            dataset_stats = _load_stats(Path(stats_path) if stats_path is not None else None)
            self.pre, self.post = make_smolvla_pre_post_processors(
                self.model_config, dataset_stats=dataset_stats
            )

        # ------------------------------------------------------------------ image mapping
        # Build the SmolVLA-key -> UniVTAC-path map dynamically so we only
        # encode the cameras the checkpoint actually expects.
        configured_keys = set(self.model_config.input_features.keys())
        if self.use_marker_variants == "auto":
            self.use_marker_variants = any(key.endswith("_marker") for key in configured_keys)
        else:
            self.use_marker_variants = bool(self.use_marker_variants)
        self._image_map: dict[str, tuple[str, str, str, bool]] = {}
        for batch_key, (root, cam, key, is_tactile) in self._DEFAULT_IMAGE_MAP.items():
            if batch_key not in configured_keys:
                continue
            if (not self.use_marker_variants) and batch_key.endswith("_marker"):
                continue
            self._image_map[batch_key] = (root, cam, key, is_tactile)
        print(f"[smolvla] image streams: {list(self._image_map.keys())}")

        self._instruction_set = False
        self._cached_instruction: str | None = None
        self._reset_temporal_aggregation()

    def _reset_temporal_aggregation(self) -> None:
        import torch

        self.t = 0
        self.all_time_actions = torch.zeros(
            (
                self.temporal_agg_max_timesteps,
                self.temporal_agg_max_timesteps,
                self.temporal_agg_action_dim,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        self.all_time_actions_populated = torch.zeros(
            (
                self.temporal_agg_max_timesteps,
                self.temporal_agg_max_timesteps,
            ),
            device=self.device,
            dtype=torch.bool,
        )

    def _aggregate_action_chunk(self, action_chunk) -> np.ndarray:
        import torch

        action_chunk = torch.as_tensor(action_chunk, device=self.device, dtype=torch.float32)
        if action_chunk.ndim == 3:
            action_chunk = action_chunk[0]
        if action_chunk.ndim != 2:
            raise ValueError(f"expected action chunk shaped (T, D) or (1, T, D), got {tuple(action_chunk.shape)}")

        if action_chunk.shape[-1] != self.temporal_agg_action_dim:
            raise ValueError(
                "SmolVLA action dim does not match temporal aggregation buffer: "
                f"chunk has {action_chunk.shape[-1]}, buffer has {self.temporal_agg_action_dim}"
            )
        if self.t >= self.temporal_agg_max_timesteps:
            raise RuntimeError(
                f"SmolVLA temporal aggregation buffer exhausted at step {self.t}; "
                f"increase smolvla_temporal_agg_max_timesteps above {self.temporal_agg_max_timesteps}"
            )

        chunk_len = min(action_chunk.shape[0], self.temporal_agg_max_timesteps - self.t)
        end = self.t + chunk_len
        self.all_time_actions[self.t, self.t:end] = action_chunk[:chunk_len]
        self.all_time_actions_populated[self.t, self.t:end] = True

        populated = self.all_time_actions_populated[:, self.t]
        actions_for_curr_step = self.all_time_actions[populated, self.t]
        if actions_for_curr_step.numel() == 0:
            raise RuntimeError(f"no SmolVLA action predictions available for step {self.t}")

        weights = torch.exp(
            -self.temporal_agg_k
            * torch.arange(actions_for_curr_step.shape[0], device=self.device, dtype=torch.float32)
        )
        weights = weights / weights.sum()
        action = (actions_for_curr_step * weights.unsqueeze(1)).sum(dim=0)
        self.t += 1
        return action.detach().cpu().numpy()

    def _feature_hw(self, default: tuple[int, int], tactile: bool) -> tuple[int, int]:
        for key, feature in self.model_config.input_features.items():
            if tactile != ("tactile" in key):
                continue
            shape = tuple(getattr(feature, "shape", ()))
            if len(shape) == 3:
                return int(shape[-2]), int(shape[-1])
        return default

    def _infer_dataset_stats_path(self, ckpt_dir: Path) -> Path | None:
        train_config_path = ckpt_dir / "train_config.json"
        if train_config_path.exists():
            with open(train_config_path, "r") as f:
                train_config = json.load(f)
            dataset_root = train_config.get("dataset", {}).get("root")
            if dataset_root:
                stats_path = Path(dataset_root) / "meta" / "stats.json"
                if stats_path.exists():
                    return stats_path
        for candidate in (ckpt_dir / "meta" / "stats.json", ckpt_dir.parent / "meta" / "stats.json"):
            if candidate.exists():
                return candidate
        return None

    # ----------------------------------------------------------------------
    # Observation encoding
    # ----------------------------------------------------------------------

    def _pick_image(self, observation: dict, root: str, cam: str, key: str):
        """Fetch a HWC image tensor from a UniVTAC observation, falling back
        gracefully across the variants seen in the codebase."""
        if root in observation and cam in observation[root] and key in observation[root][cam]:
            return observation[root][cam][key]
        # Legacy aliases used by the HDF5 dumps (``left_gsmini``).
        alias = cam.replace("_tactile", "_gsmini")
        if root in observation and alias in observation[root] and key in observation[root][alias]:
            return observation[root][alias][key]
        raise KeyError(f"observation missing {root}.{cam}.{key}")

    def encode_obs(self, observation: dict) -> dict:
        """Convert UniVTAC observation to SmolVLA's raw (un-tokenized) batch.

        Input (UniVTAC, see ``envs/_base_task.py``)::
            observation['observation']['head']['rgb']      HWC uint8 0-255
            observation['observation']['wrist']['rgb']     HWC uint8 0-255
            observation['tactile']['left_tactile']['rgb']  HWC uint8 0-255
            observation['tactile']['right_tactile']['rgb'] HWC uint8 0-255
            observation['embodiment']['ee']                (7,)
            observation['embodiment']['joint']             (9,)

        Output (SmolVLA, pre-processor will then tokenize / normalize)::
            observation.images.head           (3, H, W) float32 in [0, 1]
            observation.images.wrist          (3, H, W) float32 in [0, 1]
            observation.images.tactile_ll     (3, H, W) float32 in [0, 1]
            observation.images.tactile_lr     (3, H, W) float32 in [0, 1]
            observation.state                 (16,)     float32  -- [ee(7), joint(9)]
            task                              str       -- language instruction
        """
        import torch

        batch: dict[str, torch.Tensor | str] = {}

        for batch_key, (root, cam, key, is_tactile) in self._image_map.items():
            # Honor the deploy.yml camera_type setting for *visual* cameras:
            # the ACT recipe lets the operator pick head-only / wrist-only.
            if (not is_tactile) and self.camera_type != "all" and cam != self.camera_type:
                continue
            img = self._pick_image(observation, root, cam, key)
            img = _to_chw_float(img, image_scale=self.image_scale)
            target = self.tactile_size if is_tactile else self.image_size
            img = _resize_chw(img, target)
            batch[batch_key] = img.to(self.device)

        # State = [ee(7), joint(9)] following the preprocessing convention
        # in scripts/preprocess_hdf5_to_lerobot.py.
        ee = torch.as_tensor(observation["embodiment"]["ee"], device=self.device).float().view(-1)
        joint = torch.as_tensor(observation["embodiment"]["joint"], device=self.device).float().view(-1)
        batch["observation.state"] = torch.cat([ee, joint], dim=0)

        return batch

    # ----------------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------------

    def eval(self, task, observation):
        """Run a single env step with SmolVLA and dispatch the action.

        Mirrors :meth:`policy.ACT.deploy_policy.Policy.eval` so the harness
        treats SmolVLA identically to ACT. We run chunk inference every step
        and aggregate overlapping predictions with exponential weights.
        """
        import torch

        if not self._instruction_set:
            self._cached_instruction = _instruction_for(
                task.instruction, self.task_name, self.use_env_instruction
            )
            self.model.reset()
            self._instruction_set = True

        batch = self.encode_obs(observation)
        batch["task"] = self._cached_instruction

        # Run the SmolVLA pre-processor (adds batch dim, tokenizes language,
        # device-places, normalizes state).
        batch = self.pre(batch)

        action_np = self._next_action(batch)

        action_t = torch.from_numpy(action_np).to(task.device).float()
        # SmolVLA action is the next-step joint pose; UniVTAC uses 'qpos' for joint targets.
        exec_succ, eval_succ = task.take_action(action_t, action_type="qpos")
        return exec_succ, eval_succ

    def get_action(self, observation: dict, instruction: str | None) -> np.ndarray:
        """Return one action for a serialized UniVTAC observation."""
        import torch

        if not self._instruction_set:
            self._cached_instruction = instruction or self.task_name.replace("_", " ")
            self.model.reset()
            self._instruction_set = True

        batch = self.encode_obs(observation)
        batch["task"] = self._cached_instruction
        batch = self.pre(batch)

        return self._next_action(batch)

    def _next_action(self, batch) -> np.ndarray:
        """Action for this step: temporal aggregation, or the next queued action of the last chunk.

        With RTC (synchronous, inference delay 0) the new chunk is guided towards
        the part of the previous chunk that was not executed, so consecutive
        chunks join up instead of jumping at the switch.
        """
        import torch

        if self.replan_every > 0 and self._action_queue:
            return self._action_queue.popleft()
        kwargs = {}
        if self.use_rtc and self._prev_chunk is not None:
            left_over = self._prev_chunk[:, self.replan_every :]
            kwargs = {
                "inference_delay": 0,
                "prev_chunk_left_over": left_over,
                "execution_horizon": left_over.shape[1],
            }
        with torch.no_grad():  # RTC re-enables grad internally for its guidance term
            raw_chunk = self.model.predict_action_chunk(batch, **kwargs)
        self._prev_chunk = raw_chunk.detach()
        action_chunk = self.post(raw_chunk.detach())
        if self.replan_every <= 0:
            return self._aggregate_action_chunk(action_chunk)
        chunk = torch.as_tensor(action_chunk, dtype=torch.float32).reshape(-1, self.temporal_agg_action_dim)
        self._action_queue.extend(a.cpu().numpy() for a in chunk[: self.replan_every])
        return self._action_queue.popleft()

    def reset(self):
        """Reset SmolVLA's internal action queue and clear cached instruction."""
        self._instruction_set = False
        self._cached_instruction = None
        self._reset_temporal_aggregation()
        self._action_queue.clear()
        self._prev_chunk = None
        if hasattr(self.model, "reset"):
            self.model.reset()


# ---------------------------------------------------------------------------
# FastAPI client used by the IsaacLab evaluation process
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _to_wire_payload(value: Any):
    """Convert tensors/arrays to JSON-safe containers for cross-env HTTP."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, dict):
        return {str(k): _to_wire_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wire_payload(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return _to_wire_payload(np.asarray(value))


def _from_wire_payload(value: Any):
    """Restore JSON-safe ndarray payloads on the receiving side."""
    if isinstance(value, dict):
        if value.get("__ndarray__") is True:
            data = base64.b64decode(value["data"].encode("ascii"))
            array = np.frombuffer(data, dtype=np.dtype(value["dtype"]))
            return array.reshape(value["shape"]).copy()
        return {k: _from_wire_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_wire_payload(v) for v in value]
    return value


class _SmolVLAFastAPIClient:
    def __init__(self, args: dict):
        self.smolvla_dir = _SMOLVLA_DIR
        self.repo_root = self.smolvla_dir.parents[1]
        self.host = str(args.get("smolvla_host", os.environ.get("SMOLVLA_HOST", "127.0.0.1")))
        self.port = int(args.get("smolvla_port", os.environ.get("SMOLVLA_PORT", 0)) or 0)
        self.python = Path(args.get("smolvla_python", self.smolvla_dir / ".venv" / "bin" / "python"))
        self.startup_timeout = float(args.get("smolvla_startup_timeout", 300))
        self.request_timeout = float(args.get("smolvla_request_timeout", 300))
        self.authkey = str(args.get("smolvla_authkey", os.environ.get("SMOLVLA_AUTHKEY", "")))
        self._external = self.port != 0
        self._process: subprocess.Popen | None = None
        self._closed = False

        if not self._external:
            self.port = _find_free_port()
            self.authkey = secrets.token_hex(16)
            self._start_server()

        self.base_url = f"http://{self.host}:{self.port}"
        self._wait_until_ready()
        self._post_json("/init", {"authkey": self.authkey, "args": _to_wire_payload(args)})

    def _start_server(self) -> None:
        if not self.python.exists():
            raise FileNotFoundError(f"SmolVLA python not found: {self.python}")
        server = self.smolvla_dir / "smolvla_server.py"
        self._process = subprocess.Popen(
            [
                str(self.python),
                str(server),
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--authkey",
                self.authkey,
            ],
            cwd=str(self.repo_root),
        )

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                response = self._request("GET", "/health")
                if response.get("status") == "ok":
                    return
            except BaseException as exc:
                last_error = exc
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError(f"SmolVLA server exited with code {self._process.returncode}") from exc
            time.sleep(0.25)
        raise TimeoutError(f"timed out waiting for SmolVLA server: {last_error}")

    def _request(self, method: str, path: str, data: bytes | None = None, content_type: str | None = None):
        headers = {"X-SmolVLA-Auth": self.authkey}
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                body = resp.read()
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SmolVLA server HTTP {exc.code}: {detail}") from exc

    def _post_json(self, path: str, payload: Any):
        response = self._request(
            "POST",
            path,
            data=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
        )
        if isinstance(response, dict) and response.get("type") == "error":
            raise RuntimeError(
                "SmolVLA server error:\n"
                f"{response.get('error')}\n"
                f"{response.get('traceback')}"
            )
        return response

    def get_action(self, observation: dict, instruction: str | None) -> np.ndarray:
        payload = {
            "authkey": self.authkey,
            "instruction": instruction,
            "observation": _to_wire_payload(observation),
        }
        response = self._post_json("/act", payload)
        return np.asarray(_from_wire_payload(response["action"]), dtype=np.float32)

    def reset(self) -> None:
        self._post_json("/reset", {"authkey": self.authkey})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._external:
                self._post_json("/reset", {"authkey": self.authkey})
            else:
                self._post_json("/shutdown", {"authkey": self.authkey})
        except Exception:
            pass
        if self._process is not None and self._process.poll() is None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()


class Policy(BasePolicy):
    """Deploy-time SmolVLA wrapper for UniVTAC via a FastAPI inference server."""

    def __init__(self, args: dict):
        self.task_name = args["task_name"]
        self.use_env_instruction = bool(args.get("smolvla_use_env_instruction", False))
        self.model = _SmolVLAFastAPIClient(args)
        self._instruction_set = False
        self._cached_instruction: str | None = None

    def encode_obs(self, observation: dict) -> dict:
        return observation

    def eval(self, task, observation):
        import torch

        if not self._instruction_set:
            self._cached_instruction = _instruction_for(
                task.instruction, self.task_name, self.use_env_instruction
            )
            self._instruction_set = True

        action_np = self.model.get_action(observation, self._cached_instruction).reshape(-1)
        # Trained on embodiment/joint: 7 arm joints + 2 mirrored finger joints.
        # take_action("qpos") expects 7 arm joints + 1 gripper value (as ACT's [:8]).
        action_np = action_np[:8].copy()
        action_t = torch.from_numpy(action_np).to(task.device).float()
        exec_succ, eval_succ = task.take_action(action_t, action_type="qpos")
        return exec_succ, eval_succ

    def reset(self):
        self._instruction_set = False
        self._cached_instruction = None
        self.model.reset()

    def close(self):
        self.model.close()
