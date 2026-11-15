#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import copy
import json
import math
from collections import deque
from pathlib import Path
from typing import TypedDict, Unpack

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import get_safe_dtype
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from ..rtc.modeling_rtc import RTCProcessor
from ..utils import (
    populate_queues,
)
from .configuration_smolvla import SmolVLAConfig
from .smolvlm_with_expert import SmolVLMWithExpertModel
from .vitac_encoder import FiLMHead, TactileAlignmentHead, build_vitac_encoder


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        require_package("transformers", extra="smolvla")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        # World action model: latents of the last frames, one per env step (see wam_observe).
        self._wam_buffer = deque(maxlen=1 - min(self.config.wam_history_offsets, default=0))

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        tactile_images, tactile_masks = self.prepare_tactile(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        wam_window = self._wam_window(batch) if self.config.wam_enabled else None
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            tactile_images=tactile_images, tactile_masks=tactile_masks,
            noise=noise, wam_window=wam_window, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        tactile_images, tactile_masks = self.prepare_tactile(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        loss_dict = {}
        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time,
            tactile_images=tactile_images, tactile_masks=tactile_masks,
            wam_batch=self.prepare_wam_batch(batch),
        )
        wam_losses = {}
        if self.config.wam_enabled:
            losses, wam_losses = losses
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        # World-action-model terms: each modality is averaged on its own (over
        # its tokens and the samples whose future frame lies inside the
        # episode), then weighted, so token counts do not set its share.
        wam_weights = {
            "vision": self.config.wam_vision_loss_weight,
            "tactile": self.config.wam_tactile_loss_weight,
        }

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over valid (time, action) entries
            if actions_is_pad is None:
                per_sample_loss = losses.mean(dim=(1, 2))
            else:
                num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample_loss = losses.sum(dim=(1, 2)) / num_valid
            loss_dict["action_loss"] = per_sample_loss.mean().item()
            for name, (sample_loss, sample_valid) in wam_losses.items():
                per_sample_loss = per_sample_loss + wam_weights[name] * sample_loss * sample_valid
                loss_dict[f"wam_{name}_loss"] = (
                    (sample_loss * sample_valid).sum() / sample_valid.sum().clamp_min(1)
                ).item()
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss over valid (time, action) entries
            if actions_is_pad is None:
                loss = losses.mean()
            else:
                num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                loss = losses.sum() / num_valid
            loss_dict["action_loss"] = loss.item()
            for name, (sample_loss, sample_valid) in wam_losses.items():
                wam_loss = (sample_loss * sample_valid).sum() / sample_valid.sum().clamp_min(1)
                loss = loss + wam_weights[name] * wam_loss
                loss_dict[f"wam_{name}_loss"] = wam_loss.item()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        # Strip tactile streams out of the regular image list whenever they
        # are handled by the dedicated ViTacEncoder pathway:
        #   * cross_attn / film: tactile is always routed externally;
        #   * token fusion: only routed externally when a non-SigLIP encoder
        #     is configured (``vitac_encoder_name != "siglip"``).
        tactile_keys_set = set(self.config.tactile_keys) if self.config.use_tactile else set()
        use_dedicated_encoder = (
            self.config.use_tactile
            and (self.config.vitac_encoder_name or "siglip").lower() != "siglip"
        )
        skip_tactile = self.config.use_tactile and (
            self.config.tactile_fusion in ("cross_attn", "film")
            or (self.config.tactile_fusion == "token" and use_dedicated_encoder)
        )
        present_img_keys = [
            key for key in self.config.image_features
            if key in batch and not (skip_tactile and key in tactile_keys_set)
        ]
        missing_img_keys = [
            key for key in self.config.image_features
            if key not in batch and not (skip_tactile and key in tactile_keys_set)
        ]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = self._current_frame(batch, key)
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, 0, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_tactile(self, batch):
        """Collect raw tactile images for cross-attention, FiLM, or dedicated-encoder token mode.

        Returns:
            tactile_images: list of (B, 3, H, W) float tensors in [0,1], one
                per tactile key that is present in the batch.  Empty list when
                the SigLIP token-fusion path is active (``vitac_encoder_name``
                is "siglip" with ``tactile_fusion == "token"``).
            tactile_masks: list of bool (B,) tensors; True = stream is present.
        """
        use_dedicated_encoder = (
            self.config.use_tactile
            and (self.config.vitac_encoder_name or "siglip").lower() != "siglip"
        )
        # Activate the tactile path when:
        #   - cross_attn / film fusion modes (always use dedicated encoder), OR
        #   - token fusion but with a non-SigLIP dedicated encoder.
        if not self.config.use_tactile:
            return [], []
        if self.config.tactile_fusion == "token" and not use_dedicated_encoder:
            return [], []
        tactile_images = []
        tactile_masks = []
        for key in self.config.tactile_keys:
            if key not in batch:
                continue
            img = self._current_frame(batch, key)
            bsize = img.shape[0]
            device = img.device
            tactile_images.append(img)
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            tactile_masks.append(mask)
        return tactile_images, tactile_masks

    def _current_frame(self, batch, key):
        """Offset-0 frame of an observation that may carry extra offsets (WAM online tactile)."""
        x = batch[key]
        if x.ndim != 5:
            return x
        offsets = self.config.observation_delta_indices_for(key)
        return x[:, offsets.index(0) if len(offsets) == x.shape[1] else 0]

    def prepare_wam_batch(self, batch):
        """Inputs of the world-action-model latents for a training batch (None when off).

        Latents come from the precomputed store by global frame ``index``; the
        online tactile mode also passes the tactile frames at their loaded
        offsets (window and future) with their ``_is_pad`` masks.
        """
        if not self.config.wam_enabled:
            return None
        if "index" not in batch:
            raise KeyError("world action model training needs the dataset 'index' in the batch")
        out = {"index": batch["index"].reshape(-1)}
        if self.config.wam_use_tactile and not self.config.vitac_freeze_backbone:
            out["tactile_offsets"] = self.config.observation_delta_indices_for(self.config.tactile_keys[0])
            out["tactile_frames"] = {k: batch[k] for k in self.config.tactile_keys}
            out["tactile_is_pad"] = {k: batch[f"{k}_is_pad"] for k in self.config.tactile_keys}
        return out

    def _wam_current(self, batch):
        """Raw latents of the current frame of a (preprocessed) batch."""
        vision, tactile = [], []
        if self.config.wam_use_vision:
            for key in self.config.wam_vision_keys:
                img = self._current_frame(batch, key)
                if self.config.resize_imgs_with_padding is not None:
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
                vision.append(img * 2.0 - 1.0)
        if self.config.wam_use_tactile:
            tactile = [self._current_frame(batch, key) for key in self.config.tactile_keys]
        return self.model.wam_current_raw(vision, tactile)

    @torch.no_grad()
    def wam_observe(self, batch):
        """Push the current frame into the observation window without predicting.

        The window holds one frame per env step (``wam_history_stride`` steps
        apart, as the dataset frames in training). ``predict_action_chunk``
        appends its frame itself, so a deploy that predicts a chunk only every
        few steps calls this on the other steps (not on the predicting ones).
        """
        if self.config.wam_enabled:
            self._wam_buffer.append(self._wam_current(batch))

    @torch.no_grad()
    def _wam_window(self, batch):
        """Inference observation window: latents of the last env steps.

        The current frame is appended here (``wam_observe`` covers the steps
        without a prediction); before the window is full the oldest frame
        repeats, as the training window is clamped at the episode start.
        """
        self._wam_buffer.append(self._wam_current(batch))
        last = len(self._wam_buffer) - 1
        steps = [max(0, last + o) for o in self.config.wam_history_offsets]
        raw = {
            name: torch.stack([self._wam_buffer[i][name] for i in steps], dim=1)
            for name in self._wam_buffer[last]
        }
        return self.model.wam_window_from_raw(raw)

    def update(self):
        """Called by the trainer after each optimizer step: EMA of the tactile target."""
        if self.config.wam_enabled:
            self.model.update_wam_ema()

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        # wrap_with_peft freezes every base parameter, so the randomly
        # initialized tactile modules must be fully trained (and saved with
        # the adapter); otherwise they stay at their random init forever.
        tactile_modules = [
            name for name in ("vitac_encoder", "tactile_align_head", "film_head")
            if getattr(self.model, name, None) is not None
        ]
        return {
            "target_modules": target_modules,
            "modules_to_save": tactile_modules,
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        if not self.config.pretrained_path:
            super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class _WAMLatentStore:
    """Memory-mapped latents written by scripts/precompute_wam_latents.py, rows = global frame index."""

    def __init__(self, latent_dir: str | Path, keys: list[str]):
        latent_dir = Path(latent_dir)
        self.info = json.loads((latent_dir / "info.json").read_text())
        self.arrays = {k: np.load(latent_dir / f"{k}.npy", mmap_mode="r") for k in keys}
        frames = np.load(latent_dir / "frames.npz")
        self.episode_from = frames["episode_from"]
        self.episode_to = frames["episode_to"]

    def check(self, config: SmolVLAConfig, tactile: bool) -> None:
        """The latents must come from the encoders and input sizes the policy uses."""
        expected = {"vlm": config.vlm_model_name, "image_size": config.resize_imgs_with_padding[0]}
        if tactile:
            expected |= {
                "vitac_ckpt": config.vitac_pretrained_path,
                "tactile_size": tuple(config.vitac_input_size)[0],
            }
        wrong = {k: (self.info.get(k), v) for k, v in expected.items() if self.info.get(k) != v}
        if wrong:
            raise ValueError(f"precomputed latents do not match the policy config (stored, expected): {wrong}")

    def gather(self, index: np.ndarray, offsets: list[int], keys) -> tuple[dict, np.ndarray]:
        """Latents at ``index + offsets`` clamped to each sample's episode.

        Returns key -> (B, len(offsets), ...) arrays and valid (B, len(offsets)),
        False where the offset falls outside the episode.
        """
        frame = index[:, None] + np.asarray(offsets, dtype=np.int64)[None, :]
        lo = self.episode_from[index][:, None]
        hi = self.episode_to[index][:, None] - 1
        valid = (frame >= lo) & (frame <= hi)
        flat = np.clip(frame, lo, hi).reshape(-1)
        out = {k: np.asarray(self.arrays[k][flat]).reshape(*frame.shape, *self.arrays[k].shape[1:]) for k in keys}
        return out, valid


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        # ---- Visuo-tactile heads -----------------------------------------
        # ViTacEncoder yields raw tokens at ``vitac_feature_dim``. A separate
        # MLP (TactileAlignmentHead) lifts those into the SmolVLA prefix
        # space (cross-attn / token fusion) or the expert hidden space (FiLM).
        # FiLM mode adds a small head that maps pooled aligned features into
        # (gamma, beta) modulation parameters applied to the action-time
        # embedding inside the action expert.
        self.vitac_encoder = None
        self.tactile_align_head = None
        self.film_head = None

        encoder_name = (self.config.vitac_encoder_name or "siglip").lower()
        # We construct the dedicated encoder whenever:
        #   * fusion is cross_attn or film (always required), OR
        #   * fusion is token AND a non-SigLIP encoder was requested.
        use_dedicated_encoder = self.config.use_tactile and (
            self.config.tactile_fusion in ("cross_attn", "film")
            or (self.config.tactile_fusion == "token" and encoder_name != "siglip")
        )

        if use_dedicated_encoder:
            # Cross_attn / film never use the SigLIP path; if the user left the
            # encoder name as "siglip" in those modes, fall back to "stub".
            effective_name = encoder_name if encoder_name != "siglip" else "stub"
            self.vitac_encoder = build_vitac_encoder(
                name=effective_name,
                feature_dim=self.config.vitac_feature_dim,
                num_tokens_per_stream=self.config.vitac_num_tokens_per_stream,
                input_size=tuple(self.config.vitac_input_size),
                backbone=self.config.vitac_backbone,
                latent_dims=self.config.vitac_latent_dims,
                pretrained_path=self.config.vitac_pretrained_path,
                freeze_backbone=self.config.vitac_freeze_backbone,
            )
            if self.config.tactile_fusion == "film":
                # Project tactile tokens into the action-expert hidden space
                # then condense them into FiLM modulation parameters.
                self.tactile_align_head = TactileAlignmentHead(
                    in_dim=self.config.vitac_feature_dim,
                    out_dim=self.vlm_with_expert.expert_hidden_size,
                    hidden_mult=self.config.vitac_align_hidden_mult,
                )
                self.film_head = FiLMHead(
                    in_dim=self.vlm_with_expert.expert_hidden_size,
                    target_dim=self.vlm_with_expert.expert_hidden_size,
                    hidden_mult=self.config.vitac_film_hidden_mult,
                )
            else:
                # token fusion (with dedicated encoder) and cross_attn both
                # need the prefix-space alignment head. SmolVLM image
                # embeddings live at ``text_config.hidden_size``, so we map
                # tactile tokens into the same dim.
                self.tactile_align_head = TactileAlignmentHead(
                    in_dim=self.config.vitac_feature_dim,
                    out_dim=self.vlm_with_expert.config.text_config.hidden_size,
                    hidden_mult=self.config.vitac_align_hidden_mult,
                )

        self._init_wam()
        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    # ------------------------------------------------------------------ world action model
    # Each modality m ("vision", "tactile") is a stream of per-frame latents with
    # ``per_frame`` tokens. The suffix holds the observation window (clean,
    # flow time 0), then the noisy future frames, then the actions:
    #   [ window: vision | tactile ][ future: vision | tactile ][ action x chunk ]
    # Shapes below: latents (B, frames, per_frame, D).

    def _init_wam(self):
        self.wam_specs: dict[str, tuple[int, int]] = {}  # name -> (latent dim, tokens per frame)
        self.wam_tactile_target = None
        self.wam_store = None
        if not self.config.wam_enabled:
            return
        cfg = self.config
        hidden = self.vlm_with_expert.expert_hidden_size
        if cfg.wam_use_vision:
            ph, pw = cfg.wam_vision_pool
            self.wam_specs["vision"] = (
                self.vlm_with_expert.config.text_config.hidden_size,
                len(cfg.wam_vision_keys) * ph * pw,
            )
        if cfg.wam_use_tactile:
            backbone = getattr(self.vitac_encoder, "backbone", None)
            if backbone is None:
                raise ValueError("wam_use_tactile needs the UniVTAC tactile encoder")
            if not cfg.vitac_freeze_backbone:
                # Fine-tuned encoder: the targets come from an EMA copy of it.
                self.wam_tactile_target = copy.deepcopy(backbone)
                for p in self.wam_tactile_target.parameters():
                    p.requires_grad = False
            self.wam_specs["tactile"] = (cfg.vitac_latent_dims, len(cfg.tactile_keys))

        self.wam_in_proj = nn.ModuleDict({m: nn.Linear(d, hidden) for m, (d, _) in self.wam_specs.items()})
        self.wam_time_mlp_in = nn.ModuleDict({m: nn.Linear(2 * hidden, hidden) for m in self.wam_specs})
        self.wam_time_mlp_out = nn.ModuleDict({m: nn.Linear(hidden, hidden) for m in self.wam_specs})
        self.wam_slot_emb = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(n, hidden) * 0.02) for m, (_, n) in self.wam_specs.items()}
        )
        # Frame offset (window < 0 <= future), continuous so it can later be conditioned on.
        self.wam_offset_proj = nn.ModuleDict({m: nn.Linear(hidden, hidden) for m in self.wam_specs})
        for proj in self.wam_offset_proj.values():
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        self.wam_out_proj = nn.ModuleDict(
            {m: nn.Linear(hidden, d) for m, (d, _) in self.wam_specs.items() if self.config.wam_future_offsets(m)}
        )

    @property
    def wam_future_names(self) -> list[str]:
        return [m for m in self.wam_specs if self.config.wam_future_offsets(m)]

    def _wam_num_tokens(self) -> tuple[int, int]:
        """(window tokens, future tokens) over all modalities."""
        window = sum(n * self.config.wam_history for _, n in self.wam_specs.values())
        future = sum(self.wam_specs[m][1] * len(self.config.wam_future_offsets(m)) for m in self.wam_future_names)
        return window, future

    # ---- latents
    def _wam_vision_tokens(self, raw):
        """(B, F, cameras, 64, D) connector tokens -> pooled, normalized (B, F, cameras*ph*pw, D)."""
        bsize, frames, cams, length, dim = raw.shape
        side = math.isqrt(length)
        if side * side != length:
            raise ValueError(f"expected a square connector token grid, got {length} tokens")
        grid = raw.float().reshape(-1, length, dim).transpose(1, 2).reshape(-1, dim, side, side)
        pooled = F.adaptive_avg_pool2d(grid, self.config.wam_vision_pool).flatten(2).transpose(1, 2)
        pooled = pooled.reshape(bsize, frames, -1, dim)
        return F.layer_norm(pooled, (dim,))

    @staticmethod
    def _wam_tactile_tokens(raw):
        """(B, F, streams, D) backbone features -> normalized tokens."""
        return F.layer_norm(raw.float(), raw.shape[-1:])

    def _wam_tactile_features(self, backbone, images):
        """[0, 1] images (N, 3, H, W) -> (N, latent_dims) with the UniVTAC input size."""
        size = tuple(self.config.vitac_input_size)
        if images.shape[-2:] != size:
            images = F.interpolate(images, size=size, mode="bilinear", align_corners=False)
        return backbone(images)

    def _wam_open_store(self):
        if self.wam_store is not None:
            return self.wam_store
        if not self.config.wam_latent_dir:
            raise ValueError("training the world action model needs --policy.wam_latent_dir "
                             "(output of scripts/precompute_wam_latents.py)")
        keys = list(self.config.wam_vision_keys) if "vision" in self.wam_specs else []
        if "tactile" in self.wam_specs and self.config.vitac_freeze_backbone:
            keys += list(self.config.tactile_keys)
        self.wam_store = _WAMLatentStore(self.config.wam_latent_dir, keys)
        self.wam_store.check(self.config, tactile="tactile" in self.wam_specs and self.config.vitac_freeze_backbone)
        return self.wam_store

    def wam_training_latents(self, wam_batch):
        """Window and future latents of a training batch.

        Returns ``window[m]`` (B, H, P, D), ``future[m]`` (B, K, P, D) and
        ``future_valid[m]`` (B, K); futures past the episode end are invalid.
        The window is clamped at the episode start (the first frame repeats).
        """
        cfg = self.config
        device = wam_batch["index"].device
        index = wam_batch["index"].cpu().numpy().astype(np.int64)
        hist = cfg.wam_history_offsets
        window, future, future_valid = {}, {}, {}
        store = self._wam_open_store()

        if "vision" in self.wam_specs:
            fut = cfg.wam_future_offsets("vision")
            raw, valid = store.gather(index, hist + fut, cfg.wam_vision_keys)
            raw = torch.from_numpy(np.stack([raw[k] for k in cfg.wam_vision_keys], axis=2)).to(device)
            tokens = self._wam_vision_tokens(raw)  # (B, H+K, P, D)
            window["vision"], future["vision"] = tokens[:, : len(hist)], tokens[:, len(hist) :]
            future_valid["vision"] = torch.from_numpy(valid[:, len(hist) :]).to(device)

        if "tactile" in self.wam_specs:
            fut = cfg.wam_future_offsets("tactile")
            if cfg.vitac_freeze_backbone:
                raw, valid = store.gather(index, hist + fut, cfg.tactile_keys)
                raw = torch.from_numpy(np.stack([raw[k] for k in cfg.tactile_keys], axis=2)).to(device)
                tokens = self._wam_tactile_tokens(raw)
                window["tactile"], future["tactile"] = tokens[:, : len(hist)], tokens[:, len(hist) :]
                future_valid["tactile"] = torch.from_numpy(valid[:, len(hist) :]).to(device)
            else:
                # Online: frames (B, offsets, 3, H, W) per stream at the loaded offsets.
                offsets = wam_batch["tactile_offsets"]
                hist_idx = [offsets.index(o) for o in hist]
                fut_idx = [offsets.index(o) for o in fut]
                win_feats, fut_feats, fut_valid = [], [], []
                self.wam_tactile_target.eval()
                for key in cfg.tactile_keys:
                    frames = wam_batch["tactile_frames"][key]
                    bsize = frames.shape[0]
                    win = frames[:, hist_idx].flatten(0, 1)
                    win_feats.append(self._wam_tactile_features(self.vitac_encoder.backbone, win).view(bsize, len(hist), -1))
                    if fut_idx:
                        with torch.no_grad():
                            tgt = frames[:, fut_idx].flatten(0, 1)
                            fut_feats.append(
                                self._wam_tactile_features(self.wam_tactile_target, tgt).view(bsize, len(fut), -1)
                            )
                        fut_valid.append(~wam_batch["tactile_is_pad"][key][:, fut_idx].bool())
                window["tactile"] = self._wam_tactile_tokens(torch.stack(win_feats, dim=2))
                if fut_idx:
                    future["tactile"] = self._wam_tactile_tokens(torch.stack(fut_feats, dim=2))
                    future_valid["tactile"] = torch.stack(fut_valid, dim=0).all(dim=0)
        return window, future, future_valid

    @torch.no_grad()
    def wam_current_raw(self, vision_images, tactile_images):
        """Latents of the current frame for the inference window.

        vision_images: SigLIP-ready (B, 3, H, W) per key in ``wam_vision_keys``.
        tactile_images: [0, 1] (B, 3, H, W) per key in ``tactile_keys``.
        Returns name -> raw latent of one frame: vision (B, cameras, 64, D),
        tactile (B, streams, latent_dims).
        """
        raw = {}
        if "vision" in self.wam_specs:
            raw["vision"] = torch.stack([self.vlm_with_expert.embed_image(img).float() for img in vision_images], 1)
        if "tactile" in self.wam_specs:
            backbone = self.vitac_encoder.backbone  # the same encoder as the prefix, frozen or fine-tuned
            raw["tactile"] = torch.stack(
                [self._wam_tactile_features(backbone, img).float() for img in tactile_images], 1
            )
        return raw

    def wam_window_from_raw(self, raw_window):
        """name -> raw (B, H, ...) window -> processed tokens (B, H, P, D)."""
        out = {}
        if "vision" in raw_window:
            out["vision"] = self._wam_vision_tokens(raw_window["vision"])
        if "tactile" in raw_window:
            out["tactile"] = self._wam_tactile_tokens(raw_window["tactile"])
        return out

    @torch.no_grad()
    def update_wam_ema(self):
        """EMA of the fine-tuned tactile backbone into the target copy (fine-tuned encoder only)."""
        if self.wam_tactile_target is None:
            return
        decay = self.config.wam_tactile_ema_decay
        online = self.vitac_encoder.backbone
        for p_t, p_o in zip(self.wam_tactile_target.parameters(), online.parameters(), strict=True):
            p_t.lerp_(p_o.detach(), 1.0 - decay)
        for b_t, b_o in zip(self.wam_tactile_target.buffers(), online.buffers(), strict=True):
            b_t.copy_(b_o)

    # ---- tokens
    def _wam_embed(self, name, latents, offsets, time):
        """latents (B, F, P, D) at frame ``offsets`` with flow time (B,) -> (B, F*P, hidden)."""
        hidden = self.vlm_with_expert.expert_hidden_size
        bsize, frames, per_frame, _ = latents.shape
        x = self.wam_in_proj[name](latents)
        time_emb = create_sinusoidal_pos_embedding(
            time, hidden, self.config.min_period, self.config.max_period, device=x.device
        ).type(x.dtype)
        emb = torch.cat([x, time_emb[:, None, None, :].expand_as(x)], dim=-1)
        emb = self.wam_time_mlp_out[name](F.silu(self.wam_time_mlp_in[name](emb)))
        # Offsets in units of the action chunk, shifted to stay positive.
        off = torch.tensor([1.0 + o / self.config.chunk_size for o in offsets], dtype=torch.float32, device=x.device)
        off_emb = create_sinusoidal_pos_embedding(
            off, hidden, self.config.min_period, self.config.max_period, device=x.device
        ).type(x.dtype)
        emb = emb + self.wam_slot_emb[name][None, None] + self.wam_offset_proj[name](off_emb)[None, :, None, :]
        return emb.reshape(bsize, frames * per_frame, hidden)

    def embed_wam_tokens(self, window, wam_x, wam_time):
        """Window tokens (clean, time 0) and noisy future tokens, each as one block."""
        bsize = next(iter(window.values())).shape[0]
        device = next(iter(window.values())).device
        zero = torch.zeros(bsize, dtype=torch.float32, device=device)
        win = [self._wam_embed(m, window[m], self.config.wam_history_offsets, zero) for m in self.wam_specs]
        fut = [
            self._wam_embed(m, wam_x[m], self.config.wam_future_offsets(m), wam_time[m])
            for m in self.wam_future_names
        ]
        return torch.cat(win, dim=1), (torch.cat(fut, dim=1) if fut else None)

    def wam_velocities(self, suffix_out):
        """Velocities of the future tokens per modality, shaped like the latents (B, K, P, D)."""
        n_window, _ = self._wam_num_tokens()
        out, start = {}, n_window
        for name in self.wam_future_names:
            k, p = len(self.config.wam_future_offsets(name)), self.wam_specs[name][1]
            v = self.wam_out_proj[name](suffix_out[:, start : start + k * p].to(dtype=torch.float32))
            out[name] = v.view(v.shape[0], k, p, -1)
            start += k * p
        return out

    def _wam_allow_obs_to_actions(self, att_2d_masks, query_offset, key_offset):
        """Let the future-latent rows also attend to the action key columns."""
        if not self.wam_specs or not self.config.wam_obs_attend_actions:
            return att_2d_masks
        n_window, n_future = self._wam_num_tokens()
        att_2d_masks = att_2d_masks.clone()
        rows = slice(query_offset + n_window, query_offset + n_window + n_future)
        att_2d_masks[:, rows, key_offset + n_window + n_future :] = True
        return att_2d_masks

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None,
        tactile_images=None, tactile_masks=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.

        When ``tactile_images`` is provided and the cross-attention fusion mode
        is active, the ViTacEncoder is applied to each tactile stream. The
        resulting tokens are appended as a separate prefix block *after* the
        state token so that:

        * The VLM image/language context attends to them with a block-boundary
          mask (``att_masks = 1``), preventing leakage back to early tokens.
        * The action expert can attend to the tactile condition tokens because
          they share the full prefix key-value cache.

        This implements the "cross-attention condition token" strategy described
        in CLAUDE.md without requiring structural changes to the action expert.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)

        # ---- Tactile via dedicated ViTacEncoder --------------------------
        # * cross_attn -- append tactile tokens as a *new* prefix attention
        #   block AFTER the state token (att_masks starts with [1]).
        # * token      -- splice tactile tokens INTO the image block (no new
        #   block boundary, att_masks contribution is all zeros) so SmolVLM
        #   treats them just like image patches.
        # * film       -- handled inside :meth:`embed_suffix`; no tokens are
        #   appended here.
        if (
            self.vitac_encoder is not None
            and self.config.tactile_fusion in ("cross_attn", "token")
            and tactile_images is not None
            and len(tactile_images) > 0
        ):
            tactile_feat, tactile_pad = self.vitac_encoder(
                tactile_images, tactile_masks=tactile_masks
            )
            # Alignment MLP: ViTacEncoder feature space -> prefix token space.
            tactile_tokens = self.tactile_align_head(tactile_feat)
            tactile_tokens = tactile_tokens.to(dtype=embs[0].dtype, device=device)

            if self.config.tactile_fusion == "cross_attn":
                embs.append(tactile_tokens)
                pad_masks.append(tactile_pad.to(device=device))
                # New attention block: tactile tokens see prior context, and
                # the action expert can attend to them through the shared
                # KV cache.
                att_masks += [1] + [0] * (tactile_tokens.shape[1] - 1)
            else:  # token fusion via dedicated encoder
                # No block boundary: tactile tokens behave like additional
                # image patches and share the image block's attention mask.
                # We insert them before the state token, which we already
                # appended above; rebuild the order by popping & re-appending
                # state_emb so that tactile tokens sit between language and
                # state (mirroring the layout used for SigLIP image tokens).
                # Pop the state we just appended (last 3 entries: state_emb,
                # state_mask, state att_mask 1's).
                state_emb_popped = embs.pop()
                state_mask_popped = pad_masks.pop()
                # state contributed exactly ``states_seq_len`` "1"s.
                att_masks = att_masks[: -states_seq_len]
                embs.append(tactile_tokens)
                pad_masks.append(tactile_pad.to(device=device))
                att_masks += [0] * tactile_tokens.shape[1]
                # Re-append state so it remains the last prefix token.
                embs.append(state_emb_popped)
                pad_masks.append(state_mask_popped)
                att_masks += [1] * states_seq_len

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def _compute_film_params(self, tactile_images, tactile_masks):
        """Pool ViTacEncoder outputs and produce FiLM (gamma, beta).

        Returns ``None`` when FiLM is not active or no tactile streams are
        provided. Gamma/beta shapes are ``(B, expert_hidden_size)`` and the
        FiLMHead is zero-initialized, so untrained FiLM behaves as identity.
        """
        if (
            self.film_head is None
            or self.tactile_align_head is None
            or tactile_images is None
            or len(tactile_images) == 0
        ):
            return None
        tactile_feat, tactile_pad = self.vitac_encoder(
            tactile_images, tactile_masks=tactile_masks
        )
        aligned = self.tactile_align_head(tactile_feat)  # (B, T, expert_hidden)
        # Masked mean-pool over the token dimension so padded streams do not
        # bias the modulation.
        mask = tactile_pad.to(dtype=aligned.dtype, device=aligned.device).unsqueeze(-1)
        pooled = (aligned * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        gamma, beta = self.film_head(pooled)  # (B, expert_hidden) each
        return gamma, beta

    def embed_suffix(
        self, noisy_actions, timestep, film_params=None, wam_window=None, wam_x=None, wam_time=None
    ):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing.

        When ``film_params`` is provided (``(gamma, beta)`` of shape
        ``(B, expert_hidden)``) the action-time embedding is FiLM-modulated::

            out = (1 + gamma)[:, None, :] * x + beta[:, None, :]

        With the world action model the suffix starts with two bidirectional
        blocks, the clean observation window and the noisy future latents,
        followed by the action tokens with their original causal mask::

            [ window | future | action_0 ... action_{chunk-1} ]

        The future block sees the window; the actions see both (see
        ``wam_obs_attend_actions`` for future -> action attention).
        """
        embs = []
        pad_masks = []
        att_masks = []

        if wam_window is not None:
            window_emb, future_emb = self.embed_wam_tokens(wam_window, wam_x, wam_time)
            for emb in (window_emb, future_emb):
                if emb is None:
                    continue
                embs.append(emb)
                pad_masks.append(torch.ones(emb.shape[:2], dtype=torch.bool, device=emb.device))
                att_masks += [1] + [0] * (emb.shape[1] - 1)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # FiLM modulation: (1 + gamma) * x + beta, applied after the expert MLP
        # so that tactile conditioning gates the denoising trajectory without
        # interfering with the time/action embedding structure.
        if film_params is not None:
            gamma, beta = film_params
            action_time_emb = (1.0 + gamma[:, None, :]) * action_time_emb + beta[:, None, :]

        # Add to input tokens
        embs.append(action_time_emb.to(dtype=embs[0].dtype) if embs else action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None,
        tactile_images=None, tactile_masks=None, wam_batch=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors).

        With the world action model the return value is ``(losses, wam_losses)``,
        where ``wam_losses[name] = (per-sample loss (B,), sample valid (B,))``.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Future latents: independent flow time per modality (UWM); with
        # probability wam_marginal_prob all of them are pure noise for a sample.
        wam_window = wam_x = wam_time = wam_u = wam_valid = None
        if self.wam_specs:
            bsize, device = actions.shape[0], actions.device
            wam_window, future, wam_valid = self.wam_training_latents(wam_batch)
            marginal = torch.rand(bsize, device=device) < self.config.wam_marginal_prob
            wam_x, wam_time, wam_u = {}, {}, {}
            for name, z in future.items():
                tau = torch.where(marginal, torch.ones_like(time), self.sample_time(bsize, device))
                eps = self.sample_noise(z.shape, device)
                t = tau[:, None, None, None]
                wam_x[name] = t * eps + (1 - t) * z
                wam_time[name] = tau
                wam_u[name] = eps - z

        film_params = self._compute_film_params(tactile_images, tactile_masks)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state,
            tactile_images=tactile_images, tactile_masks=tactile_masks,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t, time, film_params=film_params, wam_window=wam_window, wam_x=wam_x, wam_time=wam_time
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        prefix_len = prefix_pad_masks.shape[1]
        att_2d_masks = self._wam_allow_obs_to_actions(att_2d_masks, prefix_len, prefix_len)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        wam_out = suffix_out
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        if not self.wam_specs:
            return losses

        wam_losses = {}
        for name, v in self.wam_velocities(wam_out).items():
            frame_loss = F.mse_loss(wam_u[name], v, reduction="none").mean(dim=(2, 3))  # (B, K)
            valid = wam_valid[name].to(frame_loss.dtype)
            count = valid.sum(dim=1)
            wam_losses[name] = ((frame_loss * valid).sum(dim=1) / count.clamp_min(1), (count > 0).to(frame_loss.dtype))
        return losses, wam_losses

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        tactile_images=None,
        tactile_masks=None,
        wam_window=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors).

        ``wam_window``: name -> (B, H, P, D) observation-window latents (world action model).
        """
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        film_params = self._compute_film_params(tactile_images, tactile_masks)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state,
            tactile_images=tactile_images, tactile_masks=tactile_masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        if self.wam_specs:
            return self._sample_actions_wam(
                noise, prefix_pad_masks, past_key_values, film_params, num_steps, dt, wam_window, **kwargs
            )

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                    film_params=film_params,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def _sample_actions_wam(
        self, noise, prefix_pad_masks, past_key_values, film_params, num_steps, dt, wam_window, **kwargs
    ):
        """Euler integration of the actions together with the future latents.

        "joint": all modalities share the time and are denoised together.
        "marginal": the future latents stay pure noise at time 1 (the
        condition trained with wam_marginal_prob); only the actions move.
        With RTC the action velocity is guided towards the unexecuted part of
        the previous chunk (``kwargs`` of ``RTCProcessor.denoise_step``); the
        future latents use the velocity of the same network call.
        The predicted future latents are kept in ``self.last_wam_latents``.
        """
        bsize, device = noise.shape[0], noise.device
        joint = self.config.wam_inference_mode == "joint"
        wam_x = {
            name: self.sample_noise(
                (bsize, len(self.config.wam_future_offsets(name)), self.wam_specs[name][1], self.wam_specs[name][0]),
                device,
            )
            for name in self.wam_future_names
        }
        x_t = noise
        for step in range(num_steps):
            time = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=device)
            wam_time = {name: time if joint else torch.ones_like(time) for name in self.wam_future_names}
            outputs = {}

            def denoise(input_x_t, time=time, wam_x=wam_x, wam_time=wam_time, outputs=outputs):
                v, outputs["wam_v"] = self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=time,
                    film_params=film_params,
                    wam_window=wam_window,
                    wam_x=wam_x,
                    wam_time=wam_time,
                )
                return v

            if self._rtc_enabled():
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                    inference_delay=kwargs.get("inference_delay"),
                    time=1.0 + step * dt,
                    original_denoise_step_partial=denoise,
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            else:
                v_t = denoise(x_t)
            wam_v = {name: v.detach() for name, v in outputs["wam_v"].items()}
            x_t = x_t + dt * v_t
            if joint:
                wam_x = {name: wam_x[name] + dt * wam_v[name] for name in self.wam_future_names}
        self.last_wam_latents = wam_x if joint else None
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        film_params=None,
        wam_window=None,
        wam_x=None,
        wam_time=None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep.

        With the world action model (``wam_window``) it returns ``(v_actions, {name: v})``.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t, timestep, film_params=film_params, wam_window=wam_window, wam_x=wam_x, wam_time=wam_time
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        if wam_window is not None:
            suffix_att_2d_masks = self._wam_allow_obs_to_actions(suffix_att_2d_masks, 0, 0)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        wam_out = suffix_out
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        if wam_window is None:
            return v_t
        return v_t, self.wam_velocities(wam_out)
