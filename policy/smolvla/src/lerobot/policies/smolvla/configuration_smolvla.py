# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    # ---------------------------------------------------------------------
    # Visuo-tactile (ViTac) extension
    # ---------------------------------------------------------------------
    # Whether to use a dedicated visuo-tactile encoder. When False, tactile
    # streams (if present in the dataset) are treated as ordinary cameras and
    # flow through SmolVLM's vision encoder along with the head/wrist cameras
    # (this is the "token-fusion" strategy in CLAUDE.md). When True the
    # tactile streams are routed through :class:`ViTacEncoder` and their
    # tokens are appended to the prefix as cross-attention condition tokens.
    use_tactile: bool = True

    # Fusion mode:
    #   "token"      -- treat tactile as another image modality. By default
    #                   tactile streams are routed through SmolVLM's SigLIP
    #                   vision encoder (no ViTacEncoder used). If
    #                   ``vitac_encoder_name != "siglip"`` the dedicated
    #                   ViTacEncoder is used instead and its tokens are
    #                   appended to the prefix's image block.
    #   "cross_attn" -- ViTacEncoder + TactileAlignmentHead project tactile
    #                   into the prefix token space as condition tokens.
    #   "film"       -- ViTacEncoder + TactileAlignmentHead + FiLMHead pool
    #                   tactile into gamma/beta that modulate the action-time
    #                   embedding inside the action expert at each denoise step.
    tactile_fusion: str = "token"

    # Keys in the batch that carry tactile images. Two-arm-ready: the second
    # character is the arm ("l"/"r"), and the third is the gripper of that
    # arm ("l"/"r"). Currently only a single arm is recorded so we default to
    # left arm + (left, right) grippers.
    # The marker-overlay streams are the default because the UniVTAC tactile
    # encoder was pretrained on marked RGB. These are the ONLY tactile streams
    # the policy sees: every other ``observation.images.tactile_*`` stream in
    # the dataset is dropped in ``validate_features`` (it never reaches SigLIP).
    tactile_keys: tuple[str, ...] = (
        "observation.images.tactile_ll_marker",
        "observation.images.tactile_lr_marker",
    )

    # ViTacEncoder selection.
    #   "siglip"  -- (token-fusion only) route tactile through SmolVLM's
    #                SigLIP image encoder; no ViTacEncoder is constructed.
    #   "stub"    -- the lightweight ``ViTacEncoder`` CNN stub.
    #   "univtac" -- :class:`UniVTACEncoder` (ResNet18/34/50 + token queries),
    #                optionally initialized from a UniVTAC pretrained ckpt.
    # Cross-attn / FiLM modes always construct a ViTacEncoder, so "siglip" is
    # silently treated as "stub" in those modes.
    vitac_encoder_name: str = "siglip"
    # Native dimension of ViTacEncoder output tokens (before the alignment MLP).
    vitac_feature_dim: int = 256
    vitac_num_tokens_per_stream: int = 8
    vitac_input_size: tuple[int, int] = (224, 224)
    # TactileAlignmentHead hidden multiplier: hidden = max(in, out) * mult.
    vitac_align_hidden_mult: int = 2
    # FiLM-specific: hidden multiplier for the FiLMHead shared MLP.
    vitac_film_hidden_mult: int = 2
    # UniVTACEncoder-specific options.
    vitac_backbone: str = "resnet18"
    vitac_latent_dims: int = 512
    vitac_pretrained_path: str | None = None
    vitac_freeze_backbone: bool = False

    # ---------------------------------------------------------------------
    # World action model (UWM-style joint denoising of future latents)
    # ---------------------------------------------------------------------
    # Latents: frozen SigLIP+connector features of the cameras in
    # ``wam_vision_keys`` and frozen pretrained UniVTAC backbone features of
    # the streams in ``tactile_keys`` (scripts/precompute_wam_latents.py).
    # The action expert sees an observation window of past latents (clean) and
    # denoises the latents of several future frames together with the action
    # chunk; each modality has its own flow time.
    wam_enabled: bool = False
    wam_use_vision: bool = True
    wam_use_tactile: bool = True
    wam_vision_keys: tuple[str, ...] = (
        "observation.images.head",
        "observation.images.wrist",
    )
    # Output of scripts/precompute_wam_latents.py (<dataset root>/wam_latents);
    # used for training only, inference encodes the frames online.
    wam_latent_dir: str | None = None
    # Observation window: ``wam_history`` frames, ``wam_history_stride`` frames
    # apart, ending at the current frame (1 = current frame only).
    wam_history: int = 2
    wam_history_stride: int = 1
    # Predicted future frames: ``steps`` frames, ``stride`` frames apart, i.e.
    # offsets stride, 2*stride, ..., steps*stride (0 steps = no prediction).
    wam_vision_future_stride: int = 5
    wam_vision_future_steps: int = 6
    wam_tactile_future_stride: int = 5
    wam_tactile_future_steps: int = 6
    # The connector's 8x8 token grid is average-pooled to this grid per camera.
    wam_vision_pool: tuple[int, int] = (2, 2)
    # Tactile latents follow ``vitac_freeze_backbone``, so one encoder serves the
    # whole modality. Frozen: the prefix, the window and the targets all use the
    # frozen pretrained backbone, and window/targets are read from the
    # precomputed latents. Fine-tuned: the prefix and the window use the
    # fine-tuned backbone (window encoded online), and the targets use an EMA
    # copy of it, updated after every optimizer step (a target encoder trained
    # by the prediction loss itself could collapse to a constant).
    wam_tactile_ema_decay: float = 0.999
    wam_vision_loss_weight: float = 1.0
    wam_tactile_loss_weight: float = 1.0
    # Probability of training a sample with every future latent at pure noise
    # (tau = 1), the condition used by wam_inference_mode="marginal".
    wam_marginal_prob: float = 0.1
    # "joint": future latents are denoised together with the actions.
    # "marginal": future latents stay pure noise; only the actions are denoised.
    wam_inference_mode: str = "joint"
    # False: future-latent tokens see the context but not the action tokens,
    # and the action tokens keep their causal mask plus see the future latents.
    # True: future-latent tokens also attend to the (noisy) action tokens.
    wam_obs_attend_actions: bool = False

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.wam_enabled:
            if self.wam_inference_mode not in ("joint", "marginal"):
                raise ValueError(f"wam_inference_mode must be 'joint' or 'marginal', got {self.wam_inference_mode!r}")
            if self.wam_history < 1 or self.wam_history_stride < 1:
                raise ValueError("wam_history and wam_history_stride must be >= 1")
            if min(self.wam_vision_future_stride, self.wam_tactile_future_stride) < 1:
                raise ValueError("wam_*_future_stride must be >= 1")
            if min(self.wam_vision_future_steps, self.wam_tactile_future_steps) < 0:
                raise ValueError("wam_*_future_steps must be >= 0")
            if self.wam_use_tactile and not (
                self.use_tactile and (self.vitac_encoder_name or "").lower() == "univtac"
            ):
                raise ValueError("wam_use_tactile needs use_tactile=True and vitac_encoder_name='univtac'")

    def validate_features(self) -> None:
        # Keep only the selected tactile streams; the dataset also carries the
        # other variant (rgb vs marker) which would otherwise be fed to SigLIP
        # as an extra camera. With use_tactile=False all tactile streams are
        # left untouched and behave as ordinary cameras.
        if self.use_tactile and self.input_features:
            selected = set(self.tactile_keys)
            for key in list(self.input_features):
                if key.startswith(f"{OBS_IMAGES}.tactile_") and key not in selected:
                    del self.input_features[key]
            missing = selected - set(self.input_features)
            if missing:
                raise ValueError(
                    f"tactile_keys {sorted(missing)} not found in input features "
                    f"{sorted(self.input_features)}. Re-run process_data.py or fix --policy.tactile_keys."
                )
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    def observation_delta_indices_for(self, key: str) -> list:
        """Per-key frame offsets. Only the online (non-frozen) tactile WAM path
        loads extra frames: its streams at the window and future offsets."""
        if (
            self.wam_enabled
            and self.wam_use_tactile
            and not self.vitac_freeze_backbone
            and key in self.tactile_keys
        ):
            return sorted(set(self.wam_history_offsets) | set(self.wam_future_offsets("tactile")))
        return self.observation_delta_indices

    @property
    def wam_history_offsets(self) -> list[int]:
        """Frame offsets of the observation window, oldest first, ending at 0."""
        return [-i * self.wam_history_stride for i in range(self.wam_history - 1, -1, -1)]

    def wam_future_offsets(self, modality: str) -> list[int]:
        if modality == "vision":
            stride, steps = self.wam_vision_future_stride, self.wam_vision_future_steps
        else:
            stride, steps = self.wam_tactile_future_stride, self.wam_tactile_future_steps
        return [stride * (i + 1) for i in range(steps)]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
