from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet_1d_static_attention import (
    StaticAttentionConditionalUnet1D,
)
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DiffusionUnetHybridImageAttentionPolicy(BaseImagePolicy):
    """
    Diffusion policy using a hybrid (Robomimic / R3M) obs encoder with cross-attention UNet.

    Instead of collapsing all observation timesteps into a single FiLM conditioning vector,
    each obs timestep is encoded to an independent feature token (B, To, obs_feature_dim).
    These To tokens are passed to StaticAttentionConditionalUnet1D, which attends to them
    at every residual block via cross-attention.

    Compatible with train_diffusion_unet_hybrid_workspace_no_env.py.
    """

    def __init__(
        self,
        shape_meta: dict,
        DDPM_noise_scheduler: DDPMScheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        diffusion_step_embed_dim: int = 256,  # Size of the diffusion timestep embedding
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,  # Resnet param
        n_groups: int = 8,  # Resnet param
        num_DDPM_inference_steps: int = 100,
        num_DDIM_inference_steps: int = 10,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        use_temporal_pos_emb: bool = True,
        use_modality_emb: bool = True,
        use_range_emb: bool = True,
        obs_encoder: nn.Module = None,  # Required: Hydra-instantiated encoder (e.g. RobomimicObsEncoder, ResNetObsEncoder)
        short_range_encoder: nn.Module = None,  # Optional: separate encoder for the most recent short_range_obs_horizon frames
        short_range_obs_horizon: Optional[int] = None,  # Number of most-recent frames treated as short-range; None disables dual-encoder
        short_range_dropout: float = 0.0,  # Probability of replacing short-range tokens with a learned null token during training
        self_trained_obs_encoder: str = None,  # Path to a checkpoint file to load obs encoder weights from
        freeze_self_trained_obs_encoder: bool = False,  # Freeze encoder after loading self_trained_obs_encoder weights
        inference_loading: bool = False,  # Skip self_trained_obs_encoder loading during inference (use final checkpoint weights)
        past_action_visible: bool = False,
        # parameters passed to step
        **kwargs,
    ):
        """
        Additional Features implemented by this policy class:
         - Cross-attention conditioning: each obs timestep is a separate token, preserving
           temporal structure rather than flattening into a single FiLM vector
         - Self-trained observation encoder -- load observation encoder weights from a specified checkpoint
        """
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        if obs_encoder is None:
            raise ValueError(
                "obs_encoder must be provided. Specify it in the policy config using "
                "_target_: diffusion_policy.model.vision.robomimic_config_util.RobomimicObsEncoder "
                "(or another encoder class)."
            )

        if short_range_obs_horizon is not None:
            assert isinstance(short_range_obs_horizon, int) and short_range_obs_horizon >= 0, \
                f"short_range_obs_horizon must be a non-negative integer, got {short_range_obs_horizon}"
            assert short_range_obs_horizon < n_obs_steps, \
                f"short_range_obs_horizon ({short_range_obs_horizon}) must be strictly less than n_obs_steps ({n_obs_steps})"
            assert 0.0 <= short_range_dropout <= 1.0, \
                f"short_range_dropout must be in [0, 1], got {short_range_dropout}"
            assert short_range_encoder is not None, \
                "short_range_encoder must be provided when short_range_obs_horizon is not None"
            assert short_range_encoder.output_shape() == obs_encoder.output_shape(), \
                f"short_range_encoder output shape {short_range_encoder.output_shape()} must match obs_encoder {obs_encoder.output_shape()}"

        # obs_feature_dim: dimensionality of a single obs-timestep token.
        # Each obs timestep is encoded independently, giving one (obs_feature_dim,) token.
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim
        print(f"Input dim: {input_dim}, Obs feature dim (token dim): {obs_feature_dim}, n_obs_steps (n_tokens): {n_obs_steps}")

        # max_modalities: 0 = diffusion timestep token (null), 1 = hybrid encoder tokens
        # max_ranges:     0 = null (diffusion timestep token),  1 = LONG,  2 = SHORT
        self.model = StaticAttentionConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=obs_feature_dim,
            local_cond_dim=None,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            use_temporal_pos_emb=use_temporal_pos_emb,
            max_temporal_position=1000,
            use_modality_emb=use_modality_emb,
            max_modalities=2,  # 0=null/timestep, 1=hybrid encoder
            use_range_emb=use_range_emb,
            max_ranges=3,      # 0=null/timestep, 1=LONG, 2=SHORT
        )

        # Only load obs encoder weights from self-trained checkpoint during training.
        # During inference we skip this entirely so the final checkpoint weights are used.
        if not inference_loading and self_trained_obs_encoder is not None:
            self._load_and_freeze_obs_encoder(obs_encoder, self_trained_obs_encoder, freeze_self_trained_obs_encoder)

        # Create a DDIM sampler that mirrors the DDPM β-schedule
        DDIM_noise_scheduler = DDIMScheduler(
            num_train_timesteps=DDPM_noise_scheduler.num_train_timesteps,
            beta_start=DDPM_noise_scheduler.beta_start,
            beta_end=DDPM_noise_scheduler.beta_end,
            beta_schedule=DDPM_noise_scheduler.beta_schedule,
            clip_sample=DDPM_noise_scheduler.clip_sample,
            prediction_type=DDPM_noise_scheduler.prediction_type,
        )
        DDIM_noise_scheduler.set_timesteps(num_DDIM_inference_steps)

        self.past_action_visible = past_action_visible
        self.obs_encoder = obs_encoder
        self.short_range_encoder = short_range_encoder
        self.short_range_obs_horizon = short_range_obs_horizon
        self.short_range_dropout = short_range_dropout
        if short_range_obs_horizon is not None:
            self.short_range_null_token = nn.Parameter(torch.zeros(1, obs_feature_dim))
        self.DDPM_noise_scheduler = DDPM_noise_scheduler
        self.DDIM_noise_scheduler = DDIM_noise_scheduler
        self.num_DDPM_inference_steps = num_DDPM_inference_steps
        self.num_DDIM_inference_steps = num_DDIM_inference_steps
        self.normalizer = LinearNormalizer()  # Empty normalizer; parameters will be set later
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.use_temporal_pos_emb = use_temporal_pos_emb
        self.use_modality_emb = use_modality_emb
        self.use_range_emb = use_range_emb
        # Filter kwargs to only include valid scheduler.step() parameters
        # Remove policy-specific parameters that were already consumed
        scheduler_params = {"eta", "use_clipped_model_output", "variance_noise"}
        self.kwargs = {k: v for k, v in kwargs.items() if k in scheduler_params}

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Observation Encoder params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        if short_range_encoder is not None:
            print("Short-range Encoder params: %e" % sum(p.numel() for p in self.short_range_encoder.parameters()))

    def _load_and_freeze_obs_encoder(self, obs_encoder, checkpoint_path, freeze):
        """
        Initializes the obs encoder from a prior training run and optionally freezes it.

        The checkpoint is expected to be a full model checkpoint saved by the training workspace,
        where obs encoder weights are stored under the "module.obs_encoder.*" prefix.
        """
        # Load the full model checkpoint and extract only the obs_encoder sub-dict
        print(f"Loading obs encoder from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        full_model_state_dict = checkpoint["state_dicts"]["model"]

        prefix = "module.obs_encoder."
        obs_encoder_state_dict = {
            key[len(prefix):]: value
            for key, value in full_model_state_dict.items()
            if key.startswith(prefix)
        }

        assert len(obs_encoder_state_dict) != 0, ("No obs_encoder weights found in checkpoint."
            f"Keys present: {list(full_model_state_dict.keys())}")

        obs_encoder.load_state_dict(obs_encoder_state_dict, strict=True)
        print(f"Loaded {len(obs_encoder_state_dict)} obs_encoder parameters")

        # Optionally freeze the encoder so only the diffusion UNet trains
        if freeze:
            print("Freezing obs encoder parameters.")
            for param in obs_encoder.parameters():
                param.requires_grad = False

    def _encode_obs(self, nobs: Dict[str, torch.Tensor], B: int) -> tuple:
        """
        Encode n_obs_steps observation timesteps into a cross-attention token sequence.

        With dual encoders (short_range_obs_horizon not None): the oldest
        (n_obs_steps - short_range_obs_horizon) frames are encoded by obs_encoder as
        LONG-range tokens, and the most recent short_range_obs_horizon frames are encoded
        by short_range_encoder as SHORT-range tokens. During training, short-range tokens
        are replaced with a learned null token with probability short_range_dropout.

        Args:
            nobs: normalized obs dict, each value shape (B, To, ...)
            B:    batch size

        Returns:
            tokens:     (B, To, obs_feature_dim) — one token per obs timestep
            positions:  (B, To) — temporal index; 0 = oldest, To-1 = most recent
            modalities: (B, To) — all 1 (single hybrid-encoder modality)
            ranges:     (B, To) — 1 = LONG, 2 = SHORT
        """
        To = self.n_obs_steps

        if self.short_range_obs_horizon is None:
            # Single encoder — all tokens are LONG range
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            tokens = nobs_features.reshape(B, To, self.obs_feature_dim)
            device = tokens.device
            positions = torch.arange(To, device=device).unsqueeze(0).expand(B, -1)
            modalities = torch.ones(B, To, dtype=torch.long, device=device)
            ranges = torch.ones(B, To, dtype=torch.long, device=device)
            return tokens, positions, modalities, ranges

        # Dual encoder: all frames → long-range tokens; most-recent frames ALSO → short-range tokens.
        # Total tokens: To (long) + To_short (short) = n_obs_steps + short_range_obs_horizon.
        To_short = self.short_range_obs_horizon
        short_start = To - To_short  # index of the first short-range frame in nobs

        # All n_obs_steps frames encoded by the long-range encoder
        all_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        long_tokens = self.obs_encoder(all_nobs).reshape(B, To, self.obs_feature_dim)
        device = long_tokens.device

        # Most recent To_short frames ALSO encoded by the short-range encoder
        short_nobs = dict_apply(nobs, lambda x: x[:, short_start:To, ...].reshape(-1, *x.shape[2:]))
        short_tokens = self.short_range_encoder(short_nobs).reshape(B, To_short, self.obs_feature_dim)

        # Per-sample short-range dropout: replace each sample independently
        if self.training and self.short_range_dropout > 0.0:
            drop_mask = torch.bernoulli(
                torch.full((B,), self.short_range_dropout, device=device)
            ).bool()  # (B,) — True means replace this sample's short-range tokens
            if drop_mask.any():
                null = self.short_range_null_token.view(1, 1, -1).expand(B, To_short, -1)
                short_tokens = torch.where(drop_mask.view(B, 1, 1), null, short_tokens)

        tokens = torch.cat([long_tokens, short_tokens], dim=1)  # (B, To + To_short, D)

        # Long-range positions: 0..To-1. Short-range positions: same frames as short_start..To-1.
        long_positions = torch.arange(To, device=device).unsqueeze(0).expand(B, -1)
        short_positions = torch.arange(short_start, To, device=device).unsqueeze(0).expand(B, -1)
        positions = torch.cat([long_positions, short_positions], dim=1)

        modalities = torch.ones(B, To + To_short, dtype=torch.long, device=device)
        long_ranges = torch.ones(B, To, dtype=torch.long, device=device)
        short_ranges = torch.full((B, To_short), 2, dtype=torch.long, device=device)
        ranges = torch.cat([long_ranges, short_ranges], dim=1)

        return tokens, positions, modalities, ranges

    def get_inpaint_mask(self, shape):
        """
        Returns a boolean inpainting mask of shape (B, T, action_dim).

        True = "known/given" (will be inpainted and excluded from loss).
        False = "to predict" (model must denoise these).

        When past_action_visible=False (the default), all actions are predicted from
        scratch and the mask is all-False. When past_action_visible=True, the first
        (n_obs_steps - 1) action timesteps are treated as known, since they overlap with
        the observation horizon.

        Args:
            shape: (B, T, D) — batch size, prediction horizon, action dim.

        Returns:
            Bool tensor of shape (B, T, D); True = inpainted (fixed).
        """
        B, T, D = shape
        device = self.device
        if not self.past_action_visible:
            return torch.zeros(size=shape, dtype=torch.bool, device=device)
        action_steps = max(self.n_obs_steps - 1, 0)
        steps = torch.arange(0, T, device=device).reshape(1, T).expand(B, T)
        return (steps < action_steps).reshape(B, T, 1).expand(B, T, D)

    ####################################################################################################################
    ### Inference
    ####################################################################################################################
    def conditional_sample(
        self,
        inpaint_data,
        inpaint_mask,
        global_cond=None,
        temporal_positions=None,
        modality_indices=None,
        range_indices=None,
        generator=None,
        use_DDIM=False,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        """
        Run reverse diffusion sampling.

        Args:
            inpaint_data (torch.Tensor):       Known values for naive inpainting. Shape: [B, T, Da]
            inpaint_mask (torch.Tensor):       Boolean mask indicating which values to inpaint. Shape: [B, T, Da]
            global_cond (torch.Tensor):        Observation token sequence from encoder. Shape: [B, To, obs_feature_dim]
            temporal_positions (torch.Tensor): Temporal index per token. Shape: [B, To]
            modality_indices (torch.Tensor):   Modality index per token. Shape: [B, To]
            range_indices (torch.Tensor):      Short/long range index per token. Shape: [B, To]
            generator (torch.Generator):       RNG
            use_DDIM (bool):                   DDIM vs DDPM
            **kwargs:                          Additional arguments passed to scheduler.step()

        Returns:
            torch.Tensor: Denoised action trajectory. Shape: [B, T, Da]
        """
        if use_DDIM:
            scheduler = self.DDIM_noise_scheduler
            scheduler.set_timesteps(self.num_DDIM_inference_steps)
        else:
            scheduler = self.DDPM_noise_scheduler
            scheduler.set_timesteps(self.num_DDPM_inference_steps)

        trajectory = torch.randn(
            size=inpaint_data.shape,
            dtype=inpaint_data.dtype,
            device=inpaint_data.device,
            generator=generator,
        )  # Start with random noise trajectory

        for t in scheduler.timesteps:
            # 1. Inpaint: overwrite known positions with correctly-noised inpaint values so
            #    the UNet sees a consistent noise level across the whole trajectory.
            if inpaint_mask.any():
                t_batch = torch.full((inpaint_data.shape[0],), t, device=inpaint_data.device, dtype=torch.long)
                noised_inpaint = scheduler.add_noise(inpaint_data, torch.randn_like(inpaint_data), t_batch)
                trajectory[inpaint_mask] = noised_inpaint[inpaint_mask]
            # 2. predict model output
            model_output = self.model(
                sample=trajectory,
                timestep=t.to(trajectory.device),
                global_cond=global_cond,
                temporal_positions=temporal_positions,
                modality_indices=modality_indices,
                range_indices=range_indices,
            )
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        # Final override: snap known positions to exact clean values after the last denoising step
        trajectory[inpaint_mask] = inpaint_data[inpaint_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], use_DDIM=False) -> Dict[str, torch.Tensor]:
        """
        Args:
            obs_dict: Dict containing observation data with the following structure:
                obs_dict["obs"]: dict of obs key → (B, To, ...) tensors
            use_DDIM: Whether to use DDIM sampling instead of DDPM

        Returns:
            Dictionary containing predicted actions: {"action": torch.Tensor, "action_pred": torch.Tensor}
                action:      [B, Ta, Da] — the n_action_steps slice starting after the last observation
                action_pred: [B, T,  Da] — the full predicted horizon

        Where:
            B  = batch size
            To = n_obs_steps (number of observation timesteps, e.g., 2)
            Ta = n_action_steps (number of predicted action timesteps, e.g., 8)
            Da = action_dim (action space dimensionality)
        """
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict  # not implemented yet

        mixed_precision = getattr(self, "mixed_precision", None) or "no"
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
            enabled=mixed_precision != "no",
        ):
            # Normalize obs dict
            nobs = self.normalizer.normalize(obs_dict["obs"])
            value = next(iter(nobs.values()))
            B, To = value.shape[:2]  # Batch size
            T = self.horizon          # T  = prediction horizon
            Da = self.action_dim      # Da = action dim
            To = self.n_obs_steps     # To = obs horizon

            # Encode observations into cross-attention token sequence
            tokens, positions, modalities, ranges = self._encode_obs(nobs, B)

            # No inpainting by default
            inpaint_data = torch.zeros(size=(B, T, Da), device=self.device, dtype=self.dtype)
            inpaint_mask = torch.zeros_like(inpaint_data, dtype=torch.bool)

            # Run reverse diffusion sampling
            nsample = self.conditional_sample(
                inpaint_data,
                inpaint_mask,
                global_cond=tokens,
                temporal_positions=positions,
                modality_indices=modalities,
                range_indices=ranges,
                use_DDIM=use_DDIM,
                **self.kwargs,
            )

            # Unnormalize prediction
            naction_pred = nsample[..., :Da]
            action_pred = self.normalizer["action"].unnormalize(naction_pred)

            # Get action
            # pred_action is the full predicted action sequence; action is a slice beginning after
            # the latest observation and with length n_action_steps
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

            result = {"action": action, "action_pred": action_pred}
            return result

    ####################################################################################################################
    ### Training
    ####################################################################################################################
    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set normalizer's parameters based on dataset."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def load_state_dict(self, state_dict, strict=True):
        # Checkpoints saved with BatchNorm encoders contain running_mean / running_var /
        # num_batches_tracked buffers that don't exist in GroupNorm. Drop them so that
        # resuming from a pre-GroupNorm checkpoint doesn't raise "unexpected keys".
        own_keys = set(self.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in own_keys}
        missing = own_keys - filtered.keys()
        if missing:
            raise RuntimeError(f"Missing key(s) in state_dict: {missing}")
        super().load_state_dict(filtered, strict=True)

    def noise_trajectory(self, trajectory):
        """Add random amount of noise to the clean trajectory."""
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        batch_size = trajectory.shape[0]
        timesteps = torch.randint(
            low=0,
            high=self.DDPM_noise_scheduler.config.num_train_timesteps,
            size=(batch_size,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.DDPM_noise_scheduler.add_noise(trajectory, noise, timesteps)
        return noisy_trajectory, timesteps, noise

    def forward(self, batch, noisy_trajectory, timesteps, inpaint_mask=None):
        """
        Encode observations into a token sequence and run the cross-attention UNet forward pass.

        Called by the workspace training loop as:
            pred = self.model(batch, noisy_trajectory, timesteps)

        Returns a prediction of either the noise residual or the denoised trajectory,
        depending on the scheduler's prediction_type config.
        """
        assert "valid_mask" not in batch

        # Collect normalized observations
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        # Sizing convenience
        B = nactions.shape[0]

        # Encode observations into cross-attention token sequence
        tokens, positions, modalities, ranges = self._encode_obs(nobs, B)

        if inpaint_mask is not None:
            noisy_trajectory[inpaint_mask] = nactions[inpaint_mask]

        # Predict the noise residual (or clean trajectory, depending on prediction_type)
        return self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            global_cond=tokens,
            temporal_positions=positions,
            modality_indices=modalities,
            range_indices=ranges,
        )

    def compute_loss(self, trajectory, noise, pred, loss_mask=None):
        """
        Compute MSE loss against the noise or clean trajectory depending on prediction_type.

        Optional loss mask used during validation so that we only compute loss on the
        timesteps corresponding to future actions.
        """
        # Check whether we are trying to predict the noise added to the trajectory or the trajectory itself
        pred_type = self.DDPM_noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype) if loss_mask is not None else loss
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
