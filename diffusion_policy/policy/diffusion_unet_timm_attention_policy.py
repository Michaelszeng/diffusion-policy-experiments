from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet_1d_static_attention import (
    StaticAttentionConditionalUnet1D,
)
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DiffusionUnetTimmAttentionPolicy(BaseImagePolicy):
    """
    Diffusion policy using a timm vision encoder and cross-attention UNet.

    Observations are encoded into a structured token sequence (one token per
    key per timestep) and passed to StaticAttentionConditionalUnet1D via
    cross-attention conditioning.  All observations are assumed to share the
    same horizon (n_obs_steps) at 10 Hz.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: TimmObsEncoder,
        horizon: int,
        num_DDPM_inference_steps=None,
        num_ddim_inference_steps=10,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        num_attention_heads=8,
        attention_dropout=0.1,
        use_temporal_pos_emb=True,
        use_modality_emb=True,
        use_range_emb=True,
        input_pertub=0.1,
        short_range_encoder: Optional[TimmObsEncoder] = None,
        short_range_obs_horizon: Optional[int] = None,
        short_range_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        n_obs_steps = obs_encoder.n_obs_steps

        if short_range_obs_horizon is not None:
            assert isinstance(short_range_obs_horizon, int) and short_range_obs_horizon >= 0, \
                f"short_range_obs_horizon must be a non-negative integer, got {short_range_obs_horizon}"
            assert short_range_obs_horizon <= n_obs_steps, \
                f"short_range_obs_horizon ({short_range_obs_horizon}) must be less than or equal to n_obs_steps ({n_obs_steps})"
            assert 0.0 <= short_range_dropout <= 1.0, \
                f"short_range_dropout must be in [0, 1], got {short_range_dropout}"
            assert short_range_encoder is not None, \
                "short_range_encoder must be provided when short_range_obs_horizon is not None"
            assert short_range_encoder.output_shape() == obs_encoder.output_shape(), \
                f"short_range_encoder output shape {short_range_encoder.output_shape()} must match obs_encoder {obs_encoder.output_shape()}"

        self.model = StaticAttentionConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_encoder.feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            use_temporal_pos_emb=use_temporal_pos_emb,
            use_modality_emb=use_modality_emb,
            max_modalities=obs_encoder.n_modalities,
            use_range_emb=use_range_emb,
            max_ranges=3,  # 0: NULL (timestep), 1: LONG, 2: SHORT
        )

        # DDIM scheduler mirrors the DDPM β-schedule for fast inference
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=noise_scheduler.config.num_train_timesteps,
            beta_start=noise_scheduler.config.beta_start,
            beta_end=noise_scheduler.config.beta_end,
            beta_schedule=noise_scheduler.config.beta_schedule,
            clip_sample=noise_scheduler.config.clip_sample,
            prediction_type=noise_scheduler.config.prediction_type,
        )
        ddim_scheduler.set_timesteps(num_ddim_inference_steps)

        self.obs_encoder = obs_encoder
        self.short_range_encoder = short_range_encoder
        self.short_range_obs_horizon = short_range_obs_horizon
        self.short_range_dropout = short_range_dropout
        if short_range_obs_horizon is not None:
            # Learned replacement token used when short-range tokens are dropped during training.
            self.short_range_null_token = nn.Parameter(torch.zeros(1, obs_encoder.feature_dim))
        self.noise_scheduler = noise_scheduler
        self.ddim_noise_scheduler = ddim_scheduler
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.prediction_horizon = horizon
        self.input_pertub = input_pertub
        self.kwargs = kwargs

        if num_DDPM_inference_steps is None:
            num_DDPM_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_DDPM_inference_steps
        self.num_ddim_inference_steps = num_ddim_inference_steps

    def _encode_obs(
        self,
        nobs: Dict[str, torch.Tensor],
        cached_long: Optional[Dict[str, List[torch.Tensor]]] = None,
        cached_short: Optional[Dict[str, List[torch.Tensor]]] = None,
    ):
        """
        Returns (tokens, positions, modalities, ranges, new_long, new_short) from the obs encoder.

        If cached_long (and/or cached_short) are provided (cache mode, B=1):
            nobs[rgb_key]: List[Tensor (C, H, W)] — the raw frames that have not yet been encoded,
                                                    corresponding to the newest timesteps.
            nobs[lowdim_key]: Tensor (1, T, dim) — low dim obs from the full observation window.
            cached_long / cached_short: Dict[rgb_key -> List[Tensor (D,)]] — already-encoded visual features
                                                                             corresponding to the oldest timesteps.
        Else:
            nobs[key] is (B, T, ...) for every key.
        """
        cache_mode = cached_long is not None

        # Long-range encoding
        if cache_mode:
            nobs_rgb_raw = {key: nobs[key] for key in self.obs_encoder.rgb_keys}
            nobs_lowdim_full = {key: nobs[key] for key in self.obs_encoder.low_dim_keys}
            long_enc, new_long = self.obs_encoder.encode_with_cache(
                nobs_rgb_raw=nobs_rgb_raw,
                nobs_lowdim_full=nobs_lowdim_full,
                cached_rgb=cached_long,
                output_format="dit",
            )
        else:
            long_enc = self.obs_encoder(nobs, output_format="dit")
            new_long: Dict[str, List[torch.Tensor]] = {}

        long_tokens = long_enc["tokens"]
        long_positions = long_enc["positions"]
        long_modalities = long_enc["modality"]
        long_ranges = long_enc["range"]
        new_short: Dict[str, List[torch.Tensor]] = {}

        # Single-encoder policies stop here.
        if self.short_range_obs_horizon is None:
            return long_tokens, long_positions, long_modalities, long_ranges, new_long, new_short

        # Short-range encoding: re-encode the most recent To_short frames via the short-range encoder.
        To = self.obs_encoder.n_obs_steps
        To_short = self.short_range_obs_horizon

        if cache_mode:
            # Most recent up-to-To_short raw RGB frames 
            short_rgb_raw = {key: nobs[key][-To_short:] for key in self.short_range_encoder.rgb_keys}
            # Most recent To_short low dim obs
            short_lowdim = {key: nobs[key][:, (To-To_short):To, ...] for key in self.short_range_encoder.low_dim_keys}
            short_enc, new_short = self.short_range_encoder.encode_with_cache(
                nobs_rgb_raw=short_rgb_raw,
                nobs_lowdim_full=short_lowdim,
                cached_rgb=cached_short if cached_short else {},
                output_format="dit",
            )
        else:
            # Pre-slice each key to the most recent To_short frames so the short-range encoder
            # (configured with n_obs_steps=To_short) encodes the correct window.
            short_nobs = {key: v[:, To - To_short:To, ...] for key, v in nobs.items()}
            short_enc = self.short_range_encoder(short_nobs, output_format="dit")

        short_tokens = short_enc["tokens"]
        # The short-range encoder emits positions 0..To_short-1 (its local window); shift
        # them to align with the original n_obs_steps timeline.
        short_positions = short_enc["positions"] + (To - To_short)
        short_modalities = short_enc["modality"]
        short_ranges = torch.full_like(short_enc["range"], 2)  # 0=null/timestep, 1=LONG, 2=SHORT

        # Per-sample short-range dropout: replace this sample's short-range block with the
        # learned null token to teach the policy to function without short-range context.
        if self.training and self.short_range_dropout > 0.0:
            B = short_tokens.shape[0]
            device = short_tokens.device
            drop_mask = torch.bernoulli(
                torch.full((B,), self.short_range_dropout, device=device)
            ).bool()
            if drop_mask.any():
                null = self.short_range_null_token.view(1, 1, -1).expand_as(short_tokens)
                short_tokens = torch.where(drop_mask.view(B, 1, 1), null, short_tokens)

        tokens = torch.cat([long_tokens, short_tokens], dim=1)
        positions = torch.cat([long_positions, short_positions], dim=1)
        modalities = torch.cat([long_modalities, short_modalities], dim=1)
        ranges = torch.cat([long_ranges, short_ranges], dim=1)
        return tokens, positions, modalities, ranges, new_long, new_short

    # ── Inference ──────────────────────────────────────────────────────────────

    def conditional_sample(
        self,
        inpaint_data,
        inpaint_mask,
        global_cond=None,
        temporal_positions=None,
        modality_indices=None,
        range_indices=None,
        use_ddim=False,
        generator=None,
        **kwargs,
    ):
        if use_ddim:
            scheduler = self.ddim_noise_scheduler
            scheduler.set_timesteps(self.num_ddim_inference_steps)
        else:
            scheduler = self.noise_scheduler
            scheduler.set_timesteps(self.num_inference_steps)

        trajectory = torch.randn(
            size=inpaint_data.shape,
            dtype=inpaint_data.dtype,
            device=inpaint_data.device,
            generator=generator,
        )

        for t in scheduler.timesteps.to(trajectory.device).long():
            # Inpaint: overwrite known positions with correctly-noised inpaint values so
            # the UNet sees a consistent noise level across the whole trajectory.
            if inpaint_mask.any():
                t_batch = torch.full((inpaint_data.shape[0],), t, device=inpaint_data.device, dtype=torch.long)
                noised_inpaint = scheduler.add_noise(inpaint_data, torch.randn_like(inpaint_data), t_batch)
                trajectory = torch.where(inpaint_mask, noised_inpaint, trajectory)
            model_output = self.model(
                sample=trajectory,
                timestep=t,
                global_cond=global_cond,
                temporal_positions=temporal_positions,
                modality_indices=modality_indices,
                range_indices=range_indices,
            )
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # Final override: snap known positions to exact clean values after the last denoising step
        trajectory = torch.where(inpaint_mask, inpaint_data, trajectory)
        return trajectory

    def predict_action(
        self,
        raw_obs_dict: Dict[str, torch.Tensor],
        cached_long: Optional[Dict[str, List[torch.Tensor]]] = None,
        cached_short: Optional[Dict[str, List[torch.Tensor]]] = None,
        use_DDIM: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict an action sequence.

        If cached_long (and/or cached_short) are provided (cache mode, B=1):
            raw_obs_dict[rgb_key]: List[Tensor (C, H, W)] — the raw frames that have not yet been encoded,
                                                            corresponding to the newest timesteps.
            raw_obs_dict[lowdim_key]: Tensor (1, T, dim) — low dim obs from the full observation window.
            cached_long / cached_short: Dict[rgb_key -> List[Tensor (D,)]] — already-encoded visual features
                                                                             corresponding to the oldest timesteps.
        Else:
            raw_obs_dict[key] is (B, T, ...) for every key.
        """
        assert "past_action" not in raw_obs_dict
        cache_mode = cached_long is not None

        mixed_precision = getattr(self, "mixed_precision", None) or "no"
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
            enabled=mixed_precision != "no",
        ):
            # Normalize raw obs. In cache mode rgb keys are lists of (C,H,W) tensors that need
            # stacking before the normalizer accepts them, then unstacking back for the encoder.
            if cache_mode:
                nobs: Dict[str, torch.Tensor] = {}
                for key in self.obs_encoder.rgb_keys:
                    raws = raw_obs_dict[key]
                    if raws:
                        stacked = torch.stack(raws, dim=0).unsqueeze(0)  # (1, n_raw, C, H, W)
                        normalized = self.normalizer.normalize({key: stacked})[key]
                        nobs[key] = [normalized[0, i] for i in range(len(raws))]
                    else:
                        nobs[key] = []
                for key in self.obs_encoder.low_dim_keys:
                    nobs[key] = self.normalizer.normalize({key: raw_obs_dict[key]})[key]
                B = 1
            else:
                nobs = self.normalizer.normalize(raw_obs_dict)
                B = next(iter(nobs.values())).shape[0]

            tokens, positions, modalities, ranges, new_long, new_short = self._encode_obs(
                nobs, cached_long=cached_long, cached_short=cached_short
            )

            # Empty inpaint slots: no known/fixed actions, the UNet predicts the full horizon.
            inpaint_data = torch.zeros(B, self.prediction_horizon, self.action_dim, device=self.device, dtype=self.dtype)
            inpaint_mask = torch.zeros_like(inpaint_data, dtype=torch.bool)

            # DDIM/DDPM denoising loop conditioned on the encoded token sequence.
            nsample = self.conditional_sample(
                inpaint_data=inpaint_data,
                inpaint_mask=inpaint_mask,
                global_cond=tokens,
                temporal_positions=positions,
                modality_indices=modalities,
                range_indices=ranges,
                use_ddim=use_DDIM,
            )
            assert nsample.shape == (B, self.prediction_horizon, self.action_dim)

            result = {"action_pred": self.normalizer["action"].unnormalize(nsample)}
            if cache_mode:
                result["new_features_long"] = new_long
                result["new_features_short"] = new_short
            return result

    # ── Training ───────────────────────────────────────────────────────────────

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        tokens, positions, modalities, ranges, _, _ = self._encode_obs(nobs)

        trajectory = nactions
        noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=self.dtype)
        # Input perturbation to alleviate exposure bias (https://github.com/forever208/DDPM-IP)
        noise_new = noise + self.input_pertub * torch.randn_like(noise)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (nactions.shape[0],),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise_new, timesteps).detach()

        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            global_cond=tokens,
            temporal_positions=positions,
            modality_indices=modalities,
            range_indices=ranges,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        target = noise if pred_type == "epsilon" else trajectory
        loss = reduce(F.mse_loss(pred, target, reduction="none"), "b ... -> b (...)", "mean").mean()
        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
