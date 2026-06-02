from typing import Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet_1d_static_attention import (
    StaticAttentionConditionalUnet1D,
)
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder, _RANGE_SHORT
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.rtc_utils import build_rtc_tensors, pigdm_eps_correction


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
        short_range_encoder: Optional[TimmObsEncoder] = None,
        short_range_obs_horizon: Optional[int] = None,
        short_range_dropout: float = 0.0,
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
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

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

    def _encode_obs(self, nobs: Dict[str, torch.Tensor]):
        """Returns tokens, positions, modalities, ranges from the obs encoder(s) (dit mode).

        When a short_range_encoder is configured, the most recent short_range_obs_horizon
        frames are also encoded and appended as SHORT-range tokens.  During training a
        per-sample dropout mask zeroes out those tokens with probability short_range_dropout.
        """
        # The dataset returns each obs key with length == prediction horizon (because the
        # sampler pads up to sequence_length); only the first n_obs_steps frames are real
        # observations — the rest is NaN/zero padding from key_first_k. Slice down here so
        # the encoder produces exactly n_keys * n_obs_steps tokens (matching the
        # positions/modalities/ranges it generates from self.n_obs_steps).
        To = self.obs_encoder.n_obs_steps
        long_nobs = {k: v[:, :To] for k, v in nobs.items()}
        enc = self.obs_encoder(long_nobs, output_format="dit")
        tokens    = enc["tokens"]     # (B, N_long, D)
        positions = enc["positions"]  # (B, N_long)
        modalities = enc["modality"]  # (B, N_long)
        ranges    = enc["range"]      # (B, N_long)

        if self.short_range_encoder is not None:
            h = self.short_range_obs_horizon
            # Take the last h frames of the *valid* observation window, not the last h
            # frames of the padded tensor (which would be NaN padding).
            short_nobs = {k: v[:, To - h:To] for k, v in nobs.items()}
            s_enc = self.short_range_encoder(short_nobs, output_format="dit")
            s_tokens     = s_enc["tokens"]
            s_positions  = s_enc["positions"]
            s_modalities = s_enc["modality"]
            # The policy knows this encoder is short-range; assign the tag here,
            # not inside the encoder (which has no knowledge of its own role).
            s_ranges = torch.full(
                s_tokens.shape[:2], _RANGE_SHORT, dtype=torch.long, device=s_tokens.device
            )

            # Per-sample dropout: zero out all short-range tokens for dropped samples
            if self.training and self.short_range_dropout > 0.0:
                B = s_tokens.shape[0]
                keep = (torch.rand(B, device=s_tokens.device) >= self.short_range_dropout)
                s_tokens = s_tokens * keep.float().view(B, 1, 1)

            tokens     = torch.cat([tokens,     s_tokens],     dim=1)
            positions  = torch.cat([positions,  s_positions],  dim=1)
            modalities = torch.cat([modalities, s_modalities], dim=1)
            ranges     = torch.cat([ranges,     s_ranges],     dim=1)

        return tokens, positions, modalities, ranges

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
        rtc_target=None,
        rtc_weights=None,
        rtc_max_guidance_weight=5.0,
        rtc_sigma_d=1.0,
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

        # ── RTC (ΠGDM guidance) path ──────────────────────────────────────────
        if rtc_target is not None:
            return self._rtc_conditional_sample(
                scheduler, trajectory, global_cond,
                temporal_positions, modality_indices, range_indices,
                rtc_target, rtc_weights, rtc_max_guidance_weight, rtc_sigma_d,
                generator=generator, **kwargs,
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

    def _rtc_conditional_sample(
        self, scheduler, trajectory, global_cond,
        temporal_positions, modality_indices, range_indices,
        rtc_target, rtc_weights, rtc_max_guidance_weight, rtc_sigma_d,
        generator=None, **kwargs,
    ):
        """ΠGDM-guided reverse diffusion. ``rtc_target`` is (B, P, A) and ``rtc_weights``
        is (P,), both pre-placed in the full prediction-horizon frame (zeros outside the
        guided slice). Conditioning is detached so backward flows only through the UNet."""
        alphas_cumprod = scheduler.alphas_cumprod.to(trajectory.device)
        # Backward must NOT flow into the encoder/conditioning tokens.
        global_cond = global_cond.detach() if global_cond is not None else None

        for t in scheduler.timesteps.to(trajectory.device).long():
            abar = alphas_cumprod[t]
            x_t = trajectory.detach().requires_grad_(True)
            with torch.enable_grad():
                model_output = self.model(
                    sample=x_t,
                    timestep=t,
                    global_cond=global_cond,
                    temporal_positions=temporal_positions,
                    modality_indices=modality_indices,
                    range_indices=range_indices,
                )
                eps_cond = pigdm_eps_correction(
                    x_t, model_output, abar, rtc_target, rtc_weights,
                    rtc_max_guidance_weight, rtc_sigma_d,
                )
            trajectory = scheduler.step(
                eps_cond, t, x_t.detach(), generator=generator, **kwargs
            ).prev_sample

        # ΠGDM is a soft constraint — no final hard snap (document Algorithm 3).
        return trajectory

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        use_DDIM: bool = False,
        rtc_target: Optional[torch.Tensor] = None,
        rtc_inference_delay: Optional[int] = None,
        rtc_execution_horizon: Optional[int] = None,
        rtc_schedule: str = "exp",
        rtc_max_guidance_weight: float = 5.0,
        rtc_sigma_d: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict an action sequence. obs_dict[key] is (B, T, ...) for every key.

        RTC (ΠGDM real-time chunking): when ``rtc_target`` is provided it is the NORMALIZED
        previous-chunk target for the future-action window, shape (B, H, A) with
        H = prediction_horizon - (n_obs_steps - 1). The first ``rtc_inference_delay`` slots are
        hard-frozen, tapering to zero at ``rtc_execution_horizon`` per ``rtc_schedule``.
        """
        assert "past_action" not in obs_dict

        mixed_precision = getattr(self, "mixed_precision", None) or "no"
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
            enabled=mixed_precision != "no",
        ):
            nobs = self.normalizer.normalize(obs_dict)
            B = next(iter(nobs.values())).shape[0]
            tokens, positions, modalities, ranges = self._encode_obs(nobs)

            inpaint_data = torch.zeros(
                B, self.prediction_horizon, self.action_dim, device=self.device, dtype=self.dtype
            )
            inpaint_mask = torch.zeros_like(inpaint_data, dtype=torch.bool)

            # Build full-horizon RTC target/weights from the future-window target (if any).
            target_full, weights_full = build_rtc_tensors(
                rtc_target, B, self.obs_encoder.n_obs_steps, self.prediction_horizon,
                self.action_dim, rtc_inference_delay, rtc_execution_horizon, rtc_schedule,
                self.device, self.dtype,
            )
            if target_full is not None:
                tokens = tokens.detach()  # free encoder graph; cond is forward-only

            # DDIM/DDPM denoising loop conditioned on the encoded token sequence.
            nsample = self.conditional_sample(
                inpaint_data=inpaint_data,
                inpaint_mask=inpaint_mask,
                global_cond=tokens,
                temporal_positions=positions,
                modality_indices=modalities,
                range_indices=ranges,
                use_ddim=use_DDIM,
                rtc_target=target_full,
                rtc_weights=weights_full,
                rtc_max_guidance_weight=rtc_max_guidance_weight,
                rtc_sigma_d=rtc_sigma_d,
            )
            assert nsample.shape == (B, self.prediction_horizon, self.action_dim)
            return {"action_pred": self.normalizer["action"].unnormalize(nsample)}

    # ── Training ───────────────────────────────────────────────────────────────

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        tokens, positions, modalities, ranges = self._encode_obs(nobs)

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
