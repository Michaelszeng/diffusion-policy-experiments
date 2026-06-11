from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.rtc_utils import build_rtc_tensors, pigdm_eps_correction


class DiffusionUnetTimmFilmPolicy(BaseImagePolicy):
    """
    Diffusion policy using a timm vision encoder and FiLM-conditioned UNet.

    Observations are encoded into a flat feature vector (cat mode) and passed
    as global conditioning to ConditionalUnet1D.  All observations are assumed
    to share the same horizon (n_obs_steps) at 10 Hz.
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
        input_pertub=0.1,
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_encoder.cat_output_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
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

    # ── Inference ──────────────────────────────────────────────────────────────

    def conditional_sample(
        self,
        inpaint_data,
        inpaint_mask,
        global_cond=None,
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
            model_output = self.model(trajectory, t, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # Final override: snap known positions to exact clean values after the last denoising step
        trajectory = torch.where(inpaint_mask, inpaint_data, trajectory)
        return trajectory

    def _rtc_conditional_sample(
        self, scheduler, trajectory, global_cond,
        rtc_target, rtc_weights, rtc_max_guidance_weight, rtc_sigma_d,
        generator=None, **kwargs,
    ):
        """ΠGDM-guided reverse diffusion (FiLM conditioning). ``rtc_target`` is (B, P, A) and
        ``rtc_weights`` is (P,), both pre-placed in the full prediction-horizon frame. The
        FiLM ``global_cond`` is detached so backward flows only through the UNet."""
        alphas_cumprod = scheduler.alphas_cumprod.to(trajectory.device)
        global_cond = global_cond.detach() if global_cond is not None else None

        for t in scheduler.timesteps.to(trajectory.device).long():
            abar = alphas_cumprod[t]
            x_t = trajectory.detach().requires_grad_(True)
            with torch.enable_grad():
                model_output = self.model(x_t, t, global_cond=global_cond)
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
        cached_long: Optional[Dict[str, List[torch.Tensor]]] = None,
        cached_short: Optional[Dict[str, List[torch.Tensor]]] = None,
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
        Predict an action sequence.

        Standard path (cached_long is None): obs_dict[k] is (B, T, ...); normalized internally.

        RTC (ΠGDM real-time chunking): when ``rtc_target`` is provided it is the NORMALIZED
        previous-chunk target for the future-action window, shape (B, H, A) with
        H = prediction_horizon - (n_obs_steps - 1). See rtc_utils for the guidance math.

        Cache-aware path (cached_long provided): B=1 inference. obs_dict[rgb_key] is a
        List[Tensor (C,H,W)] of raw un-normalized frames at the newest positions;
        obs_dict[lowdim_key] is a (1, T, dim) un-normalized tensor. cached_long[k] is
        List[Tensor (D,)] of cached features (in encoder space — produced by a prior call)
        at the oldest positions. cached_short is accepted for API symmetry with the
        attention policy but unused (FiLM has no short-range encoder). The response carries
        'new_features_long' (per-key lists of newly-encoded features) and an empty
        'new_features_short'.
        """
        del cached_short  # FiLM has no short-range encoder
        cache_mode = cached_long is not None

        mixed_precision = getattr(self, "mixed_precision", None) or "no"
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
            enabled=mixed_precision != "no",
        ):
            if cache_mode:
                # Normalize raw obs. RGB keys are lists of (C,H,W) tensors that need stacking
                # before the normalizer accepts them, then unstacking back for the encoder.
                nobs: Dict[str, torch.Tensor] = {}
                for k in self.obs_encoder.rgb_keys:
                    raws = obs_dict[k]
                    if raws:
                        stacked = torch.stack(raws, dim=0).unsqueeze(0)  # (1, n_raw, C, H, W)
                        normalized = self.normalizer.normalize({k: stacked})[k]
                        nobs[k] = [normalized[0, i] for i in range(len(raws))]
                    else:
                        nobs[k] = []
                for k in self.obs_encoder.low_dim_keys:
                    nobs[k] = self.normalizer.normalize({k: obs_dict[k]})[k]
                B = 1
                global_cond, new_long = self.obs_encoder.encode_with_cache(
                    nobs_rgb_raw={k: nobs[k] for k in self.obs_encoder.rgb_keys},
                    nobs_lowdim_full={k: nobs[k] for k in self.obs_encoder.low_dim_keys},
                    cached_rgb=cached_long,
                    output_format="cat",
                )
            else:
                nobs = self.normalizer.normalize(obs_dict)
                B = next(iter(nobs.values())).shape[0]
                # Keep only the n_obs_steps real frames (see compute_loss for why).
                To = self.obs_encoder.n_obs_steps
                nobs = {k: v[:, :To] for k, v in nobs.items()}
                global_cond = self.obs_encoder(nobs, output_format="cat")
                new_long: Dict[str, List[torch.Tensor]] = {}

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
                global_cond = global_cond.detach()  # free encoder graph; cond is forward-only

            # DDIM/DDPM denoising loop with FiLM-style global conditioning.
            nsample = self.conditional_sample(
                inpaint_data=inpaint_data,
                inpaint_mask=inpaint_mask,
                global_cond=global_cond,
                use_ddim=use_DDIM,
                rtc_target=target_full,
                rtc_weights=weights_full,
                rtc_max_guidance_weight=rtc_max_guidance_weight,
                rtc_sigma_d=rtc_sigma_d,
            )
            assert nsample.shape == (B, self.prediction_horizon, self.action_dim)

            result = {"action_pred": self.normalizer["action"].unnormalize(nsample)}
            if cache_mode:
                result["new_features_long"] = new_long
                result["new_features_short"] = {}  # FiLM has no short-range encoder
            return result

    # ── Training ───────────────────────────────────────────────────────────────

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        # The dataset returns each obs key at length == prediction horizon (the sampler
        # pads up to sequence_length); only the first n_obs_steps frames are real obs —
        # the rest is NaN/zero padding from key_first_k. Slice to n_obs_steps so the
        # encoder emits exactly n_keys * n_obs_steps features (matching cat_output_dim).
        # (Mirrors the slice in diffusion_unet_timm_attention_policy._encode_obs.)
        To = self.obs_encoder.n_obs_steps
        nobs = {k: v[:, :To] for k, v in nobs.items()}
        global_cond = self.obs_encoder(nobs, output_format="cat")
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
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        target = noise if pred_type == "epsilon" else trajectory
        loss = reduce(F.mse_loss(pred, target, reduction="none"), "b ... -> b (...)", "mean").mean()
        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
