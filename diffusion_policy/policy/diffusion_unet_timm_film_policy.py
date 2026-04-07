from typing import Dict

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


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
            model_output = self.model(trajectory, t, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # Final override: snap known positions to exact clean values after the last denoising step
        trajectory = torch.where(inpaint_mask, inpaint_data, trajectory)
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], use_DDIM=False, **kwargs) -> Dict[str, torch.Tensor]:

        mixed_precision = getattr(self, "mixed_precision", None) or "no"
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
            enabled=mixed_precision != "no",
        ):
            nobs = self.normalizer.normalize(obs_dict)
            B = next(iter(nobs.values())).shape[0]
            global_cond = self.obs_encoder(nobs, output_format="cat")

            inpaint_data = torch.zeros(
                B, self.prediction_horizon, self.action_dim, device=self.device, dtype=self.dtype
            )
            inpaint_mask = torch.zeros_like(inpaint_data, dtype=torch.bool)

            nsample = self.conditional_sample(
                inpaint_data=inpaint_data,
                inpaint_mask=inpaint_mask,
                global_cond=global_cond,
                use_ddim=use_DDIM,
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
