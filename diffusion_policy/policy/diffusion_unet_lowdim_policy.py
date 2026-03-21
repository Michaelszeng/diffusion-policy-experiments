"""
NOT WELL TESTED.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_DDPM_inference_steps=None,
        num_DDIM_inference_steps=10,
        diffusion_step_embed_dim=128,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        use_target_cond=False,
        target_dim=None,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()
        if use_target_cond:
            assert target_dim is not None

        # Parse shape_meta to get action_dim and obs_dim
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_dim = 0
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            assert len(shape) == 1, f"Lowdim policy only supports 1D observations, got {key}: {shape}"
            obs_dim += shape[0]

        # Construct the ConditionalUnet1D model
        input_dim = action_dim
        global_cond_dim = obs_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            target_dim=target_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.DDPM_noise_scheduler = noise_scheduler

        # Create DDIM scheduler with same config as DDPM
        # Filter out parameters that are not valid for DDIMScheduler
        ddim_config = dict(noise_scheduler.config)
        ddim_config.pop("variance_type", None)  # variance_type is only for DDPM
        self.DDIM_noise_scheduler = DDIMScheduler(**ddim_config)

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.use_target_cond = use_target_cond
        self.target_dim = target_dim
        self.kwargs = kwargs

        self.num_DDPM_inference_steps = num_DDPM_inference_steps
        self.num_DDIM_inference_steps = num_DDIM_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Global condition dim: %e" % self.model.global_cond_dim)
        print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")

    # ========= inference  ============
    def conditional_sample(
        self,
        inpaint_data,
        inpaint_mask,
        local_cond=None,
        global_cond=None,
        target_cond=None,
        generator=None,
        sample_for_vis=False,
        use_DDIM=False,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        if use_DDIM:
            scheduler = self.DDIM_noise_scheduler
            scheduler.set_timesteps(self.num_DDIM_inference_steps)
        else:
            scheduler = self.DDPM_noise_scheduler
            scheduler.set_timesteps(self.num_DDPM_inference_steps)

        trajectory = torch.randn(
            size=inpaint_data.shape, dtype=inpaint_data.dtype, device=inpaint_data.device, generator=generator
        )

        if sample_for_vis:
            trajectories = []

        for t in scheduler.timesteps:
            # 1. Inpaint: overwrite known positions with correctly-noised inpaint values so
            #    the UNet sees a consistent noise level across the whole trajectory.
            if inpaint_mask.any():
                t_batch = torch.full((inpaint_data.shape[0],), t, device=inpaint_data.device, dtype=torch.long)
                noised_inpaint = scheduler.add_noise(inpaint_data, torch.randn_like(inpaint_data), t_batch)
                trajectory[inpaint_mask] = noised_inpaint[inpaint_mask]

            # 2. predict model output
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond, target_cond=target_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

            if sample_for_vis:
                if t % 5 == 0 or t < 10:
                    trajectories.append(trajectory.clone())

        # Final override: snap known positions to exact clean values after the last denoising step
        trajectory[inpaint_mask] = inpaint_data[inpaint_mask]

        if sample_for_vis:
            return trajectory, trajectories

        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], sample_for_vis: bool = False, use_DDIM: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        obs_dict: must include "target" key if use_target_cond is True
        use_DDIM: if True, use DDIM scheduler; otherwise use DDPM scheduler
        result: must include "action" key
        """

        assert "obs" in obs_dict
        assert "past_action" not in obs_dict  # not implemented yet
        if self.use_target_cond:
            assert "target" in obs_dict

        mixed_precision = getattr(self, "mixed_precision", None) or "no"
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
            enabled=mixed_precision != "no",
        ):
            # Normalize obs dictionary (flat normalizer structure)
            nobs_dict = self.normalizer.normalize(obs_dict["obs"])
            # Concatenate obs dictionary into a single tensor
            nobs_list = [nobs_dict[key] for key in sorted(nobs_dict.keys())]
            nobs = torch.cat(nobs_list, dim=-1)

            ntarget = None
            if self.use_target_cond:
                ntarget = self.normalizer["target"].normalize(obs_dict["target"])
            B, _, Do = nobs.shape
            To = self.n_obs_steps
            assert Do == self.obs_dim
            T = self.horizon
            Da = self.action_dim

            # build input
            device = self.device
            dtype = self.dtype

            # condition through global feature
            global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            inpaint_data = torch.zeros(size=shape, device=device, dtype=dtype)
            inpaint_mask = torch.zeros_like(inpaint_data, dtype=torch.bool)
            local_cond = None

            # handle target conditioning
            target_cond = None
            if self.use_target_cond:
                target_cond = ntarget.reshape(ntarget.shape[0], -1)  # B, D_t

            # run sampling
            nsample = self.conditional_sample(
                inpaint_data,  # always inactive masks unless impainting
                inpaint_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                target_cond=target_cond,
                sample_for_vis=sample_for_vis,
                use_DDIM=use_DDIM,
                **self.kwargs,
            )

            if sample_for_vis:
                nsample, trajectories = nsample
            # unnormalize prediction
            naction_pred = nsample[..., :Da]
            action_pred = self.normalizer["action"].unnormalize(naction_pred)

            unnormalized_trajectories = []
            if sample_for_vis:
                for traj in trajectories:
                    ntraj = traj[..., :Da]
                    traj_action_pred = self.normalizer["action"].unnormalize(ntraj)
                    unnormalized_trajectories.append(traj_action_pred)

            # get action
            start = To - 1  # re-predict actions in observation horizon
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

            result = {"action": action, "action_pred": action_pred}

            if sample_for_vis:
                return result, unnormalized_trajectories

            return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # Normalize input
        assert "valid_mask" not in batch

        # Normalize obs dictionary and action
        nobs_dict = self.normalizer.normalize(batch["obs"])
        naction = self.normalizer["action"].normalize(batch["action"])

        # Normalize target if needed
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer["target"].normalize(batch["target"])

        # Concatenate obs dictionary into a single tensor
        obs_list = [nobs_dict[key] for key in sorted(nobs_dict.keys())]
        obs = torch.cat(obs_list, dim=-1)
        action = naction

        # condition through global feature
        local_cond = None
        slice_end = self.n_obs_steps
        global_cond = obs[:, :slice_end, :].reshape(obs.shape[0], -1)
        trajectory = action

        # handle target conditioning
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(ntarget.shape[0], -1)  # B, D_t

        # All actions are to be predicted; no inpainting needed
        inpaint_mask = torch.zeros(trajectory.shape, dtype=torch.bool, device=trajectory.device)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~inpaint_mask

        # apply conditioning
        noisy_trajectory[inpaint_mask] = trajectory[inpaint_mask]

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond, target_cond=target_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
