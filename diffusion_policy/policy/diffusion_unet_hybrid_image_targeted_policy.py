from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.robomimic_config_util import get_robomimic_obs_encoder
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DiffusionUnetHybridImageTargetedPolicy(BaseImagePolicy):
    """
    Diffusion policy model architecture that uses a UNet encoder, hybrid (Robomimic-pretrained Resnet) encoder,
    and conditions on images and target states/goals
    """

    def __init__(
        self,
        shape_meta: dict,
        DDPM_noise_scheduler: DDPMScheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        one_hot_encoding_dim: int = 0,  # Dimension of 1-hot encoding (used for i.e. task specification)
        use_target_cond: bool = False,
        target_dim: int = None,
        crop_shape: tuple = (76, 76),  # NOTE: crop size is handled here, in the policy class, since there is no explicit observation encoder class.
        diffusion_step_embed_dim: int = 256,  # Size of the diffusion timestep embedding
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,  # Resnet param
        n_groups: int = 8,  # Resnet param
        num_DDPM_inference_steps: int = 100,
        num_DDIM_inference_steps: int = 10,
        pretrained_encoder: bool = False,  # Use Robomimic-pretrained encoder
        self_trained_obs_encoder: str = None,  # Path to a checkpoint file containing the weights for the self-trained obs encoder
        freeze_encoder: bool = False,  # Freeze the encoder parameters
        inference_loading: bool = False,  # Flag to set during inference to skip loading the self-trained obs encoder weights (use the actual checkpoint weights instead)
        past_action_visible: bool = False,
        # DEPRECATED PARAMETERS FOR BACKWARD COMPATIBILITY
        cond_predict_scale = None,
        obs_encoder_group_norm = None,
        eval_fixed_crop = None,
        freeze_pretrained_encoder = None,
        freeze_self_trained_obs_encoder = None,
        # parameters passed to step
        **kwargs,
    ):
        """
        Additional Features implemented by this policy class:
         - Targeted conditioning on a target state/goal
         - One-hot encoding of task specifications
         - Self-trained observation encoder -- load observation encoder weights from a specified checkpoint
        """
        super().__init__()

        # BACKWARD COMPATIBILITY
        if freeze_self_trained_obs_encoder is not None:
            freeze_encoder = freeze_self_trained_obs_encoder

        if use_target_cond:
            assert target_dim is not None
        assert one_hot_encoding_dim >= 0

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        # each list contains the keys of the corresponding modality
        # ex. {low_dim: [agent_pos], rgb: [image], depth: [], scan: []}
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = dict()
        # ex. {agent_pos: shape, image: shape}
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            typee = attr.get("type", "low_dim")
            if typee == "rgb":
                obs_config["rgb"].append(key)
            elif typee == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # Get observation encoder from Robomimic
        obs_encoder = get_robomimic_obs_encoder(
            obs_config=obs_config,
            obs_key_shapes=obs_key_shapes,
            action_dim=action_dim,
            pretrained_encoder=pretrained_encoder,
            freeze_encoder=freeze_encoder,
            crop_shape=crop_shape,
        )

        obs_feature_dim = obs_encoder.output_shape()[0]
        global_cond_dim = obs_feature_dim * n_obs_steps + one_hot_encoding_dim
        input_dim = action_dim
        print(f"Input dim: {input_dim}, Global cond dim: {global_cond_dim}")
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            target_dim=target_dim if use_target_cond else None,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

        # Only load obs encoder weights from self-trained checkpoint during training. During inference, we want to use
        # the weights from the final checkpoint (rather than override w/ the self-trained weights).
        if not inference_loading:
            self._maybe_load_and_freeze_obs_encoder(obs_encoder, self_trained_obs_encoder, freeze_encoder)

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
        self.model = model
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
        self.one_hot_encoding_dim = one_hot_encoding_dim
        self.use_target_cond = use_target_cond
        # Filter kwargs to only include valid scheduler.step() parameters
        # Remove policy-specific parameters that were already consumed
        scheduler_params = {"eta", "use_clipped_model_output", "variance_noise"}
        self.kwargs = {k: v for k, v in kwargs.items() if k in scheduler_params}

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Observation Encoder params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    def _maybe_load_and_freeze_obs_encoder(self, obs_encoder, checkpoint_path, freeze):
        """
        Optionally initializes the obs encoder from a prior training run and/or freezes it.

        The checkpoint is expected to be a full model checkpoint saved by the training workspace,
        where obs encoder weights are stored under the "module.obs_encoder.*" prefix.
        """
        if checkpoint_path is None:
            return

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

    def get_inpaint_mask(self, shape):
        """
        Returns a boolean inpainting mask of shape (B, T, action_dim).

        True = "known/given" (will be inpainted and excluded from loss).
        False = "to predict" (model must denoise these).

        When past_action_visible=False (the default), all actions are predicted from
        scratch and the mask is all-False. When past_action_visible=True, the first
        (n_obs_steps - 1) action timesteps are treated as known, since they overlap with
        the observation horizon.
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
        target_cond=None,
        generator=None,
        use_DDIM=False,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        """
        Run reverse diffusion sampling.

        Args:
            inpaint_data (torch.Tensor): Known values for naive inpainting. Shape: [B, T, Da]
            inpaint_mask (torch.Tensor): Boolean mask indicating which values to inpaint. Shape: [B, T, Da]
            global_cond (torch.Tensor): Observation features from vision encoder. Shape: [B, Do]
            target_cond (torch.Tensor): Goal/target conditioning. Shape: [B, target_dim]
            generator (torch.Generator): RNG
            use_DDIM (bool): DDIM vs DDPM
            **kwargs: Additional arguments passed to scheduler.step()

        Returns:
            torch.Tensor: Denoised action trajectory. Shape: [B, T, Da]
        """
        model = self.model
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
            model_output = model(trajectory, t.to(trajectory.device), global_cond=global_cond, target_cond=target_cond)
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        # Final override: snap known positions to exact clean values after the last denoising step
        trajectory[inpaint_mask] = inpaint_data[inpaint_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], use_DDIM=False) -> Dict[str, torch.Tensor]:
        """
        Args:
            obs_dict: Dict containing observation data with the following structure:
            use_DDIM: Whether to use DDIM sampling instead of DDPM

        Returns:
            Dictionary containing predicted actions: {"action": torch.Tensor}  # Shape: [B, Ta, Da]

        Where:
            B = batch size
            To = n_obs_steps (number of observation timesteps, e.g., 2)
            Ta = n_action_steps (number of predicted action timesteps, e.g., 8)
            Da = action_dim (action space dimensionality, e.g., 13)
            target_dim = dimensionality of goal/target (e.g., 3 for 3D position)

        Note:
            - The specific keys and shapes in obs_dict["obs"] are determined by the shape_meta configuration
        """
        assert "obs" in obs_dict
        if self.use_target_cond:
            assert "target" in obs_dict
        assert "past_action" not in obs_dict  # not implemented yet

        mixed_precision = getattr(self, "mixed_precision", None) or "no"
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
            enabled=mixed_precision != "no",
        ):
            # Normalize obs dict
            nobs = self.normalizer.normalize(obs_dict["obs"])
            ntarget = None
            if self.use_target_cond:
                # Normalize target tensor
                ntarget = self.normalizer["target"].normalize(obs_dict["target"])
            value = next(iter(nobs.values()))
            B, To = value.shape[:2]  # Batch size
            T = self.horizon  # T = prediction horizon
            Da = self.action_dim  # Da = action dim
            Do = self.obs_feature_dim  # Do = observation feature dim
            To = self.n_obs_steps  # To = obs horizon

            # Encode observations into global conditioning vector
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # Reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)

            # No inpainting
            inpaint_data = torch.zeros(size=(B, T, Da), device=self.device, dtype=self.dtype)
            inpaint_mask = torch.zeros_like(inpaint_data, dtype=torch.bool)

            # Append one hot encoding to global conditioning vector
            if self.one_hot_encoding_dim > 0:
                one_hot_encoding = obs_dict["one_hot_encoding"]
                global_cond = torch.cat([global_cond, one_hot_encoding], dim=-1)

            # Build target conditioning vector
            target_cond = None
            if self.use_target_cond:
                target_cond = ntarget.reshape(B, -1)  # B, D_t

            # Run reverse diffusion sampling
            nsample = self.conditional_sample(
                inpaint_data,
                inpaint_mask,
                global_cond=global_cond,
                target_cond=target_cond,
                use_DDIM=use_DDIM,
                **self.kwargs,
            )

            # Unnormalize prediction
            naction_pred = nsample[..., :Da]
            action_pred = self.normalizer["action"].unnormalize(naction_pred)

            # Get action
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

            # pred_action is the full predicted action sequence; action is a slice beginning after the latest observation
            # and with length n_action_steps
            result = {"action": action, "action_pred": action_pred}
            return result

    ####################################################################################################################
    ### Training
    ####################################################################################################################
    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set normalizer's parameters based on dataset."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def noise_trajectory(self, trajectory):
        """Add random amount of noise to the clean images"""
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
        Build input vectors (noisy trajectory, global conditioning vector, target conditioning vector) and run them
        through the model to get a prediction (of either the noise vector or the denoised trajectory depending on the
        prediction type config).
        """
        assert "valid_mask" not in batch

        # Collect normalized observations/actions/targets
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer["target"].normalize(batch["target"])

        # Sizing convenience
        B = nactions.shape[0]
        horizon = nactions.shape[1]

        # encode observations into global conditioning vector
        global_cond = None
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)  # B, Do

        # append one hot encoding to global conditioning vector
        if self.one_hot_encoding_dim > 0:
            one_hot_encoding = batch["one_hot_encoding"]
            global_cond = torch.cat([global_cond, one_hot_encoding], dim=-1)

        # build target conditioning vector
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(B, -1)  # B, D_t

        if inpaint_mask is not None:
            noisy_trajectory[inpaint_mask] = nactions[inpaint_mask]

        # Predict the noise residual
        return self.model(
            noisy_trajectory,
            timesteps,
            global_cond=global_cond,
            target_cond=target_cond,
        )

    def compute_loss(self, trajectory, noise, pred, loss_mask=None):
        """
        Optional loss mask used during validation so that we only compute loss on the timesteps corresponding to future
        actions.
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
