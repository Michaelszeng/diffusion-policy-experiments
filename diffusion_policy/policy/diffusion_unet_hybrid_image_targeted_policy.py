from typing import Dict

import robomimic.models.obs_core as rmobsc
import robomimic.utils.obs_utils as ObsUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo

import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.mlp.mlp import MLP
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DiffusionUnetHybridImageTargetedPolicy(BaseImagePolicy):
    """
    Diffusion policy model architecture that uses a UNet encoder, and conditions on
    images and target states/goals
    """

    def __init__(
        self,
        shape_meta: dict,
        DDPM_noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        one_hot_encoding_dim=0,
        use_target_cond=False,
        target_dim=None,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_embedding_dim=None,
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
        num_DDPM_inference_steps=100,
        num_DDIM_inference_steps=10,
        pretrained_obs_encoder=False,
        freeze_pretrained_obs_encoder=False,
        self_trained_obs_encoder=None,
        freeze_self_trained_obs_encoder=False,
        inference_loading=False,
        past_action_visible=False,
        # parameters passed to step
        **kwargs,
    ):
        """
        Args:
            one_hot_encoding_dim: number of datasets (>=1),

        """
        super().__init__()

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

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name="bc_rnn",
            hdf5_type="image",
            task_name="square",
            dataset_type="ph",
            pretrained_obs_encoder=pretrained_obs_encoder,
            freeze_pretrained_obs_encoder=freeze_pretrained_obs_encoder,
        )

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # We use the image encoder from robomimic
        # This isn't clean, but we create a robomimic policy object and then
        # just extract the image encoder.
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            # This is a hack to get around the fact that the image encoder uses
            # batch norm. We replace it with group norm.
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        if eval_fixed_crop:
            # This is a hack to get around the fact that the image encoder uses
            # crop randomizer. We replace it with our own crop randomizer.
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmobsc.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )

        if obs_embedding_dim is not None:
            obs_feature_dim = obs_embedding_dim
            self.obs_embedding_projector = MLP(obs_encoder.output_shape()[0], [], obs_feature_dim)
            self.obs_embedding_projector.to("cuda" if torch.cuda.is_available() else "cpu")
            project_obs_embedding = True
        else:
            obs_feature_dim = obs_encoder.output_shape()[0]
            project_obs_embedding = False

        # create diffusion model
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps + one_hot_encoding_dim
        print(f"Input dim: {input_dim}, Global cond dim: {global_cond_dim}")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            target_dim=target_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        if not inference_loading:
            if self_trained_obs_encoder is not None:
                print(f"Loading obs encoder from {self_trained_obs_encoder}")
                checkpoint = torch.load(self_trained_obs_encoder, map_location="cpu")  # Load the full checkpoint

                # Extract the model state dict from the checkpoint
                full_model_state_dict = checkpoint["state_dicts"]["model"]

                # Extract only obs_encoder weights
                # Keys are "module.obs_encoder.*"
                obs_encoder_state_dict = {}
                prefix = "module.obs_encoder."
                for key, value in full_model_state_dict.items():
                    if key.startswith(prefix):
                        # Remove the "module.obs_encoder." prefix
                        new_key = key[len(prefix) :]
                        obs_encoder_state_dict[new_key] = value

                if len(obs_encoder_state_dict) == 0:
                    raise ValueError(
                        f"No obs_encoder weights found in checkpoint. Keys present: {list(full_model_state_dict.keys())}"
                    )

                print(f"Loaded {len(obs_encoder_state_dict)} obs_encoder parameters")
                obs_encoder.load_state_dict(obs_encoder_state_dict, strict=True)

                if freeze_self_trained_obs_encoder:
                    for param in obs_encoder.parameters():
                        param.requires_grad = False

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

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,  # Using Global Observation Conditioning so obs_dim per timestep is 0
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=past_action_visible,
        )
        self.obs_encoder = obs_encoder
        self.model = model
        self.DDPM_noise_scheduler = DDPM_noise_scheduler
        self.DDIM_noise_scheduler = DDIM_noise_scheduler
        self.num_DDPM_inference_steps = num_DDPM_inference_steps
        self.num_DDIM_inference_steps = num_DDIM_inference_steps
        self.normalizer = LinearNormalizer()  # Empty normalizer; parameters will be set later
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.project_obs_embedding = project_obs_embedding
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.one_hot_encoding_dim = one_hot_encoding_dim
        self.use_target_cond = use_target_cond
        self.kwargs = kwargs

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        if project_obs_embedding:
            print("Vision projector params: %e" % sum(p.numel() for p in self.obs_embedding_projector.parameters()))

    ####################################################################################################################
    ### Inference
    ####################################################################################################################
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        global_cond=None,
        target_cond=None,
        generator=None,
        use_DDIM=False,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        """
        DDPM sampling.

        Args:
            condition_data (torch.Tensor): Known values to condition on (e.g., observed actions)
                                        Shape: [B, T, Da] where some timesteps may be known
            condition_mask (torch.Tensor): Boolean mask indicating which values in condition_data
                                        are known/should be enforced. Shape: [B, T, Da]
            global_cond (torch.Tensor): Observation features from vision encoder. Shape: [B, Do]
            target_cond (torch.Tensor): Goal/target conditioning. Shape: [B, target_dim]
            generator (torch.Generator): RNG
            use_DDIM (bool): DDIM vs DDPM
            **kwargs: Additional arguments passed to scheduler.step()

        Returns:
            torch.Tensor: Generated action trajectory. Shape: [B, T, Da]
                        Clean action sequence ready for robot execution

        Note:
            - This implements classifier-free guidance through conditioning
            - condition_mask typically masks out future actions, leaving model to predict them
            - global_cond encodes "what I see" (current observations)
            - target_cond encodes "what I want" (goal state)
            - The result is "what I should do" (action sequence)
        """
        model = self.model
        if use_DDIM:
            scheduler = self.DDIM_noise_scheduler
            scheduler.set_timesteps(self.num_DDIM_inference_steps)
        else:
            scheduler = self.DDPM_noise_scheduler
            scheduler.set_timesteps(self.num_DDPM_inference_steps)

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )  # Start with random noise trajectory

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(
                trajectory,
                t,
                local_cond=None,
                global_cond=global_cond,
                target_cond=target_cond,
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], use_DDIM=False) -> Dict[str, torch.Tensor]:
        """
        Args:
            obs_dict: Dict containing observation data with the following structure:
                {
                    "obs": {
                        # Low-dimensional observations (robot state)
                        "agent_pos": torch.Tensor,     # Shape: [B, To, 13] - robot state (6 DOF position + 6 DOF velocity + 1 DOF gripper)

                        # RGB image observations
                        "overhead_camera": torch.Tensor,  # Shape: [B, To, 3, 128, 128] - overhead RGB images
                        "wrist_camera": torch.Tensor,     # Shape: [B, To, 3, 128, 128] - wrist RGB images
                    },

                    # only required if self.use_target_cond=True
                    "target": torch.Tensor,            # Shape: [B, target_dim] - target
                }

            use_DDIM: Whether to use DDIM sampling instead of DDPM

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing predicted actions:
                {
                    "action": torch.Tensor  # Shape: [B, Ta, Da] - predicted action sequence
                }

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

        # Normalize obs dict
        nobs = self.normalizer.normalize(obs_dict["obs"])
        ntarget = None
        if self.use_target_cond:
            # Normalize target tensor
            ntarget = self.normalizer["target"].normalize(obs_dict["target"])
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        if self.project_obs_embedding:
            nobs_features = self.obs_embedding_projector(nobs_features)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            one_hot_encoding = obs_dict["one_hot_encoding"]
            global_cond = torch.cat([global_cond, one_hot_encoding], dim=-1)

        # handle target conditioning
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(B, -1)  # B, D_t

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            global_cond=global_cond,
            target_cond=target_cond,
            use_DDIM=use_DDIM,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
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

    def forward(self, batch, noisy_trajectory, timesteps, condition_mask=None):
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
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # encode observations into global conditioning vector
        global_cond = None
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        if self.project_obs_embedding:
            nobs_features = self.obs_embedding_projector(nobs_features)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)  # B, Do

        # append one hot encoding to global conditioning vector
        if self.one_hot_encoding_dim > 0:
            one_hot_encoding = batch["one_hot_encoding"]
            global_cond = torch.cat([global_cond, one_hot_encoding], dim=-1)

        # build target conditioning vector
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(batch_size, -1)  # B, D_t

        if condition_mask is not None:
            noisy_trajectory[condition_mask] = nactions[condition_mask]

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
