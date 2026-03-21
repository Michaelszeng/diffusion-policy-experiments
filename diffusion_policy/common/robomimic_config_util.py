import robomimic.scripts.generate_paper_configs as gpc
from robomimic.config import config_factory
from robomimic.scripts.generate_paper_configs import (
    modify_config_for_dataset,
    modify_config_for_default_image_exp,
    modify_config_for_default_low_dim_exp,
)
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.models.base_nets as rmobsc
import torch
import torch.nn as nn
from diffusion_policy.common.pytorch_util import replace_submodules
import diffusion_policy.model.vision.crop_randomizer as dmvc


def get_robomimic_config(
    algo_name="bc_rnn",
    hdf5_type="low_dim",
    task_name="square",
    dataset_type="ph",
    pretrained_encoder=False,
    freeze_pretrained_encoder=False,
):
    if freeze_pretrained_encoder:
        assert pretrained_encoder, "Should not freeze encoder if encoder is not pretrained"

    base_dataset_dir = "/tmp/null"
    filter_key = None

    # decide whether to use low-dim or image training defaults
    modifier_for_obs = modify_config_for_default_image_exp
    if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
        modifier_for_obs = modify_config_for_default_low_dim_exp

    algo_config_name = "bc" if algo_name == "bc_rnn" else algo_name
    config = config_factory(algo_name=algo_config_name)

    # turn into default config for observation modalities (e.g.: low-dim or rgb)
    config = modifier_for_obs(config)

    # add in config based on the dataset
    config = modify_config_for_dataset(
        config=config,
        task_name=task_name,
        dataset_type=dataset_type,
        hdf5_type=hdf5_type,
        base_dataset_dir=base_dataset_dir,
        filter_key=filter_key,
    )

    if pretrained_encoder:
        # config.observation.encoder.rgb.core_kwargs.backbone_class = 'R3MConv'                         # R3M backbone for image observations (unused if no image observations)
        # config.observation.encoder.rgb.core_kwargs.backbone_kwargs.r3m_model_class = 'resnet18'       # R3M model class (resnet18, resnet34, resnet50)

        config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = True
        config.observation.encoder.rgb.core_kwargs.backbone_kwargs.freeze = (
            freeze_pretrained_encoder  # whether to freeze network during training or allow finetuning
        )
        # config.observation.encoder.rgb.core_kwargs.pool_class = None

    # add in algo hypers based on dataset
    algo_config_modifier = getattr(gpc, f"modify_{algo_name}_config_for_dataset")
    config = algo_config_modifier(
        config=config,
        task_name=task_name,
        dataset_type=dataset_type,
        hdf5_type=hdf5_type,
    )
    return config

def get_robomimic_obs_encoder(
    obs_config,
    obs_key_shapes,
    action_dim,
    pretrained_encoder=False,
    freeze_encoder=False,
    crop_shape=None,
):
    # Get raw Robomimic config for image encoder
    config = get_robomimic_config(
        algo_name="bc_rnn",
        hdf5_type="image",
        task_name="square",
        dataset_type="ph",
        pretrained_encoder=pretrained_encoder,
        freeze_pretrained_encoder=(freeze_encoder and pretrained_encoder),
    )

    # Override the Robomimic observation keys with our own
    with config.unlocked():
        config.observation.modalities.obs = obs_config

        # Set random crop parameters of Robomimic CropRandomizer
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

    # Create a robomimic policy object and then extract the image encoder
    policy: PolicyAlgo = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=action_dim,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

    # We replace Robomimic's BatchNorm with GroupNorm.
    replace_submodules(
        root_module=obs_encoder,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
    )

    # We replace Robomimic's CropRandomizer with our own CropRandomizer with the same parameters.
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
    return obs_encoder
