"""Modular components for TimmObsEncoderV2."""

from diffusion_policy.model.vision.timm_obs_v2.types import (
    EncoderCoreOutputs,
    ModalityType,
    RangeType,
    TokenBundle,
    VisualFeatureSet,
    VisualModality,
    VisualStreamFeatures,
)
from diffusion_policy.model.vision.timm_obs_v2.backbone_utils import (
    create_and_process_encoder,
    get_feature_dim_for_model,
    get_normalization_for_model,
    modify_first_conv_to_single_channel,
)
from diffusion_policy.model.vision.timm_obs_v2.depth_module import DepthModule
from diffusion_policy.model.vision.timm_obs_v2.force_module import ForceModule
from diffusion_policy.model.vision.timm_obs_v2.lowdim_module import LowDimModule
from diffusion_policy.model.vision.timm_obs_v2.metadata_utils import get_key_metadata
from diffusion_policy.model.vision.timm_obs_v2.rgb_module import RGBModule
from diffusion_policy.model.vision.timm_obs_v2.encoder_registry import (
    EncoderRegistryBuildOutput,
    build_encoder_registry,
)
from diffusion_policy.model.vision.timm_obs_v2.standard_visual_aggregator import (
    AttentionPool1d,
    AttentionPool2d,
)
from diffusion_policy.model.vision.timm_obs_v2.visual_fusion import (
    VisualFusionModule,
)

__all__ = [
    "EncoderCoreOutputs",
    "ModalityType",
    "RangeType",
    "TokenBundle",
    "VisualFeatureSet",
    "VisualModality",
    "VisualStreamFeatures",
    "create_and_process_encoder",
    "get_feature_dim_for_model",
    "get_normalization_for_model",
    "modify_first_conv_to_single_channel",
    "DepthModule",
    "ForceModule",
    "LowDimModule",
    "get_key_metadata",
    "RGBModule",
    "EncoderRegistryBuildOutput",
    "build_encoder_registry",
    "AttentionPool1d",
    "AttentionPool2d",
    "VisualFusionModule",
]
