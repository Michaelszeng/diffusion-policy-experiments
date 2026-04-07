"""Encoder registry builder for TimmObsEncoderV2."""

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch.nn as nn

from diffusion_policy.model.vision.timm_obs_v2.backbone_utils import (
    create_and_process_encoder,
    get_normalization_for_model,
    modify_first_conv_to_single_channel,
)
from diffusion_policy.model.vision.timm_obs_v2.config import ObsEncoderInitConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RGBGroupResolvedConfig:
    model_name: str
    pretrained: bool
    frozen: bool
    use_group_norm: bool
    imagenet_norm: bool
    use_lora: bool
    lora_r: int
    lora_alpha: float
    lora_dropout: float
    lora_target_modules: Optional[list]
    share_within_group: bool
    camera_group: str


@dataclass(frozen=True)
class _RGBBuiltEntry:
    model: nn.Module
    feature_dim: int
    normalize: nn.Module


@dataclass(frozen=True)
class EncoderRegistryBuildOutput:
    default_rgb_model: nn.Module
    default_depth_model: nn.Module
    default_feature_dim: int
    default_depth_feature_dim: int
    default_normalize_transform: nn.Module
    rgb_keys: List[str]
    depth_keys: List[str]
    low_dim_keys: List[str]
    force_keys: List[str]
    key_shape_map: Dict[str, Tuple[int, ...]]
    key_feature_dim_map: Dict[str, int]
    key_normalize_map: Dict[str, nn.Module]
    key_model_map: nn.ModuleDict


def _validate_short_range_horizon(key: str, attr: dict, short_window: int) -> None:
    this_horizon = attr["horizon"]
    if this_horizon <= short_window:
        raise ValueError(
            f"short_range_obs_window ({short_window}) must be < horizon ({this_horizon}) for {key}"
        )


def _resolve_rgb_group_config(
    key: str,
    attr: dict,
    cfg: ObsEncoderInitConfig,
) -> Optional[_RGBGroupResolvedConfig]:
    camera_group = attr.get("camera_group")
    if not camera_group:
        return None

    encoder_by_group = cfg.rgb.encoder_by_group
    if encoder_by_group and camera_group in encoder_by_group:
        group_config = encoder_by_group[camera_group]
        logger.info("Camera %s using group '%s' encoder config", key, camera_group)
        return _RGBGroupResolvedConfig(
            model_name=group_config.get("model_name", cfg.rgb.model_name),
            pretrained=group_config.get("pretrained", cfg.rgb.pretrained),
            frozen=group_config.get("frozen", cfg.rgb.frozen),
            use_group_norm=group_config.get("use_group_norm", cfg.rgb.use_group_norm),
            imagenet_norm=group_config.get("imagenet_norm", cfg.rgb.imagenet_norm),
            use_lora=group_config.get("use_lora", cfg.lora.enabled),
            lora_r=group_config.get("lora_r", cfg.lora.r),
            lora_alpha=group_config.get("lora_alpha", cfg.lora.alpha),
            lora_dropout=group_config.get("lora_dropout", cfg.lora.dropout),
            lora_target_modules=group_config.get("lora_target_modules", cfg.lora.target_modules),
            share_within_group=group_config.get("share_within_group", False),
            camera_group=camera_group,
        )

    if encoder_by_group:
        logger.warning(
            "Camera %s has group '%s' but no encoder config found. Using default.",
            key,
            camera_group,
        )
    return None


def _create_rgb_encoder(cfg: ObsEncoderInitConfig, resolved: _RGBGroupResolvedConfig) -> Tuple[nn.Module, int]:
    return create_and_process_encoder(
        model_name=resolved.model_name,
        pretrained=resolved.pretrained,
        frozen=resolved.frozen,
        downsample_ratio=cfg.vision_features.downsample_ratio,
        use_group_norm=resolved.use_group_norm,
        use_lora=resolved.use_lora,
        lora_r=resolved.lora_r,
        lora_alpha=resolved.lora_alpha,
        lora_dropout=resolved.lora_dropout,
        lora_target_modules=resolved.lora_target_modules,
    )


def _get_or_create_group_single(
    cache: dict,
    cache_key: str,
    share: bool,
    create_fn: Callable[[], Tuple[nn.Module, int]],
) -> Tuple[nn.Module, int]:
    if share and cache_key in cache:
        return cache[cache_key]

    model, feature_dim = create_fn()
    if share:
        cache[cache_key] = (model, feature_dim)
    return model, feature_dim


def _get_or_create_group_pair(
    cache: dict,
    long_key: str,
    short_key: str,
    share: bool,
    create_fn: Callable[[], Tuple[nn.Module, int]],
) -> Tuple[nn.Module, nn.Module, int]:
    if share and long_key in cache:
        model_long, feature_dim = cache[long_key]
        model_short, _ = cache[short_key]
        return model_long, model_short, feature_dim

    model_long, feature_dim = create_fn()
    model_short, _ = create_fn()
    if share:
        cache[long_key] = (model_long, feature_dim)
        cache[short_key] = (model_short, feature_dim)
    return model_long, model_short, feature_dim


def _build_default_rgb_model(
    short_range: bool,
    share_model: bool,
    default_rgb_model: nn.Module,
    shared_long_range_model: Optional[nn.Module],
) -> nn.Module:
    if not short_range:
        return default_rgb_model if share_model else copy.deepcopy(default_rgb_model)

    model_short = default_rgb_model if share_model else copy.deepcopy(default_rgb_model)
    model_long = (
        shared_long_range_model
        if (share_model and shared_long_range_model is not None)
        else copy.deepcopy(default_rgb_model)
    )
    return nn.ModuleList([model_long, model_short])


def _build_default_depth_model(
    short_range: bool,
    share_model: bool,
    default_depth_model: nn.Module,
    shared_long_range_depth_model: Optional[nn.Module],
) -> nn.Module:
    if not short_range:
        return default_depth_model if share_model else copy.deepcopy(default_depth_model)

    model_short = default_depth_model if share_model else copy.deepcopy(default_depth_model)
    model_long = (
        shared_long_range_depth_model
        if (share_model and shared_long_range_depth_model is not None)
        else copy.deepcopy(default_depth_model)
    )
    return nn.ModuleList([model_long, model_short])


def _build_rgb_model(
    *,
    cfg: ObsEncoderInitConfig,
    group_config: Optional[_RGBGroupResolvedConfig],
    use_short_range: bool,
    group_model_cache: dict,
    default_rgb_model: nn.Module,
    default_feature_dim: int,
    default_normalize_transform: nn.Module,
    shared_long_range_model: Optional[nn.Module],
) -> _RGBBuiltEntry:
    if group_config is None:
        return _RGBBuiltEntry(
            model=_build_default_rgb_model(
                short_range=use_short_range,
                share_model=cfg.rgb.share_model,
                default_rgb_model=default_rgb_model,
                shared_long_range_model=shared_long_range_model,
            ),
            feature_dim=default_feature_dim,
            normalize=default_normalize_transform,
        )

    create_fn = lambda: _create_rgb_encoder(cfg, group_config)
    if use_short_range:
        long_key = f"{group_config.camera_group}_long"
        short_key = f"{group_config.camera_group}_short"
        model_long, model_short, feature_dim = _get_or_create_group_pair(
            cache=group_model_cache,
            long_key=long_key,
            short_key=short_key,
            share=group_config.share_within_group,
            create_fn=create_fn,
        )
        model = nn.ModuleList([model_long, model_short])
        normalize_model = model_long
    else:
        model, feature_dim = _get_or_create_group_single(
            cache=group_model_cache,
            cache_key=group_config.camera_group,
            share=group_config.share_within_group,
            create_fn=create_fn,
        )
        normalize_model = model

    normalize = get_normalization_for_model(
        group_config.model_name,
        group_config.pretrained,
        group_config.imagenet_norm,
        model=normalize_model,
    )
    return _RGBBuiltEntry(model=model, feature_dim=feature_dim, normalize=normalize)


def build_encoder_registry(shape_meta: dict, cfg: ObsEncoderInitConfig) -> EncoderRegistryBuildOutput:
    """Build all modality backbones and per-key registry maps."""

    rgb_keys: List[str] = []
    depth_keys: List[str] = []
    low_dim_keys: List[str] = []
    force_keys: List[str] = []
    key_shape_map: Dict[str, Tuple[int, ...]] = {}
    key_feature_dim_map: Dict[str, int] = {}
    key_normalize_map: Dict[str, nn.Module] = {}
    key_model_map = nn.ModuleDict()
    group_model_cache = {}

    default_rgb_model, default_feature_dim = create_and_process_encoder(
        model_name=cfg.rgb.model_name,
        pretrained=cfg.rgb.pretrained,
        frozen=cfg.rgb.frozen,
        downsample_ratio=cfg.vision_features.downsample_ratio,
        use_group_norm=cfg.rgb.use_group_norm,
        use_lora=cfg.lora.enabled,
        lora_r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        lora_target_modules=cfg.lora.target_modules,
    )
    default_normalize_transform = get_normalization_for_model(
        cfg.rgb.model_name,
        cfg.rgb.pretrained,
        cfg.rgb.imagenet_norm,
        model=default_rgb_model,
    )

    default_depth_model, default_depth_feature_dim = create_and_process_encoder(
        model_name=cfg.depth.model_name,
        pretrained=cfg.depth.pretrained,
        frozen=cfg.depth.frozen,
        downsample_ratio=cfg.vision_features.downsample_ratio,
        use_group_norm=cfg.depth.use_group_norm,
    )
    depth_modified = modify_first_conv_to_single_channel(default_depth_model)
    if not depth_modified:
        logger.warning("Could not modify depth model first layer to single-channel input.")

    shared_long_range_model = (
        copy.deepcopy(default_rgb_model) if (cfg.rgb.share_model and cfg.short_range.obs_window is not None) else None
    )
    shared_long_range_depth_model = (
        copy.deepcopy(default_depth_model)
        if (cfg.depth.share_model and cfg.short_range.obs_window is not None)
        else None
    )
    use_short_range = cfg.short_range.obs_window is not None

    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = tuple(attr["shape"])
        obs_type = attr.get("type", "low_dim")
        key_shape_map[key] = shape

        if obs_type == "rgb":
            if attr.get("ignore_by_policy", False):
                continue
            rgb_keys.append(key)
            if use_short_range:
                _validate_short_range_horizon(key, attr, cfg.short_range.obs_window)

            group_config = _resolve_rgb_group_config(key, attr, cfg)
            rgb_entry = _build_rgb_model(
                cfg=cfg,
                group_config=group_config,
                use_short_range=use_short_range,
                group_model_cache=group_model_cache,
                default_rgb_model=default_rgb_model,
                default_feature_dim=default_feature_dim,
                default_normalize_transform=default_normalize_transform,
                shared_long_range_model=shared_long_range_model,
            )
            key_feature_dim_map[key] = rgb_entry.feature_dim
            key_normalize_map[key] = rgb_entry.normalize
            key_model_map[key] = rgb_entry.model

        elif obs_type == "depth":
            if attr.get("ignore_by_policy", False):
                continue
            depth_keys.append(key)
            key_feature_dim_map[key] = default_depth_feature_dim
            if use_short_range:
                _validate_short_range_horizon(key, attr, cfg.short_range.obs_window)

            key_model_map[key] = _build_default_depth_model(
                short_range=use_short_range,
                share_model=cfg.depth.share_model,
                default_depth_model=default_depth_model,
                shared_long_range_depth_model=shared_long_range_depth_model,
            )

        elif obs_type == "low_dim":
            if not attr.get("ignore_by_policy", False):
                low_dim_keys.append(key)
        elif obs_type == "force":
            if not attr.get("ignore_by_policy", False):
                force_keys.append(key)
        else:
            raise RuntimeError(f"Unsupported obs type: {obs_type}")

    return EncoderRegistryBuildOutput(
        default_rgb_model=default_rgb_model,
        default_depth_model=default_depth_model,
        default_feature_dim=default_feature_dim,
        default_depth_feature_dim=default_depth_feature_dim,
        default_normalize_transform=default_normalize_transform,
        rgb_keys=sorted(rgb_keys),
        depth_keys=sorted(depth_keys),
        low_dim_keys=sorted(low_dim_keys),
        force_keys=sorted(force_keys),
        key_shape_map=key_shape_map,
        key_feature_dim_map=key_feature_dim_map,
        key_normalize_map=key_normalize_map,
        key_model_map=key_model_map,
    )
