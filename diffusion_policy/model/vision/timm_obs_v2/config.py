from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RGBConfig:
    model_name: str
    pretrained: bool
    frozen: bool
    transforms: list
    token_mode: str
    use_group_norm: bool
    share_model: bool
    imagenet_norm: bool
    encoder_by_group: Optional[dict]


@dataclass(frozen=True)
class DepthConfig:
    model_name: str
    pretrained: bool
    frozen: bool
    share_model: bool
    use_group_norm: bool


@dataclass(frozen=True)
class VisionFeaturesConfig:
    feature_mode: str
    feature_aggregation: Optional[str]
    position_encoding: str
    downsample_ratio: int
    use_cross_image_attn: bool
    cross_image_attn_heads: int


@dataclass(frozen=True)
class ForceConfig:
    fusion_mode: Optional[str]
    cross_attn_heads: int
    cross_attn_kv_mode: str
    modality_dropout_p: float
    proj_dropout_p: float


@dataclass(frozen=True)
class LoraConfig:
    enabled: bool
    r: int
    alpha: float
    dropout: float
    target_modules: Optional[list]


@dataclass(frozen=True)
class ShortRangeConfig:
    obs_window: Optional[int]
    dropout: float


@dataclass(frozen=True)
class LowdimConfig:
    dropout_p: float
    project: bool


@dataclass(frozen=True)
class CompileConfig:
    backbone: bool
    mode: str


@dataclass(frozen=True)
class ProjectionConfig:
    target_feature_dim: Optional[int]


@dataclass(frozen=True)
class ObsEncoderInitConfig:
    rgb: RGBConfig
    depth: DepthConfig
    vision_features: VisionFeaturesConfig
    force: ForceConfig
    lora: LoraConfig
    short_range: ShortRangeConfig
    lowdim: LowdimConfig
    compile: CompileConfig
    projection: ProjectionConfig


def _section(
    section_value: Optional[dict],
    *,
    defaults: dict,
    required_keys: List[str],
    section_name: str,
) -> dict:
    if section_value is None:
        section = {}
    elif isinstance(section_value, dict) or hasattr(section_value, "items"):
        section = dict(section_value.items())
    else:
        raise TypeError(
            f"Section '{section_name}' must be a mapping or null, got {type(section_value).__name__}"
        )

    unknown = sorted(set(section.keys()) - set(defaults.keys()))
    if unknown:
        raise ValueError(
            f"Unknown keys in section '{section_name}': {unknown}. "
            f"Allowed keys: {sorted(defaults.keys())}"
        )

    merged = dict(defaults)
    merged.update(section)

    missing = [k for k in required_keys if merged.get(k) is None]
    if missing:
        raise ValueError(f"Missing required keys in section '{section_name}': {missing}")

    return merged


def parse_obs_encoder_init_config(
    rgb: dict,
    depth: Optional[dict] = None,
    vision_features: Optional[dict] = None,
    force: Optional[dict] = None,
    lora: Optional[dict] = None,
    short_range: Optional[dict] = None,
    lowdim: Optional[dict] = None,
    compile: Optional[dict] = None,
    projection: Optional[dict] = None,
) -> ObsEncoderInitConfig:
    rgb_cfg = _section(
        rgb,
        defaults={
            "model_name": None,
            "pretrained": None,
            "frozen": None,
            "transforms": None,
            "token_mode": "all_tokens",
            "use_group_norm": False,
            "share_model": False,
            "imagenet_norm": False,
            "encoder_by_group": None,
        },
        required_keys=["model_name", "pretrained", "frozen", "transforms"],
        section_name="rgb",
    )
    depth_cfg = _section(
        depth,
        defaults={
            "model_name": "resnet18",
            "pretrained": False,
            "frozen": False,
            "share_model": False,
            "use_group_norm": rgb_cfg["use_group_norm"],
        },
        required_keys=[],
        section_name="depth",
    )
    vision_cfg = _section(
        vision_features,
        defaults={
            "feature_mode": "raw_tokens",
            "feature_aggregation": None,
            "position_encoding": "learnable",
            "downsample_ratio": 32,
            "use_cross_image_attn": False,
            "cross_image_attn_heads": 8,
        },
        required_keys=[],
        section_name="vision_features",
    )
    force_cfg = _section(
        force,
        defaults={
            "fusion_mode": None,
            "cross_attn_heads": 4,
            "cross_attn_kv_mode": "mean",
            "modality_dropout_p": 0.0,
            "proj_dropout_p": 0.0,
        },
        required_keys=[],
        section_name="force",
    )
    lora_cfg = _section(
        lora,
        defaults={
            "enabled": False,
            "r": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "target_modules": None,
        },
        required_keys=[],
        section_name="lora",
    )
    short_range_cfg = _section(
        short_range,
        defaults={
            "obs_window": None,
            "dropout": 0.0,
        },
        required_keys=[],
        section_name="short_range",
    )
    lowdim_cfg = _section(
        lowdim,
        defaults={
            "dropout_p": 0.0,
            "project": False,
        },
        required_keys=[],
        section_name="lowdim",
    )
    compile_cfg = _section(
        compile,
        defaults={
            "backbone": False,
            "mode": "reduce-overhead",
        },
        required_keys=[],
        section_name="compile",
    )
    projection_cfg = _section(
        projection,
        defaults={
            "target_feature_dim": None,
        },
        required_keys=[],
        section_name="projection",
    )

    valid_rgb_token_modes = {"all_tokens", "patch_tokens", "cls"}
    if rgb_cfg["token_mode"] not in valid_rgb_token_modes:
        raise ValueError(
            f"Unsupported rgb.token_mode={rgb_cfg['token_mode']}. "
            f"Expected one of {sorted(valid_rgb_token_modes)}"
        )

    valid_vision_modes = {"raw_tokens", "cross_image_attention", "standard_aggregation"}
    if vision_cfg["feature_mode"] not in valid_vision_modes:
        raise ValueError(
            f"Unsupported vision_features.feature_mode={vision_cfg['feature_mode']}. "
            f"Expected one of {sorted(valid_vision_modes)}"
        )

    if vision_cfg["feature_mode"] == "standard_aggregation":
        valid_aggregations = {"avg", "attention_pool"}
        if vision_cfg["feature_aggregation"] not in valid_aggregations:
            raise ValueError(
                "vision_features.feature_mode='standard_aggregation' requires "
                f"vision_features.feature_aggregation in {sorted(valid_aggregations)}, "
                f"got {vision_cfg['feature_aggregation']}."
            )

    valid_force_kv_modes = {"mean", "cls", "all_tokens", "patch_tokens"}
    if force_cfg["cross_attn_kv_mode"] == "tokens":
        raise ValueError(
            "force.cross_attn_kv_mode='tokens' is no longer supported. "
            "Use 'all_tokens' instead."
        )
    if force_cfg["cross_attn_kv_mode"] not in valid_force_kv_modes:
        raise ValueError(
            f"Unsupported force.cross_attn_kv_mode={force_cfg['cross_attn_kv_mode']}. "
            f"Expected one of {sorted(valid_force_kv_modes)}"
        )

    return ObsEncoderInitConfig(
        rgb=RGBConfig(
            model_name=rgb_cfg["model_name"],
            pretrained=rgb_cfg["pretrained"],
            frozen=rgb_cfg["frozen"],
            transforms=rgb_cfg["transforms"],
            token_mode=rgb_cfg["token_mode"],
            use_group_norm=rgb_cfg["use_group_norm"],
            share_model=rgb_cfg["share_model"],
            imagenet_norm=rgb_cfg["imagenet_norm"],
            encoder_by_group=rgb_cfg["encoder_by_group"],
        ),
        depth=DepthConfig(
            model_name=depth_cfg["model_name"],
            pretrained=depth_cfg["pretrained"],
            frozen=depth_cfg["frozen"],
            share_model=depth_cfg["share_model"],
            use_group_norm=depth_cfg["use_group_norm"],
        ),
        vision_features=VisionFeaturesConfig(
            feature_mode=vision_cfg["feature_mode"],
            feature_aggregation=vision_cfg["feature_aggregation"],
            position_encoding=vision_cfg["position_encoding"],
            downsample_ratio=vision_cfg["downsample_ratio"],
            use_cross_image_attn=vision_cfg["use_cross_image_attn"],
            cross_image_attn_heads=vision_cfg["cross_image_attn_heads"],
        ),
        force=ForceConfig(
            fusion_mode=force_cfg["fusion_mode"],
            cross_attn_heads=force_cfg["cross_attn_heads"],
            cross_attn_kv_mode=force_cfg["cross_attn_kv_mode"],
            modality_dropout_p=force_cfg["modality_dropout_p"],
            proj_dropout_p=force_cfg["proj_dropout_p"],
        ),
        lora=LoraConfig(
            enabled=lora_cfg["enabled"],
            r=lora_cfg["r"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
        ),
        short_range=ShortRangeConfig(
            obs_window=short_range_cfg["obs_window"],
            dropout=short_range_cfg["dropout"],
        ),
        lowdim=LowdimConfig(
            dropout_p=lowdim_cfg["dropout_p"],
            project=lowdim_cfg["project"],
        ),
        compile=CompileConfig(
            backbone=compile_cfg["backbone"],
            mode=compile_cfg["mode"],
        ),
        projection=ProjectionConfig(
            target_feature_dim=projection_cfg["target_feature_dim"],
        ),
    )
