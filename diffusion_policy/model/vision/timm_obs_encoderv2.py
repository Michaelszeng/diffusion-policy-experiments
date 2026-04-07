import logging
from typing import Dict, Optional

import timm
import timm.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.timm_obs_v2.depth_module import DepthModule
from diffusion_policy.model.vision.timm_obs_v2.force_module import ForceModule
from diffusion_policy.model.vision.timm_obs_v2.lowdim_module import LowDimModule
from diffusion_policy.model.vision.timm_obs_v2.metadata_utils import get_key_metadata
from diffusion_policy.model.vision.timm_obs_v2.rgb_module import RGBModule
from diffusion_policy.model.vision.timm_obs_v2.config import parse_obs_encoder_init_config
from diffusion_policy.model.vision.timm_obs_v2.encoder_registry import build_encoder_registry
from diffusion_policy.model.vision.timm_obs_v2.types import (
    EncoderCoreOutputs,
    ModalityType,
    RangeType,
    TokenBundle,
    VisualFeatureSet,
    VisualModality,
    VisualStreamFeatures,
)
from diffusion_policy.model.vision.timm_obs_v2.visual_fusion import VisualFusionModule

logger = logging.getLogger(__name__)


class TimmObsEncoder(ModuleAttrMixin):
    """Modular Timm observation encoder.

    Shape legend:
    B = batch, T = horizon/time, N = tokens, D = embedding dim
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb: dict,
        depth: Optional[dict] = None,
        vision_features: Optional[dict] = None,
        force: Optional[dict] = None,
        lora: Optional[dict] = None,
        short_range: Optional[dict] = None,
        lowdim: Optional[dict] = None,
        compile: Optional[dict] = None,
        projection: Optional[dict] = None,
        forward_output: str = "cat",
        n_obs_steps: Optional[int] = None,
    ):
        super().__init__()

        cfg = parse_obs_encoder_init_config(
            rgb,
            depth,
            vision_features,
            force,
            lora,
            short_range,
            lowdim,
            compile,
            projection,
        )

        if not (0.0 <= cfg.force.modality_dropout_p <= 1.0):
            raise ValueError(f"force.modality_dropout_p must be in [0,1], got {cfg.force.modality_dropout_p}")
        if not (0.0 <= cfg.force.proj_dropout_p <= 1.0):
            raise ValueError(f"force.proj_dropout_p must be in [0,1], got {cfg.force.proj_dropout_p}")
        valid_rgb_token_modes = {"all_tokens", "patch_tokens", "cls"}
        if cfg.rgb.token_mode not in valid_rgb_token_modes:
            raise ValueError(
                f"Unsupported rgb.token_mode={cfg.rgb.token_mode}. "
                f"Expected one of {sorted(valid_rgb_token_modes)}"
            )
        if cfg.force.cross_attn_kv_mode == "tokens":
            raise ValueError(
                "force.cross_attn_kv_mode='tokens' is no longer supported. "
                "Use 'all_tokens' instead."
            )
        valid_force_kv_modes = {"mean", "cls", "all_tokens", "patch_tokens"}
        if cfg.force.cross_attn_kv_mode not in valid_force_kv_modes:
            raise ValueError(
                f"Unsupported force.cross_attn_kv_mode={cfg.force.cross_attn_kv_mode}. "
                f"Expected one of {sorted(valid_force_kv_modes)}"
            )

        self.shape_meta = shape_meta
        self.model_name = cfg.rgb.model_name
        self.depth_model_name = cfg.depth.model_name
        self.encoder_by_group = cfg.rgb.encoder_by_group or {}

        self.short_range_obs_window = cfg.short_range.obs_window
        self.short_range_dropout = cfg.short_range.dropout
        self.forward_output = forward_output
        self.force_fusion_mode = cfg.force.fusion_mode
        self.force_cross_attn_kv_mode = cfg.force.cross_attn_kv_mode

        # Infer canonical image shape from RGB keys.
        image_shape = None
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            shape = tuple(attr["shape"])
            if obs_type == "rgb":
                if image_shape is None:
                    image_shape = shape[1:]
                elif image_shape != shape[1:]:
                    raise ValueError(
                        f"All RGB keys must share same image shape. Got {image_shape} and {shape[1:]}"
                    )

        registry = build_encoder_registry(shape_meta=shape_meta, cfg=cfg)

        self._setup_img_augmentation_transforms(
            image_shape=image_shape,
            transforms=cfg.rgb.transforms,
            model=registry.default_rgb_model,
            pretrained=cfg.rgb.pretrained,
            imagenet_norm=cfg.rgb.imagenet_norm,
            normalize_transform=registry.default_normalize_transform,
        )
        self.center_crop = torchvision.transforms.CenterCrop(size=self.crop_size)
        self.final_resize = torchvision.transforms.Resize(size=image_shape[0], antialias=True)

        self.rgb_keys = registry.rgb_keys
        self.depth_keys = registry.depth_keys
        self.low_dim_keys = registry.low_dim_keys
        self.force_keys = registry.force_keys
        self.key_shape_map = registry.key_shape_map
        self.key_feature_dim_map = registry.key_feature_dim_map
        self.key_normalize_map = registry.key_normalize_map
        self.key_model_map = registry.key_model_map
        self.visual_prefix_tokens_by_key = self._build_visual_prefix_tokens_by_key()
        if cfg.rgb.token_mode in {"patch_tokens", "cls"}:
            incompatible_rgb_keys = []
            for key in self.rgb_keys:
                prefix_tokens = int(self.visual_prefix_tokens_by_key.get(key, 0) or 0)
                if prefix_tokens < 1:
                    incompatible_rgb_keys.append(key)
            if len(incompatible_rgb_keys) > 0:
                raise ValueError(
                    f"rgb.token_mode='{cfg.rgb.token_mode}' requires RGB streams with prefix tokens (CLS/registers). "
                    f"The following RGB keys have num_prefix_tokens=0 and must use rgb.token_mode='all_tokens': "
                    f"{incompatible_rgb_keys}"
                )
        self.share_rgb_model = cfg.rgb.share_model
        self.share_depth_model = cfg.depth.share_model

        logger.info("rgb keys: %s", self.rgb_keys)
        logger.info("depth keys: %s", self.depth_keys)
        logger.info("low_dim keys: %s", self.low_dim_keys)
        logger.info("force keys: %s", self.force_keys)

        self.cam_down_sample_steps = 1
        for key, attr in shape_meta["obs"].items():
            if attr.get("type") == "rgb":
                self.cam_down_sample_steps = int(attr.get("down_sample_steps", 1))
                break

        feature_dim = registry.default_feature_dim
        key_feature_dim_map = registry.key_feature_dim_map
        all_feature_dims = set(key_feature_dim_map.values())
        self.feature_dim = feature_dim
        self.target_feature_dim = cfg.projection.target_feature_dim if cfg.projection.target_feature_dim else feature_dim

        if len(all_feature_dims) > 1 or (
            len(all_feature_dims) == 1 and list(all_feature_dims)[0] != self.target_feature_dim
        ):
            logger.info(
                "Multiple feature dims %s detected. RGB/Depth modules project to target_feature_dim=%d",
                all_feature_dims,
                self.target_feature_dim,
            )

        self.short_range_dropout_param_map = nn.ParameterDict()
        if self.short_range_obs_window is not None and self.short_range_dropout > 0:
            for key in self.rgb_keys + self.depth_keys:
                native_dim = key_feature_dim_map[key]
                self.short_range_dropout_param_map[key] = nn.Parameter(torch.randn(1, 1, native_dim) * 0.05)

        self.max_obs_steps = 0
        for _, attr in shape_meta["obs"].items():
            if attr.get("ignore_by_policy", False):
                continue
            horizon = attr.get("horizon", 1)
            if horizon is None:
                continue
            down_steps = int(attr.get("down_sample_steps", 1))
            obs_steps = (horizon - 1) * down_steps
            self.max_obs_steps = max(self.max_obs_steps, obs_steps)

        self.visual_fusion = VisualFusionModule(
            target_feature_dim=self.target_feature_dim,
            visual_feature_mode=cfg.vision_features.feature_mode,
            use_cross_image_attn=cfg.vision_features.use_cross_image_attn,
            cross_image_attn_heads=cfg.vision_features.cross_image_attn_heads,
            rgb_token_mode=cfg.rgb.token_mode,
            feature_aggregation=cfg.vision_features.feature_aggregation,
            visual_prefix_tokens_by_key=self.visual_prefix_tokens_by_key,
        )

        self.force_module = ForceModule(
            force_fusion_mode=cfg.force.fusion_mode,
            force_keys=self.force_keys,
            shape_meta=shape_meta,
            target_feature_dim=self.target_feature_dim,
            force_cross_attn_heads=cfg.force.cross_attn_heads,
            force_cross_attn_kv_mode=cfg.force.cross_attn_kv_mode,
            force_modality_dropout_p=cfg.force.modality_dropout_p,
            force_proj_dropout_p=cfg.force.proj_dropout_p,
        )

        self.lowdim_module = LowDimModule(
            low_dim_keys=self.low_dim_keys,
            shape_meta=shape_meta,
            target_feature_dim=self.target_feature_dim,
            lowdim_dropout_p=cfg.lowdim.dropout_p,
            project=cfg.lowdim.project,
        )

        self.rgb_module = RGBModule(
            rgb_keys=self.rgb_keys,
            short_range_obs_window=self.short_range_obs_window,
            key_feature_dim_map=key_feature_dim_map,
            default_feature_dim=feature_dim,
            target_feature_dim=self.target_feature_dim,
        )
        self.depth_module = DepthModule(
            depth_keys=self.depth_keys,
            short_range_obs_window=self.short_range_obs_window,
            key_feature_dim_map=key_feature_dim_map,
            default_feature_dim=feature_dim,
            target_feature_dim=self.target_feature_dim,
        )

        # n_obs_steps is optional but required for cat_output_dim and n_modalities properties
        self._n_obs_steps = n_obs_steps

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        if cfg.compile.backbone:
            self._compile_backbones(mode=cfg.compile.mode)

    @property
    def cat_output_dim(self) -> int:
        """Flat output dimension for output_format='cat'.

        Valid when rgb.token_mode='cls' (1 CLS token per camera per step) and
        lowdim.project=True (lowdim projected to target_feature_dim).
        Requires n_obs_steps to be set at init.
        """
        if self._n_obs_steps is None:
            raise ValueError(
                "cat_output_dim requires n_obs_steps to be passed at init. "
                "Add 'n_obs_steps: ${n_obs_steps}' to the obs_encoder config."
            )
        return (len(self.rgb_keys) + len(self.low_dim_keys)) * self._n_obs_steps * self.target_feature_dim

    @property
    def n_modalities(self) -> int:
        """Size of modality embedding table (max modality index + 1)."""
        return self.get_max_modalities()

    def _setup_img_augmentation_transforms(
        self,
        image_shape,
        transforms,
        model,
        pretrained: bool,
        imagenet_norm: bool,
        normalize_transform: Optional[nn.Module] = None,
    ):
        """Set up RGB-specific augmentation stack and normalization transform."""

        ratio = 1.0
        if transforms is not None and len(transforms) > 0 and not isinstance(transforms[0], torch.nn.Module):
            if transforms[0].type != "RandomCrop":
                raise ValueError("First transform config must be RandomCrop when using config-style transforms.")
            ratio = transforms[0].ratio

        self.crop_size = int(image_shape[0] * ratio)

        self.rgb_specific_transforms = nn.Identity()
        if transforms is not None and len(transforms) > 1:
            self.rgb_specific_transforms = torch.nn.Sequential(*transforms[1:])

        if normalize_transform is not None:
            self.normalize_transform = normalize_transform
        else:
            self.normalize_transform = nn.Identity()
            if imagenet_norm and pretrained:
                data_config = timm.data.resolve_model_data_config(model)
                mean = data_config.get("mean", (0.485, 0.456, 0.406))
                std = data_config.get("std", (0.229, 0.224, 0.225))
                self.normalize_transform = torchvision.transforms.Normalize(mean=mean, std=std)

    def _compile_backbones(self, mode: str = "reduce-overhead"):
        """Compile backbone modules in key_model_map."""

        compile_count = 0
        for key in list(self.key_model_map.keys()):
            model = self.key_model_map[key]
            if isinstance(model, nn.ModuleList):
                self.key_model_map[key] = nn.ModuleList(
                    [
                        torch.compile(model[0], mode=mode, dynamic=False),
                        torch.compile(model[1], mode=mode, dynamic=False),
                    ]
                )
                compile_count += 2
            else:
                self.key_model_map[key] = torch.compile(model, mode=mode, dynamic=False)
                compile_count += 1
        logger.info("Compiled %d backbone(s) with mode='%s'", compile_count, mode)

    def _build_visual_prefix_tokens_by_key(self) -> Dict[str, int]:
        """Infer per-key visual prefix-token count (CLS/register tokens)."""

        prefix_tokens = {}
        for key in self.rgb_keys + self.depth_keys:
            if key not in self.key_model_map:
                continue
            model = self.key_model_map[key]
            backbone = model[0] if isinstance(model, nn.ModuleList) else model
            count = int(getattr(backbone, "num_prefix_tokens", 0) or 0)
            if count < 0:
                raise ValueError(f"num_prefix_tokens for key={key} must be non-negative, got {count}")
            prefix_tokens[key] = count
        return prefix_tokens

    def _get_key_metadata(self, key: str, batch_size: int, device: torch.device, this_max_obs_horizon: int):
        return get_key_metadata(
            key=key,
            batch_size=batch_size,
            device=device,
            this_max_obs_horizon=this_max_obs_horizon,
            shape_meta=self.shape_meta,
            max_obs_steps=self.max_obs_steps,
            rgb_keys=self.rgb_keys,
            depth_keys=self.depth_keys,
            low_dim_keys=self.low_dim_keys,
            force_keys=self.force_keys,
            short_range_obs_window=self.short_range_obs_window,
        )

    def _normalize_visual_inputs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize RGB/depth inputs once before modality encoding.

        Returns:
            key -> normalized tensor, each with shape (B, T, C, H, W)
        """

        normalized = {}
        processed_depth = set()

        # Process paired RGB + depth using identical crop parameters.
        for key in self.rgb_keys:
            img = obs_dict[key]  # (B, T, 3, H, W)
            batch_size, horizon = img.shape[:2]
            if tuple(img.shape[2:]) != tuple(self.key_shape_map[key]):
                raise ValueError(
                    f"RGB key {key} has shape {tuple(img.shape[2:])}, expected {self.key_shape_map[key]}"
                )

            base_key = key.rsplit("_rgb", 1)[0] if key.endswith("_rgb") else key
            depth_key = f"{base_key}_depth"
            has_depth = (depth_key in self.depth_keys and depth_key in obs_dict)

            depth_img = None
            if has_depth:
                depth_img = obs_dict[depth_key]  # (B, T, 1, H, W)
                depth_img = depth_img.flatten(0, 1)  # (B*T, 1, H, W)

                rgb_h, rgb_w = img.shape[3], img.shape[4]
                depth_h, depth_w = depth_img.shape[2], depth_img.shape[3]
                if (depth_h != rgb_h) or (depth_w != rgb_w):
                    depth_img = F.interpolate(
                        depth_img,
                        size=(rgb_h, rgb_w),
                        mode="bilinear",
                        align_corners=False,
                    )

            img = img.flatten(0, 1)  # (B*T, 3, H, W)
            normalize_fn = self.key_normalize_map.get(key, self.normalize_transform)

            if self.training:
                # Shared random crop params for paired rgb/depth streams.
                i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                    img,
                    output_size=(self.crop_size, self.crop_size),
                )

                img = TF.crop(img, i, j, h, w)
                img = self.final_resize(img)
                img = self.rgb_specific_transforms(img)
                img = normalize_fn(img)

                if has_depth:
                    depth_img = TF.crop(depth_img, i, j, h, w)
                    depth_img = self.final_resize(depth_img)
            else:
                img = self.center_crop(img)
                img = self.final_resize(img)
                img = normalize_fn(img)

                if has_depth:
                    depth_img = self.center_crop(depth_img)
                    depth_img = self.final_resize(depth_img)

            normalized[key] = img.view(batch_size, horizon, *img.shape[1:])  # (B, T, 3, H, W)
            if has_depth:
                normalized[depth_key] = depth_img.view(batch_size, horizon, *depth_img.shape[1:])
                processed_depth.add(depth_key)

        # Process depth streams not paired with an RGB key.
        for key in self.depth_keys:
            if key in processed_depth:
                continue
            depth_img = obs_dict[key]  # (B, T, 1, H, W)
            batch_size, horizon = depth_img.shape[:2]
            if tuple(depth_img.shape[2:]) != tuple(self.key_shape_map[key]):
                raise ValueError(
                    f"Depth key {key} has shape {tuple(depth_img.shape[2:])}, expected {self.key_shape_map[key]}"
                )

            depth_img = depth_img.flatten(0, 1)
            if self.training:
                i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                    depth_img,
                    output_size=(self.crop_size, self.crop_size),
                )
                depth_img = TF.crop(depth_img, i, j, h, w)
                depth_img = self.final_resize(depth_img)
            else:
                depth_img = self.center_crop(depth_img)
                depth_img = self.final_resize(depth_img)

            normalized[key] = depth_img.view(batch_size, horizon, *depth_img.shape[1:])

        return normalized

    def _encode_core(self, obs_dict: Dict[str, torch.Tensor]) -> EncoderCoreOutputs:
        """Shared token-first encoding trunk used by both cat and dit adapters."""

        drop_short_range = False
        if self.training and self.short_range_dropout > 0:
            if torch.rand(1, device=self.device) < self.short_range_dropout:
                drop_short_range = True

        batch_size = next(iter(obs_dict.values())).shape[0]

        normalized_visual = self._normalize_visual_inputs(obs_dict)

        # Encoder-owned per-key visual pipeline:
        # each key returns BTND tokens (B*T, N, D) which includes any key-specific prefix tokens (e.g. CLS tokens) and is not yet fused across keys.
        rgb_features_by_key = self.rgb_module.encode(
            normalized_obs=normalized_visual,
            key_model_map=self.key_model_map,
            short_range_dropout_param_map=self.short_range_dropout_param_map,
            drop_short_range=drop_short_range,
        )
        depth_features_by_key = self.depth_module.encode(
            normalized_obs=normalized_visual,
            key_model_map=self.key_model_map,
            short_range_dropout_param_map=self.short_range_dropout_param_map,
            drop_short_range=drop_short_range,
        )

        # Deterministic visual stream order: all RGB keys, then all depth keys.
        visual_features_by_key: VisualFeatureSet = {}
        for key in self.rgb_keys:
            if key not in rgb_features_by_key:
                continue
            visual_features_by_key[key] = VisualStreamFeatures(
                tokens=rgb_features_by_key[key],
                modality=VisualModality.RGB,
                num_prefix_tokens=int(self.visual_prefix_tokens_by_key.get(key, 0) or 0),
            )
        for key in self.depth_keys:
            if key not in depth_features_by_key:
                continue
            visual_features_by_key[key] = VisualStreamFeatures(
                tokens=depth_features_by_key[key],
                modality=VisualModality.DEPTH,
                num_prefix_tokens=int(self.visual_prefix_tokens_by_key.get(key, 0) or 0),
            )

        force_features_by_key = self.force_module.encode(
            obs_dict=obs_dict,
            key_shape_map=self.key_shape_map,
            visual_features_by_key=visual_features_by_key,
            batch_size=batch_size,
            short_range_obs_window=self.short_range_obs_window,
            visual_fusion=self.visual_fusion,
            device=self.device,
        )

        lowdim_features_by_key = self.lowdim_module.encode(obs_dict, self.key_shape_map)

        return EncoderCoreOutputs(
            batch_size=batch_size,
            visual_features_by_key=visual_features_by_key,
            force_features_by_key=force_features_by_key,
            lowdim_features_by_key=lowdim_features_by_key,
        )

    def _build_dit_bundle(self, core: EncoderCoreOutputs, obs_dict: Dict[str, torch.Tensor]) -> TokenBundle:
        """Assemble canonical DIT token bundle from shared core outputs."""

        token_list = []
        pos_list = []
        mod_list = []
        range_list = []

        def append_key_block(
            key: str,
            tokens_btnd: torch.Tensor,
            obs_horizon: int,
            modality_override: Optional[int] = None,
        ) -> None:
            if len(tokens_btnd.shape) != 3:
                raise ValueError(
                    f"Expected BTND tokens for key={key}, got shape={tuple(tokens_btnd.shape)}"
                )
            if (tokens_btnd.shape[0] % core.batch_size) != 0:
                raise ValueError(
                    f"Token batch mismatch for key {key}: got {tokens_btnd.shape[0]} and B={core.batch_size}"
                )

            steps_from_feat = tokens_btnd.shape[0] // core.batch_size
            tokens_per_step = tokens_btnd.shape[1]
            token_dim = tokens_btnd.shape[2]

            pos, mod, rng = self._get_key_metadata(
                key=key,
                batch_size=core.batch_size,
                device=tokens_btnd.device,
                this_max_obs_horizon=obs_horizon,
            )
            if pos.shape[1] != steps_from_feat:
                raise ValueError(
                    f"Metadata mismatch for {key}: metadata={pos.shape[1]} vs feature={steps_from_feat}"
                )

            if modality_override is not None:
                mod = torch.full_like(mod, modality_override)

            if tokens_per_step > 1:
                pos = pos.repeat_interleave(tokens_per_step, dim=1)
                mod = mod.repeat_interleave(tokens_per_step, dim=1)
                rng = rng.repeat_interleave(tokens_per_step, dim=1)

            token_list.append(tokens_btnd.reshape(core.batch_size, steps_from_feat * tokens_per_step, token_dim))
            pos_list.append(pos)
            mod_list.append(mod)
            range_list.append(rng)

        # visual tokens are selected as cls/patch tokens/all_tokens
        visual_tokens, visual_tokens_by_key = self.visual_fusion.aggregate_by_mode(
            visual_features=core.visual_features_by_key,
            output_format="dit",
            batch_size=core.batch_size,
            return_by_key=True,
        )
        visual_keys = [key for key in (self.rgb_keys + self.depth_keys) if key in core.visual_features_by_key]
        if len(visual_keys) > 0:
            if self.visual_fusion.visual_feature_mode == "cross_image_attention":
                if visual_tokens is None:
                    raise ValueError("Expected fused visual tokens for cross_image_attention mode.")
                first_key = visual_keys[0]
                pooled_visual_btnd = visual_tokens.reshape(-1, visual_tokens.shape[-1]).unsqueeze(1)
                append_key_block(
                    key=first_key,
                    tokens_btnd=pooled_visual_btnd,
                    obs_horizon=obs_dict[first_key].shape[1],
                    modality_override=ModalityType.NULL,
                )
            else:
                if visual_tokens_by_key is None:
                    raise ValueError("Expected per-key visual tokens for non-cross visual mode.")
                for key in visual_keys:
                    append_key_block(
                        key=key,
                        tokens_btnd=visual_tokens_by_key[key],
                        obs_horizon=obs_dict[key].shape[1],
                    )

        for key in self.force_keys:
            if key not in core.force_features_by_key:
                continue
            append_key_block(
                key=key,
                tokens_btnd=core.force_features_by_key[key],
                obs_horizon=obs_dict[key].shape[1],
            )

        for key in self.low_dim_keys:
            if key not in core.lowdim_features_by_key:
                continue
            append_key_block(
                key=key,
                tokens_btnd=core.lowdim_features_by_key[key],
                obs_horizon=obs_dict[key].shape[1],
            )

        if len(token_list) == 0:
            raise ValueError("No tokens produced by encoder.")

        tokens = torch.cat(token_list, dim=1)  # (B, N, D)
        positions = torch.cat(pos_list, dim=1) if pos_list else None
        modalities = torch.cat(mod_list, dim=1) if mod_list else None
        ranges = torch.cat(range_list, dim=1) if range_list else None

        if positions is not None and positions.shape[1] != tokens.shape[1]:
            raise ValueError(
                f"DIT metadata/token mismatch: positions={positions.shape[1]} vs tokens={tokens.shape[1]}"
            )
        if modalities is not None and modalities.shape[1] != tokens.shape[1]:
            raise ValueError(
                f"DIT metadata/token mismatch: modality={modalities.shape[1]} vs tokens={tokens.shape[1]}"
            )
        if ranges is not None and ranges.shape[1] != tokens.shape[1]:
            raise ValueError(f"DIT metadata/token mismatch: range={ranges.shape[1]} vs tokens={tokens.shape[1]}")

        return TokenBundle(
            tokens=tokens,
            positions=positions,
            modality=modalities,
            range=ranges,
        )

    def _bundle_to_dit_output(self, bundle: TokenBundle) -> Dict[str, torch.Tensor]:
        return {
            "tokens": bundle.tokens,
            "positions": bundle.positions,
            "modality": bundle.modality,
            "range": bundle.range,
        }

    def _core_to_cat(self, core: EncoderCoreOutputs) -> torch.Tensor:
        """Convert shared core outputs to cat-mode flat feature vector."""

        features = []

        visual_cat, _ = self.visual_fusion.aggregate_by_mode(
            visual_features=core.visual_features_by_key,
            output_format="cat",
            batch_size=core.batch_size,
        )
        if visual_cat is not None:
            features.append(visual_cat)

        for key in self.force_keys:
            if key in core.force_features_by_key:
                features.append(core.force_features_by_key[key].reshape(core.batch_size, -1))

        for key in self.low_dim_keys:
            if key in core.lowdim_features_by_key:
                features.append(core.lowdim_features_by_key[key].reshape(core.batch_size, -1))

        if len(features) == 0:
            return torch.zeros(core.batch_size, 0, device=self.device, dtype=self.dtype)
        return torch.cat(features, dim=-1)

    def forward(self, obs_dict: Dict[str, torch.Tensor], output_format: str = "cat"):
        """Encode observations.

        output_format:
            - 'cat': returns (B, F)
            - 'dit': returns dict {'tokens','positions','modality','range'}
        """

        if output_format == "dict":
            raise NotImplementedError(
                "output_format='dict' is intentionally removed in this refactor. "
                "Supported output formats are 'cat' and 'dit'."
            )
        if output_format not in {"cat", "dit"}:
            raise ValueError(f"Unsupported output_format: {output_format}")

        core = self._encode_core(obs_dict=obs_dict)

        if output_format == "cat":
            return self._core_to_cat(core)

        bundle = self._build_dit_bundle(core, obs_dict)
        return self._bundle_to_dit_output(bundle)

    def get_max_modalities(self):
        return ModalityType.FORCE + 1

    def get_max_ranges(self):
        return RangeType.SHORT + 1

    def get_all_valid_keys(self):
        all_keys = self.rgb_keys + self.depth_keys + self.low_dim_keys + self.force_keys
        return sorted(list(set(all_keys)))

    @torch.no_grad()
    def output_shape(self, mode: str = "cat"):
        """Return encoder output shape.

        mode='cat' -> (1, F)
        mode='dit' -> (1, N, D)
        """

        if mode not in {"cat", "dit"}:
            raise ValueError(f"Unsupported mode={mode}. Expected 'cat' or 'dit'.")

        example_obs_dict = {}
        for key, attr in self.shape_meta["obs"].items():
            horizon = attr.get("horizon", None)
            if horizon is None:
                raise ValueError(
                    f"horizon must be specified for key {key} in shape_meta to call output_shape()."
                )
            shape = tuple(attr["shape"])
            example_obs_dict[key] = torch.zeros(
                (1, horizon) + shape,
                dtype=self.dtype,
                device=self.device,
            )

        example_output = self.forward(example_obs_dict, output_format=mode)
        if mode == "dit":
            tokens = example_output["tokens"]
            if tokens.shape[0] != 1:
                raise ValueError(f"Unexpected output batch in output_shape(mode='dit'): {tokens.shape}")
            return tokens.shape

        if example_output.shape[0] != 1:
            raise ValueError(f"Unexpected output batch in output_shape(mode='cat'): {example_output.shape}")
        return example_output.shape
