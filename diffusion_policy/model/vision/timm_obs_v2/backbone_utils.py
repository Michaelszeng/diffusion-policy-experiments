"""Backbone and normalization utilities for TimmObsEncoderV2."""

from typing import List, Optional, Tuple
import logging

import timm
import timm.data
import torch
import torch.nn as nn
import torchvision

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.common.lora import apply_lora_to_model, print_trainable_parameters

logger = logging.getLogger(__name__)


def get_feature_dim_for_model(model_name: str, downsample_ratio: int = 32) -> int:
    """Get output feature dimension for model architecture."""

    if model_name.startswith("resnet18") or model_name.startswith("resnet34"):
        return 512 if downsample_ratio == 32 else 256
    if model_name.startswith("resnet50") or model_name.startswith("resnet101") or model_name.startswith("resnet152"):
        return 2048 if downsample_ratio == 32 else 1024
    if model_name.startswith("convnext"):
        return 1024
    if model_name.startswith("vit_tiny"):
        return 192
    if model_name.startswith("vit_small"):
        return 384
    if model_name.startswith("vit_base"):
        return 768
    if model_name.startswith("vit_large"):
        return 1024
    if model_name.startswith("efficientnet_b0"):
        return 1280
    if model_name.startswith("efficientnet_b3"):
        return 1536
    if model_name.startswith("dinov2"):
        if "vits" in model_name:
            return 384
        if "vitb" in model_name:
            return 768
        if "vitl" in model_name:
            return 1024
        if "vitg" in model_name:
            return 1536
        return 768

    logger.warning("Unknown model %s, creating temporary instance to infer feature dim", model_name)
    temp_model = timm.create_model(model_name, pretrained=False, num_classes=0)
    if hasattr(temp_model, "embed_dim"):
        return int(temp_model.embed_dim)
    if hasattr(temp_model, "num_features"):
        return int(temp_model.num_features)
    return 768


def create_and_process_encoder(
    model_name: str,
    pretrained: bool,
    frozen: bool,
    downsample_ratio: int,
    use_group_norm: bool,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, int]:
    """Create timm encoder and apply optional architecture transforms."""

    create_kwargs = {
        "model_name": model_name,
        "pretrained": pretrained,
        "global_pool": "",
        "num_classes": 0,
    }
    if "vit" in model_name.lower() or "dinov2" in model_name.lower():
        create_kwargs["img_size"] = 224

    model = timm.create_model(**create_kwargs)

    if use_lora:
        if not pretrained:
            logger.warning("Using LoRA on non-pretrained model %s", model_name)
        model = apply_lora_to_model(
            model=model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
            freeze_non_lora=True,
        )
        print_trainable_parameters(model, model_name)
    elif frozen:
        if not pretrained:
            logger.warning("Freezing non-pretrained model %s", model_name)
        for param in model.parameters():
            param.requires_grad = False

    feature_dim: Optional[int] = None
    if model_name.startswith("resnet"):
        if downsample_ratio == 32:
            model = nn.Sequential(*list(model.children())[:-2])
            feature_dim = 512 if ("resnet18" in model_name or "resnet34" in model_name) else 2048
        elif downsample_ratio == 16:
            model = nn.Sequential(*list(model.children())[:-3])
            feature_dim = 256 if ("resnet18" in model_name or "resnet34" in model_name) else 1024
        else:
            raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
    elif model_name.startswith("convnext"):
        if downsample_ratio == 32:
            model = nn.Sequential(*list(model.children())[:-2])
            feature_dim = 1024
        else:
            raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
    elif model_name.startswith("vit") or model_name.startswith("dinov2"):
        feature_dim = int(model.embed_dim)
    elif model_name.startswith("efficientnet"):
        feature_dim = int(model.num_features)
    else:
        feature_dim = get_feature_dim_for_model(model_name, downsample_ratio)

    if use_group_norm and not pretrained:
        model = replace_submodules(
            root_module=model,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                num_channels=x.num_features,
            ),
        )

    return model, int(feature_dim)


def modify_first_conv_to_single_channel(model: nn.Module) -> bool:
    """Modify first conv/patch projection layer to accept one input channel."""

    if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        return True

    if isinstance(model, nn.Sequential) and len(model) > 0 and isinstance(model[0], nn.Conv2d):
        old_conv = model[0]
        model[0] = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        return True

    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
        old_conv = model.patch_embed.proj
        model.patch_embed.proj = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        return True

    return False


def get_normalization_for_model(
    model_name: str,
    pretrained: bool,
    imagenet_norm: bool,
    model: Optional[nn.Module] = None,
) -> nn.Module:
    """Get model-specific normalization transform."""

    if (not imagenet_norm) or (not pretrained):
        return nn.Identity()

    if model is not None:
        data_config = timm.data.resolve_model_data_config(model)
    else:
        temp_model = timm.create_model(model_name, pretrained=False, num_classes=0)
        data_config = timm.data.resolve_model_data_config(temp_model)

    mean = data_config.get("mean", (0.485, 0.456, 0.406))
    std = data_config.get("std", (0.229, 0.224, 0.225))
    return torchvision.transforms.Normalize(mean=mean, std=std)
