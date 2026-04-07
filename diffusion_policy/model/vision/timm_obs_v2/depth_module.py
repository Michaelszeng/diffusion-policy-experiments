"""Depth modality encoder module for TimmObsEncoderV2."""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DepthModule(nn.Module):
    """Encode preprocessed depth tensors into BTND visual tokens.

    Shape legend:
    B = batch, T = horizon/time, N = tokens, D = embedding dim
    """

    def __init__(
        self,
        depth_keys,
        short_range_obs_window: Optional[int],
        key_feature_dim_map: Dict[str, int],
        default_feature_dim: int,
        target_feature_dim: int,
    ):
        super().__init__()
        self.depth_keys = list(depth_keys)
        self.short_range_obs_window = short_range_obs_window
        self.feature_projections = nn.ModuleDict()

        for key in self.depth_keys:
            src_dim = key_feature_dim_map.get(key, default_feature_dim)
            if src_dim != target_feature_dim:
                self.feature_projections[key] = nn.Linear(src_dim, target_feature_dim)
                logger.info("Depth projection %s: %d -> %d", key, src_dim, target_feature_dim)

    @staticmethod
    def _encode(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        return model(x)

    @staticmethod
    def _to_tokens(feature: torch.Tensor) -> torch.Tensor:
        if len(feature.shape) == 3:
            return feature
        if len(feature.shape) == 4:
            # CNN: (B*T, C, H, W) -> (B*T, H*W, C)
            return torch.flatten(feature, start_dim=-2).transpose(1, 2)
        raise ValueError(f"Unsupported visual feature rank for tokenization: {tuple(feature.shape)}")

    def _maybe_project(self, feature: torch.Tensor, key: str) -> torch.Tensor:
        if key in self.feature_projections:
            return self.feature_projections[key](feature)
        return feature

    def _encode_visual(
        self,
        key: str,
        img_long: torch.Tensor,
        img_short: Optional[torch.Tensor],
        batch_size: int,
        drop_short_range: bool,
        key_model_map: nn.ModuleDict,
        short_range_dropout_param_map: nn.ParameterDict,
    ) -> torch.Tensor:
        model = key_model_map[key]
        if not isinstance(model, nn.ModuleList):
            return self._encode(model, img_long)

        short_window = self.short_range_obs_window
        feat_long = self._encode(model[0], img_long)

        if img_short is not None and not drop_short_range:
            feat_short = self._encode(model[1], img_short)
        else:
            if key not in short_range_dropout_param_map:
                raise ValueError(f"Short-range dropout parameter for {key} not found.")
            param = short_range_dropout_param_map[key]  # (1, 1, D)
            if len(feat_long.shape) == 3:
                feat_short = param.expand(batch_size * short_window, feat_long.shape[1], -1).contiguous()
            else:
                channels = param.shape[-1]
                feat_short = (
                    param.view(1, channels, 1, 1)
                    .expand(batch_size * short_window, channels, *feat_long.shape[2:])
                    .contiguous()
                )

        t_long = feat_long.shape[0] // batch_size
        feat_long = feat_long.view(batch_size, t_long, *feat_long.shape[1:])
        feat_short = feat_short.view(batch_size, short_window, *feat_short.shape[1:])
        return torch.cat([feat_long, feat_short], dim=1).flatten(0, 1)

    def encode(
        self,
        normalized_obs: Dict[str, torch.Tensor],
        key_model_map: nn.ModuleDict,
        short_range_dropout_param_map: nn.ParameterDict,
        drop_short_range: bool,
    ) -> Dict[str, torch.Tensor]:
        """Encode all depth keys and return full BTND tokens per key."""

        out = {}
        for key in self.depth_keys:
            img = normalized_obs[key]  # (B, T, 1, H, W)
            batch_size = img.shape[0]

            img_long = img.flatten(0, 1)
            img_short = None
            if self.short_range_obs_window is not None and not drop_short_range:
                img_short = img[:, -self.short_range_obs_window :].flatten(0, 1)

            out[key] = self._encode_visual(
                key=key,
                img_long=img_long,
                img_short=img_short,
                batch_size=batch_size,
                drop_short_range=drop_short_range,
                key_model_map=key_model_map,
                short_range_dropout_param_map=short_range_dropout_param_map,
            )

            out[key] = self._to_tokens(out[key])

            out[key] = self._maybe_project(out[key], key)

            if len(out[key].shape) != 3:
                raise ValueError(f"Depth key {key} did not produce BTND output. Got shape {tuple(out[key].shape)}")

        return out
