"""Visual token selection and aggregation for TimmObsEncoderV2."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from diffusion_policy.model.vision.timm_obs_v2.standard_visual_aggregator import (
    AttentionPool1d,
    StandardVisualAggregator,
)
from diffusion_policy.model.vision.timm_obs_v2.types import VisualFeatureSet, VisualModality, VisualStreamFeatures

logger = logging.getLogger(__name__)


class VisualFusionModule(nn.Module):
    """Visual feature processing and output-mode-aware aggregation."""

    def __init__(
        self,
        target_feature_dim: int,
        visual_feature_mode: str,
        use_cross_image_attn: bool,
        cross_image_attn_heads: int,
        rgb_token_mode: str,
        feature_aggregation: Optional[str] = None,
        visual_prefix_tokens_by_key: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.target_feature_dim = target_feature_dim
        self.visual_feature_mode = visual_feature_mode
        self.rgb_token_mode = rgb_token_mode
        self.feature_aggregation = feature_aggregation
        self.visual_prefix_tokens_by_key = dict(visual_prefix_tokens_by_key or {})
        self.standard_aggregator = None

        valid_modes = {"raw_tokens", "cross_image_attention", "standard_aggregation"}
        if self.visual_feature_mode not in valid_modes:
            raise ValueError(
                f"Unsupported visual_feature_mode={self.visual_feature_mode}. Expected one of {sorted(valid_modes)}"
            )
        valid_rgb_token_modes = {"all_tokens", "patch_tokens", "cls"}
        if self.rgb_token_mode not in valid_rgb_token_modes:
            raise ValueError(
                f"Unsupported rgb_token_mode={self.rgb_token_mode}. Expected one of {sorted(valid_rgb_token_modes)}"
            )

        if use_cross_image_attn and self.visual_feature_mode == "raw_tokens":
            logger.warning(
                "use_cross_image_attn=True is deprecated. Using visual_feature_mode='cross_image_attention'."
            )
            self.visual_feature_mode = "cross_image_attention"

        self.use_cross_image_attn = self.visual_feature_mode == "cross_image_attention"
        if self.use_cross_image_attn:
            self.cross_image_pool = AttentionPool1d(
                embed_dim=target_feature_dim,
                num_heads=cross_image_attn_heads,
            )

        if self.visual_feature_mode == "standard_aggregation":
            if self.rgb_token_mode == "cls":
                raise ValueError(
                    "visual_feature_mode='standard_aggregation' cannot be combined with rgb_token_mode='cls'. "
                    "Standard aggregation over CLS-only streams is undefined."
                )
            self.standard_aggregator = StandardVisualAggregator(
                feature_aggregation=feature_aggregation,
                target_feature_dim=target_feature_dim,
            )

    @staticmethod
    def _ensure_btnd(key: str, tokens: torch.Tensor) -> None:
        if len(tokens.shape) != 3:
            raise ValueError(f"Expected BTND visual tokens for key {key}, got shape {tuple(tokens.shape)}")

    def _get_prefix_tokens(self, key: str, stream: VisualStreamFeatures) -> int:
        self._ensure_btnd(key, stream.tokens)
        count = int(stream.num_prefix_tokens)
        fallback = self.visual_prefix_tokens_by_key.get(key)
        if fallback is not None and fallback != count:
            logger.warning(
                "Prefix-token mismatch for key=%s: stream=%s, fallback=%s. Using stream value.",
                key,
                count,
                fallback,
            )
        if count < 0:
            raise ValueError(f"num_prefix_tokens must be non-negative for key {key}, got {count}")
        if count > stream.tokens.shape[1]:
            raise ValueError(
                f"num_prefix_tokens={count} exceeds token count={stream.tokens.shape[1]} for key {key}"
            )
        return count

    def _select_rgb_tokens(self, key: str, stream: VisualStreamFeatures, token_mode: str) -> torch.Tensor:
        tokens = stream.tokens
        prefix_tokens = self._get_prefix_tokens(key, stream)

        if token_mode == "all_tokens":
            return tokens

        if prefix_tokens < 1:
            raise ValueError(
                f"RGB key {key} does not expose prefix tokens (num_prefix_tokens={prefix_tokens}). "
                f"rgb_token_mode='{token_mode}' is not supported for this stream; use 'all_tokens'."
            )

        if token_mode == "patch_tokens":
            if prefix_tokens >= tokens.shape[1]:
                raise ValueError(
                    f"Requested patch_tokens for key {key}, but num_prefix_tokens={prefix_tokens} "
                    f"and total_tokens={tokens.shape[1]} (no patch tokens remain)."
                )
            return tokens[:, prefix_tokens:, :]

        if token_mode == "cls":
            return tokens[:, :1, :]

        raise ValueError(
            f"Unsupported RGB token_mode={token_mode}. Expected one of ['all_tokens', 'patch_tokens', 'cls']"
        )

    def _select_stream_tokens(self, key: str, stream: VisualStreamFeatures, rgb_token_mode: str) -> torch.Tensor:
        self._ensure_btnd(key, stream.tokens)
        if stream.modality == VisualModality.DEPTH:
            return stream.tokens
        if stream.modality == VisualModality.RGB:
            return self._select_rgb_tokens(key=key, stream=stream, token_mode=rgb_token_mode)
        raise ValueError(
            f"Unsupported visual modality '{stream.modality}' for key {key}. "
            f"Expected one of [{VisualModality.RGB}, {VisualModality.DEPTH}]"
        )

    def select_visual_features(
        self,
        visual_features: VisualFeatureSet,
        rgb_token_mode: str,
    ) -> Dict[str, torch.Tensor]:
        selected = {}
        for key, stream in visual_features.items():
            selected[key] = self._select_stream_tokens(key=key, stream=stream, rgb_token_mode=rgb_token_mode)
        return selected

    def get_concat_tokens(
        self,
        visual_features: VisualFeatureSet,
        rgb_token_mode: str = "all_tokens",
    ) -> torch.Tensor:
        """Concatenate per-key selected visual tokens.

        Args:
            visual_features: key -> VisualStreamFeatures(B*T, N, D)

        Returns:
            (B*T, N_total, D)
        """

        selected = self.select_visual_features(visual_features, rgb_token_mode=rgb_token_mode)
        return torch.cat(list(selected.values()), dim=1)

    def get_pooled_tokens(
        self,
        visual_features: VisualFeatureSet,
        rgb_token_mode: str,
    ) -> torch.Tensor:
        """Mean-pool selected per-key token streams and stack as sequence.

        Args:
            visual_features: key -> VisualStreamFeatures(B*T, N, D)

        Returns:
            (B*T, N_streams, D)
        """

        selected = self.select_visual_features(visual_features, rgb_token_mode=rgb_token_mode)
        pooled = [feat.mean(dim=1) for feat in selected.values()]
        return torch.stack(pooled, dim=1)

    def build_force_kv_tokens(
        self,
        visual_features: VisualFeatureSet,
        kv_mode: str,
    ) -> torch.Tensor:
        """Build K/V tensor for force cross-attention from full visual streams."""

        valid_kv_modes = {"mean", "cls", "all_tokens", "patch_tokens"}
        if kv_mode not in valid_kv_modes:
            raise ValueError(
                f"Unsupported force_cross_attn_kv_mode={kv_mode}. Expected one of {sorted(valid_kv_modes)}"
            )

        if kv_mode in {"all_tokens", "patch_tokens", "cls"}:
            return self.get_concat_tokens(visual_features=visual_features, rgb_token_mode=kv_mode)

        # mean mode pools each key separately after modality-aware selection.
        return self.get_pooled_tokens(
            visual_features=visual_features,
            rgb_token_mode=self.rgb_token_mode,
        )

    def aggregate_by_mode(
        self,
        visual_features: VisualFeatureSet,
        output_format: str,
        batch_size: int,
        return_by_key: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """Aggregate visual features for cat/dit adapters."""

        if output_format not in {"cat", "dit"}:
            raise ValueError(f"Unsupported output_format: {output_format}")

        if len(visual_features) == 0:
            return None, ({} if return_by_key else None)

        if self.visual_feature_mode == "raw_tokens":
            selected_by_key = (
                self.select_visual_features(visual_features, rgb_token_mode=self.rgb_token_mode)
                if return_by_key
                else None
            )
            if output_format == "dit" and selected_by_key is not None:
                return None, selected_by_key

            concat_tokens = self.get_concat_tokens(
                visual_features=visual_features,
                rgb_token_mode=self.rgb_token_mode,
            )
            all_tokens = concat_tokens.reshape(batch_size, -1, concat_tokens.shape[-1])  # (B, T*N, D)
            if output_format == "cat":
                return all_tokens.reshape(batch_size, -1), selected_by_key
            return all_tokens, selected_by_key

        if self.visual_feature_mode == "cross_image_attention":
            concat_tokens = self.get_concat_tokens(
                visual_features=visual_features,
                rgb_token_mode=self.rgb_token_mode,
            )
            pooled = self.cross_image_pool(concat_tokens)  # (B*T, D)
            pooled = pooled.reshape(batch_size, -1, pooled.shape[-1])  # (B, T, D)
            if output_format == "cat":
                return pooled.reshape(batch_size, -1), None
            return pooled, None

        if self.standard_aggregator is None:
            raise ValueError("standard_aggregator must be initialized for visual_feature_mode='standard_aggregation'")

        selected = self.select_visual_features(visual_features, rgb_token_mode=self.rgb_token_mode)
        aggregated_by_key = {}
        for key, feat in selected.items():
            if feat.shape[1] < 1:
                raise ValueError(f"Expected at least one token for standard aggregation on key {key}")
            aggregated_by_key[key] = self.standard_aggregator(feat).unsqueeze(1)  # (B*T, 1, D)

        by_key = aggregated_by_key if return_by_key else None
        if output_format == "dit" and return_by_key:
            return None, by_key

        if output_format == "cat":
            chunks = [feat.reshape(batch_size, -1) for feat in aggregated_by_key.values()]
            return torch.cat(chunks, dim=1), by_key

        visual_tokens = [feat.reshape(batch_size, -1, feat.shape[-1]) for feat in aggregated_by_key.values()]
        return torch.cat(visual_tokens, dim=1), by_key
