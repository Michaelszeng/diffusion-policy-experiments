"""Per-stream standard visual aggregation utilities."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool2d(nn.Module):
    """Attention pooling for 2D feature maps."""

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: Optional[int] = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # (H*W, B, C)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (H*W+1, B, C)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)  # (B, D)


class AttentionPool1d(nn.Module):
    """Attention pooling for token sequences."""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) / embed_dim**0.5)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, D)
        out, _ = self.attn(query, x, x)
        return out.squeeze(1)  # (B, D)


class StandardVisualAggregator(nn.Module):
    """Token-space visual aggregator for standard_aggregation mode.

    Input shape: (B*T, N, D)
    Output shape: (B*T, D)
    """

    def __init__(
        self,
        feature_aggregation: Optional[str],
        target_feature_dim: int,
    ):
        super().__init__()
        if feature_aggregation is None:
            raise ValueError("visual_feature_mode='standard_aggregation' requires feature_aggregation")
        if isinstance(feature_aggregation, str) and feature_aggregation.strip().lower() in {"", "null"}:
            raise ValueError("feature_aggregation must be non-empty for standard_aggregation")

        valid_aggregations = {"avg", "attention_pool"}
        if feature_aggregation not in valid_aggregations:
            raise ValueError(
                f"Unsupported feature_aggregation={feature_aggregation}. "
                f"Expected one of {sorted(valid_aggregations)}"
            )

        self.feature_aggregation = feature_aggregation
        if self.feature_aggregation == "attention_pool":
            self.attention_pool_1d = AttentionPool1d(embed_dim=target_feature_dim, num_heads=8)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Aggregate token streams into one embedding per stream.

        Args:
            feature: (B*T, N, D)
        Returns:
            (B*T, D)
        """

        if len(feature.shape) != 3:
            raise ValueError(
                f"StandardVisualAggregator expects rank-3 token input (B*T,N,D), got {tuple(feature.shape)}"
            )

        if self.feature_aggregation == "avg":
            return torch.mean(feature, dim=1)
        return self.attention_pool_1d(feature)
