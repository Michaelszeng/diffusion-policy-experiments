import logging
import math
from typing import Optional, Union

import einops
import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.conv1d_components import (
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for fixed-length observation token sequences.
    """

    def __init__(self, embed_dim: int, max_position: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_position = max_position

        position = torch.arange(max_position).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-(math.log(10000.0) / embed_dim)))

        pe = torch.zeros(max_position, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if positions.shape[:2] != x.shape[:2]:
            raise ValueError(f"positions shape {positions.shape} must match token shape {x.shape[:2]}")
        pos_embed = self.pe[positions.long()].to(dtype=x.dtype, device=x.device)
        if token_mask is not None:
            pos_embed = pos_embed * token_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x + pos_embed


class CrossAttentionConditioning(nn.Module):
    """
    Cross-attention block where trajectory features attend to conditioning tokens.
    """

    def __init__(
        self,
        embed_dim: int,
        cond_token_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_temporal_pos_emb: bool = True,
        max_temporal_position: int = 1000,
        use_modality_emb: bool = True,
        max_modalities: int = 8,
        use_range_emb: bool = True,
        max_ranges: int = 3,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.cond_proj = nn.Linear(cond_token_dim, embed_dim)
        self.embed_scale = math.sqrt(embed_dim)
        self.use_temporal_pos_emb = use_temporal_pos_emb
        self.use_modality_emb = use_modality_emb
        self.use_range_emb = use_range_emb

        if self.use_temporal_pos_emb:
            self.temporal_pos_emb = TemporalPositionalEncoding(
                embed_dim=embed_dim,
                max_position=max_temporal_position,
            )
        if self.use_modality_emb:
            self.modality_emb = nn.Embedding(max_modalities, embed_dim)
        if self.use_range_emb:
            # 0: NULL, 1: LONG, 2: SHORT
            self.range_emb = nn.Embedding(max_ranges, embed_dim)

        self.query_norm = nn.LayerNorm(embed_dim)
        self.cond_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        temporal_positions: Optional[torch.Tensor] = None,
        modality_indices: Optional[torch.Tensor] = None,
        range_indices: Optional[torch.Tensor] = None,
        obs_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, T, C), cond_tokens: (B, N, D_cond)
        cond = self.cond_proj(cond_tokens)
        cond = cond * self.embed_scale

        if self.use_modality_emb:
            if modality_indices is None:
                raise ValueError("modality_indices required when use_modality_emb=True")
            cond = cond + self.modality_emb(modality_indices).to(dtype=cond.dtype)

        if self.use_range_emb:
            if range_indices is None:
                raise ValueError("range_indices required when use_range_emb=True")
            cond = cond + self.range_emb(range_indices).to(dtype=cond.dtype)

        if self.use_temporal_pos_emb:
            if temporal_positions is None:
                raise ValueError("temporal_positions required when use_temporal_pos_emb=True")
            cond = self.temporal_pos_emb(
                x=cond,
                positions=temporal_positions,
                token_mask=obs_token_mask,
            )

        attn_out, _ = self.cross_attn(
            query=self.query_norm(x),
            key=self.cond_norm(cond),
            value=self.cond_norm(cond),
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class AttentionConditionalResidualBlock1D(nn.Module):
    """
    Residual Conv1D block with cross-attention conditioning between the conv blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_token_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        use_temporal_pos_emb: bool = True,
        max_temporal_position: int = 1000,
        use_modality_emb: bool = True,
        max_modalities: int = 8,
        use_range_emb: bool = True,
        max_ranges: int = 3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.cross_attention = CrossAttentionConditioning(
            embed_dim=out_channels,
            cond_token_dim=cond_token_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            use_temporal_pos_emb=use_temporal_pos_emb,
            max_temporal_position=max_temporal_position,
            use_modality_emb=use_modality_emb,
            max_modalities=max_modalities,
            use_range_emb=use_range_emb,
            max_ranges=max_ranges,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        temporal_positions: Optional[torch.Tensor] = None,
        modality_indices: Optional[torch.Tensor] = None,
        range_indices: Optional[torch.Tensor] = None,
        obs_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.blocks[0](x)
        out = self.cross_attention(
            x=out.transpose(1, 2).contiguous(),
            cond_tokens=cond_tokens,
            temporal_positions=temporal_positions,
            modality_indices=modality_indices,
            range_indices=range_indices,
            obs_token_mask=obs_token_mask,
        )
        out = out.transpose(1, 2).contiguous()
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class StaticAttentionConditionalUnet1D(nn.Module):
    """
    1D U-Net with attention conditioning for fixed, static observation horizons.
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        local_cond_dim: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: list = [256, 512, 1024],
        kernel_size: int = 3,
        n_groups: int = 8,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        use_temporal_pos_emb: bool = True,
        max_temporal_position: int = 1000,
        use_modality_emb: bool = True,
        max_modalities: int = 8,
        use_range_emb: bool = True,
        max_ranges: int = 3,
    ):
        super().__init__()

        self.global_cond_dim = global_cond_dim
        self.use_temporal_pos_emb = use_temporal_pos_emb
        self.use_modality_emb = use_modality_emb
        self.use_range_emb = use_range_emb

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dim=diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        self.timestep_to_token = nn.Linear(diffusion_step_embed_dim, global_cond_dim)

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            local_cond_encoder = nn.ModuleList(
                [
                    AttentionConditionalResidualBlock1D(
                        local_cond_dim,
                        dim_out,
                        cond_token_dim=global_cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        num_attention_heads=num_attention_heads,
                        attention_dropout=attention_dropout,
                        use_temporal_pos_emb=use_temporal_pos_emb,
                        max_temporal_position=max_temporal_position,
                        use_modality_emb=use_modality_emb,
                        max_modalities=max_modalities,
                        use_range_emb=use_range_emb,
                        max_ranges=max_ranges,
                    ),
                    AttentionConditionalResidualBlock1D(
                        local_cond_dim,
                        dim_out,
                        cond_token_dim=global_cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        num_attention_heads=num_attention_heads,
                        attention_dropout=attention_dropout,
                        use_temporal_pos_emb=use_temporal_pos_emb,
                        max_temporal_position=max_temporal_position,
                        use_modality_emb=use_modality_emb,
                        max_modalities=max_modalities,
                        use_range_emb=use_range_emb,
                        max_ranges=max_ranges,
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                AttentionConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_token_dim=global_cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    num_attention_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                    use_temporal_pos_emb=use_temporal_pos_emb,
                    max_temporal_position=max_temporal_position,
                    use_modality_emb=use_modality_emb,
                    max_modalities=max_modalities,
                    use_range_emb=use_range_emb,
                    max_ranges=max_ranges,
                ),
                AttentionConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_token_dim=global_cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    num_attention_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                    use_temporal_pos_emb=use_temporal_pos_emb,
                    max_temporal_position=max_temporal_position,
                    use_modality_emb=use_modality_emb,
                    max_modalities=max_modalities,
                    use_range_emb=use_range_emb,
                    max_ranges=max_ranges,
                ),
            ]
        )

        down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        AttentionConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_token_dim=global_cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            num_attention_heads=num_attention_heads,
                            attention_dropout=attention_dropout,
                            use_temporal_pos_emb=use_temporal_pos_emb,
                            max_temporal_position=max_temporal_position,
                            use_modality_emb=use_modality_emb,
                            max_modalities=max_modalities,
                            use_range_emb=use_range_emb,
                            max_ranges=max_ranges,
                        ),
                        AttentionConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_token_dim=global_cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            num_attention_heads=num_attention_heads,
                            attention_dropout=attention_dropout,
                            use_temporal_pos_emb=use_temporal_pos_emb,
                            max_temporal_position=max_temporal_position,
                            use_modality_emb=use_modality_emb,
                            max_modalities=max_modalities,
                            use_range_emb=use_range_emb,
                            max_ranges=max_ranges,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        AttentionConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_token_dim=global_cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            num_attention_heads=num_attention_heads,
                            attention_dropout=attention_dropout,
                            use_temporal_pos_emb=use_temporal_pos_emb,
                            max_temporal_position=max_temporal_position,
                            use_modality_emb=use_modality_emb,
                            max_modalities=max_modalities,
                            use_range_emb=use_range_emb,
                            max_ranges=max_ranges,
                        ),
                        AttentionConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_token_dim=global_cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            num_attention_heads=num_attention_heads,
                            attention_dropout=attention_dropout,
                            use_temporal_pos_emb=use_temporal_pos_emb,
                            max_temporal_position=max_temporal_position,
                            use_modality_emb=use_modality_emb,
                            max_modalities=max_modalities,
                            use_range_emb=use_range_emb,
                            max_ranges=max_ranges,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules

        logger.info(
            "StaticAttentionConditionalUnet1D parameters: %e",
            sum(p.numel() for p in self.parameters()),
        )

    def _expand_timesteps(
        self,
        timestep: Union[torch.Tensor, float, int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(device)
        else:
            timesteps = timesteps.to(device)
        return timesteps.expand(batch_size)

    def _prepare_cond_tokens(
        self,
        timestep: Union[torch.Tensor, float, int],
        global_cond: Optional[torch.Tensor],
        temporal_positions: Optional[torch.Tensor],
        modality_indices: Optional[torch.Tensor],
        range_indices: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> tuple:
        # Observation tokens and metadata
        if global_cond is None:
            model_dtype = self.timestep_to_token.weight.dtype
            obs_tokens = torch.zeros(batch_size, 0, self.global_cond_dim, device=device, dtype=model_dtype)
            obs_temporal_positions = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            obs_modality_indices = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            obs_range_indices = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        else:
            if global_cond.ndim == 2:
                global_cond = global_cond.unsqueeze(1)
            if global_cond.ndim != 3:
                raise ValueError(f"global_cond must have shape (B, N, D) or (B, D), got {global_cond.shape}")
            if global_cond.shape[0] != batch_size:
                raise ValueError(f"global_cond batch {global_cond.shape[0]} != sample batch {batch_size}")
            if global_cond.shape[-1] != self.global_cond_dim:
                raise ValueError(f"global_cond dim {global_cond.shape[-1]} != expected {self.global_cond_dim}")
            obs_tokens = global_cond
            n_obs_tokens = obs_tokens.shape[1]

            if self.use_temporal_pos_emb:
                if temporal_positions is None:
                    raise ValueError("temporal_positions is required when use_temporal_pos_emb=True")
                obs_temporal_positions = temporal_positions
            else:
                obs_temporal_positions = torch.zeros(batch_size, n_obs_tokens, dtype=torch.long, device=device)
            if self.use_modality_emb:
                if modality_indices is None:
                    raise ValueError("modality_indices is required when use_modality_emb=True")
                obs_modality_indices = modality_indices
            else:
                obs_modality_indices = torch.zeros(batch_size, n_obs_tokens, dtype=torch.long, device=device)
            if self.use_range_emb:
                if range_indices is None:
                    raise ValueError("range_indices is required when use_range_emb=True")
                obs_range_indices = range_indices
            else:
                obs_range_indices = torch.zeros(batch_size, n_obs_tokens, dtype=torch.long, device=device)

        # Diffusion timestep token
        timesteps = self._expand_timesteps(timestep, batch_size, device)
        timestep_embed = self.diffusion_step_encoder(timesteps)
        timestep_token = self.timestep_to_token(timestep_embed).unsqueeze(1)
        timestep_zeros = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        timestep_obs_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        # Keep token ordering consistent with conditional_unet1d_attention:
        # [timestep_token, obs_tokens...]
        cond_tokens = torch.cat([timestep_token, obs_tokens], dim=1)
        temporal_positions = torch.cat([timestep_zeros, obs_temporal_positions], dim=1)
        modality_indices = torch.cat([timestep_zeros, obs_modality_indices], dim=1)
        range_indices = torch.cat([timestep_zeros, obs_range_indices], dim=1)
        obs_token_mask = torch.cat(
            [
                timestep_obs_mask,
                torch.ones(batch_size, obs_tokens.shape[1], dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        return cond_tokens, temporal_positions, modality_indices, range_indices, obs_token_mask

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        temporal_positions: Optional[torch.Tensor] = None,
        modality_indices: Optional[torch.Tensor] = None,
        range_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            sample: (B, T, input_dim)
            timestep: (B,) or scalar diffusion step
            global_cond: (B, N, global_cond_dim) observation tokens
            temporal_positions: (B, N) temporal indices for each observation token
            modality_indices: (B, N) modality indices for each observation token
            range_indices: (B, N) short/long range indices for each observation token
        Returns:
            (B, T, input_dim)
        """
        sample = einops.rearrange(sample, "b h t -> b t h").contiguous()
        batch_size = sample.shape[0]

        (
            cond_tokens,
            cond_temporal_positions,
            cond_modality_indices,
            cond_range_indices,
            cond_obs_token_mask,
        ) = self._prepare_cond_tokens(
            timestep=timestep,
            global_cond=global_cond,
            temporal_positions=temporal_positions,
            modality_indices=modality_indices,
            range_indices=range_indices,
            batch_size=batch_size,
            device=sample.device,
        )
        cond_tokens = cond_tokens.to(dtype=sample.dtype, device=sample.device)

        h_local = []
        if local_cond is not None and self.local_cond_encoder is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h").contiguous()
            resnet, resnet2 = self.local_cond_encoder
            x_local = resnet(
                local_cond,
                cond_tokens,
                temporal_positions=cond_temporal_positions,
                modality_indices=cond_modality_indices,
                range_indices=cond_range_indices,
                obs_token_mask=cond_obs_token_mask,
            )
            h_local.append(x_local)
            x_local = resnet2(
                local_cond,
                cond_tokens,
                temporal_positions=cond_temporal_positions,
                modality_indices=cond_modality_indices,
                range_indices=cond_range_indices,
                obs_token_mask=cond_obs_token_mask,
            )
            h_local.append(x_local)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(
                x,
                cond_tokens,
                temporal_positions=cond_temporal_positions,
                modality_indices=cond_modality_indices,
                range_indices=cond_range_indices,
                obs_token_mask=cond_obs_token_mask,
            )
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(
                x,
                cond_tokens,
                temporal_positions=cond_temporal_positions,
                modality_indices=cond_modality_indices,
                range_indices=cond_range_indices,
                obs_token_mask=cond_obs_token_mask,
            )
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(
                x,
                cond_tokens,
                temporal_positions=cond_temporal_positions,
                modality_indices=cond_modality_indices,
                range_indices=cond_range_indices,
                obs_token_mask=cond_obs_token_mask,
            )

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(
                x,
                cond_tokens,
                temporal_positions=cond_temporal_positions,
                modality_indices=cond_modality_indices,
                range_indices=cond_range_indices,
                obs_token_mask=cond_obs_token_mask,
            )
            if idx == (len(self.up_modules) - 1) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(
                x,
                cond_tokens,
                temporal_positions=cond_temporal_positions,
                modality_indices=cond_modality_indices,
                range_indices=cond_range_indices,
                obs_token_mask=cond_obs_token_mask,
            )
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t").contiguous()
        return x
