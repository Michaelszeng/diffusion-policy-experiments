"""Force modality module for TimmObsEncoderV2."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from diffusion_policy.model.vision.timm_obs_v2.types import VisualFeatureSet, VisualStreamFeatures


class ForceModule(nn.Module):
    """Process force observations and optionally fuse with visual context.

    Shape legend:
    B = batch, T = horizon/time, N = tokens, D = embedding dim
    """

    def __init__(
        self,
        force_fusion_mode: Optional[str],
        force_keys,
        shape_meta: dict,
        target_feature_dim: int,
        force_cross_attn_heads: int,
        force_cross_attn_kv_mode: str,
        force_modality_dropout_p: float,
        force_proj_dropout_p: float,
    ):
        super().__init__()

        self.force_fusion_mode = force_fusion_mode
        self.force_keys = list(force_keys)
        self.force_cross_attn_kv_mode = force_cross_attn_kv_mode
        self.force_modality_dropout_p = float(force_modality_dropout_p)

        self.force_proj = nn.ModuleDict()
        self.force_cross_attn = None

        if self.force_fusion_mode is not None and len(self.force_keys) > 0:
            for key in self.force_keys:
                input_dim = int(shape_meta["obs"][key]["shape"][0])
                self.force_proj[key] = nn.Sequential(
                    nn.Dropout(p=force_proj_dropout_p),
                    nn.Linear(input_dim, target_feature_dim),
                )

            if self.force_fusion_mode == "cross_attention":
                self.force_cross_attn = nn.MultiheadAttention(
                    embed_dim=target_feature_dim,
                    num_heads=force_cross_attn_heads,
                    batch_first=True,
                )

    def encode(
        self,
        obs_dict: Dict[str, torch.Tensor],
        key_shape_map: Dict[str, tuple],
        visual_features_by_key: VisualFeatureSet,
        batch_size: int,
        short_range_obs_window: Optional[int],
        visual_fusion,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Encode force streams and optionally cross-attend over visual context.

        Returns:
            key -> (B*T, 1, D)
        """

        if self.force_fusion_mode is None or len(self.force_keys) == 0:
            return {}
        kv_mode = self.force_cross_attn_kv_mode
        concat_token_modes = {"all_tokens", "patch_tokens"}
        pooled_token_modes = {"mean", "cls"}
        if kv_mode not in concat_token_modes and kv_mode not in pooled_token_modes:
            raise ValueError(
                f"Unsupported force_cross_attn_kv_mode={kv_mode}. "
                "Expected one of ['mean', 'cls', 'all_tokens', 'patch_tokens']"
            )

        force_modality_keep_mask = None
        if self.training and self.force_modality_dropout_p > 0.0:
            # (B, 1, 1): one keep/drop decision per sample across force streams.
            force_modality_keep_mask = (
                torch.rand(batch_size, 1, 1, device=device) >= self.force_modality_dropout_p
            )

        force_embeds = []
        force_keep_mask_flat = {}
        force_horizon = None

        for key in self.force_keys:
            data = obs_dict[key]  # (B, T, D_force)
            batch, horizon = data.shape[:2]
            if batch != batch_size:
                raise ValueError(f"Force key {key} has batch={batch}, expected {batch_size}")
            if tuple(data.shape[2:]) != tuple(key_shape_map[key]):
                raise ValueError(
                    f"Force key {key} has shape {tuple(data.shape[2:])}, expected {key_shape_map[key]}"
                )

            if force_horizon is None:
                force_horizon = horizon

            if force_modality_keep_mask is not None:
                force_keep_mask_flat[key] = force_modality_keep_mask.expand(-1, horizon, -1).reshape(batch_size * horizon, 1)

            data_flat = data.reshape(batch_size * horizon, -1)  # (B*T, D_force)
            force_embeds.append(self.force_proj[key](data_flat))  # (B*T, D)

        force_features_2d_by_key = {}

        if self.force_fusion_mode == "concat":
            for i, key in enumerate(self.force_keys):
                force_features_2d_by_key[key] = force_embeds[i]

        elif self.force_fusion_mode == "cross_attention":
            if self.force_cross_attn is None:
                raise ValueError("force_cross_attn must be initialized when force_fusion_mode='cross_attention'")

            if short_range_obs_window is not None:
                short_window = short_range_obs_window
                if short_window > force_horizon:
                    raise ValueError(
                        f"short_range_obs_window={short_window} exceeds force horizon={force_horizon}. "
                        "Force short-range cross-attention requires short_window <= horizon."
                    )
                visual_kv_long = {}
                visual_kv_short = {}
                for key, stream in visual_features_by_key.items():
                    feat = stream.tokens
                    if (feat.shape[0] % batch_size) != 0:
                        raise ValueError(
                            f"Visual key {key} has invalid first dim={feat.shape[0]} for batch_size={batch_size}"
                        )
                    steps_total = feat.shape[0] // batch_size
                    if steps_total < force_horizon:
                        raise ValueError(
                            f"Visual key {key} has steps={steps_total}, but force horizon is {force_horizon}"
                        )
                    if steps_total < (force_horizon + short_window):
                        raise ValueError(
                            f"Visual key {key} has steps={steps_total}, but force short-range path requires "
                            f"{force_horizon + short_window} steps."
                        )

                    feat_btnd = feat.reshape(batch_size, steps_total, feat.shape[1], feat.shape[2])
                    long_tokens = feat_btnd[:, :force_horizon].reshape(batch_size * force_horizon, feat.shape[1], feat.shape[2])
                    short_tokens = feat_btnd[
                        :,
                        force_horizon : force_horizon + short_window,
                    ].reshape(batch_size * short_window, feat.shape[1], feat.shape[2])

                    visual_kv_long[key] = VisualStreamFeatures(
                        tokens=long_tokens,
                        modality=stream.modality,
                        num_prefix_tokens=stream.num_prefix_tokens,
                    )
                    visual_kv_short[key] = VisualStreamFeatures(
                        tokens=short_tokens,
                        modality=stream.modality,
                        num_prefix_tokens=stream.num_prefix_tokens,
                    )

                kv_long = visual_fusion.build_force_kv_tokens(
                    visual_features=visual_kv_long,
                    kv_mode=kv_mode,
                )
                kv_short = visual_fusion.build_force_kv_tokens(
                    visual_features=visual_kv_short,
                    kv_mode=kv_mode,
                )

                force_stack = torch.stack(force_embeds, dim=1)  # (B*T, N_force, D)
                out_long, _ = self.force_cross_attn(force_stack, kv_long, kv_long)

                force_stack_short = force_stack.view(batch_size, force_horizon, *force_stack.shape[1:])[
                    :,
                    -short_window:,
                ].flatten(0, 1)
                out_short, _ = self.force_cross_attn(force_stack_short, kv_short, kv_short)

                out_long = out_long.view(batch_size, force_horizon, *out_long.shape[1:])
                out_long[:, -short_window:] += out_short.view(batch_size, short_window, *out_short.shape[1:])
                force_attended = out_long.flatten(0, 1)  # (B*T, N_force, D)
            else:
                visual_kv = visual_fusion.build_force_kv_tokens(
                    visual_features=visual_features_by_key,
                    kv_mode=kv_mode,
                )

                force_stack = torch.stack(force_embeds, dim=1)  # (B*T, N_force, D)
                force_attended, _ = self.force_cross_attn(
                    query=force_stack,
                    key=visual_kv,
                    value=visual_kv,
                )

            for i, key in enumerate(self.force_keys):
                force_features_2d_by_key[key] = force_attended[:, i, :]

        else:
            raise ValueError(f"Unsupported force_fusion_mode: {self.force_fusion_mode}")

        if force_modality_keep_mask is not None:
            for key in self.force_keys:
                if key in force_keep_mask_flat:
                    force_features_2d_by_key[key] = force_features_2d_by_key[key] * force_keep_mask_flat[key].to(
                        dtype=force_features_2d_by_key[key].dtype,
                        device=force_features_2d_by_key[key].device,
                    )

        force_features_by_key = {
            key: force_features_2d_by_key[key].unsqueeze(1)
            for key in self.force_keys
            if key in force_features_2d_by_key
        }
        return force_features_by_key
