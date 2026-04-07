"""Low-dimensional modality module for TimmObsEncoderV2."""

from typing import Dict, List

import torch
import torch.nn as nn


class LowDimModule(nn.Module):
    """Process low-dimensional observations for cat/dit adapters.

    Shape legend:
    B = batch, T = horizon/time, N = tokens, D = embedding dim
    """

    def __init__(
        self,
        low_dim_keys: List[str],
        shape_meta: dict,
        target_feature_dim: int,
        lowdim_dropout_p: float,
        project: bool,
    ):
        super().__init__()
        self.low_dim_keys = list(low_dim_keys)
        self.project = bool(project)
        self.lowdim_proj = nn.ModuleDict()

        for key in self.low_dim_keys:
            input_dim = int(shape_meta["obs"][key]["shape"][0])
            projection = nn.Identity()
            if input_dim != target_feature_dim:
                projection = nn.Linear(input_dim, target_feature_dim)
            self.lowdim_proj[key] = nn.Sequential(
                nn.Dropout(p=lowdim_dropout_p),
                projection,
            )

    def encode(self, obs_dict: Dict[str, torch.Tensor], key_shape_map: Dict[str, tuple]) -> Dict[str, torch.Tensor]:
        """Return lowdim features for shared BTND core.

        Returns:
            key -> (B*T, 1, D_raw) when project=False
            key -> (B*T, 1, D) when project=True
        """

        out = {}
        for key in self.low_dim_keys:
            data = obs_dict[key]  # (B, T, D_raw)
            if tuple(data.shape[2:]) != tuple(key_shape_map[key]):
                raise ValueError(
                    f"Low-dim key {key} has shape {tuple(data.shape[2:])}, expected {key_shape_map[key]}"
                )
            batch_size, horizon = data.shape[:2]
            feat = data.reshape(batch_size * horizon, 1, -1)
            if self.project:
                feat = self.lowdim_proj[key](feat)
            out[key] = feat
        return out
