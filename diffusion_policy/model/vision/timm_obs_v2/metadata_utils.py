"""Token metadata utilities for TimmObsEncoderV2."""

from typing import Sequence, Tuple

import torch

from diffusion_policy.model.vision.timm_obs_v2.types import ModalityType, RangeType


def get_key_metadata(
    key: str,
    batch_size: int,
    device: torch.device,
    this_max_obs_horizon: int,
    shape_meta: dict,
    max_obs_steps: int,
    rgb_keys: Sequence[str],
    depth_keys: Sequence[str],
    low_dim_keys: Sequence[str],
    force_keys: Sequence[str],
    short_range_obs_window: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build temporal/modality/range metadata for one observation key.

    Shape legend:
    B = batch, T = horizon/time, N = tokens
    """

    modality_idx = ModalityType.NULL
    if key in rgb_keys:
        modality_idx = ModalityType.RGB
    elif key in depth_keys:
        modality_idx = ModalityType.DEPTH
    elif key in low_dim_keys:
        modality_idx = ModalityType.LOW_DIM
    elif key in force_keys:
        modality_idx = ModalityType.FORCE

    stride = int(shape_meta["obs"][key].get("down_sample_steps", 1))

    # (T,) relative steps from oldest->newest aligned to global max window.
    steps_rel = torch.arange(this_max_obs_horizon - 1, -1, -1, device=device) * stride * -1
    steps_abs = (steps_rel + (max_obs_steps - 1)).clamp(0, max_obs_steps - 1)

    pos_tensor = steps_abs.unsqueeze(0).expand(batch_size, -1)
    mod_tensor = torch.full(
        (batch_size, this_max_obs_horizon),
        modality_idx,
        dtype=torch.long,
        device=device,
    )
    range_tensor = torch.full(
        (batch_size, this_max_obs_horizon),
        RangeType.LONG,
        dtype=torch.long,
        device=device,
    )

    if short_range_obs_window is not None and (key in rgb_keys or key in depth_keys):
        pos_tensor = torch.cat([pos_tensor, pos_tensor[:, -short_range_obs_window:]], dim=1)
        mod_tensor = torch.cat([mod_tensor, mod_tensor[:, -short_range_obs_window:]], dim=1)
        short_range_tensor = torch.full(
            (batch_size, short_range_obs_window),
            RangeType.SHORT,
            dtype=torch.long,
            device=device,
        )
        range_tensor = torch.cat([range_tensor, short_range_tensor], dim=1)

    return pos_tensor, mod_tensor, range_tensor
