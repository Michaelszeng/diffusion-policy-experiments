"""
FurnitureBench state-only (lowdim) dataset.  Reads from the same zarr stores
as FurnitureBenchDataset but exposes only robot_state (and any other lowdim
obs keys) — no image tensors.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseZarrLowdimDataset


class FurnitureBenchLowdimDataset(BaseZarrLowdimDataset):
    """
    State-based dataset for FurnitureBench one-leg (and other) tasks.

    Reads ``robot_state`` (and any other lowdim obs keys declared in
    ``shape_meta``) plus ``action`` from zarr stores.  Supports co-training
    over multiple zarr files with optional per-dataset sampling weights and
    train/val splits.

    Each entry in ``zarr_configs`` is a dict with:
        path (str)
        max_train_episodes (int, optional)
        sampling_weight (float, optional) — all or none must provide this
        val_ratio (float, optional) — overrides the global val_ratio

    shape_meta example::

        action: {shape: [10]}
        obs:
          robot_state: {shape: [16], type: low_dim}
    """

    def __init__(
        self,
        zarr_configs: List[Dict],
        shape_meta: Dict,
        horizon: int = 1,
        n_obs_steps: Optional[int] = None,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
    ):
        super().__init__(
            zarr_configs=zarr_configs,
            shape_meta=shape_meta,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=val_ratio,
        )

    def _get_buffer_keys(self) -> List[str]:
        print(self.lowdim_keys)
        return list(self.lowdim_keys) + ["action"]

    def _lowdim_key_map(self) -> Dict[str, str]:
        # identity map — zarr key names match the model key names
        return {"action": "action", **{k: k for k in self.lowdim_keys}}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.num_datasets == 1:
            data = self.samplers[0].sample_data(idx)
        else:
            i = int(np.random.choice(self.num_datasets, p=self.sample_probabilities))
            data = self.samplers[i].sample_data(idx % len(self.samplers[i]))

        return dict_apply(data, torch.from_numpy)
