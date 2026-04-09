"""
FurnitureBench dataset. Reads from zarr stores produced by imitation-juicer,
after first translating them to the standard ReplayBuffer layout with
translate_furniture_bench.py.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseZarrImageDataset


class FurnitureBenchDataset(BaseZarrImageDataset):
    """
    Image + delta-action dataset for FurnitureBench. Supports multiple zarr
    datasets, co-training with sampling weights, per-dataset train/val splits,

    Each entry in zarr_configs is a dict with:
        path (str)
        max_train_episodes (int, optional)
        sampling_weight (float, optional) — all or none must provide this
        val_ratio (float, optional) — overrides the global val_ratio

    shape_meta example:
        action: {shape: [10]}
        obs:
          color_image1: {type: rgb, shape: [3, 240, 320]}
          color_image2: {type: rgb, shape: [3, 240, 320]}
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
        color_jitter: Optional[Dict] = None,
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
            color_jitter=color_jitter,
        )

    def _get_buffer_keys(self) -> List[str]:
        return self.rgb_keys + self.lowdim_keys + ["action"]

    def _lowdim_key_map(self) -> Dict[str, str]:
        return {"action": "action", **{k: k for k in self.lowdim_keys}}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.num_datasets == 1:
            data = self.samplers[0].sample_data(idx)
        else:
            i = int(np.random.choice(self.num_datasets, p=self.sample_probabilities))
            data = self.samplers[i].sample_data(idx % len(self.samplers[i]))

        if self.transforms is not None and self.rgb_keys:
            data = self._apply_color_jitter(data)

        return dict_apply(data, torch.from_numpy)
