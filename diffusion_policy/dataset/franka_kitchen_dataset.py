"""
Image dataset for the Franka Kitchen (relay-policy-learning) zarr format.

Expected zarr layout (per kitchen_demos.zarr):
    data/
        action            (T, 9)              float64   # 7 arm joints + 2 gripper
        state             (T, 60)             float64   # robot + env state
        scene             (T, 240, 320, 3)    uint8     # external scene camera
        wrist             (T, 240, 320, 3)    uint8     # wrist camera
    meta/
        episode_ends      (E,)                int64

Routing (handled by ImprovedDatasetSampler):
    * zarr "state"   -> obs["agent_pos"]
    * rgb keys       -> obs[key]            (uint8)
    * other obs keys -> obs[key]            (float32)
    * "action"       -> top-level "action"

Unlike ManiskillZarrDataset, there is no "target" key (no goal conditioning).
"""

from typing import Dict, List

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseZarrImageDataset


class FrankaKitchenDataset(BaseZarrImageDataset):
    """
    Minimal zarr-backed dataset for Franka Kitchen data.

    Loads the standard ReplayBuffer zarr layout (``data/<key>`` arrays +
    ``meta/episode_ends``). Observation keys are determined entirely by
    ``shape_meta.obs``; this class only fixes the action key.
    """

    def _get_buffer_keys(self) -> List[str]:
        # rgb_keys + lowdim_keys come from shape_meta.obs.
        # "state" is the buffer key behind the model-side "agent_pos";
        # add it explicitly when (and only when) the user asks for
        # "agent_pos" in shape_meta.
        keys = list(self.rgb_keys) + list(self.lowdim_keys) + ["action"]
        if "agent_pos" in self.lowdim_keys:
            keys.remove("agent_pos")
            keys.append("state")
        # dedup, preserve order
        seen = set()
        return [k for k in keys if not (k in seen or seen.add(k))]

    def _lowdim_key_map(self) -> Dict[str, str]:
        # Maps model-side normalizer keys to the buffer keys actually present.
        m: Dict[str, str] = {"action": "action"}
        for k in self.lowdim_keys:
            m[k] = "state" if k == "agent_pos" else k
        return m

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.num_datasets == 1:
            data = self.samplers[0].sample_data(idx)
        else:
            i = int(np.random.choice(self.num_datasets, p=self.sample_probabilities))
            data = self.samplers[i].sample_data(idx % len(self.samplers[i]))

        if self.transforms is not None and self.rgb_keys:
            data = self._apply_color_jitter(data)

        return dict_apply(data, torch.from_numpy)
