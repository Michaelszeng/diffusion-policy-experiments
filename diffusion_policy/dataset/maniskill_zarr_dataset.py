"""
Image dataset for the ManiSkill zarr format.

Expected zarr layout (per maniskill_planar_push_t.zarr):
    data/
        action            (T, 2)             float64   # 2D ee delta pose
        state             (T, 3)             float64   # robot ee state
        target            (T, 3)             float64   # goal T pose
        slider_state      (T, 3)             float64   # T (slider) pose
        overhead_camera   (T, 128, 128, 3)   uint8
        wrist_camera      (T, 128, 128, 3)   uint8
    meta/
        episode_ends      (E,)               int64

Routing (handled by ImprovedDatasetSampler):
    * zarr "state"      -> obs["agent_pos"]
    * rgb keys          -> obs[key]            (uint8)
    * other obs keys    -> obs[key]            (float32, e.g. "slider_state")
    * zarr "target"     -> top-level "target"  (single frame; for goal conditioning)
    * "action"          -> top-level "action"
"""

from typing import Dict, List

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseZarrImageDataset


class ManiskillZarrDataset(BaseZarrImageDataset):
    """
    Minimal zarr-backed dataset for ManiSkill data.

    Loads the standard ReplayBuffer zarr layout (``data/<key>`` arrays +
    ``meta/episode_ends``). Observation keys are determined entirely by
    ``shape_meta.obs``; this class only fixes the action key and the
    auxiliary ``target`` key used for goal conditioning.
    """

    def _get_buffer_keys(self) -> List[str]:
        # rgb_keys + lowdim_keys come from shape_meta.obs.
        # "state" is the buffer key behind the model-side "agent_pos";
        # add it explicitly when (and only when) the user asks for
        # "agent_pos" in shape_meta.
        keys = list(self.rgb_keys) + list(self.lowdim_keys) + ["action", "target"]
        if "agent_pos" in self.lowdim_keys:
            keys.remove("agent_pos")
            keys.append("state")
        # dedup, preserve order
        seen = set()
        return [k for k in keys if not (k in seen or seen.add(k))]

    def _lowdim_key_map(self) -> Dict[str, str]:
        # Maps model-side normalizer keys to the buffer keys actually present.
        m: Dict[str, str] = {"action": "action", "target": "target"}
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
