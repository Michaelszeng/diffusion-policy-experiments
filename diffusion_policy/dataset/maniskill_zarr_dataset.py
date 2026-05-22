"""
Image dataset for the ManiSkill zarr format.

Expected zarr layout (per maniskill_planar_push_t.zarr):
    data/
        action            (T, 2)             float64   # 2D ee delta pose
        state             (T, 3)             float64   # robot ee state
        target            (T, 3)             float64   # goal T pose
        slider_state      (T, 3)             float64   # T (slider) pose
        base_camera       (T, 128, 128, 3)   uint8
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

from typing import Dict, List, Optional

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

    ``downsample_steps`` subsamples the raw sequence before returning it,
    allowing a policy trained at a lower control frequency (e.g. 10 Hz) to
    consume data recorded at a higher rate (e.g. 30 Hz).  Set it to the
    integer ratio between the recording Hz and the target policy Hz.
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
        downsample_steps: int = 1,
    ):
        self.downsample_steps = downsample_steps
        super().__init__(
            zarr_configs=zarr_configs,
            shape_meta=shape_meta,
            horizon=horizon * downsample_steps,
            n_obs_steps=n_obs_steps * downsample_steps if n_obs_steps is not None else None,
            pad_before=pad_before * downsample_steps,
            pad_after=pad_after * downsample_steps,
            seed=seed,
            val_ratio=val_ratio,
            color_jitter=color_jitter,
        )

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

        if self.downsample_steps > 1:
            data["action"] = data["action"][:: self.downsample_steps]
            for k in data["obs"]:
                data["obs"][k] = data["obs"][k][:: self.downsample_steps]

        if self.transforms is not None and self.rgb_keys:
            data = self._apply_color_jitter(data)

        return dict_apply(data, torch.from_numpy)