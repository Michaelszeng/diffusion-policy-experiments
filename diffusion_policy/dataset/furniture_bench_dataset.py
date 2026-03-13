"""
FurnitureBench dataset. Reads from zarr stores produced by imitation-juicer:
    <root>/processed/sim/<furniture>/<split>/<quality>/success.zarr

Exposes delta actions and camera images only; all other data is privileged.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import ImprovedDatasetSampler
from diffusion_policy.dataset.base_dataset import BaseZarrImageDataset

# FurnitureBench zarr stores use "action/delta" for the action key and keep
# episode_ends at the root rather than under meta/.  This function re-maps
# those paths into the data/meta layout that ReplayBuffer expects.
_ZARR_KEY_MAP = {"action": "action/delta"}


def _open_as_replay_buffer(zarr_path: str, keys: List[str], load_into_memory: bool) -> ReplayBuffer:
    group = zarr.open(os.path.expanduser(zarr_path), mode="r")
    episode_ends = group["episode_ends"][:].astype(np.int64)
    if load_into_memory:
        data = {key: group[_ZARR_KEY_MAP.get(key, key)][:] for key in keys}
    else:
        data = {key: group[_ZARR_KEY_MAP.get(key, key)] for key in keys}
    return ReplayBuffer(root={"data": data, "meta": {"episode_ends": episode_ends}})


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
        load_into_memory: bool = False,
    ):
        super().__init__()
        self._validate_zarr_configs(zarr_configs)

        obs_meta = shape_meta["obs"]
        self.rgb_keys = [k for k, v in obs_meta.items() if v.get("type") == "rgb"]
        self.lowdim_keys = [k for k, v in obs_meta.items() if v.get("type") != "rgb"]
        zarr_keys = self.rgb_keys + self.lowdim_keys + ["action"]
        key_first_k = self._build_key_first_k(zarr_keys, n_obs_steps, horizon)

        self.num_datasets = len(zarr_configs)
        self.replay_buffers: List[ReplayBuffer] = []
        self.train_masks: List[np.ndarray] = []
        self.val_masks: List[np.ndarray] = []
        self.samplers: List[ImprovedDatasetSampler] = []
        self.sample_probabilities = np.zeros(len(zarr_configs))
        self.zarr_paths: List[str] = []

        for i, cfg in enumerate(zarr_configs):
            zarr_path = os.path.expanduser(cfg["path"])
            buf = _open_as_replay_buffer(zarr_path, zarr_keys, load_into_memory)
            train_mask, val_mask = self._build_episode_masks(
                n_episodes=buf.n_episodes,
                val_ratio=cfg.get("val_ratio", val_ratio),
                max_train_episodes=cfg.get("max_train_episodes", None),
                seed=seed,
            )
            self.replay_buffers.append(buf)
            self.zarr_paths.append(zarr_path)
            self.train_masks.append(train_mask)
            self.val_masks.append(val_mask)
            self.samplers.append(
                ImprovedDatasetSampler(
                    replay_buffer=buf,
                    sequence_length=horizon,
                    shape_meta=shape_meta,
                    pad_before=pad_before,
                    pad_after=pad_after,
                    episode_mask=train_mask,
                    key_first_k=key_first_k,
                )
            )
            self.sample_probabilities[i] = cfg.get("sampling_weight") or np.sum(train_mask)

        self.sample_probabilities = self._normalize_sample_probabilities(self.sample_probabilities)
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta

        self.transforms = self.get_default_color_jitter(color_jitter)

    def _lowdim_key_map(self) -> Dict[str, str]:
        """
        Maps keys used by the normalizer to keys in the replay buffer.
        """
        key_map = {"action": "action"}
        for key in self.lowdim_keys:
            key_map[key] = key
        return key_map

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.num_datasets == 1:
            data = self.samplers[0].sample_data(idx)
        else:
            i = int(np.random.choice(self.num_datasets, p=self.sample_probabilities))
            data = self.samplers[i].sample_data(idx % len(self.samplers[i]))

        if self.transforms is not None and self.rgb_keys:
            data = self._apply_color_jitter(data)

        return dict_apply(data, torch.from_numpy)
