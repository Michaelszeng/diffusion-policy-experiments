"""
FurnitureBench dataset for the pre-computed action-chunk zarr format.

This zarr format is used for data generated using DART, when sampling action sequences using a sliding window
is bad (because the actions at consecutive timesteps represent noisy intermediate states; what we actually want
is an action sequence corresponding to clean intermediate states).

The zarr layout expected:
  data/
    action          (T, 10)      float32
    action_chunk    (T, chunk_size, 10)  float32
    color_image1    (T, H, W, 3) uint8
    color_image2    (T, H, W, 3) uint8
    robot_state     (T, 16)      float32
    ...
  meta/
    episode_ends    (E,)         int64

The outputted observation/action sequence pairs contains:
 - Observations from n_obs_steps timesteps
 - Past noisy actions action[t - n_obs_steps + 1 : t] that connect the observations together
 - Future clean actions action_chunk[t][:horizon - n_obs_steps + 1]
"""

import copy
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import ImprovedDatasetSampler
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseZarrImageDataset


class FurnitureBenchChunkDataset(BaseZarrImageDataset):
    """
    Key differences from FurnitureBenchDataset:
      - The sampler window spans n_obs_steps frames (not pred_horizon), so
        buffer_end_idx - 1 is always the current timestep t.
      - The action target is assembled from past noisy actions (zero-padded at
        episode start) and action_chunk[t], rather than horizon consecutive
        raw actions.
      - pad_after is always 0 because action_chunk[t] already covers the full
        lookahead for every t, including the last frame of each episode.

    shape_meta example:
        action:
          shape: [10]
        obs:
          color_image1: {type: rgb,     shape: [240, 320, 3]}
          color_image2: {type: rgb,     shape: [240, 320, 3]}
          robot_state:  {type: low_dim, shape: [16]}
    """

    def __init__(
        self,
        zarr_configs: List[Dict],
        shape_meta: Dict,
        horizon: int = 16,
        n_obs_steps: int = 1,
        pad_before: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        color_jitter: Optional[Dict] = None,
    ):
        # Skip BaseZarrImageDataset.__init__: it builds samplers with sequence_length=horizon,
        # but we need sequence_length=n_obs_steps so buffer_end_idx-1 == current timestep t.
        # So, we call BaseImageDataset.__init__ and then override the horizon and n_obs_steps.
        BaseImageDataset.__init__(self)
        self._validate_zarr_configs(zarr_configs)

        obs_meta = shape_meta["obs"]
        self.rgb_keys = [k for k, v in obs_meta.items() if v.get("type") == "rgb"]
        self.lowdim_keys = [k for k, v in obs_meta.items() if v.get("type") != "rgb"]
        self.shape_meta = shape_meta
        self.pred_horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.pad_before = pad_before
        self.pad_after = 0

        buf_keys = self._get_buffer_keys()

        self.num_datasets = len(zarr_configs)
        self.replay_buffers: List[ReplayBuffer] = []
        self.action_chunk_arrays: List[np.ndarray] = []
        self.train_masks: List[np.ndarray] = []
        self.val_masks: List[np.ndarray] = []
        self.samplers: List[ImprovedDatasetSampler] = []
        self.zarr_paths: List[str] = []
        self.sample_probabilities = np.zeros(len(zarr_configs))

        for i, cfg in enumerate(zarr_configs):
            zarr_path = os.path.expanduser(cfg["path"])

            buf = ReplayBuffer.copy_from_path(
                zarr_path=zarr_path,
                store=zarr.MemoryStore(),
                keys=buf_keys,
            )

            # Load action_chunk separately — it doesn't follow the sliding-window sampling pattern, 
            # so must not go through the sampler.
            z = zarr.open(zarr_path, "r")
            action_chunk = z["data/action_chunk"][:]  # (T, chunk_size, Da)

            train_mask, val_mask = self._build_episode_masks(
                n_episodes=buf.n_episodes,
                val_ratio=cfg.get("val_ratio", val_ratio),
                max_train_episodes=cfg.get("max_train_episodes", None),
                seed=seed,
            )

            sampler = ImprovedDatasetSampler(
                replay_buffer=buf,
                sequence_length=n_obs_steps,
                shape_meta=shape_meta,
                pad_before=pad_before,
                pad_after=0,
                keys=buf_keys,
                episode_mask=train_mask,
            )

            self.replay_buffers.append(buf)
            self.action_chunk_arrays.append(action_chunk)
            self.zarr_paths.append(zarr_path)
            self.train_masks.append(train_mask)
            self.val_masks.append(val_mask)
            self.samplers.append(sampler)
            self.sample_probabilities[i] = cfg.get("sampling_weight") or np.sum(train_mask)

        self.sample_probabilities = self._normalize_sample_probabilities(self.sample_probabilities)
        # Set horizon == n_obs_steps so the inherited get_validation_dataset() recreates samplers correctly.
        self.horizon = n_obs_steps
        self.transforms = self.get_default_color_jitter(color_jitter)

    def _get_buffer_keys(self) -> List[str]:
        return self.rgb_keys + self.lowdim_keys + ["action"]

    def _lowdim_key_map(self) -> Dict[str, str]:
        return {"action": "action", **{k: k for k in self.lowdim_keys}}

    def get_validation_dataset(self, index: Optional[int] = None) -> "FurnitureBenchChunkDataset":
        val_set = super().get_validation_dataset(index)
        if index is None:
            index = 0
        val_set.action_chunk_arrays = [self.action_chunk_arrays[index]]
        return val_set

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.num_datasets == 1:
            sampler = self.samplers[0]
            action_chunks = self.action_chunk_arrays[0]
        else:
            i = int(np.random.choice(self.num_datasets, p=self.sample_probabilities))
            sampler = self.samplers[i]
            action_chunks = self.action_chunk_arrays[i]

        local_idx = idx % len(sampler)
        _, buffer_end_idx, _, _ = sampler.indices[local_idx]
        t = buffer_end_idx - 1  # current timestep in the buffer

        data = sampler.sample_data(local_idx)

        # Assemble action: (C-1) zero-padded past actions + (H-C+1) clean chunk.
        # The sampler zero-pads non-obs keys at episode start, so past_action is
        # already correctly zeroed for padded slots.
        past_count = self.n_obs_steps - 1
        future_count = self.pred_horizon - past_count
        past_action = data["action"][:past_count]       # (C-1, Da)
        chunk_action = action_chunks[t][:future_count]  # (H-C+1, Da)
        data["action"] = np.concatenate([past_action, chunk_action], axis=0)  # (H, Da)

        if self.transforms is not None and self.rgb_keys:
            data = self._apply_color_jitter(data)

        return dict_apply(data, torch.from_numpy)
