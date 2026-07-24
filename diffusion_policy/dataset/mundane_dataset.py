"""
Dataset for Mundane bimanual manipulation data (zarr format).

Expected zarr layout (e.g. data/diffusion_experiments/mundane/pill_v2.zarr):
    data/
        action              (T, 14)         float32  leader-side actions
        action_follower     (T, 14)         float32  follower executed poses
        camera_*_rgb        (T, H, W, 3)   uint8
        robot0_eef_pos      (T, 3)         float32
        robot0_eef_rpy      (T, 3)         float32
        robot0_gripper_width (T, 1)        float32
        robot1_eef_pos      (T, 3)         float32
        robot1_eef_rpy      (T, 3)         float32
        robot1_gripper_width (T, 1)        float32
        ...                               (other proprioception arrays)
    meta/
        episode_ends        (E,)           int64
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseZarrImageDataset


class MundaneDataset(BaseZarrImageDataset):
    """
    Zarr-backed dataset for Mundane bimanual manipulation recordings.

    Args:
        zarr_configs:     List of dicts with keys:
                            path              (str)             zarr directory path
                            max_train_episodes (int | null)     cap on training episodes
                            sampling_weight   (float | null)    relative sampling weight
                            val_ratio         (float | null)    per-dataset val fraction
        shape_meta:       Shape metadata for actions and observations.
        horizon:          Action prediction horizon in *policy* timesteps.
        n_obs_steps:      Observation conditioning window in *policy* timesteps.
        pad_before:       Episode-start padding in *policy* timesteps.
        pad_after:        Episode-end padding in *policy* timesteps.
        seed:             RNG seed for train/val splits.
        val_ratio:        Default validation fraction (overridden per zarr_config).
        color_jitter:     Optional color-jitter config dict (passed to base class).
        random_rotation:  Optional random-rotation config dict (passed to base class).
                          Scalar or dict with ``degrees``/``fill``/``expand``.
        action_key:       Which zarr array to use as the training target:
                            ``"action"``          leader-side actions (default)
                            ``"action_follower"`` follower executed poses
        downsample_steps: Stride for temporal subsampling.  Raw data is recorded
                          at 30 Hz; set to 3 for a 10 Hz policy.
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
        random_rotation: Optional[Dict] = None,
        action_key: str = "action",
        downsample_steps: int = 1,
    ):
        if action_key not in ("action", "action_follower"):
            raise ValueError(f"action_key must be 'action' or 'action_follower', got {action_key!r}")
        self._action_zarr_key = action_key
        self.downsample_steps = downsample_steps

        # Scale all temporal parameters to the native recording frequency so the
        # sampler draws windows at the original rate; __getitem__ strides them down.
        s = downsample_steps
        super().__init__(
            zarr_configs=zarr_configs,
            shape_meta=shape_meta,
            horizon=horizon * s,
            n_obs_steps=n_obs_steps * s if n_obs_steps is not None else None,
            pad_before=pad_before * s,
            pad_after=pad_after * s,
            seed=seed,
            val_ratio=val_ratio,
            color_jitter=color_jitter,
            random_rotation=random_rotation,
        )

    # ── BaseZarrImageDataset hooks ──────────────────────────────────────────────

    def _get_buffer_keys(self) -> List[str]:
        keys = list(self.rgb_keys) + list(self.lowdim_keys) + [self._action_zarr_key]
        seen: set = set()
        return [k for k in keys if not (k in seen or seen.add(k))]

    def _load_buffer_and_masks(
        self,
        zarr_path: str,
        keys: List[str],
        cfg: Dict,
        seed: int,
        default_val_ratio: float,
    ) -> Tuple[ReplayBuffer, np.ndarray, np.ndarray]:
        buf, train_mask, val_mask = super()._load_buffer_and_masks(
            zarr_path=zarr_path,
            keys=keys,
            cfg=cfg,
            seed=seed,
            default_val_ratio=default_val_ratio,
        )
        # Rename the chosen action array to the canonical "action" key so the
        # rest of the pipeline (samplers, normalizer, workspaces) is unaffected.
        if self._action_zarr_key != "action":
            data = buf.root["data"]
            if isinstance(data, dict):
                data["action"] = data.pop(self._action_zarr_key)
            else:
                data.move(self._action_zarr_key, "action")
        return buf, train_mask, val_mask

    def _lowdim_key_map(self) -> Dict[str, str]:
        m: Dict[str, str] = {"action": "action"}
        for k in self.lowdim_keys:
            m[k] = k
        return m

    # ── Sampling ────────────────────────────────────────────────────────────────

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

        if self.rotation_transforms is not None and self.rgb_keys:
            data = self._apply_random_rotation(data)

        return dict_apply(data, torch.from_numpy)
