"""
Image dataset for the Isaac Sim zarr format.

Expected zarr layout (per data/diffusion_experiments/isaac_sim/*.zarr):
    data/
        actions             (T, 7)             float32/64   # 7D ee delta pose + gripper
        eef_pos             (T, 3)             float32
        eef_quat            (T, 4)             float32
        gripper_pos         (T, 2)             float32
        joint_pos           (T, 9)             float32
        joint_vel           (T, 9)             float32
        scene_cam_front     (T, 128, 128, 3)   uint8
        scene_cam_rear_left (T, 128, 128, 3)   uint8
        wrist_cam           (T, 128, 128, 3)   uint8
        # ...plus task-specific object pose arrays (e.g. gear_pos, gear_quat, shaft_pos)
    meta/
        episode_ends        (E,)               int64

Differences from the ManiSkill zarr layout this codebase originally
targets:
    * the action array is named ``actions`` (plural), not ``action``;
    * there is no ``target`` / goal array (only `use_target_cond: false`
      policies are supported).

The class loads ``actions`` from disk and renames the array to
``"action"`` in the in-memory ReplayBuffer so that the rest of the
pipeline (workspaces, samplers, normalizer) — which all assume an
``"action"`` key — works unchanged.

Routing (handled by ImprovedDatasetSampler):
    * rgb keys          -> obs[key]            (uint8)
    * other obs keys    -> obs[key]            (float32, e.g. "eef_pos")
    * "action"          -> top-level "action"
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseZarrImageDataset


class IsaacSimZarrDataset(BaseZarrImageDataset):
    """
    Minimal zarr-backed dataset for Isaac Sim data.

    The observation keys are determined entirely by ``shape_meta.obs``;
    the dataset adds the action array (``actions`` on disk, exposed as
    ``"action"`` in the buffer) on top of those.

    ``downsample_steps`` subsamples the raw sequence before returning it,
    allowing a policy trained at a lower control frequency (e.g. 10 Hz) to
    consume data recorded at a higher rate (e.g. 30 Hz).  Set it to the
    integer ratio between the recording Hz and the target policy Hz.
    """

    # Name of the action array as stored in the Isaac Sim zarr files.
    _ACTION_ZARR_KEY = "actions"

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
        # The Isaac Sim zarr has no goal/target array; the only non-obs
        # array we need is the action sequence.
        keys = list(self.rgb_keys) + list(self.lowdim_keys) + [self._ACTION_ZARR_KEY]
        # dedup, preserve order
        seen = set()
        return [k for k in keys if not (k in seen or seen.add(k))]

    def _load_buffer_and_masks(
        self,
        zarr_path: str,
        keys: List[str],
        cfg: Dict,
        seed: int,
        default_val_ratio: float,
    ) -> Tuple[ReplayBuffer, np.ndarray, np.ndarray]:
        # Delegate the heavy lifting (incl. the max_train_episodes
        # selective-copy optimisation) to the base class, then rename the
        # disk-side "actions" array to the canonical "action" key so the
        # rest of the pipeline (samplers, normalizer, workspaces) is
        # oblivious to the on-disk naming convention.
        buf, train_mask, val_mask = super()._load_buffer_and_masks(
            zarr_path=zarr_path,
            keys=keys,
            cfg=cfg,
            seed=seed,
            default_val_ratio=default_val_ratio,
        )
        data = buf.root["data"]
        if isinstance(data, dict):
            data["action"] = data.pop(self._ACTION_ZARR_KEY)
        else:
            data.move(self._ACTION_ZARR_KEY, "action")
        return buf, train_mask, val_mask

    def _lowdim_key_map(self) -> Dict[str, str]:
        # Maps model-side normalizer keys to the (post-rename) buffer keys.
        m: Dict[str, str] = {"action": "action"}
        for k in self.lowdim_keys:
            m[k] = k
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
