"""
FurnitureBench state-only (lowdim) dataset.  Reads from the same zarr stores
as FurnitureBenchDataset but exposes only robot_state (and any other lowdim
obs keys) -- no image tensors.

parts_poses is stored in the zarr as [x, y, z, qx, qy, qz, qw] x n_parts.
On load, quaternions are converted once to the 6D rotation representation
(first two columns of the rotation matrix), giving [x, y, z, r0..r5] x n_parts.
This avoids the antipodal ambiguity of quaternions and is safe to normalize
linearly.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseZarrLowdimDataset

# parts_poses zarr layout: n_parts x [x, y, z, qx, qy, qz, qw]
_PARTS_POSE_IN_DIM = 7   # xyz (3) + quaternion (4)
_PARTS_POSE_OUT_DIM = 9  # xyz (3) + 6D rotation (6)


def _quat_xyzw_to_6d(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternions [qx, qy, qz, qw] to 6D rotation (first two columns
    of the rotation matrix, concatenated).

    Input:  (..., 4)
    Output: (..., 6)
    """
    qx, qy, qz, qw = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    col0 = np.stack([
        1 - 2*(qy**2 + qz**2),
        2*(qx*qy + qz*qw),
        2*(qx*qz - qy*qw),
    ], axis=-1)
    col1 = np.stack([
        2*(qx*qy - qz*qw),
        1 - 2*(qx**2 + qz**2),
        2*(qy*qz + qx*qw),
    ], axis=-1)
    return np.concatenate([col0, col1], axis=-1)


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
        sampling_weight (float, optional) -- all or none must provide this
        val_ratio (float, optional) -- overrides the global val_ratio

    shape_meta example::

        action: {shape: [10]}
        obs:
          robot_state: {shape: [16], type: low_dim}
          parts_poses: {shape: [45], type: low_dim}  # 5 parts x 9 (xyz + 6D rot)
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

        # One-time in-memory conversion: replace the quaternion-based parts_poses
        # arrays in every replay buffer with their 6D rotation equivalents.
        if "parts_poses" in self.lowdim_keys:
            self._convert_parts_poses_to_6d()

    def _convert_parts_poses_to_6d(self) -> None:
        """
        Replace each replay buffer's parts_poses array (N x n_parts*7) with
        its 6D-rotation equivalent (N x n_parts*9).  Runs once at init.
        """
        for buf in self.replay_buffers:
            raw = buf["parts_poses"][:]              # (N, n_parts*7)
            N = raw.shape[0]
            in_dim = raw.shape[1]
            assert in_dim % _PARTS_POSE_IN_DIM == 0
            n_parts = in_dim // _PARTS_POSE_IN_DIM

            parts = raw.reshape(N, n_parts, _PARTS_POSE_IN_DIM)
            xyz  = parts[..., :3]   # (N, n_parts, 3)
            quat = parts[..., 3:]   # (N, n_parts, 4) -- [qx, qy, qz, qw]
            rot6d = _quat_xyzw_to_6d(quat)           # (N, n_parts, 6)

            converted = np.concatenate([xyz, rot6d], axis=-1)   # (N, n_parts, 9)
            converted = converted.reshape(N, n_parts * _PARTS_POSE_OUT_DIM).astype(np.float32)

            buf.root["data"].array("parts_poses", converted, dtype=np.float32, overwrite=True)
            print(f"[FurnitureBenchLowdimDataset] Converted parts_poses: "
                  f"{raw.shape} -> {converted.shape}")

    def _get_buffer_keys(self) -> List[str]:
        print(self.lowdim_keys)
        return list(self.lowdim_keys) + ["action"]

    def _lowdim_key_map(self) -> Dict[str, str]:
        # identity map -- zarr key names match the model key names
        return {"action": "action", **{k: k for k in self.lowdim_keys}}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.num_datasets == 1:
            data = self.samplers[0].sample_data(idx)
        else:
            i = int(np.random.choice(self.num_datasets, p=self.sample_probabilities))
            data = self.samplers[i].sample_data(idx % len(self.samplers[i]))

        return dict_apply(data, torch.from_numpy)
