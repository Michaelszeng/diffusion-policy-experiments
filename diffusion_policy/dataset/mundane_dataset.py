"""
Mundane bimanual-teleop dataset (e.g. ``pill_v1.zarr``).

Expected zarr layout:
    data/
        action                  (T, 14)            float32   # leader command
        action_follower         (T, 14)            float32   # follower-executed pose
        camera_<serial>_rgb     (T, 224, 224, 3)   uint8     # one per RealSense
        joint_pos_L             (T, 6)             float32   # follower (robot) joints
        joint_pos_R             (T, 6)             float32
        L_joint_pos_L           (T, 6)             float32   # leader joints
        L_joint_pos_R           (T, 6)             float32
        robot0_eef_pos          (T, 3)             float32
        robot0_eef_rpy          (T, 3)             float32
        robot0_gripper_width    (T, 1)             float32
        robot0_wrench           (T, 6)             float32
        robot1_eef_pos          (T, 3)             float32
        robot1_eef_rpy          (T, 3)             float32
        robot1_gripper_width    (T, 1)             float32
        robot1_wrench           (T, 6)             float32
        gripper_pos_L           (T,)               float32
        gripper_pos_R           (T,)               float32
        L_gripper_pos_L         (T,)               float32
        L_gripper_pos_R         (T,)               float32
        gripper_contact_L       (T, 1)             int8
        gripper_contact_R       (T, 1)             int8
        gripper_tau_external_L  (T, 1)             float32
        gripper_tau_external_R  (T, 1)             float32
        joints_tau_external_L   (T, 6)             float32
        joints_tau_external_R   (T, 6)             float32
        timestamps              (T,)               float64
    meta/
        episode_ends            (E,)               int64

shape_meta example for two cameras + dual-arm proprio:
    action:
      shape: [14]
    obs:
      camera_335122271682_rgb:
        type: rgb
        shape: [224, 224, 3]
      camera_419522072281_rgb:
        type: rgb
        shape: [224, 224, 3]
      joint_pos_L:
        type: low_dim
        shape: [6]
      joint_pos_R:
        type: low_dim
        shape: [6]
      robot0_eef_pos:
        type: low_dim
        shape: [3]
      robot1_eef_pos:
        type: low_dim
        shape: [3]
      robot0_gripper_width:
        type: low_dim
        shape: [1]
      robot1_gripper_width:
        type: low_dim
        shape: [1]

Notes:
    - The action key used for supervision is selectable via ``action_key`` (defaults to ``"action"``, the leader command).
      Pass ``"action_follower"`` to train against the follower's actually executed pose. 
      The action is internally stored under the standard ``"action"`` key in the in-memory ReplayBuffer so downstream 
      samplers, normalizers, and policies don't need to know which source was used.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseZarrImageDataset


VALID_ACTION_KEYS = ("action", "action_follower")


class MundaneDataset(BaseZarrImageDataset):
    """Image + low-dim dataset for the Mundane bimanual-teleop zarr format.

    Supports multiple zarr files with per-dataset sampling weights,
    train/val splits, and ``max_train_episodes`` capping, inheriting
    that machinery from ``BaseZarrImageDataset``.

    Each entry in ``zarr_configs`` is a dict with:
        path (str)                       — path to a Mundane ``*.zarr`` directory
        max_train_episodes (int, opt.)   — cap on training episodes
        sampling_weight (float, opt.)    — all or none must provide this
        val_ratio (float, opt.)          — overrides the global ``val_ratio``
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
        action_key: str = "action",
        downsample_steps: int = 1,
    ):
        """
        downsample_steps: Stride between successive frames inside a sampled
            window. The Mundane zarrs are recorded at 30 Hz; set
            ``downsample_steps: 3`` to train a 10 Hz policy (each obs/action
            sequence is read with stride 3 from the buffer).
        """
        if action_key not in VALID_ACTION_KEYS:
            raise ValueError(
                f"action_key must be one of {VALID_ACTION_KEYS}, got '{action_key}'"
            )
        self.action_key = action_key

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
            downsample_steps=downsample_steps,
        )

    def _get_buffer_keys(self) -> List[str]:
        # Use a set + dedup-preserving list in case ``action_key`` happens
        # to coincide with one of the user-supplied obs keys.
        keys = self.rgb_keys + self.lowdim_keys + [self.action_key]
        seen = set()
        deduped = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                deduped.append(k)
        return deduped

    def _lowdim_key_map(self) -> Dict[str, str]:
        # Normalizer/model always reference the action under the key "action";
        # remap from the zarr-side name (which may be "action_follower").
        return {"action": self.action_key, **{k: k for k in self.lowdim_keys}}

    def load_replay_buffer(
        self, path: str, keys: List[str], config: Dict
    ) -> ReplayBuffer:
        """Load a Mundane zarr into a ReplayBuffer.

        When ``self.action_key`` is not ``"action"`` we still want the
        in-memory buffer to expose it under the standard ``"action"``
        key so the rest of the pipeline (samplers, normalizer, policy)
        doesn't have to know which source was selected. We do that by
        reading the requested array and copying it into a new
        ``"action"`` slot before returning.
        """
        if self.action_key == "action":
            return ReplayBuffer.copy_from_path(
                zarr_path=path, store=zarr.MemoryStore(), keys=keys
            )

        # Read everything except the standard "action" key from disk;
        # then materialize the chosen action stream as the buffer's "action".
        read_keys = [k for k in keys if k != "action"]
        buf = ReplayBuffer.copy_from_path(
            zarr_path=path, store=zarr.MemoryStore(), keys=read_keys
        )
        src = zarr.open(path, mode="r")
        action_arr = src["data"][self.action_key][:]
        # Use the same chunking as a typical low-dim array in the buffer.
        buf.data.create_dataset(
            "action",
            data=action_arr,
            chunks=(action_arr.shape[0], *action_arr.shape[1:]),
            compressor=None,
            overwrite=True,
        )
        return buf

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.num_datasets == 1:
            data = self.samplers[0].sample_data(idx)
        else:
            i = int(np.random.choice(self.num_datasets, p=self.sample_probabilities))
            data = self.samplers[i].sample_data(idx % len(self.samplers[i]))

        if self.transforms is not None and self.rgb_keys:
            data = self._apply_color_jitter(data)

        return dict_apply(data, torch.from_numpy)
