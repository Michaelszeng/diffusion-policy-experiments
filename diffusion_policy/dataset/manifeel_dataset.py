"""
Image dataset for the manifeel zarr format (e.g. data/pih_quan_June06,
data/plug_quan_Aug02).

On-disk layout:
    data/
        action          (T, 6)              float32   # 6D ee delta pose
        state           (T, 7)              float32   # eef pos (3) + quat (4)
        front           (T, 256, 256, 3)    float32   # RGB in [0, 1]
        side            (T, 256, 256, 3)    float32
        wrist           (T, 256, 256, 3)    float32
        wrist_2         (T, 256, 256, 3)    float32
        left_tactile_camera_taxim   (T, 320, 240, 3)  float32
        right_tactile_camera_taxim  (T, 320, 240, 3)  float32
        tactile_depth_right         (T, 10, 14)       float32
        tactile_force_field_right   (T, 10, 14, 3)    float32
    meta/
        episode_ends    (E,)                int64

Conversions handled by this dataset (sampler is shared across datasets):
    * RGB float32 in [0, 1] -> uint8 [0, 255] after load, so the
      ImprovedDatasetSampler's blind ``astype(uint8)`` keeps the range.
    * On-disk key ``state`` is loaded, then exposed to the model as
      ``agent_pos`` (the sampler hard-codes that rename).

Expected shape_meta:
    action.shape   = [6]
    obs keys must match on-disk camera names (front, side, wrist, wrist_2);
    proprio key must be ``agent_pos`` with shape [7].
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseZarrImageDataset


class ManifeelImageDataset(BaseZarrImageDataset):
    _AGENT_POS_KEY = "agent_pos"
    _STATE_DISK_KEY = "state"
    _ACTION_KEY = "action"

    def _get_buffer_keys(self) -> List[str]:
        # Translate model-side key names to on-disk key names.
        disk_keys: List[str] = list(self.rgb_keys)
        for k in self.lowdim_keys:
            disk_keys.append(self._STATE_DISK_KEY if k == self._AGENT_POS_KEY else k)
        disk_keys.append(self._ACTION_KEY)
        seen = set()
        return [k for k in disk_keys if not (k in seen or seen.add(k))]

    def _lowdim_key_map(self) -> Dict[str, str]:
        m: Dict[str, str] = {self._ACTION_KEY: self._ACTION_KEY}
        for k in self.lowdim_keys:
            m[k] = self._STATE_DISK_KEY if k == self._AGENT_POS_KEY else k
        return m

    def _load_buffer_and_masks(
        self,
        zarr_path: str,
        keys: List[str],
        cfg: Dict,
        seed: int,
        default_val_ratio: float,
    ) -> Tuple[ReplayBuffer, np.ndarray, np.ndarray]:
        buf, train_mask, val_mask = super()._load_buffer_and_masks(
            zarr_path=zarr_path, keys=keys, cfg=cfg,
            seed=seed, default_val_ratio=default_val_ratio,
        )
        data = buf.root["data"]
        for rgb_key in self.rgb_keys:
            arr = data[rgb_key][:]
            if arr.dtype == np.uint8:
                continue
            arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            if isinstance(data, dict):
                data[rgb_key] = arr_u8
            else:
                # zarr Group in MemoryStore: overwrite the array in place.
                data.array(rgb_key, arr_u8, chunks=data[rgb_key].chunks, overwrite=True)
        return buf, train_mask, val_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.num_datasets == 1:
            data = self.samplers[0].sample_data(idx)
        else:
            i = int(np.random.choice(self.num_datasets, p=self.sample_probabilities))
            data = self.samplers[i].sample_data(idx % len(self.samplers[i]))

        if self.transforms is not None and self.rgb_keys:
            data = self._apply_color_jitter(data)

        return dict_apply(data, torch.from_numpy)
