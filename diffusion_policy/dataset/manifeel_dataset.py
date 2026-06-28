"""
Image dataset for the manifeel zarr format (e.g. data/pih_quan_June06,
data/plug_quan_Aug02, data/nutbolt_quan_July1).

On-disk layout:
    data/
        action          (T, 6 or 7)         float32   # 6D ee delta pose (+ gripper for nut-bolt)
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

Two manifeel-specific conversions done at load time (not by the shared sampler):
    * RGB float32 in [0, 1] -> uint8 [0, 255], so the obs encoder's input
      matches what get_image_passthrough_normalizer expects.
    * Buffer rename: on-disk ``state`` -> ``agent_pos``. This is the only
      manifeel proprio key, and shape_meta uses the model-side name
      ``agent_pos``. Renaming in the buffer means the shared sampler
      iterates an obs key that matches ``shape_meta.obs``, so the standard
      pre-pad path (``key in self.obs_dict_keys``) covers it. Avoids
      depending on the sampler's legacy ``state``-specific branch.

Expected shape_meta:
    action.shape = [6] (peg-in-hole, plug-insert) or [7] (nut-bolt)
    obs keys must match on-disk camera names (front, side, wrist, wrist_2)
    or the sampler-side ``tactile_force_field_right``; proprio key must be
    ``agent_pos`` with shape [7].
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseZarrImageDataset


class ManifeelImageDataset(BaseZarrImageDataset):
    _AGENT_POS_KEY = "agent_pos"
    _STATE_DISK_KEY = "state"
    _ACTION_KEY = "action"

    def _get_buffer_keys(self) -> List[str]:
        # Names used to fetch arrays from the on-disk zarr. Only the proprio
        # key differs (model-side "agent_pos" lives under disk-side "state").
        disk_keys: List[str] = list(self.rgb_keys)
        for k in self.lowdim_keys:
            disk_keys.append(self._STATE_DISK_KEY if k == self._AGENT_POS_KEY else k)
        disk_keys.append(self._ACTION_KEY)
        seen = set()
        return [k for k in disk_keys if not (k in seen or seen.add(k))]

    def _lowdim_key_map(self) -> Dict[str, str]:
        # After _load_buffer_and_masks renames state -> agent_pos in the buffer,
        # the model-side name and the buffer-side name match for every lowdim key.
        m: Dict[str, str] = {self._ACTION_KEY: self._ACTION_KEY}
        for k in self.lowdim_keys:
            m[k] = k
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
            zarr_path=zarr_path,
            keys=keys,
            cfg=cfg,
            seed=seed,
            default_val_ratio=default_val_ratio,
        )
        data = buf.root["data"]

        # Rename buffer key: state -> agent_pos. After this the sampler
        # iterates "agent_pos" (which is in shape_meta.obs), so the standard
        # obs pre-pad path covers it and no "state"-specific sampler branch
        # is exercised by this dataset.
        if self._AGENT_POS_KEY in self.lowdim_keys and self._STATE_DISK_KEY in data:
            if isinstance(data, dict):
                data[self._AGENT_POS_KEY] = data.pop(self._STATE_DISK_KEY)
            else:
                data.move(self._STATE_DISK_KEY, self._AGENT_POS_KEY)

        # RGB float [0, 1] -> uint8 [0, 255] (the obs encoder + passthrough
        # normalizer expect [0, 255]).
        for rgb_key in self.rgb_keys:
            arr = data[rgb_key][:]
            if arr.dtype == np.uint8:
                continue
            arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            if isinstance(data, dict):
                data[rgb_key] = arr_u8
            else:
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
