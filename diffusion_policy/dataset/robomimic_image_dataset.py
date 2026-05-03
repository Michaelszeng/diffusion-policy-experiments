"""
Robomimic HDF5 image dataset for relative 7D actions.

Normalization contract
----------------------
Images   : stored as uint8 HWC in zarr. The passthrough normalizer (inherited
           from BaseZarrImageDataset.get_normalizer) leaves them as float32 HWC
           [0, 255]. RobomimicObsEncoder converts internally to CHW [-1, 1].
Actions  : stored as raw float32; range-normalised to [-1, 1] by get_normalizer.
Low-dim  : stored as raw float32; range-normalised to [-1, 1] by get_normalizer.

Image shapes in shape_meta must use HWC order, e.g. [240, 240, 3].
"""

import concurrent.futures
import multiprocessing
from typing import Dict, List, Optional

import h5py
import numcodecs
import numpy as np
import torch
import zarr
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseZarrImageDataset


def _hdf5_to_replay(dataset_path: str, shape_meta: Dict) -> ReplayBuffer:
    """
    Convert a Robomimic HDF5 file to an in-memory zarr ReplayBuffer.

    Low-dim data and actions are stored as float32 (uncompressed).
    Images are stored as uint8 HWC with Blosc/LZ4 compression.
    Image shapes in shape_meta must be [H, W, C].
    """
    rgb_keys = [k for k, v in shape_meta["obs"].items() if v.get("type") == "rgb"]
    lowdim_keys = [k for k, v in shape_meta["obs"].items() if v.get("type") != "rgb"]

    store = zarr.MemoryStore()
    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    n_workers = multiprocessing.cpu_count()
    max_inflight = n_workers * 5

    with h5py.File(dataset_path, "r") as f:
        demos = f["data"]
        n_demos = len(demos)

        # compute episode boundaries
        episode_ends = []
        prev_end = 0
        for i in range(n_demos):
            prev_end += demos[f"demo_{i}"]["actions"].shape[0]
            episode_ends.append(prev_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]

        meta_group.array("episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True)

        # low-dim observations and actions
        for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim"):
            hdf5_key = f"obs/{key}" if key != "action" else "actions"
            chunks = [demos[f"demo_{i}"][hdf5_key][:].astype(np.float32) for i in range(n_demos)]
            arr = np.concatenate(chunks, axis=0)
            expected = tuple(shape_meta["action"]["shape"]) if key == "action" else tuple(shape_meta["obs"][key]["shape"])
            assert arr.shape == (n_steps,) + expected, (
                f"Key '{key}': expected shape {(n_steps,) + expected}, got {arr.shape}"
            )
            data_group.array(name=key, data=arr, shape=arr.shape, chunks=arr.shape,
                             compressor=None, dtype=arr.dtype)

        # images: HDF5 stores (T, H, W, C) uint8; zarr mirrors this layout
        def _img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                _ = zarr_arr[zarr_idx]  # verify round-trip decode
                return True
            except Exception:
                return False

        img_compressor = numcodecs.Blosc(cname="lz4", clevel=5)
        with tqdm(total=n_steps * len(rgb_keys), desc="Loading images") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    h, w, c = shape_meta["obs"][key]["shape"]
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=img_compressor,
                        dtype=np.uint8,
                    )
                    for ep_idx in range(n_demos):
                        hdf5_arr = demos[f"demo_{ep_idx}"]["obs"][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight:
                                done, futures = concurrent.futures.wait(
                                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                                )
                                for fut in done:
                                    if not fut.result():
                                        raise RuntimeError("Failed to copy image frame!")
                                pbar.update(len(done))
                            zarr_idx = episode_starts[ep_idx] + hdf5_idx
                            futures.add(executor.submit(_img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                done, futures = concurrent.futures.wait(futures)
                for fut in done:
                    if not fut.result():
                        raise RuntimeError("Failed to copy image frame!")
                pbar.update(len(done))

    return ReplayBuffer(root)


class RobomimicImageDataset(BaseZarrImageDataset):
    """
    Image-based Robomimic dataset with 7D relative actions.

    Reads from a Robomimic HDF5 file and converts to an in-memory zarr
    ReplayBuffer on init. Train/val splitting, ImprovedDatasetSampler, and
    normalizer construction are inherited from BaseZarrImageDataset.

    Image shapes in shape_meta must use HWC order, e.g. [240, 240, 3].
    """

    def __init__(
        self,
        dataset_path: str,
        shape_meta: Dict,
        horizon: int = 1,
        n_obs_steps: Optional[int] = None,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
        color_jitter: Optional[Dict] = None,
    ):
        # BaseZarrImageDataset expects a list of zarr_configs dicts; we wrap
        # the single HDF5 path in that format and override load_replay_buffer
        # to perform the HDF5 → zarr conversion.
        zarr_configs = [{
            "path": dataset_path,
            "val_ratio": val_ratio,
            "max_train_episodes": max_train_episodes,
            "sampling_weight": None,
        }]
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
        )

    def _get_buffer_keys(self) -> List[str]:
        return self.rgb_keys + self.lowdim_keys + ["action"]

    def _lowdim_key_map(self) -> Dict[str, str]:
        # zarr key names match shape_meta key names directly
        return {"action": "action", **{k: k for k in self.lowdim_keys}}

    def load_replay_buffer(self, path: str, keys: List[str], config: Dict) -> ReplayBuffer:
        # path here is the HDF5 file path; convert to in-memory zarr
        return _hdf5_to_replay(path, self.shape_meta)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.samplers[0].sample_data(idx)
        if self.transforms is not None and self.rgb_keys:
            data = self._apply_color_jitter(data)
        return dict_apply(data, torch.from_numpy)
