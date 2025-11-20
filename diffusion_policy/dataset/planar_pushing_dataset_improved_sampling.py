import copy
import math
import os
import random
import time
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torchvision import transforms

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import ImprovedDatasetSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset, gaussian_kernel, low_pass_filter
from diffusion_policy.model.common.normalizer import LinearNormalizer


class PlanarPushingDataset(BaseImageDataset):
    """
    Dataset for planar pushing that supports:
    - hybrid observations (images + end effector state)
    - multi cameras
    - cotraining with multiple datasets (datasets must share input output space)
    - dataset/loss scaling
    """

    def __init__(
        self,
        zarr_configs,
        shape_meta,
        use_one_hot_encoding=False,
        horizon=1,
        n_obs_steps=None,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        color_jitter=None,
        low_pass_on_wrist=False,
        low_pass_on_overhead=False,
    ):
        super().__init__()
        self._validate_zarr_configs(zarr_configs)

        self.low_pass_on_wrist = low_pass_on_wrist
        if low_pass_on_wrist:
            self.wrist_kernel = gaussian_kernel(kernel_size=9, sigma=3, channels=3)
        self.low_pass_on_overhead = low_pass_on_overhead
        if low_pass_on_overhead:
            self.overhead_kernel = gaussian_kernel(kernel_size=9, sigma=3, channels=3)

        # Set up dataset keys
        self.rgb_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "")
            if type == "rgb":
                self.rgb_keys.append(key)

        keys = self.rgb_keys + ["state", "action", "target"]

        # trick for saving ram
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in keys:
                key_first_k[key] = n_obs_steps
        key_first_k["action"] = horizon

        # Load in all the zarr datasets
        self.num_datasets = len(zarr_configs)
        self.replay_buffers = []
        self.train_masks = []
        self.val_masks = []
        self.samplers = []
        self.sample_probabilities = np.zeros(len(zarr_configs))
        self.zarr_paths = []

        for i, zarr_config in enumerate(zarr_configs):
            # Extract config info
            zarr_path = os.path.expanduser(zarr_config["path"])
            max_train_episodes = zarr_config.get("max_train_episodes", None)
            sampling_weight = zarr_config.get("sampling_weight", None)

            # Set up replay buffer
            self.replay_buffers.append(
                ReplayBuffer.copy_from_path(zarr_path=zarr_path, store=zarr.MemoryStore(), keys=keys)
            )
            n_episodes = self.replay_buffers[-1].n_episodes

            # Set up masks
            if "val_ratio" in zarr_config and zarr_config["val_ratio"] is not None:
                dataset_val_ratio = zarr_config["val_ratio"]
            else:
                dataset_val_ratio = val_ratio
            val_mask = get_val_mask(n_episodes=n_episodes, val_ratio=dataset_val_ratio, seed=seed)
            train_mask = ~val_mask
            # Note max_train_episodes is the max number of training episodes
            # not the total number of train and val episodes!
            train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

            self.train_masks.append(train_mask)
            self.val_masks.append(val_mask)

            # Set up sampler
            self.samplers.append(
                ImprovedDatasetSampler(
                    replay_buffer=self.replay_buffers[-1],
                    sequence_length=horizon,
                    shape_meta=shape_meta,
                    pad_before=pad_before,
                    pad_after=pad_after,
                    episode_mask=train_mask,
                    key_first_k=key_first_k,
                )
            )

            # Set up sample probabilities and zarr paths
            if sampling_weight is not None:
                self.sample_probabilities[i] = sampling_weight
            else:
                self.sample_probabilities[i] = np.sum(train_mask)
            self.zarr_paths.append(zarr_path)
        # Normalize sample_probabilities
        self.sample_probabilities = self._normalize_sample_probabilities(self.sample_probabilities)

        # Set up color jitter
        self.color_jitter = color_jitter
        if color_jitter is not None:
            self.transforms = transforms.ColorJitter(
                brightness=self.color_jitter.get("brightness", 0),
                contrast=self.color_jitter.get("contrast", 0),
                saturation=self.color_jitter.get("saturation", 0),
                hue=self.color_jitter.get("hue", 0),
            )

        # Load other variables
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.use_one_hot_encoding = use_one_hot_encoding
        self.one_hot_encoding = None  # if val dataset, this will not be None

    def get_validation_dataset(self, index=None):
        # Create validation dataset
        val_set = copy.copy(self)

        if index is None:
            assert self.num_datasets == 1, "Must specify validation dataset index if multiple datasets"
            index = 0
        else:
            val_set.replay_buffers = [self.replay_buffers[index]]
            val_set.train_masks = [self.train_masks[index]]
            val_set.val_masks = [self.val_masks[index]]
            val_set.zarr_paths = [self.zarr_paths[index]]
        val_set.num_datasets = 1
        val_set.sample_probabilities = np.array([1.0])

        # Set one hot encoding
        val_set.one_hot_encoding = np.zeros(self.num_datasets).astype(np.float32)
        val_set.one_hot_encoding[index] = 1

        val_set.samplers = [
            ImprovedDatasetSampler(
                replay_buffer=self.replay_buffers[index],
                sequence_length=self.horizon,
                shape_meta=self.shape_meta,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=self.val_masks[index],
            )
        ]

        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        # compute mins and maxes
        assert mode == "limits", "Only supports limits mode"
        low_dim_keys = ["action", "agent_pos", "target"]
        input_stats = {}
        for replay_buffer in self.replay_buffers:
            data = {
                "action": replay_buffer["action"],
                "agent_pos": replay_buffer["state"],
                "target": replay_buffer["target"],
            }
            normalizer = LinearNormalizer()
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

            # Update mins and maxes
            for key in low_dim_keys:
                _max = normalizer[key].params_dict.input_stats.max
                _min = normalizer[key].params_dict.input_stats.min

                if key not in input_stats:
                    input_stats[key] = {"max": _max, "min": _min}
                else:
                    input_stats[key]["max"] = torch.maximum(input_stats[key]["max"], _max)
                    input_stats[key]["min"] = torch.minimum(input_stats[key]["min"], _min)

        # Create normalizer
        # Normalizer is a PyTorch parameter dict containing normalizers for all the keys
        normalizer = LinearNormalizer()
        normalizer.fit_from_input_stats(input_stats_dict=input_stats)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_sample_probabilities(self):
        return self.sample_probabilities

    def get_num_datasets(self):
        return self.num_datasets

    def get_num_episodes(self, index=None):
        if index is None:
            num_episodes = 0
            for i in range(self.num_datasets):
                num_episodes += self.replay_buffers[i].n_episodes
            return num_episodes
        else:
            return self.replay_buffers[index].n_episodes

    def __len__(self) -> int:
        length = 0
        for sampler in self.samplers:
            length += len(sampler)
        return length

    def _sample_to_data(self, sample, sampler_idx):
        target = sample["target"][0].astype(np.float32)
        agent_pos = sample["state"].astype(np.float32)

        data = {
            "obs": {
                "agent_pos": agent_pos,  # T_obs, 3
            },
            "target": target,  # 3
            "action": sample["action"].astype(np.float32),  # T, 2
        }

        if self.use_one_hot_encoding:
            if self.one_hot_encoding is None:
                data["one_hot_encoding"] = np.zeros(self.num_datasets).astype(np.float32)
                data["one_hot_encoding"][sampler_idx] = 1
            else:
                data["one_hot_encoding"] = self.one_hot_encoding

        # Add images to data
        if self.color_jitter is None:
            for key in self.rgb_keys:
                data["obs"][key] = np.moveaxis(sample[key], -1, 1) / 255.0
                if self.low_pass_on_wrist and key == "wrist_camera":
                    data["obs"][key] = low_pass_filter(
                        torch.from_numpy(data["obs"][key]), self.wrist_kernel.to(dtype=torch.float64)
                    ).numpy()
                if self.low_pass_on_overhead and key == "overhead_camera":
                    data["obs"][key] = low_pass_filter(
                        torch.from_numpy(data["obs"][key]), self.overhead_kernel.to(dtype=torch.float64)
                    ).numpy()
                del sample[key]
        else:
            # Stack images and apply color jitter to ensure
            # all cameras have consistent color jitter
            keys = self.rgb_keys
            length = sample[keys[0]].shape[0]

            imgs = np.moveaxis(np.vstack([sample[key] for key in keys]), -1, 1) / 255.0
            for i in range(3):
                scale = np.random.uniform(0.75, 1.25)  # TODO: these are hardcoded
                imgs[:, i, :, :] = np.clip(scale * imgs[:, i, :, :], 0, 1)

            # imgs = np.vstack([sample[key] for key in keys])
            imgs = self.transforms(torch.from_numpy(imgs)).numpy()
            for i, key in enumerate(keys):
                data["obs"][key] = imgs[i * length : (i + 1) * length]
                del sample[key]

        return data

    def _validate_zarr_configs(self, zarr_configs):
        """
        Validate the zarr configs for the PlanarPushingDataset.
        - Check if the zarr path exists
        - Check if the max_train_episodes is greater than 0
        - Check if the sampling_weight is greater than or equal to 0
        - Check if all or none of the zarr_configs have a sampling_weight
        """
        num_null_sampling_weights = 0
        N = len(zarr_configs)

        for zarr_config in zarr_configs:
            zarr_path = os.path.expanduser(zarr_config["path"])
            if not os.path.exists(zarr_path):
                raise ValueError(f"path {zarr_path} does not exist")

            max_train_episodes = zarr_config.get("max_train_episodes", None)
            if max_train_episodes is not None and max_train_episodes <= 0:
                raise ValueError(f"max_train_episodes must be greater than 0, got {max_train_episodes}")

            sampling_weight = zarr_config.get("sampling_weight", None)
            if sampling_weight is None:
                num_null_sampling_weights += 1
            elif sampling_weight < 0:
                raise ValueError(f"sampling_weight must be greater than or equal to 0, got {sampling_weight}")

        if num_null_sampling_weights not in [0, N]:
            raise ValueError("Either all or none of the zarr_configs must have a sampling_weight")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # To sample a sequence, first sample a dataset,
        # then sample a sequence from that dataset
        # Note that this implementation does not guarantee that each unique
        # sequence is sampled on every epoch!

        # Get sample
        if self.num_datasets == 1:
            sampler_idx = 0
            sampler = self.samplers[sampler_idx]
            data = sampler.sample_data(idx)
        else:
            sampler_idx = np.random.choice(self.num_datasets, p=self.sample_probabilities)
            sampler = self.samplers[sampler_idx]
            data = sampler.sample_data(idx % len(sampler))

        # Process sample
        # data = self._sample_to_data(sample, sampler_idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def minimum_enclosing_circle(P, r, eps=1e-9):
    """
    Helper function to check if a given set of points fits within a circle of radius r.

    Used to determine whether an action sequence is "idle".
    """
    P = np.asarray(P, dtype=np.float64)
    n = P.shape[0]

    if n <= 1:
        return True

    r2 = (r + eps) ** 2
    max_pair_dist2 = (2 * r + eps) ** 2

    # Cheap vectorised rejection
    # Compute all-pair squared distances; stop early if any exceed allowable max
    diff = P[:, None, :] - P[None, :, :]
    dist2 = np.einsum("...i,...i->...", diff, diff)
    if np.any(dist2 > max_pair_dist2):
        return False

    # Generate candidate circle centres (≤ O(n²))
    candidates = []

    # every point itself is a potential centre
    candidates.extend(P)

    # for every *distinct* pair generate up to two intersection centres
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = P[i], P[j]
            dx, dy = p2 - p1
            d = math.hypot(dx, dy)

            # If the points are too far apart, their radius-r circles do not intersect
            if d > 2 * r + eps:
                continue

            # Mid-point between p1 and p2
            mx, my = (p1 + p2) * 0.5

            # Distance from midpoint to each possible centre along the perpendicular
            h2 = r * r - 0.25 * d * d
            if h2 < 0:  # numerical safeguard
                continue
            if d == 0:  # coincident points, the midpoint itself is the only option
                candidates.append(np.array([mx, my]))
                continue

            h = math.sqrt(h2)
            # Unit vector perpendicular to (p2 - p1)
            ux, uy = -dy / d, dx / d
            offset = np.array([ux * h, uy * h])
            candidates.append(np.array([mx, my]) + offset)
            candidates.append(np.array([mx, my]) - offset)

    # test candidates in vectorised form
    P_expanded = P[None, :, :]  # shape (1, n, 2) broadcasted over candidates
    for c in candidates:
        # squared distances from candidate to every point
        if np.all(np.sum((P_expanded - c) ** 2, axis=-1) <= r2):
            return True

    return False


def prune_idle_actions(dataset, new_dataset_name, idle_tolerance=0.001):
    """
    Prune idle (no-op) actions from every dataset contained in the provided
    `PlanarPushingDataset` instance. Saves the pruned dataset to a new zarr file.

    Individual idle frames are removed from episodes. Episodes that are entirely
    idle (all frames are idle) are removed completely from the dataset.

    A frame is classified as "idle" if it belongs to a sliding window of consecutive
    actions (of size n_obs_steps + 1) where all actions in that window fall within
    a small circle of radius `idle_tolerance`.

    Args:
        dataset: PlanarPushingDataset object.
        new_dataset_name: Name for the new pruned dataset (will be saved at the same
                          location as original with this name).
        idle_tolerance: Maximum radius such that if actions all fall into a circle
                        of this radius, the episode is considered idle.
    """
    n_obs_steps = dataset.n_obs_steps

    start_time = time.time()

    # Iterate over each underlying dataset / replay buffer
    for dataset_idx, (replay_buffer, zarr_path) in enumerate(zip(dataset.replay_buffers, dataset.zarr_paths)):
        print(f"\nProcessing dataset {dataset_idx + 1}/{len(dataset.replay_buffers)}: {zarr_path}")

        # Get all keys in the replay buffer
        keys = list(replay_buffer.keys())

        # Create a mapping to track which frames to keep for each episode
        episode_frame_masks = []  # List of boolean masks, one per episode
        total_frames_before = 0
        total_frames_before_non_stationary_episodes = 0
        total_frames_after = 0
        episodes_removed_indices = []

        # Process each episode
        n_episodes = replay_buffer.n_episodes
        for episode_idx in range(n_episodes):
            # Get episode data
            episode_slice = replay_buffer.get_episode_slice(episode_idx)
            episode_length = episode_slice.stop - episode_slice.start

            # Get action data for this episode
            actions = replay_buffer["action"][episode_slice]  # (T, action_dim)

            # Initialize mask - keep all frames by default
            keep_mask = np.ones(episode_length, dtype=bool)

            # Check each possible window for idle actions
            # Mark frames as idle if they're part of an idle window
            for i in range(episode_length - n_obs_steps):
                action_window = actions[i : i + n_obs_steps + 1]
                if minimum_enclosing_circle(action_window, idle_tolerance):
                    # Mark these frames as idle (to be removed)
                    keep_mask[i : i + n_obs_steps + 1] = False

            # Check if entire episode is idle
            if not np.any(keep_mask):
                # Mark episode for removal by using None
                episode_frame_masks.append(None)
                episodes_removed_indices.append(episode_idx)
            else:
                episode_frame_masks.append(keep_mask)
                total_frames_before_non_stationary_episodes += episode_length
                total_frames_after += np.sum(keep_mask)

            total_frames_before += episode_length

        num_pruned = total_frames_before_non_stationary_episodes - total_frames_after
        prune_percentage = 100 * num_pruned / total_frames_before_non_stationary_episodes
        print(f"  Episodes before pruning: {n_episodes}")
        print(f"  Episodes after pruning: {n_episodes - len(episodes_removed_indices)}")
        print(f"  Episodes removed: {episodes_removed_indices}")
        print(f"  Frames in non-stationary episodes (before pruning): {total_frames_before_non_stationary_episodes}")
        print(f"  Frames after pruning: {total_frames_after}")
        print(f"  Frames pruned from non-stationary episodes: {num_pruned} ({prune_percentage:.2f}%)")
        print(f"  Total frames in dataset (before): {total_frames_before}")
        print(f"  Total frames in dataset (after): {total_frames_after}")

        # Create new pruned replay buffer using the ReplayBuffer helper class
        pruned_buffer = ReplayBuffer.create_empty_numpy()

        # Add each pruned episode to the new buffer (skip entirely idle episodes)
        for episode_idx in range(n_episodes):
            keep_mask = episode_frame_masks[episode_idx]

            # Skip episodes that are entirely idle (marked as None)
            if keep_mask is None:
                continue

            episode_data = {}

            # Extract pruned data for each key
            for key in keys:
                episode_slice = replay_buffer.get_episode_slice(episode_idx)
                original_episode = replay_buffer[key][episode_slice]
                # Keep only non-idle frames
                pruned_episode = original_episode[keep_mask]
                episode_data[key] = pruned_episode

            # Add the pruned episode to the new buffer
            pruned_buffer.add_episode(episode_data)

        # Determine the new zarr path
        original_dir = os.path.dirname(zarr_path)
        new_zarr_path = os.path.join(original_dir, new_dataset_name)

        # Save to new zarr file using ReplayBuffer's save method
        print(f"  Saving pruned dataset to: {new_zarr_path}")

        # Define compression for different data types
        compressors = {}
        for key in keys:
            # Use stronger compression for image data
            sample_data = pruned_buffer[key]
            if len(sample_data.shape) == 4:  # Image data (T, H, W, C)
                compressors[key] = "disk"  # zstd with bitshuffle
            else:  # Low-dimensional data
                compressors[key] = "default"  # lz4

        pruned_buffer.save_to_path(new_zarr_path, compressors=compressors)

        print("  Successfully saved pruned dataset!")

    total_time = time.time() - start_time
    print(f"\nTotal pruning time: {total_time:.2f} seconds")

    return dataset


if __name__ == "__main__":
    shape_meta = {
        "action": {"shape": [2]},
        "obs": {
            "agent_pos": {"type": "low_dim", "shape": [3]},
            "overhead_camera": {"type": "rgb", "shape": [3, 128, 128]},
            "wrist_camera": {"type": "rgb", "shape": [3, 128, 128]},
        },
    }
    zarr_configs = [
        # {
        #     'path': 'data/planar_pushing/underactuated_data.zarr',
        #     'max_train_episodes': None,
        #     'sampling_weight': 1.0
        # },
        {
            # 'path': 'data/planar_pushing_cotrain/visual_mean_shift/visual_mean_shift_level_2.zarr',
            # "path": "data/planar_pushing_cotrain/sim_sim_tee_data_carbon_large.zarr",
            "path": "data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large.zarr",
            "max_train_episodes": None,
            "sampling_weight": 1.0,
        }
    ]
    n_obs_steps = 2
    color_jitter = {
        "brightness": 0.15,
        # 'contrast': 0.5,
        "saturation": 0.15,
        "hue": 0.15,
    }

    dataset = PlanarPushingDataset(
        zarr_configs=zarr_configs,
        shape_meta=shape_meta,
        horizon=16,
        n_obs_steps=n_obs_steps,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.05,
        low_pass_on_wrist=True,
        # color_jitter=color_jitter
    )

    dataset.__getitem__(0)
    print("=" * 60)
    print("DATASET SIZE INFORMATION")
    print("=" * 60)
    print("Initialized dataset")
    print(f"Number of datasets: {dataset.get_num_datasets()}")
    print(f"Total episodes (train + val): {dataset.get_num_episodes()}")
    print(f"Training dataset length: {len(dataset)}")

    # Print detailed information for each dataset
    for i in range(dataset.get_num_datasets()):
        print(f"\nDataset {i}:")
        print(f"  Zarr path: {dataset.zarr_paths[i]}")
        print(f"  Total episodes: {dataset.get_num_episodes(index=i)}")
        print(f"  Training episodes: {np.sum(dataset.train_masks[i])}")
        print(f"  Validation episodes: {np.sum(dataset.val_masks[i])}")
        print(f"  Sampling weight: {dataset.sample_probabilities[i]:.4f}")
        print(f"  Sampler length: {len(dataset.samplers[i])}")

    print(f"\nPer-dataset sample probabilities: {dataset.sample_probabilities}")
    print("=" * 60)

    # Test get validation dataset
    for i in range(dataset.get_num_datasets()):
        val_dataset = dataset.get_validation_dataset(index=i)
        print(f"Got validation dataset {i} with length: {len(val_dataset)}")

    # Test normalizer
    normalizer = dataset.get_normalizer()

    # dataset = prune_idle_actions(dataset, new_dataset_name="sim_sim_tee_data_carbon_large_pruned.zarr")
    # exit()

    for i in range(10):
        idx = random.randint(0, len(dataset) - 1)
        idx = i % len(dataset)

        sample = dataset[idx]
        states = sample["obs"]["agent_pos"]
        actions = sample["action"]

        print(sample["obs"].keys())

        print(f"Sample states : {states}")
        print(f"Sample actions: {actions}")
        print(f"Sample target : {sample['target']}")
        print()
        print("Press any key to continue. Ctrl+\\ to exit.\n")

        # Just display camera (wrist and overhead) images
        for key, attr in sample["obs"].items():
            if key == "agent_pos":
                continue

            for i in range(n_obs_steps):
                image_array = attr[i].detach().numpy().transpose(1, 2, 0)

                # Convert the RGB array to BGR
                image_array[:, :, 0], image_array[:, :, 2] = image_array[:, :, 2], image_array[:, :, 0].copy()
                # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                # Display the image using OpenCV
                # cv2.imshow(f"{key}_{i}", image_array)
                # cv2.waitKey(0)  # Wait for a key press to close the image window
                # cv2.destroyAllWindows()

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=64,
    #     num_workers=1,
    #     persistent_workers=False,
    #     pin_memory=True,
    #     shuffle=True
    # )

    # for local_epoch_idx in range(50):
    #     with tqdm.tqdm(train_dataloader) as tepoch:
    #         for batch_idx, batch in enumerate(tepoch):
    #             print(local_epoch_idx, batch_idx)

    # while True:
    #     idx = random.randint(0, len(dataset)-1)
    #     sample = dataset[idx]
