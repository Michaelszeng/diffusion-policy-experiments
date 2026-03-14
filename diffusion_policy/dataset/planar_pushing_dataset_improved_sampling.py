import math
import os
import random
import time
from typing import Dict, List

import cv2
import numpy as np
import torch
import zarr
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseZarrImageDataset, gaussian_kernel

class PlanarPushingDataset(BaseZarrImageDataset):
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
        self.low_pass_on_wrist = low_pass_on_wrist
        if low_pass_on_wrist:
            self.wrist_kernel = gaussian_kernel(kernel_size=9, sigma=3, channels=3)
        self.low_pass_on_overhead = low_pass_on_overhead
        if low_pass_on_overhead:
            self.overhead_kernel = gaussian_kernel(kernel_size=9, sigma=3, channels=3)

    def _get_buffer_keys(self) -> List[str]:
        return self.rgb_keys + ["state", "action", "target"]

    def _lowdim_key_map(self):
        return {"action": "action", "agent_pos": "state", "target": "target"}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.num_datasets == 1:
            data = self.samplers[0].sample_data(idx)
        else:
            sampler_idx = np.random.choice(self.num_datasets, p=self.sample_probabilities)
            sampler = self.samplers[sampler_idx]
            data = sampler.sample_data(idx % len(sampler))

        if self.transforms is not None and self.rgb_keys:
            data = self._apply_color_jitter(data)

        return dict_apply(data, torch.from_numpy)


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
        {
            "path": "data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large_pruned_obs_horizon_3_idle_tol_0_003.zarr",
            "sampling_weight": 1.0,
            "max_train_episodes": 10,
        }
    ]
    n_obs_steps = 2
    color_jitter = {
        "brightness": 0.15,
        "contrast": 0.15,
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
        val_ratio=0.0625,
        low_pass_on_wrist=True,
        color_jitter=color_jitter
    )

    dataset.__getitem__(0)
    print("=" * 60)
    print("DATASET INFORMATION")
    print(f"Zarr structure: {zarr.tree(zarr.open_group(zarr_configs[0]['path'], mode='r'))}")
    print(f"Number of datasets: {dataset.get_num_datasets()}")
    print(f"Total episodes (train + val): {dataset.get_num_episodes()}")
    print(f"Training dataset length (after applying max_train_episodes) (number of training samples): {len(dataset)}")

    # Print detailed information for each dataset
    for i in range(dataset.get_num_datasets()):
        print(f"\nDataset {i}:")
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
