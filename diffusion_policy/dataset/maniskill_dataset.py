import copy
import json
import os
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from torchvision import transforms

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import ImprovedDatasetSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset, gaussian_kernel, low_pass_filter
from diffusion_policy.model.common.normalizer import LinearNormalizer

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class ManiskillDataset(BaseImageDataset):
    """
    Dataset compatible with ManiSkill h5/JSON trajectory files.

    ROBOT-ACCESSIBLE DATA (What we extract):
    - Robot joint positions: obs.agent.qpos (T, 7)
    - Robot joint velocities: obs.agent.qvel (T, 7)
    - End-effector pose: obs.extra.tcp_pose (T, 7)
    - RGB camera images: obs.sensor_data.base_camera.rgb (T, H, W, C)
    - Actions: actions (T, 7) - joint commands

    PRIVILEGED DATA (What we ignore):
    - Episode completion signals: terminated, truncated, success
    - Object poses: env_states.actors.Tee (CHEATING - object being manipulated)
    - Goal poses: env_states.actors.goal_Tee (CHEATING - target object pose)
    - Goal end-effector: env_states.actors.goal_ee (CHEATING - target robot pose)
    - Environment actors: env_states.actors.* (privileged simulator info)
    - Camera parameters: obs.sensor_param.* (not needed for policy)

    ROBOT STATE MODES:
    - "qpos_qvel": Joint positions + velocities (T, 14) [DEFAULT]
    - "qpos": Joint positions only (T, 7)
    - "tcp_pose": End-effector pose (T, 7)
    """

    def __init__(
        self,
        h5_configs: List[Dict],
        shape_meta: Dict,
        use_one_hot_encoding: bool = False,
        horizon: int = 1,
        n_obs_steps: int = None,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        color_jitter: Dict = None,
        low_pass_on_wrist: bool = False,
        low_pass_on_overhead: bool = False,
        state_mode: str = "qpos_qvel",  # Options: "qpos_qvel", "qpos", "tcp_pose"
        camera_keys: List[str] = None,  # List of camera names to extract (e.g., ["base_camera"])
    ):
        super().__init__()
        self._validate_h5_configs(h5_configs)
        self._validate_state_mode_shape_consistency(state_mode, shape_meta)

        # Low-pass filtering setup
        self.low_pass_on_wrist = low_pass_on_wrist
        if low_pass_on_wrist:
            self.wrist_kernel = gaussian_kernel(kernel_size=9, sigma=3, channels=3)
        self.low_pass_on_overhead = low_pass_on_overhead
        if low_pass_on_overhead:
            self.overhead_kernel = gaussian_kernel(kernel_size=9, sigma=3, channels=3)

        # Store parameters needed for data extraction
        self.state_mode = state_mode
        self.camera_keys = camera_keys or []  # Default to empty list if no cameras specified

        # Extract RGB keys from shape_meta (needed for data extraction)
        self.rgb_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type_name = attr.get("type", "")
            if type_name == "rgb":
                self.rgb_keys.append(key)

        # Keys needed for replay buffer (robot state will be mapped from articulations)
        keys = self.rgb_keys + ["state", "action"]

        # Memory optimization: only load first k observations for images
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in keys:
                key_first_k[key] = n_obs_steps
        key_first_k["action"] = horizon

        # Load all H5 datasets and convert to replay buffers
        self.num_datasets = len(h5_configs)
        self.replay_buffers = []
        self.train_masks = []
        self.val_masks = []
        self.samplers = []
        self.sample_probabilities = np.zeros(len(h5_configs))
        self.h5_paths = []

        for i, h5_config in enumerate(h5_configs):
            # Extract config info
            h5_path = h5_config["h5_path"]
            max_train_episodes = h5_config.get("max_train_episodes", None)
            sampling_weight = h5_config.get("sampling_weight", None)

            # Create replay buffer from H5 file
            replay_buffer = self._create_replay_buffer_from_h5(h5_path, keys)
            self.replay_buffers.append(replay_buffer)
            n_episodes = replay_buffer.n_episodes

            # Set up masks
            if "val_ratio" in h5_config and h5_config["val_ratio"] is not None:
                dataset_val_ratio = h5_config["val_ratio"]
            else:
                dataset_val_ratio = val_ratio
            val_mask = get_val_mask(n_episodes=n_episodes, val_ratio=dataset_val_ratio, seed=seed)
            train_mask = ~val_mask
            train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

            self.train_masks.append(train_mask)
            self.val_masks.append(val_mask)

            # Set up sampler
            sampler = ImprovedDatasetSampler(
                replay_buffer=self.replay_buffers[-1],
                sequence_length=horizon,
                shape_meta=shape_meta,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
                key_first_k=key_first_k,
            )
            self.samplers.append(sampler)

            # Set up sample probabilities and paths
            if sampling_weight is not None:
                self.sample_probabilities[i] = sampling_weight
            else:
                self.sample_probabilities[i] = np.sum(train_mask)
            self.h5_paths.append(h5_path)

        # Normalize sample probabilities
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

        # Store parameters
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.use_one_hot_encoding = use_one_hot_encoding
        self.one_hot_encoding = None  # if val dataset, this will not be None

    def _create_replay_buffer_from_h5(self, h5_path: str, keys: List[str]) -> ReplayBuffer:
        """Convert ManiSkill H5 trajectory file to ReplayBuffer format"""
        print(f"Loading ManiSkill dataset from: {h5_path}")

        replay_buffer = ReplayBuffer.create_empty_numpy()

        with h5py.File(h5_path, "r") as h5_file:
            # Get all trajectory keys (traj_0, traj_1, ...)
            traj_keys = [key for key in h5_file.keys() if key.startswith("traj_")]
            traj_keys = sorted(traj_keys, key=lambda x: int(x.split("_")[1]))  # Sort by episode ID

            print(f"Found {len(traj_keys)} episodes.")

            for traj_key in traj_keys:
                traj = h5_file[traj_key]

                # Extract episode data
                episode_data = {}

                # Actions: (T, action_dim)
                actions = traj["actions"][:]
                episode_length = len(actions)
                episode_data["action"] = actions.astype(np.float32)

                # Robot state extraction based on mode
                episode_data["state"] = self._extract_robot_state(traj, episode_length)

                # Handle RGB observations from ManiSkill's obs structure
                self._extract_rgb_observations(traj, episode_data, episode_length)

                # Add episode to replay buffer
                replay_buffer.add_episode(episode_data)

        print(f"Loaded {replay_buffer.n_episodes} episodes with {replay_buffer.n_steps} total timesteps")
        return replay_buffer

    def _extract_robot_state(self, traj, episode_length: int) -> np.ndarray:
        """Extract robot state based on configured mode"""
        if self.state_mode == "qpos_qvel":
            # Combine joint positions and velocities
            if "obs" in traj and "agent" in traj["obs"]:
                qpos = traj["obs"]["agent"]["qpos"][:episode_length].astype(np.float32)
                qvel = traj["obs"]["agent"]["qvel"][:episode_length].astype(np.float32)
                return np.concatenate([qpos, qvel], axis=1)  # (T, 14)
        elif self.state_mode == "qpos":
            # Joint positions only
            if "obs" in traj and "agent" in traj["obs"]:
                return traj["obs"]["agent"]["qpos"][:episode_length].astype(np.float32)  # (T, 7)
        elif self.state_mode == "tcp_pose":
            # TCP/end-effector pose
            if "obs" in traj and "extra" in traj["obs"]:
                return traj["obs"]["extra"]["tcp_pose"][:episode_length].astype(np.float32)  # (T, 7)
        else:
            raise ValueError(f"Invalid state mode: {self.state_mode}")

    def _extract_rgb_observations(self, traj, episode_data: Dict, episode_length: int):
        """Extract RGB observations from ManiSkill trajectory"""
        sensor_data = traj["obs"]["sensor_data"]  #

        # Map RGB keys to camera names
        # Assume rgb_key format like "base_camera" maps to sensor_data["base_camera"]["rgb"]
        for rgb_key in self.rgb_keys:
            # Extract RGB images: (T+1, C, H, W) -> (T, C, H, W)
            # NOTE: Maniskill dataset has 1 more observation than action, so we are just truncating the last observation
            rgb_images = sensor_data[rgb_key]["rgb"][:episode_length]
            episode_data[rgb_key] = rgb_images.astype(np.uint8)

    def get_validation_dataset(self, index=None):
        """Create validation dataset"""
        val_set = copy.copy(self)

        if index is None:
            assert self.num_datasets == 1, "Must specify validation dataset index if multiple datasets"
            index = 0
        else:
            val_set.replay_buffers = [self.replay_buffers[index]]
            val_set.train_masks = [self.train_masks[index]]
            val_set.val_masks = [self.val_masks[index]]
            val_set.h5_paths = [self.h5_paths[index]]
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
        """Compute normalizer from data"""
        assert mode == "limits", "Only supports limits mode"
        low_dim_keys = ["action", "agent_pos"]
        input_stats = {}

        for replay_buffer in self.replay_buffers:
            data = {
                "action": replay_buffer["action"],
                "agent_pos": replay_buffer["state"],
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

        # Create final normalizer
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

    def _validate_h5_configs(self, h5_configs):
        """Validate H5 configs"""
        num_null_sampling_weights = 0
        N = len(h5_configs)

        for h5_config in h5_configs:
            h5_path = h5_config["h5_path"]
            if not os.path.exists(h5_path):
                raise ValueError(f"H5 file path {h5_path} does not exist")

            max_train_episodes = h5_config.get("max_train_episodes", None)
            if max_train_episodes is not None and max_train_episodes <= 0:
                raise ValueError(f"max_train_episodes must be greater than 0, got {max_train_episodes}")

            sampling_weight = h5_config.get("sampling_weight", None)
            if sampling_weight is None:
                num_null_sampling_weights += 1
            elif sampling_weight < 0:
                raise ValueError(f"sampling_weight must be greater than or equal to 0, got {sampling_weight}")

        if num_null_sampling_weights not in [0, N]:
            raise ValueError("Either all or none of the h5_configs must have a sampling_weight")

    def _validate_state_mode_shape_consistency(self, state_mode: str, shape_meta: Dict):
        """Validate that shape_meta.obs.agent_pos.shape matches the selected state_mode"""
        expected_shapes = {
            "qpos_qvel": 14,  # joint positions (7) + velocities (7)
            "qpos": 7,  # joint positions only
            "tcp_pose": 7,  # end-effector pose (position + quaternion)
        }

        if state_mode not in expected_shapes:
            raise ValueError(f"Invalid state_mode '{state_mode}'. Must be one of: {list(expected_shapes.keys())}")

        expected_shape = expected_shapes[state_mode]
        actual_shape = shape_meta["obs"]["agent_pos"]["shape"][0]
        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: state_mode '{state_mode}' expects agent_pos shape [{expected_shape}], "
                f"but shape_meta specifies [{actual_shape}]. "
                f"Please update shape_meta.obs.agent_pos.shape in your config to match the state_mode."
            )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Sample a sequence from the dataset"""
        # Sample from datasets according to probabilities
        if self.num_datasets == 1:
            sampler_idx = 0
            sampler = self.samplers[sampler_idx]
            data = sampler.sample_data(idx)
        else:
            sampler_idx = np.random.choice(self.num_datasets, p=self.sample_probabilities)
            sampler = self.samplers[sampler_idx]
            data = sampler.sample_data(idx % len(sampler))

        # ImprovedDatasetSampler already formats the data correctly, just convert to tensors
        # Follow the same approach as planar_pushing_dataset_improved_sampling.py
        torch_data = dict_apply(data, torch.from_numpy)

        # NOTE: Don't normalize images here - the training workspace handles normalization
        # Images should remain as uint8 [0, 255] at this stage

        # Handle one-hot encoding if enabled
        if self.use_one_hot_encoding:
            if self.one_hot_encoding is None:
                torch_data["one_hot_encoding"] = torch.zeros(self.num_datasets).float()
                torch_data["one_hot_encoding"][sampler_idx] = 1.0
            else:
                torch_data["one_hot_encoding"] = torch.from_numpy(self.one_hot_encoding).float()

        return torch_data


if __name__ == "__main__":
    # Example usage and testing with RGB observations
    # NOTE: shape_meta must match state_mode! Examples:

    # For qpos_qvel mode (default):
    shape_meta = {
        "action": {"shape": [7]},  # 7-DOF robot arm from ManiSkill inspection
        "obs": {
            "agent_pos": {"type": "low_dim", "shape": [14]},  # qpos + qvel (7 + 7)
            "base_camera": {"type": "rgb", "shape": [3, 128, 128]},  # RGB camera from ManiSkill
        },
    }

    # For qpos or tcp_pose modes, use shape [7] instead:
    # shape_meta = {
    #     "action": {"shape": [7]},
    #     "obs": {
    #         "agent_pos": {"type": "low_dim", "shape": [7]},  # qpos only OR tcp_pose
    #         "base_camera": {"type": "rgb", "shape": [3, 128, 128]},
    #     },
    # }

    h5_configs = [
        {
            "h5_path": "/home/michzeng/.maniskill/demos/PushT-v1/rl/trajectory.rgb.pd_joint_delta_pos.physx_cuda.h5",
            "json_path": "/home/michzeng/.maniskill/demos/PushT-v1/rl/"
            "trajectory.rgb.pd_joint_delta_pos.physx_cuda.json",  # Optional
            "max_train_episodes": None,  # Use all episodes for training
            "sampling_weight": 1.0,
            "val_ratio": 0.1,  # 10% for validation
        }
    ]

    # Test dataset creation
    dataset = ManiskillDataset(
        h5_configs=h5_configs,
        shape_meta=shape_meta,
        horizon=16,
        n_obs_steps=2,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.1,
        state_mode="qpos_qvel",  # Use joint positions + velocities
        camera_keys=["base_camera"],  # Extract base_camera RGB images
    )

    print("=" * 60)
    print("MANISKILL DATASET INFORMATION")
    print("=" * 60)
    print("Dataset initialized successfully!")
    print(f"Number of datasets: {dataset.get_num_datasets()}")
    print(f"Total episodes (train + val): {dataset.get_num_episodes()}")
    print(f"Training dataset length: {len(dataset)}")

    # Print detailed information for each dataset
    for i in range(dataset.get_num_datasets()):
        print(f"\nDataset {i}:")
        print(f"  H5 path: {dataset.h5_paths[i]}")
        print(f"  Total episodes: {dataset.get_num_episodes(index=i)}")
        print(f"  Training episodes: {np.sum(dataset.train_masks[i])}")
        print(f"  Validation episodes: {np.sum(dataset.val_masks[i])}")
        print(f"  Sampling weight: {dataset.sample_probabilities[i]:.4f}")
        print(f"  Sampler length: {len(dataset.samplers[i])}")

    # Test sampling
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Observation keys: {sample['obs'].keys()}")
        print(f"Action shape: {sample['action'].shape}")
        for key, value in sample["obs"].items():
            print(f"  {key} shape: {value.shape}")

        # Print sample values for robot state and actions
        print("\nSample robot state (first 3 timesteps):")
        print(sample["obs"]["agent_pos"][:3])
        print("Sample actions (first 3 timesteps):")
        print(sample["action"][:3])

    # Test validation dataset
    val_dataset = dataset.get_validation_dataset()
    print(f"\nValidation dataset length: {len(val_dataset)}")

    # Test normalizer
    normalizer = dataset.get_normalizer()
    print("Normalizer created successfully!")
    print(f"Normalizer type: {type(normalizer)}")

    # Visualize RGB images and print obs and actions for a few samples
    print("\n" + "=" * 60)
    print("VISUALIZING RGB IMAGES")
    print("=" * 60)

    # Sample a few episodes
    num_samples = min(3, len(dataset))
    fig_width = 5 * len(dataset.rgb_keys)
    fig_height = 4 * num_samples
    fig, axes = plt.subplots(num_samples, len(dataset.rgb_keys), figsize=(fig_width, fig_height))

    # Handle single row case
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    # Handle single column case
    if len(dataset.rgb_keys) == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_samples):
        sample = dataset[i * (len(dataset) // num_samples)]

        print(f"Sample states  ({sample['obs']['agent_pos'].shape}): {sample['obs']['agent_pos']}")
        print(f"Sample actions: ({sample['action'].shape}): {sample['action']}")

        for j, rgb_key in enumerate(dataset.rgb_keys):
            # Get the first observation timestep
            img = sample["obs"][rgb_key][0]  # First timestep

            # Check the shape and convert to [H, W, C] for matplotlib
            if img.shape[0] == 3:  # [C, H, W] format
                img_np = img.permute(1, 2, 0).numpy()
            elif img.shape[1] == 3:  # [H, C, W] format
                img_np = img.permute(0, 2, 1).numpy()
            else:  # [H, W, C] format
                img_np = img.numpy()

            # Normalize to [0, 1] for display if uint8
            if img_np.dtype == np.uint8:
                img_np = img_np.astype(np.float32) / 255.0

            # Display image
            axes[i, j].imshow(img_np)
            axes[i, j].set_title(f"Sample {i} - {rgb_key}")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig("maniskill_dataset_samples.png", dpi=150, bbox_inches="tight")
    print("✓ Saved visualization to: maniskill_dataset_samples.png")
    print(f"  Displaying {num_samples} samples with {len(dataset.rgb_keys)} camera(s) each")

    try:
        plt.show()
    except Exception as e:
        print(f"  (Could not display interactively: {e})")

    print("\n✓ ManiSkill dataset successfully loaded and tested!")
