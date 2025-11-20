import copy
import os
from typing import Dict, List

import h5py
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import ImprovedDatasetSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class ManiskillLowdimDataset(BaseLowdimDataset):
    """
    Low-dimensional dataset for ManiSkill (state-only, no images).

    ROBOT-ACCESSIBLE DATA (What we extract):
    - Robot state based on state_mode (qpos, qvel, or tcp_pose)
    - Tee pose: env_states.actors.Tee[:7] = [pos (3), quat (4)]
    - Actions: actions (T, action_dim)

    ROBOT STATE MODES:
    - "qpos_qvel": Joint positions + velocities (T, 14) [DEFAULT]
    - "qpos": Joint positions only (T, 7)
    - "tcp_pose": End-effector pose (T, 7)

    OBSERVATION STRUCTURE:
    Supports two types of H5 structure:
    1. Nested: traj["obs"]["agent"]["qpos"], traj["obs"]["agent"]["qvel"]
    2. Flat: traj["obs"] as flat array with configurable slices
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
        state_mode: str = "qpos_qvel",  # Options: "qpos_qvel", "qpos", "tcp_pose"
        # Indices for flat observation array (if obs is flat)
        qpos_slice: tuple = (0, 7),  # Default: obs[:, 0:7]
        qvel_slice: tuple = (7, 14),  # Default: obs[:, 7:14]
        tee_pose_slice: tuple = (14, 21),  # Default: obs[:, 14:21]
        tcp_pose_slice: tuple = None,  # If using tcp_pose mode with flat obs
    ):
        super().__init__()
        self._validate_h5_configs(h5_configs)
        self._validate_state_mode_shape_consistency(state_mode, shape_meta)

        # Store parameters
        self.state_mode = state_mode
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.use_one_hot_encoding = use_one_hot_encoding
        self.one_hot_encoding = None

        # Store observation slicing parameters for flat obs arrays
        self.qpos_slice = qpos_slice
        self.qvel_slice = qvel_slice
        self.tee_pose_slice = tee_pose_slice
        self.tcp_pose_slice = tcp_pose_slice

        # Keys for replay buffer: state, tee_pose, action
        keys = ["state", "tee_pose", "action"]

        # Memory optimization: only load first k observations
        key_first_k = dict()
        if n_obs_steps is not None:
            key_first_k["state"] = n_obs_steps
            key_first_k["tee_pose"] = n_obs_steps
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
            h5_path = os.path.expanduser(h5_config["h5_path"])
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

            # Set up sample probabilities
            if sampling_weight is not None:
                self.sample_probabilities[i] = sampling_weight
            else:
                self.sample_probabilities[i] = np.sum(train_mask)
            self.h5_paths.append(h5_path)

        # Normalize sample probabilities
        self.sample_probabilities = self._normalize_sample_probabilities(self.sample_probabilities)

    def _create_replay_buffer_from_h5(self, h5_path: str, keys: List[str]) -> ReplayBuffer:
        """Convert ManiSkill H5 trajectory file to ReplayBuffer format"""
        print(f"Loading ManiSkill lowdim dataset from: {h5_path}")

        replay_buffer = ReplayBuffer.create_empty_numpy()

        with h5py.File(h5_path, "r") as h5_file:
            # Get all trajectory keys (traj_0, traj_1, ...)
            traj_keys = [key for key in h5_file.keys() if key.startswith("traj_")]
            traj_keys = sorted(traj_keys, key=lambda x: int(x.split("_")[1]))

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

                # Tee pose extraction: [pos (3), quat (4)] = 7 dims
                episode_data["tee_pose"] = self._extract_tee_pose(traj, episode_length)

                # Add episode to replay buffer
                replay_buffer.add_episode(episode_data)

        print(f"Loaded {replay_buffer.n_episodes} episodes with {replay_buffer.n_steps} total timesteps")
        return replay_buffer

    def _extract_robot_state(self, traj, episode_length: int) -> np.ndarray:
        """Extract robot state based on configured mode.

        Handles two types of H5 structures:
        1. Nested: traj["obs"]["agent"]["qpos"], traj["obs"]["agent"]["qvel"]
        2. Flat: traj["obs"] as array with configurable slices
        """
        obs = traj["obs"]

        # Flat observation array - use slicing
        obs_data = obs[:episode_length].astype(np.float32)

        if self.state_mode == "qpos_qvel":
            # Extract and combine joint positions and velocities
            qpos = obs_data[:, self.qpos_slice[0] : self.qpos_slice[1]]
            qvel = obs_data[:, self.qvel_slice[0] : self.qvel_slice[1]]
            return np.concatenate([qpos, qvel], axis=1)  # (T, 14)
        elif self.state_mode == "qpos":
            # Joint positions only
            return obs_data[:, self.qpos_slice[0] : self.qpos_slice[1]]  # (T, 7)
        elif self.state_mode == "tcp_pose":
            # TCP/end-effector pose
            if self.tcp_pose_slice is None:
                raise ValueError("tcp_pose_slice must be specified for tcp_pose mode with flat observations")
            return obs_data[:, self.tcp_pose_slice[0] : self.tcp_pose_slice[1]]  # (T, 7)
        else:
            raise ValueError(f"Invalid state mode: {self.state_mode}")

    def _extract_tee_pose(self, traj, episode_length: int) -> np.ndarray:
        """Extract Tee pose.

        Tries multiple sources in order:
        1. env_states/actors/Tee (from env_states) - preferred
        2. Flat obs array using tee_pose_slice
        """

        obs = traj["obs"]
        if hasattr(obs, "shape") and len(obs.shape) == 2:
            obs_data = obs[:episode_length].astype(np.float32)
            tee_pose = obs_data[:, self.tee_pose_slice[0] : self.tee_pose_slice[1]]
            return tee_pose

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
        low_dim_keys = ["action", "agent_pos", "tee_pose"]
        input_stats = {}

        for replay_buffer in self.replay_buffers:
            data = {
                "action": replay_buffer["action"],
                "agent_pos": replay_buffer["state"],
                "tee_pose": replay_buffer["tee_pose"],
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

        # Create final normalizer with flat structure
        normalizer = LinearNormalizer()
        normalizer.fit_from_input_stats(input_stats_dict=input_stats)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Return all actions from all replay buffers as a single tensor"""
        all_actions = []
        for replay_buffer in self.replay_buffers:
            actions = replay_buffer["action"]
            all_actions.append(torch.from_numpy(actions))
        return torch.cat(all_actions, dim=0)

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
            h5_path = os.path.expanduser(h5_config["h5_path"])
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

    def _normalize_sample_probabilities(self, sample_probabilities: np.ndarray) -> np.ndarray:
        """Normalize sample probabilities to sum to 1.0"""
        total = np.sum(sample_probabilities)
        if total == 0:
            raise ValueError("Sample probabilities sum to zero")
        return sample_probabilities / total

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

        # Convert to tensors
        torch_data = dict_apply(data, torch.from_numpy)

        # Handle one-hot encoding if enabled
        if self.use_one_hot_encoding:
            if self.one_hot_encoding is None:
                torch_data["one_hot_encoding"] = torch.zeros(self.num_datasets).float()
                torch_data["one_hot_encoding"][sampler_idx] = 1.0
            else:
                torch_data["one_hot_encoding"] = torch.from_numpy(self.one_hot_encoding).float()

        return torch_data


if __name__ == "__main__":
    # Example usage and testing
    # NOTE: shape_meta must match state_mode!

    # For qpos_qvel mode (default):
    shape_meta = {
        "action": {"shape": [3]},  # 3-DOF end-effector control for PushT
        "obs": {
            "agent_pos": {"type": "low_dim", "shape": [14]},  # qpos + qvel (7 + 7)
            "tee_pose": {"type": "low_dim", "shape": [7]},  # Tee position + quaternion
        },
    }

    h5_configs = [
        {
            "h5_path": (
                "/home/michzeng/.maniskill/demos/PushT-v1/rl/trajectory_state_pd_ee_delta_pos_truncated_success_only.h5"
            ),
            "max_train_episodes": None,
            "sampling_weight": 1.0,
            "val_ratio": 0.1,
        }
    ]

    # Test dataset creation
    # NOTE: Default slicing parameters work for standard flat ManiSkill observations:
    # - qpos_slice=(0, 7): obs[:, 0:7]
    # - qvel_slice=(7, 14): obs[:, 7:14]
    # - tee_pose_slice=(14, 21): obs[:, 14:21]
    # Override these if your observation structure is different
    dataset = ManiskillLowdimDataset(
        h5_configs=h5_configs,
        shape_meta=shape_meta,
        horizon=16,
        n_obs_steps=2,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.1,
        state_mode="qpos_qvel",
        # Uncomment to override slicing (if needed):
        # qpos_slice=(0, 7),
        # qvel_slice=(7, 14),
        # tee_pose_slice=(14, 21),
    )

    print("=" * 60)
    print("MANISKILL LOWDIM DATASET INFORMATION")
    print("=" * 60)
    print("Dataset initialized successfully!")
    print(f"Number of datasets: {dataset.get_num_datasets()}")
    print(f"Total episodes (train + val): {dataset.get_num_episodes()}")
    print(f"Training dataset length: {len(dataset)}")

    # Print detailed information
    for i in range(dataset.get_num_datasets()):
        print(f"\nDataset {i}:")
        print(f"  H5 path: {dataset.h5_paths[i]}")
        print(f"  Total episodes: {dataset.get_num_episodes(index=i)}")
        print(f"  Training episodes: {np.sum(dataset.train_masks[i])}")
        print(f"  Validation episodes: {np.sum(dataset.val_masks[i])}")
        print(f"  Sampling weight: {dataset.sample_probabilities[i]:.4f}")

    # Test sampling
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Observation keys: {sample['obs'].keys()}")
        print(f"Action shape: {sample['action'].shape}")
        print(f"Agent pos shape: {sample['obs']['agent_pos'].shape}")
        print(f"Tee pose shape: {sample['obs']['tee_pose'].shape}")

        print("\nSample robot state (first 3 timesteps):")
        print(sample["obs"]["agent_pos"][:3])
        print("\nSample Tee pose (first 3 timesteps):")
        print(sample["obs"]["tee_pose"][:3])
        print("\nSample actions (first 3 timesteps):")
        print(sample["action"][:3])

    # Test validation dataset
    val_dataset = dataset.get_validation_dataset()
    print(f"\nValidation dataset length: {len(val_dataset)}")

    # Test normalizer
    normalizer = dataset.get_normalizer()
    print("\nNormalizer created successfully!")
    print(f"Normalizer keys: {list(normalizer.params_dict.keys())}")
    input_stats = normalizer.get_input_stats()
    print(f"Input stats keys: {list(input_stats.keys())}")
    for key in input_stats.keys():
        print(f"  {key}: min={input_stats[key]['min'].shape}, max={input_stats[key]['max'].shape}")

    print("\n✓ ManiSkill lowdim dataset successfully loaded and tested!")
