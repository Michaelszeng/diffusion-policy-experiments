import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import zarr
from torchvision import transforms

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import ImprovedDatasetSampler, downsample_mask, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer


def gaussian_kernel(kernel_size=9, sigma=3, channels=3):
    """Create a Gaussian kernel for convolution."""
    # Create 1D Gaussian
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    # Create 2D Gaussian
    g2 = g[:, None] * g[None, :]
    kernel = g2.expand(channels, 1, kernel_size, kernel_size)
    return kernel


class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> "BaseLowdimDataset":
        # return an empty dataset by default
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def load_replay_buffer(self, path: str, keys: List[str], config: Dict):
        """Load data from path into a ReplayBuffer for one dataset config."""
        raise NotImplementedError()

    def store_replay_buffer(self, replay_buffer, path: str) -> None:
        """Save a ReplayBuffer to disk in the dataset's native on-disk format."""
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: T, Do
            action: T, Da
        """
        raise NotImplementedError()


class BaseImageDataset(torch.utils.data.Dataset):
    """
    Base class for multi-dataset image-based diffusion policy datasets.

    Subclasses must populate the following instance attributes during ``__init__``:

    Attributes:
        num_datasets (int): Number of datasets being co-trained on.
        replay_buffers (list): One buffer object per dataset (ReplayBuffer or compatible).
        samplers (list): One ``ImprovedDatasetSampler`` per dataset.
        train_masks (list[np.ndarray]): Boolean episode masks for training splits.
        val_masks (list[np.ndarray]): Boolean episode masks for validation splits.
        sample_probabilities (np.ndarray): Normalised per-dataset sampling weights.
        horizon (int): Sequence length per sample.
        pad_before (int): Episode-start padding.
        pad_after (int): Episode-end padding.
        shape_meta (dict): Shape metadata for actions and observations.
        rgb_keys (list[str]): Observation keys whose values are RGB images.
        transforms (Optional[ColorJitter]): Color jitter transform, or None.

    Subclasses must implement:
        load_replay_buffer(path, keys, config)  - load one data file into a ReplayBuffer
        store_replay_buffer(replay_buffer, path) - save a ReplayBuffer back to the native format

    Concrete methods provided here:
        get_default_color_jitter - builds a ColorJitter from a config dict
        _apply_color_jitter      - applies ``self.transforms`` to all RGB obs keys
        _build_key_first_k       - builds ImprovedDatasetSampler read-cap dict
        _normalize_sample_probabilities
        get_num_datasets         - returns ``self.num_datasets``
        get_sample_probabilities - returns ``self.sample_probabilities``
        get_num_episodes         - total or per-dataset episode count
    """

    def get_validation_dataset(self) -> "BaseLowdimDataset":
        # return an empty dataset by default
        return BaseImageDataset()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def _normalize_sample_probabilities(self, sample_probabilities):
        total = np.sum(sample_probabilities)
        assert total > 0, "Sum of sampling weights must be greater than 0"
        return sample_probabilities / total

    @staticmethod
    def get_default_color_jitter(
        color_jitter: Optional[Dict],
    ) -> Optional[transforms.ColorJitter]:
        """Build a ColorJitter transform from a config dict, or return None."""
        if color_jitter is None:
            return None
        return transforms.ColorJitter(
            brightness=color_jitter.get("brightness", 0.15),
            contrast=color_jitter.get("contrast", 0.15),
            saturation=color_jitter.get("saturation", 0.15),
            hue=color_jitter.get("hue", 0.15),
        )

    def _apply_color_jitter(self, data: Dict) -> Dict:
        """Jitter all RGB cameras with the same random transform (uint8 HWC → float32 CHW)."""
        keys = self.rgb_keys
        T = data["obs"][keys[0]].shape[0]
        stacked = np.moveaxis(np.concatenate([data["obs"][k] for k in keys], axis=0), -1, 1).astype(np.float32) / 255.0
        jittered = self.transforms(torch.from_numpy(stacked)).numpy()
        for i, key in enumerate(keys):
            data["obs"][key] = jittered[i * T : (i + 1) * T]
        return data

    @staticmethod
    def _build_key_first_k(keys: List[str], n_obs_steps: Optional[int], horizon: int) -> Dict[str, int]:
        """
        Build the ``key_first_k`` dict that tells ``ImprovedDatasetSampler``
        how many frames to read per key (read-cap optimisation).

        Without this, the sampler reads ``horizon`` frames for every key even
        though observation keys only need ``n_obs_steps`` frames.  For large
        image tensors this wastes significant time and RAM.

        Args:
            keys: All keys that will be loaded by the sampler.
            n_obs_steps: Number of observation frames to condition on.
                         If None, no cap is applied to observation keys.
            horizon: Full prediction horizon (used as the cap for ``"action"``).

        Returns:
            Dict mapping key name → max frames to read.
        """
        key_first_k: Dict[str, int] = {}
        if n_obs_steps is not None:
            for key in keys:
                key_first_k[key] = n_obs_steps
        key_first_k["action"] = horizon
        return key_first_k

    def get_num_datasets(self) -> int:
        return self.num_datasets

    def get_sample_probabilities(self) -> np.ndarray:
        return self.sample_probabilities

    def get_num_episodes(self, index: Optional[int] = None) -> int:
        if index is None:
            return sum(buf.n_episodes for buf in self.replay_buffers)
        return self.replay_buffers[index].n_episodes

    def _lowdim_key_map(self) -> Dict[str, str]:
        """
        Maps keys used by the model/normalizer to the actual keys in the dataset replay buffer.

        This acts as a translation layer, allowing the model to use standardized names
        (e.g., "agent_pos") regardless of what the underlying dataset calls them
        (e.g., "state", "proprio", "joint_angles").

        Returns:
            Dict[str, str]: { "model_key": "dataset_buffer_key" }
        """
        raise NotImplementedError()

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        assert mode == "limits", "Only supports limits (min-max) normalization mode"
        key_map = self._lowdim_key_map()
        input_stats: Dict = {}  # init dict to store global min and max for each key
        # Iterate over each dataset's replay buffer; find global min/max for each key across all datasets
        # and store in input_states
        for buf in self.replay_buffers:
            # Extract data from replay buffer for each key
            data = {norm_key: buf[buf_key] for norm_key, buf_key in key_map.items()}
            # We then fit a normalizer for each key; we don't keep this normalizer, we just do this
            # since it's the easiest way to find the min/max for this key for this dataset.
            n = LinearNormalizer()
            n.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
            # We update our global min/max for each key
            for key in key_map:
                _max = n[key].params_dict.input_stats.max
                _min = n[key].params_dict.input_stats.min
                if key not in input_stats:
                    input_stats[key] = {"max": _max, "min": _min}
                else:
                    input_stats[key]["max"] = torch.maximum(input_stats[key]["max"], _max)
                    input_stats[key]["min"] = torch.minimum(input_stats[key]["min"], _min)
        # Create the final normalizer based on the global min/max in input_states
        normalizer = LinearNormalizer()
        normalizer.fit_from_input_stats(input_stats_dict=input_stats)
        # Create a normalizer for the RGB keys
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def load_replay_buffer(self, path: str, keys: List[str], config: Dict):
        """Load data from path into a ReplayBuffer for one dataset config."""
        raise NotImplementedError()

    def store_replay_buffer(self, replay_buffer, path: str) -> None:
        """Save a ReplayBuffer to disk in the dataset's native on-disk format."""
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs:
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()


class BaseZarrImageDataset(BaseImageDataset):
    """
    Intermediate base class for zarr-backed datasets.

    Subclasses must implement:
        _get_buffer_keys()              - list of keys to load from the zarr store
        load_replay_buffer(path, keys, config) - open one zarr store as a ReplayBuffer

    __init__ handles everything else: validation, key setup, the per-config loop
    (masks, samplers, sample probabilities), and final attribute assignment.
    Subclasses call super().__init__() and add any unique post-init state.

    Concrete methods added here:
        _validate_zarr_configs  - validates zarr config dicts
        _build_episode_masks    - builds boolean train/val episode masks
        get_validation_dataset  - shallow-copies self narrowed to one val dataset
        __len__                 - total number of samples across all samplers
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
    ):
        super().__init__()
        self._validate_zarr_configs(zarr_configs)

        obs_meta = shape_meta["obs"]
        self.rgb_keys = [k for k, v in obs_meta.items() if v.get("type") == "rgb"]
        self.lowdim_keys = [k for k, v in obs_meta.items() if v.get("type") != "rgb"]
        self.shape_meta = shape_meta

        keys = self._get_buffer_keys()
        key_first_k = self._build_key_first_k(keys, n_obs_steps, horizon)

        self.num_datasets = len(zarr_configs)
        self.replay_buffers = []
        self.train_masks: List[np.ndarray] = []
        self.val_masks: List[np.ndarray] = []
        self.samplers = []
        self.sample_probabilities = np.zeros(len(zarr_configs))
        self.zarr_paths: List[str] = []

        for i, cfg in enumerate(zarr_configs):
            zarr_path = os.path.expanduser(cfg["path"])
            buf = self.load_replay_buffer(zarr_path, keys, cfg)
            train_mask, val_mask = self._build_episode_masks(
                n_episodes=buf.n_episodes,
                val_ratio=cfg.get("val_ratio", val_ratio),
                max_train_episodes=cfg.get("max_train_episodes", None),
                seed=seed,
            )
            self.replay_buffers.append(buf)
            self.zarr_paths.append(zarr_path)
            self.train_masks.append(train_mask)
            self.val_masks.append(val_mask)
            self.samplers.append(ImprovedDatasetSampler(
                replay_buffer=buf,
                sequence_length=horizon,
                shape_meta=shape_meta,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
                key_first_k=key_first_k,
            ))
            self.sample_probabilities[i] = cfg.get("sampling_weight") or np.sum(train_mask)

        self.sample_probabilities = self._normalize_sample_probabilities(self.sample_probabilities)
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.transforms = self.get_default_color_jitter(color_jitter)

    def _get_buffer_keys(self) -> List[str]:
        """Return the list of keys to load from the zarr store."""
        raise NotImplementedError()

    def load_replay_buffer(self, path: str, keys: List[str], config: Dict) -> ReplayBuffer:
        """Load a standard zarr store (data/meta layout) into memory as a ReplayBuffer."""
        return ReplayBuffer.copy_from_path(zarr_path=path, store=zarr.MemoryStore(), keys=keys)

    def store_replay_buffer(self, replay_buffer, path: str) -> None:
        """Save a ReplayBuffer to a zarr directory store (standard data/meta layout)."""
        replay_buffer.save_to_path(os.path.expanduser(path))

    @staticmethod
    def _validate_zarr_configs(zarr_configs: List[Dict]) -> None:
        """
        Validate a list of zarr dataset configs.

        Checks that every path exists, max_train_episodes is positive when
        set, sampling_weight is non-negative when set, and that either ALL
        configs supply a sampling_weight or NONE do.
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
                raise ValueError(f"sampling_weight must be >= 0, got {sampling_weight}")

        if num_null_sampling_weights not in [0, N]:
            raise ValueError("Either all or none of the zarr_configs must have a sampling_weight")

    @staticmethod
    def _build_episode_masks(
        n_episodes: int,
        val_ratio: float,
        max_train_episodes: Optional[int],
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build boolean train/val episode masks for a single dataset.

        Returns:
            train_mask: np.ndarray[bool] of shape (n_episodes,)
            val_mask:   np.ndarray[bool] of shape (n_episodes,)
        """
        val_mask = get_val_mask(n_episodes=n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        return train_mask, val_mask

    def get_validation_dataset(self, index: Optional[int] = None) -> "BaseZarrImageDataset":
        if index is None:
            assert self.num_datasets == 1, "Must specify index if num_datasets > 1"
            index = 0
        val_set = copy.copy(self)
        val_set.num_datasets = 1
        val_set.replay_buffers = [self.replay_buffers[index]]
        val_set.train_masks = [self.train_masks[index]]
        val_set.val_masks = [self.val_masks[index]]
        val_set.zarr_paths = [self.zarr_paths[index]]
        val_set.sample_probabilities = np.array([1.0])
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

    def __len__(self) -> int:
        return sum(len(s) for s in self.samplers)
