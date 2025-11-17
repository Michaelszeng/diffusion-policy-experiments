from typing import Dict

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

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


def low_pass_filter(x, kernel):
    """Apply low-pass (Gaussian blur) filter to input tensor x."""
    padding = kernel.shape[-1] // 2
    return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])


class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> "BaseLowdimDataset":
        # return an empty dataset by default
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
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
    def get_validation_dataset(self) -> "BaseLowdimDataset":
        # return an empty dataset by default
        return BaseImageDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def _normalize_sample_probabilities(self, sample_probabilities):
        total = np.sum(sample_probabilities)
        assert total > 0, "Sum of sampling weights must be greater than 0"
        return sample_probabilities / total

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


class BaseDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        raise NotImplementedError()
