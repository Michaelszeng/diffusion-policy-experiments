import torch
import torch.nn.functional as F


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
