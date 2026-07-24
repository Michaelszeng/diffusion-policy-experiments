"""
Batched, GPU-side image augmentation for observation frames.

Applies color jitter (brightness/contrast/saturation/hue) and small random
rotations to batches of RGB observation frames on the GPU. This replaces the
per-sample torchvision transforms that previously ran inside the dataset's
``__getitem__`` on CPU dataloader workers, where ColorJitter's hue term
(RGB<->HSV in elementwise ops) dominated data-loading time (~1.2 s/sample) and
starved the GPU.

Semantics match the previous CPU path:
- One random transform is sampled *per sample* and shared across every camera
  and every timestep of that sample (temporal + cross-camera consistency).
- Color ops operate on float images in [0, 1] (torchvision ``_blend`` convention).
- Rotation uses bilinear sampling with zero fill and no expansion.

Inputs/outputs are dicts mapping camera key -> (B, T, C, H, W) float tensors.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# torchvision's rgb_to_grayscale luma coefficients
_R, _G, _B = 0.2989, 0.5870, 0.1140


def _blend(x: torch.Tensor, other: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    factor = factor.view(-1, 1, 1, 1)
    return (factor * x + (1.0 - factor) * other).clamp_(0.0, 1.0)


def _grayscale(x: torch.Tensor) -> torch.Tensor:
    return _R * x[:, 0:1] + _G * x[:, 1:2] + _B * x[:, 2:3]


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img.unbind(dim=-3)
    maxc = img.max(dim=-3).values
    minc = img.min(dim=-3).values
    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = torch.fmod(h / 6.0 + 1.0, 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(torch.int32)
    p = torch.clamp(v * (1.0 - s), 0.0, 1.0)
    q = torch.clamp(v * (1.0 - s * f), 0.0, 1.0)
    t = torch.clamp(v * (1.0 - s * (1.0 - f)), 0.0, 1.0)
    i = i % 6
    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=img.device).view(-1, 1, 1)
    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)
    return torch.einsum("...ijk,...xijk->...xjk", mask.to(dtype=img.dtype), a4)


class BatchImageAugmentor:
    """Vectorized GPU color jitter + random rotation with per-sample randomness."""

    def __init__(
        self,
        color_jitter: Optional[Dict] = None,
        random_rotation=None,
    ):
        self.brightness = self._cj_range(color_jitter, "brightness", center=1.0)
        self.contrast = self._cj_range(color_jitter, "contrast", center=1.0)
        self.saturation = self._cj_range(color_jitter, "saturation", center=1.0)
        self.hue = self._cj_range(color_jitter, "hue", center=0.0)
        self.degrees = self._rotation_range(random_rotation)

    @property
    def enabled(self) -> bool:
        return any(
            r is not None
            for r in (self.brightness, self.contrast, self.saturation, self.hue, self.degrees)
        )

    @staticmethod
    def _cj_range(cfg, name: str, center: float) -> Optional[Tuple[float, float]]:
        if cfg is None:
            return None
        value = cfg.get(name, 0.0) if hasattr(cfg, "get") else cfg[name]
        value = float(value)
        if value <= 0.0:
            return None
        if center == 0.0:  # hue: symmetric range [-v, v]
            return (-value, value)
        return (max(0.0, 1.0 - value), 1.0 + value)  # brightness/contrast/saturation

    @staticmethod
    def _rotation_range(cfg) -> Optional[Tuple[float, float]]:
        if cfg is None:
            return None
        if isinstance(cfg, (int, float)):
            degrees = float(cfg)
        else:
            degrees = cfg.get("degrees", 0.0) if hasattr(cfg, "get") else cfg["degrees"]
        if isinstance(degrees, (list, tuple)) or hasattr(degrees, "__len__"):
            lo, hi = float(degrees[0]), float(degrees[1])
        else:
            degrees = float(degrees)
            if degrees <= 0.0:
                return None
            lo, hi = -degrees, degrees
        if lo == 0.0 and hi == 0.0:
            return None
        return (lo, hi)

    def _sample(self, rng: Tuple[float, float], B: int, device) -> torch.Tensor:
        return torch.empty(B, device=device).uniform_(rng[0], rng[1])

    def __call__(self, imgs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """imgs: key -> (B, T, C, H, W) float in [0, 1]. Returns augmented copy."""
        if not self.enabled:
            return imgs

        ref = next(iter(imgs.values()))
        B = ref.shape[0]
        device = ref.device

        # Sample one set of parameters per sample, shared across all keys/frames.
        b_f = self._sample(self.brightness, B, device) if self.brightness else None
        c_f = self._sample(self.contrast, B, device) if self.contrast else None
        s_f = self._sample(self.saturation, B, device) if self.saturation else None
        h_f = self._sample(self.hue, B, device) if self.hue else None
        angle = self._sample(self.degrees, B, device) if self.degrees else None
        # Random order of the color ops (single order per batch, as in torchvision).
        order = ["brightness", "contrast", "saturation", "hue"]
        order = [order[i] for i in torch.randperm(4).tolist()]

        out: Dict[str, torch.Tensor] = {}
        for key, x in imgs.items():
            b, t, c, h, w = x.shape
            xf = x.reshape(b * t, c, h, w)
            xf = self._apply(xf, t, b_f, c_f, s_f, h_f, angle, order)
            out[key] = xf.reshape(b, t, c, h, w)
        return out

    def _apply(self, x, T, b_f, c_f, s_f, h_f, angle, order):
        for name in order:
            if name == "brightness" and b_f is not None:
                x = _blend(x, torch.zeros_like(x), b_f.repeat_interleave(T))
            elif name == "contrast" and c_f is not None:
                mean = _grayscale(x).mean(dim=(-3, -2, -1), keepdim=True)
                x = _blend(x, mean, c_f.repeat_interleave(T))
            elif name == "saturation" and s_f is not None:
                x = _blend(x, _grayscale(x), s_f.repeat_interleave(T))
            elif name == "hue" and h_f is not None:
                hsv = _rgb_to_hsv(x)
                shift = h_f.repeat_interleave(T).view(-1, 1, 1)
                hue = torch.fmod(hsv[:, 0] + shift + 1.0, 1.0)
                hsv = torch.stack((hue, hsv[:, 1], hsv[:, 2]), dim=1)
                x = _hsv_to_rgb(hsv).clamp_(0.0, 1.0)

        if angle is not None:
            x = self._rotate(x, angle.repeat_interleave(T))
        return x

    @staticmethod
    def _rotate(x: torch.Tensor, angle_deg: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        rad = angle_deg * (math.pi / 180.0)
        cos, sin = torch.cos(rad), torch.sin(rad)
        theta = torch.zeros(N, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
