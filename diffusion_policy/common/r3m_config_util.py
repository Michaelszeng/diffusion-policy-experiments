"""
ResNet-based observation encoders following the JUICER paper
(Ankile et al., "JUICER: Data-Efficient Imitation Learning for Robotic Assembly", 2024).

Two encoder classes are provided:

  ResNetObsEncoder — ImageNet-pretrained ResNet18 (IMAGENET1K_V1).
  R3MObsEncoder    — ResNet18 with weights from the R3M objective on Ego4D (best performer
                     in the paper: 77% vs 59% average success on the one-leg task).

Both share the same architecture (one ResNet18 per camera, projected to `projection_dim`)
and the same augmentation pipeline:
  Front camera (color_image2): ColorJitter + GaussianBlur + CenterCrop(240,280) + RandomCrop(224,224)
  Wrist camera (color_image1): ColorJitter + GaussianBlur + Resize(224,224)
At eval, both cameras are center-cropped / resized to (224,224) without augmentation.

Expected image format: float32 HWC (B*T, H, W, C) in [0, 1] range.
Use `normalize_images: false` in the dataset config so the dataset applies the
`get_image_to_float_normalizer` (maps uint8 [0,255] → float [0,1]) rather than
the default `get_image_range_normalizer` (maps [0,1] → [-1,1]).

Note on R3M input range: R3M's forward() internally divides by 255 and applies its own
normalization, so it expects CHW input in [0, 255]. R3MObsEncoder scales [0,1] → [0,255]
before calling the R3M backbone.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.models as tvm
from torchvision.transforms import v2 as T

from diffusion_policy.common.pytorch_util import replace_submodules


def _build_resnet18(pretrained: bool, use_groupnorm: bool) -> nn.Module:
    """
    Build a ResNet18 backbone with the classification head removed (output dim = 512).

    Optionally replaces all BatchNorm2d layers with GroupNorm, which is more
    stable under the small per-GPU batch sizes typical in imitation learning.
    """
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    resnet = tvm.resnet18(weights=weights)
    resnet.fc = nn.Identity()  # global avg pool output: (B, 512)

    if use_groupnorm:
        replace_submodules(
            root_module=resnet,
            predicate=lambda m: isinstance(m, nn.BatchNorm2d),
            func=lambda m: nn.GroupNorm(
                num_groups=m.num_features // 16,
                num_channels=m.num_features,
            ),
        )
    return resnet


class ResNetObsEncoder(nn.Module):
    """
    JUICER-style ResNet18 + global average pooling observation encoder.

    One ResNet18 per RGB camera key. Each camera's features are projected to
    `projection_dim` and concatenated with (already-normalized) low-dim
    observations.

    Args:
        shape_meta:        Hydra shape_meta dict (action + obs keys).
        pretrained:        Load IMAGENET1K_V1 weights.
        freeze_encoder:            Freeze encoder weights (default False — fine-tune).
        projection_dim:    Projection output size per camera (default 128).
        use_groupnorm:     Replace BatchNorm2d with GroupNorm.
        front_camera_key:  Key of the front-facing camera that receives random
                           crop augmentation.  Defaults to the last sorted RGB
                           key (i.e. "color_image2" for FurnitureBench).
    """

    RESNET_OUT_DIM = 512  # ResNet18 global avg pool output

    def __init__(
        self,
        shape_meta: dict,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        projection_dim: int = 128,
        use_groupnorm: bool = True,
        front_camera_key: Optional[str] = None,
    ):
        super().__init__()

        obs_meta = shape_meta["obs"]
        self.rgb_keys: List[str] = sorted(
            [k for k, v in obs_meta.items() if v.get("type") == "rgb"]
        )
        self.lowdim_keys: List[str] = sorted(
            [k for k, v in obs_meta.items() if v.get("type") != "rgb"]
        )
        assert len(self.rgb_keys) > 0, "ResNetObsEncoder requires at least one RGB key"

        # Identify front vs wrist cameras.
        # Front camera gets random-crop augmentation; wrist gets resize only.
        if front_camera_key is None:
            # Default: last sorted key (color_image2 for FurnitureBench)
            front_camera_key = self.rgb_keys[-1]
        assert front_camera_key in self.rgb_keys, (
            f"front_camera_key '{front_camera_key}' not in rgb_keys {self.rgb_keys}"
        )
        self.front_camera_key = front_camera_key

        # ── Per-camera encoders ────────────────────────────────────────────────
        self.encoders = nn.ModuleDict(
            {key: _build_resnet18(pretrained, use_groupnorm) for key in self.rgb_keys}
        )

        # ── Per-camera projection layers ───────────────────────────────────────
        self.projectors = nn.ModuleDict(
            {key: nn.Linear(self.RESNET_OUT_DIM, projection_dim) for key in self.rgb_keys}
        )

        # ── Low-dim output dimension ───────────────────────────────────────────
        lowdim_dim = sum(obs_meta[k]["shape"][0] for k in self.lowdim_keys)
        self._obs_feature_dim: int = len(self.rgb_keys) * projection_dim + lowdim_dim

        # ── ImageNet normalization ─────────────────────────────────────────────
        self.imagenet_norm = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # ── Camera-specific augmentation transforms ────────────────────────────
        # Front camera: remove 20 px side margins → random/center crop to 224×224
        # Wrist camera: resize to 224×224
        self.front_train_transform = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            T.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
            T.CenterCrop((240, 280)),   # (H=240, W=280): remove 20 px from each side
            T.RandomCrop((224, 224)),
        ])
        self.front_eval_transform = T.CenterCrop((224, 224))

        self.wrist_train_transform = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            T.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
            T.Resize((224, 224), antialias=True),
        ])
        self.wrist_eval_transform = T.Resize((224, 224), antialias=True)

        # ── Optionally freeze encoder ──────────────────────────────────────────
        if freeze_encoder:
            for encoder in self.encoders.values():
                for param in encoder.parameters():
                    param.requires_grad = False

    def output_shape(self) -> Tuple[int, ...]:
        """Return (obs_feature_dim,) — compatible with robomimic obs_encoder interface."""
        return (self._obs_feature_dim,)

    def _encode_image(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """
        Encode a single camera's images.

        Args:
            x:   (B*T, H, W, C) float32 in [0, 1]  — HWC, output of the
                 get_image_to_float_normalizer pipeline.
            key: camera key name (determines which transform to apply)

        Returns:
            (B*T, projection_dim) float32
        """
        # HWC → CHW
        x = x.permute(0, 3, 1, 2).contiguous()  # (B*T, C, H, W)

        # Camera-specific spatial transform + augmentation
        is_front = key == self.front_camera_key
        if self.training:
            x = self.front_train_transform(x) if is_front else self.wrist_train_transform(x)
        else:
            x = self.front_eval_transform(x) if is_front else self.wrist_eval_transform(x)

        # ImageNet normalization
        x = self.imagenet_norm(x)

        # ResNet18 forward → (B*T, 512)
        feat = self.encoders[key](x)

        # Project → (B*T, projection_dim)
        return self.projectors[key](feat)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode all observations into a single flat feature vector.

        Args:
            obs_dict: dict mapping each obs key to a tensor.
                      RGB keys: (B*T, H, W, C) float32 in [0, 1]
                      Low-dim keys: (B*T, d) float32 (already normalized)

        Returns:
            (B*T, obs_feature_dim) float32
        """
        features = []
        for key in self.rgb_keys:
            features.append(self._encode_image(obs_dict[key].float(), key))
        for key in self.lowdim_keys:
            features.append(obs_dict[key].float())
        return torch.cat(features, dim=-1)


def _build_r3m_resnet18(freeze: bool) -> nn.Module:
    """
    Load the full R3M-pretrained ResNet18 model (Ego4D objective).

    Weights are downloaded from Google Drive on first use (~95 MB) and cached in ~/.r3m/.
    load_r3m() strips the language head; the returned R3M module's forward() handles
    /255 and ImageNet normalization internally and outputs a 512-dim feature vector.

    The caller is responsible for providing CHW input in [0, 255].
    """
    from r3m import load_r3m

    r3m = load_r3m("resnet18").module  # unwrap nn.DataParallel → R3M model

    if freeze:
        for param in r3m.parameters():
            param.requires_grad = False

    return r3m


class R3MObsEncoder(nn.Module):
    """
    JUICER-style R3M-pretrained ResNet18 observation encoder.

    Identical structure to ResNetObsEncoder but uses weights from the R3M objective
    (Nair et al., 2022) trained on Ego4D egocentric video rather than ImageNet.
    Per the JUICER paper this achieves 77% average success on the one-leg task vs
    59% for ImageNet pretraining.

    R3M's backbone handles its own normalization internally (divides by 255 and applies
    ImageNet stats). This encoder scales the [0,1] float images back to [0,255] before
    calling the backbone.

    Args:
        shape_meta:       Hydra shape_meta dict (action + obs keys).
        freeze_encoder:   Freeze R3M backbone weights (default False — fine-tune).
        projection_dim:   Projection output size per camera (default 128).
        front_camera_key: Key of the front-facing camera. Defaults to the last sorted
                          RGB key ("color_image2" for FurnitureBench).
    """

    R3M_OUT_DIM = 512  # R3M ResNet18 output dim

    def __init__(
        self,
        shape_meta: dict,
        freeze_encoder: bool = False,
        projection_dim: int = 128,
        front_camera_key: Optional[str] = None,
    ):
        super().__init__()

        obs_meta = shape_meta["obs"]
        self.rgb_keys: List[str] = sorted(
            [k for k, v in obs_meta.items() if v.get("type") == "rgb"]
        )
        self.lowdim_keys: List[str] = sorted(
            [k for k, v in obs_meta.items() if v.get("type") != "rgb"]
        )
        assert len(self.rgb_keys) > 0, "R3MObsEncoder requires at least one RGB key"

        if front_camera_key is None:
            front_camera_key = self.rgb_keys[-1]
        assert front_camera_key in self.rgb_keys, (
            f"front_camera_key '{front_camera_key}' not in rgb_keys {self.rgb_keys}"
        )
        self.front_camera_key = front_camera_key

        # ── Per-camera R3M backbones ───────────────────────────────────────────
        self.encoders = nn.ModuleDict(
            {key: _build_r3m_resnet18(freeze_encoder) for key in self.rgb_keys}
        )

        # ── Per-camera projection layers ───────────────────────────────────────
        self.projectors = nn.ModuleDict(
            {key: nn.Linear(self.R3M_OUT_DIM, projection_dim) for key in self.rgb_keys}
        )

        # ── Output dimension ───────────────────────────────────────────────────
        lowdim_dim = sum(obs_meta[k]["shape"][0] for k in self.lowdim_keys)
        self._obs_feature_dim: int = len(self.rgb_keys) * projection_dim + lowdim_dim

        # ── Camera-specific augmentation transforms (same as ResNetObsEncoder) ─
        self.front_train_transform = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            T.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
            T.CenterCrop((240, 280)),
            T.RandomCrop((224, 224)),
        ])
        self.front_eval_transform = T.CenterCrop((224, 224))

        self.wrist_train_transform = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            T.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
            T.Resize((224, 224), antialias=True),
        ])
        self.wrist_eval_transform = T.Resize((224, 224), antialias=True)

    def output_shape(self) -> Tuple[int, ...]:
        return (self._obs_feature_dim,)

    def _encode_image(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """
        Args:
            x:   (B*T, H, W, C) float32 in [0, 1]
            key: camera key name

        Returns:
            (B*T, projection_dim) float32
        """
        # HWC → CHW
        x = x.permute(0, 3, 1, 2).contiguous()  # (B*T, C, H, W)

        # Camera-specific augmentation (operates on CHW float [0, 1])
        is_front = key == self.front_camera_key
        if self.training:
            x = self.front_train_transform(x) if is_front else self.wrist_train_transform(x)
        else:
            x = self.front_eval_transform(x) if is_front else self.wrist_eval_transform(x)

        # R3M expects CHW in [0, 255]; our pipeline produces [0, 1]
        x = x * 255.0

        # R3M forward: divides by 255, applies ImageNet normalization, runs ResNet18 → (B*T, 512)
        feat = self.encoders[key](x)

        return self.projectors[key](feat)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs_dict: RGB keys → (B*T, H, W, C) float32 in [0, 1]
                      Low-dim keys → (B*T, d) float32

        Returns:
            (B*T, obs_feature_dim) float32
        """
        features = []
        for key in self.rgb_keys:
            features.append(self._encode_image(obs_dict[key].float(), key))
        for key in self.lowdim_keys:
            features.append(obs_dict[key].float())
        return torch.cat(features, dim=-1)
