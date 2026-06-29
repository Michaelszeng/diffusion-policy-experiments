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

Expected image format: float32 HWC (B*T, H, W, C) in [0, 255] range (output of
`get_image_passthrough_normalizer`). Each encoder converts to the range required
by its backbone internally.

Note on R3M: R3M's forward() internally divides by 255 and applies its own
normalization, expecting CHW input in [0, 255]. R3MObsEncoder divides to [0,1]
for the camera transforms, then scales back to [0, 255] before calling the R3M backbone.
"""

import contextlib
import fcntl
import os
from typing import Dict, List, Optional, Tuple, Union

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
    observations. Encoder weights are shared across timesteps (B*T frames are
    processed as a flat batch with no temporal state in the encoder).

    Args:
        shape_meta:        Hydra shape_meta dict (action + obs keys).
        pretrained:        Load IMAGENET1K_V1 weights.
        freeze_encoder:            Freeze encoder weights (default False — fine-tune).
        projection_dim:    Projection output size per camera (default 128).
        use_groupnorm:     Replace BatchNorm2d with GroupNorm.
        front_camera_keys: Key or list of keys for fixed/scene cameras that receive
                           crop augmentation.  Defaults to the last sorted RGB key
                           (i.e. "color_image2" for FurnitureBench).  A single string
                           is accepted for backward compatibility.
    """

    RESNET_OUT_DIM = 512  # ResNet18 global avg pool output

    def __init__(
        self,
        shape_meta: dict,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        projection_dim: int = 128,
        use_groupnorm: bool = True,
        front_camera_keys: Optional[Union[str, List[str]]] = None,
        # Deprecated alias kept for backward compatibility
        front_camera_key: Optional[str] = None,
    ):
        super().__init__()

        obs_meta = shape_meta["obs"]
        self.rgb_keys: List[str] = sorted([k for k, v in obs_meta.items() if v.get("type") == "rgb"])
        self.lowdim_keys: List[str] = sorted([k for k, v in obs_meta.items() if v.get("type") != "rgb"])
        assert len(self.rgb_keys) > 0, "ResNetObsEncoder requires at least one RGB key"

        # Resolve front camera keys (support legacy single-key arg).
        if front_camera_keys is None and front_camera_key is not None:
            front_camera_keys = front_camera_key
        if front_camera_keys is None:
            front_camera_keys = self.rgb_keys[-1]
        if isinstance(front_camera_keys, str):
            front_camera_keys = [front_camera_keys]
        for k in front_camera_keys:
            assert k in self.rgb_keys, f"front_camera_keys entry '{k}' not in rgb_keys {self.rgb_keys}"
        self.front_camera_keys: List[str] = list(front_camera_keys)

        # ── Per-camera encoders ────────────────────────────────────────────────
        self.encoders = nn.ModuleDict({key: _build_resnet18(pretrained, use_groupnorm) for key in self.rgb_keys})

        # ── Per-camera projection layers ───────────────────────────────────────
        self.projectors = nn.ModuleDict({key: nn.Linear(self.RESNET_OUT_DIM, projection_dim) for key in self.rgb_keys})

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
        self.front_train_transform = T.Compose(
            [
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                T.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
                T.CenterCrop((240, 280)),  # (H=240, W=280): remove 20 px from each side
                T.RandomCrop((224, 224)),
            ]
        )
        self.front_eval_transform = T.CenterCrop((224, 224))

        self.wrist_train_transform = T.Compose(
            [
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                T.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
                T.Resize((224, 224), antialias=True),
            ]
        )
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
            x:   (B*T, H, W, C) float32 in [0, 255] — HWC, output of the
                 get_image_passthrough_normalizer pipeline.
            key: camera key name (determines which transform to apply)

        Returns:
            (B*T, projection_dim) float32
        """
        # HWC → CHW, scale to [0, 1] for torchvision transforms and ImageNet norm
        x = x.permute(0, 3, 1, 2).contiguous() / 255.0  # (B*T, C, H, W)

        # Camera-specific spatial transform + augmentation
        is_front = key in self.front_camera_keys
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
                      RGB keys: (B*T, H, W, C) float32 in [0, 255]
                      Low-dim keys: (B*T, d) float32 (already normalized)

        Returns:
            (B*T, obs_feature_dim) float32
        """
        features = []
        for key in self.rgb_keys:
            features.append(self._encode_image(obs_dict[key].float(), key))  # (B*T, projection_dim) each
        for key in self.lowdim_keys:
            features.append(obs_dict[key].float())  # (B*T, d) each, passed through unchanged
        # cat → (B*T, num_cameras*projection_dim + lowdim_dim) == (B*T, obs_feature_dim)
        return torch.cat(features, dim=-1)


@contextlib.contextmanager
def _r3m_cache_lock():
    """
    File-lock the R3M cache directory across processes on the same node.

    Upstream `r3m.load_r3m` has a TOCTOU race (`if not exists: makedirs`) plus an
    unguarded `gdown.download` of the weights. Under `torchrun --nproc_per_node>1`
    the two ranks hit these blocks in parallel: one wins the `mkdir`, the other
    raises `FileExistsError`; both can also race to write `model.pt`. We serialize
    the whole `load_r3m` call with a per-node file lock so only one rank performs
    the actual download / directory creation at a time.
    """
    cache_root = os.path.join(os.path.expanduser("~"), ".r3m")
    os.makedirs(cache_root, exist_ok=True)
    # Pre-create the per-model dir so r3m's buggy `if not exists: makedirs` is skipped.
    os.makedirs(os.path.join(cache_root, "r3m_18"), exist_ok=True)

    lock_path = os.path.join(cache_root, "r3m_18.lock")
    with open(lock_path, "w") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def _build_r3m_resnet18(freeze: bool) -> nn.Module:
    """
    Load the full R3M-pretrained ResNet18 model (Ego4D objective).

    Weights are downloaded from Google Drive on first use (~95 MB) and cached in ~/.r3m/.
    load_r3m() strips the language head; the returned R3M module's forward() handles
    /255 and ImageNet normalization internally and outputs a 512-dim feature vector.

    The caller is responsible for providing CHW input in [0, 255].
    """
    from r3m import load_r3m

    with _r3m_cache_lock():
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

    One R3M backbone per camera; encoder weights are shared across timesteps (B*T flat batch).

    R3M's backbone handles its own normalization internally (divides by 255 and applies
    ImageNet stats). This encoder scales the [0,1] float images back to [0,255] before
    calling the backbone. However, the images passed to forward() should be uint8[0,255].

    Args:
        shape_meta:        Hydra shape_meta dict (action + obs keys).
        freeze_encoder:    Freeze backbone and projector weights (default False).
        use_groupnorm:     Replace BatchNorm2d in the R3M backbone with GroupNorm.
        projection_dim:    Projection output size per camera (default 128).
        front_camera_keys: Key or list of keys for fixed/scene cameras that receive
                           crop augmentation.  Defaults to the last sorted RGB key
                           ("color_image2" for FurnitureBench).  A single string is
                           accepted for backward compatibility.
    """

    R3M_OUT_DIM = 512  # R3M ResNet18 output dim

    def __init__(
        self,
        shape_meta: dict,
        freeze_encoder: bool = False,
        use_groupnorm: bool = False,
        projection_dim: int = 128,
        front_camera_keys: Optional[Union[str, List[str]]] = None,
        # Deprecated alias kept for backward compatibility
        front_camera_key: Optional[str] = None,
    ):
        super().__init__()

        obs_meta = shape_meta["obs"]
        self.rgb_keys: List[str] = sorted([k for k, v in obs_meta.items() if v.get("type") == "rgb"])
        self.lowdim_keys: List[str] = sorted([k for k, v in obs_meta.items() if v.get("type") != "rgb"])
        assert len(self.rgb_keys) > 0, "R3MObsEncoder requires at least one RGB key"

        # Resolve front camera keys (support legacy single-key arg).
        if front_camera_keys is None and front_camera_key is not None:
            front_camera_keys = front_camera_key
        if front_camera_keys is None:
            front_camera_keys = self.rgb_keys[-1]
        if isinstance(front_camera_keys, str):
            front_camera_keys = [front_camera_keys]
        for k in front_camera_keys:
            assert k in self.rgb_keys, f"front_camera_keys entry '{k}' not in rgb_keys {self.rgb_keys}"
        self.front_camera_keys: List[str] = list(front_camera_keys)

        # ── Per-camera R3M backbones ───────────────────────────────────────────
        self.encoders = nn.ModuleDict({key: _build_r3m_resnet18(freeze_encoder) for key in self.rgb_keys})

        if use_groupnorm and not freeze_encoder:
            # Replace BatchNorm with GroupNorm for bf16 stability. Only applied when
            # the backbone is trainable; frozen BN layers have no gradient path anyway.
            for encoder in self.encoders.values():
                replace_submodules(
                    root_module=encoder,
                    predicate=lambda m: isinstance(m, nn.BatchNorm2d),
                    func=lambda m: nn.GroupNorm(
                        num_groups=m.num_features // 16,
                        num_channels=m.num_features,
                    ),
                )

        # ── Per-camera projection layers ───────────────────────────────────────
        self.projectors = nn.ModuleDict({key: nn.Linear(self.R3M_OUT_DIM, projection_dim) for key in self.rgb_keys})

        if freeze_encoder:
            for proj in self.projectors.values():
                for param in proj.parameters():
                    param.requires_grad = False

        # ── Output dimension ───────────────────────────────────────────────────
        lowdim_dim = sum(obs_meta[k]["shape"][0] for k in self.lowdim_keys)
        self._obs_feature_dim: int = len(self.rgb_keys) * projection_dim + lowdim_dim

        # ── Camera-specific augmentation transforms (same as ResNetObsEncoder) ─
        self.front_train_transform = T.Compose(
            [
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                T.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
                T.CenterCrop((240, 280)),
                T.RandomCrop((224, 224)),
            ]
        )
        self.front_eval_transform = T.CenterCrop((224, 224))

        self.wrist_train_transform = T.Compose(
            [
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                T.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
                T.Resize((224, 224), antialias=True),
            ]
        )
        self.wrist_eval_transform = T.Resize((224, 224), antialias=True)

    def output_shape(self) -> Tuple[int, ...]:
        return (self._obs_feature_dim,)

    def _encode_image(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """
        Args:
            x:   (B*T, H, W, C) float32 in [0, 255]
            key: camera key name

        Returns:
            (B*T, projection_dim) float32
        """
        # HWC → CHW, scale to [0, 1] for torchvision transforms
        x = x.permute(0, 3, 1, 2).contiguous() / 255.0  # (B*T, C, H, W)

        # Camera-specific augmentation (operates on CHW float [0, 1])
        is_front = key in self.front_camera_keys
        if self.training:
            x = self.front_train_transform(x) if is_front else self.wrist_train_transform(x)
        else:
            x = self.front_eval_transform(x) if is_front else self.wrist_eval_transform(x)

        # R3M expects CHW in [0, 255]; scale back from [0, 1]
        x = x * 255.0

        # R3M forward: divides by 255, applies ImageNet normalization, runs ResNet18 → (B*T, 512)
        feat = self.encoders[key](x)

        return self.projectors[key](feat)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs_dict: RGB keys → (B*T, H, W, C) float32 in [0, 255]
                      Low-dim keys → (B*T, d) float32

        Returns:
            (B*T, obs_feature_dim) float32
        """
        features = []
        for key in self.rgb_keys:
            features.append(self._encode_image(obs_dict[key].float(), key))  # (B*T, projection_dim) each
        for key in self.lowdim_keys:
            features.append(obs_dict[key].float())  # (B*T, d) each, passed through unchanged
        # cat → (B*T, num_cameras*projection_dim + lowdim_dim) = (B*T, obs_feature_dim)
        return torch.cat(features, dim=-1)


class _TactileResNetEncoder(nn.Module):
    """
    ResNet18 from scratch + GroupNorm + ImageNet input normalization,
    mirroring the rgb_model path of ~/manifeel's MultiImageObsEncoder
    (``get_resnet(resnet18, weights=null)`` + ``use_group_norm=True`` +
    ``imagenet_norm=True``). Output is the raw 512-D ResNet18 feature
    (no projection), matching manifeel's concat behavior.

    Accepts CHW float input. Applies the standard ImageNet mean/std
    normalization before the backbone.
    """

    OUT_DIM = 512  # ResNet18 global avg pool output

    def __init__(self):
        super().__init__()
        # _build_resnet18(pretrained=False, use_groupnorm=True) is exactly
        # get_resnet(name='resnet18', weights=null) followed by the BN→GN
        # replacement that MultiImageObsEncoder does when use_group_norm=True.
        self.resnet = _build_resnet18(pretrained=False, use_groupnorm=True)
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) — caller is responsible for permuting HWC→CHW.
        x = (x - self.imagenet_mean) / self.imagenet_std
        return self.resnet(x)


class R3MTactileHybridObsEncoder(R3MObsEncoder):
    """
    R3MObsEncoder + a from-scratch ResNet18 tactile branch.

    shape_meta entry types:
        type="rgb"      → R3M (pretrained) per camera
        type="tactile"  → from-scratch ResNet18 + GroupNorm per tactile key
                          (mirrors ~/manifeel's MultiImageObsEncoder choice)
        type="low_dim"  → pass through

    Tactile keys are declared in shape_meta with type="tactile" and shape
    [H, W, C] (HWC, as stored in the manifeel zarrs). The encoder permutes
    to CHW internally. Values are kept in their dataset-normalized range
    (the per-channel LinearNormalizer brings them to [-1, 1]).
    """

    def __init__(
        self,
        shape_meta: dict,
        freeze_encoder: bool = False,
        use_groupnorm: bool = False,
        projection_dim: int = 128,
        front_camera_keys: Optional[Union[str, List[str]]] = None,
        front_camera_key: Optional[str] = None,
    ):
        super().__init__(
            shape_meta=shape_meta,
            freeze_encoder=freeze_encoder,
            use_groupnorm=use_groupnorm,
            projection_dim=projection_dim,
            front_camera_keys=front_camera_keys,
            front_camera_key=front_camera_key,
        )

        obs_meta = shape_meta["obs"]
        self.tactile_keys: List[str] = sorted(
            [k for k, v in obs_meta.items() if v.get("type") == "tactile"]
        )
        # R3MObsEncoder put tactile keys into lowdim_keys; pull them out.
        self.lowdim_keys = [k for k in self.lowdim_keys if k not in self.tactile_keys]

        self.tactile_encoders = nn.ModuleDict()
        for k in self.tactile_keys:
            shape = obs_meta[k]["shape"]
            assert len(shape) == 3 and shape[-1] == 3, (
                f"tactile key '{k}' must have HWC shape with C=3 (matching the "
                f"ResNet18 input channels manifeel uses), got {shape}"
            )
            self.tactile_encoders[k] = _TactileResNetEncoder()

        lowdim_dim = sum(obs_meta[k]["shape"][0] for k in self.lowdim_keys)
        self._obs_feature_dim = (
            len(self.rgb_keys) * projection_dim
            + len(self.tactile_keys) * _TactileResNetEncoder.OUT_DIM
            + lowdim_dim
        )

    def _encode_tactile(self, x: torch.Tensor, key: str) -> torch.Tensor:
        # (B*T, H, W, C) → (B*T, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous().float()
        return self.tactile_encoders[key](x)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        for key in self.rgb_keys:
            features.append(self._encode_image(obs_dict[key].float(), key))
        for key in self.tactile_keys:
            features.append(self._encode_tactile(obs_dict[key], key))
        for key in self.lowdim_keys:
            features.append(obs_dict[key].float())
        return torch.cat(features, dim=-1)
