import logging
from typing import Dict, Optional

import numpy as np
import timm
import timm.data
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

# Range index 0 is reserved for the diffusion timestep token (assigned internally by the UNet).
# Observation tokens use 1=LONG range (default until short-range encoders are implemented).
_RANGE_LONG = 1


class TimmObsEncoder(ModuleAttrMixin):
    """
    Observation encoder using timm backbones (CLIP, DINOv2, etc.).

    Encodes RGB images via a shared ViT backbone and low-dim observations via
    linear projections. All observations are assumed to have the same horizon
    (n_obs_steps) at 10 Hz.

    Each (key, timestep) pair produces one token of size feature_dim.
    Modality index 0 is reserved for the diffusion timestep token (internal to the UNet);
    observation keys are assigned indices 1..n_keys in deterministic sorted order.

    Output formats:
        'cat' : (B, n_keys * n_obs_steps * feature_dim) — use with FiLM UNets
        'dit' : dict('tokens', 'positions', 'modality', 'range'), each (B, N, *)
                where N = n_keys * n_obs_steps — use with attention UNets
    """

    def __init__(
        self,
        shape_meta: dict,
        model_name: str,
        pretrained: bool = True,
        n_obs_steps: int = 2,
        feature_aggregation: str = "cls",  # "cls" or "mean" over patch tokens
        crop_ratio: float = 0.95,
        imagenet_norm: bool = True,
    ):
        super().__init__()

        obs_shape_meta = shape_meta["obs"]
        rgb_keys: list = []
        low_dim_keys: list = []
        key_shape_map: Dict[str, tuple] = {}
        image_shape: Optional[tuple] = None

        for key, attr in sorted(obs_shape_meta.items()):  # sorted for determinism
            shape = tuple(attr["shape"])
            obs_type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if obs_type == "rgb":
                rgb_keys.append(key)
                if image_shape is None:
                    image_shape = shape[1:]  # (H, W)
                elif image_shape != shape[1:]:
                    raise ValueError(
                        f"All RGB keys must share the same (H, W). "
                        f"Got {image_shape} and {shape[1:]}"
                    )
            elif obs_type == "low_dim":
                low_dim_keys.append(key)

        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.n_obs_steps = n_obs_steps
        self.feature_aggregation = feature_aggregation

        # ── Backbone (shared across all RGB keys) ──────────────────────────────
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",  # return all tokens; we aggregate ourselves
            dynamic_img_size=True,  # interpolate PE for any input resolution
        )
        self.feature_dim: int = self.backbone.num_features
        self.num_prefix_tokens: int = int(getattr(self.backbone, "num_prefix_tokens", 1))

        # ── Image transforms ────────────────────────────────────────────────────
        # With dynamic_img_size=True the backbone accepts any resolution, but the
        # spatial dims must still be divisible by the patch size. Snap the crop
        # size down to the nearest multiple of patch_size so this always holds.
        data_cfg = timm.data.resolve_model_data_config(self.backbone)
        patch_size = self.backbone.patch_embed.patch_size[0]
        if image_shape is not None:
            self.crop_size = (int(image_shape[0] * crop_ratio) // patch_size) * patch_size
            self.center_crop = torchvision.transforms.CenterCrop(self.crop_size)

        if imagenet_norm and pretrained and image_shape is not None:
            mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
            std = data_cfg.get("std", (0.229, 0.224, 0.225))
            self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        else:
            self.normalize = nn.Identity()

        # ── Low-dim projections → feature_dim ──────────────────────────────────
        self.lowdim_projs = nn.ModuleDict(
            {
                key: nn.Linear(int(np.prod(key_shape_map[key])), self.feature_dim)
                for key in self.low_dim_keys
            }
        )

        # ── Modality indices ────────────────────────────────────────────────────
        # Index 0 is reserved for the diffusion timestep token (set by the UNet).
        # Keys are assigned 1..n_keys in deterministic (sorted) order so each
        # observation modality has a unique learnable embedding.
        all_keys = self.rgb_keys + self.low_dim_keys
        self.key_to_modality: Dict[str, int] = {k: i + 1 for i, k in enumerate(all_keys)}
        self.n_modalities: int = len(all_keys) + 1  # +1 to include the reserved index 0

        logger.info(
            "TimmObsEncoder | backbone=%s feature_dim=%d rgb=%s lowdim=%s "
            "n_obs_steps=%d n_modalities=%d",
            model_name,
            self.feature_dim,
            self.rgb_keys,
            self.low_dim_keys,
            n_obs_steps,
            self.n_modalities,
        )
        logger.info("parameters: %e", sum(p.numel() for p in self.parameters()))

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _process_rgb(self, imgs: torch.Tensor) -> torch.Tensor:
        """Crop, normalise, run backbone, and aggregate to one vector per (B, T) pair.

        Args:
            imgs: (B, T, C, H, W) float in [0, 1]
        Returns:
            (B, T, feature_dim)
        """
        B, T = imgs.shape[:2]
        imgs = imgs.flatten(0, 1)  # (B*T, C, H, W)

        if self.training:
            # One shared crop for the entire batch (temporal consistency).
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                imgs, output_size=(self.crop_size, self.crop_size)
            )
            imgs = TF.crop(imgs, i, j, h, w)
        else:
            imgs = self.center_crop(imgs)

        imgs = self.normalize(imgs)
        tokens = self.backbone(imgs)  # (B*T, N_tok, D)

        if self.feature_aggregation == "cls":
            feat = tokens[:, 0, :]
        elif self.feature_aggregation == "mean":
            feat = tokens[:, self.num_prefix_tokens :, :].mean(dim=1)
        else:
            raise ValueError(f"Unknown feature_aggregation: {self.feature_aggregation!r}")

        return feat.reshape(B, T, self.feature_dim)

    # ── Public interface ────────────────────────────────────────────────────────

    @property
    def cat_output_dim(self) -> int:
        """Flat output dimension for output_format='cat'."""
        return (len(self.rgb_keys) + len(self.low_dim_keys)) * self.n_obs_steps * self.feature_dim

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        output_format: str = "cat",
    ):
        """Encode observations.

        Args:
            obs_dict      : dict mapping key → (B, T_obs, ...) tensor
            output_format : 'cat' or 'dit'

        Returns:
            'cat' → (B, cat_output_dim)
            'dit' → dict with keys:
                'tokens'    (B, N, feature_dim)
                'positions' (B, N)   temporal index per token (0 = oldest)
                'modality'  (B, N)   per-key modality index
                'range'     (B, N)   1 = LONG range (SHORT not yet implemented)
            where N = n_keys * n_obs_steps.
        """
        assert output_format in ("cat", "dit"), f"Unknown output_format: {output_format!r}"

        device = next(self.parameters()).device
        B = next(iter(obs_dict.values())).shape[0]

        # Encode every key → (B, T_obs, feature_dim)
        encoded: Dict[str, torch.Tensor] = {}
        for key in self.rgb_keys:
            encoded[key] = self._process_rgb(obs_dict[key])
        for key in self.low_dim_keys:
            x = obs_dict[key].float().flatten(2)  # (B, T_obs, d)
            encoded[key] = self.lowdim_projs[key](x)

        all_keys = self.rgb_keys + self.low_dim_keys

        if output_format == "cat":
            return torch.cat([encoded[k].reshape(B, -1) for k in all_keys], dim=-1)

        # DIT: assemble token blocks with metadata
        token_list, pos_list, mod_list, range_list = [], [], [], []
        # Positions: 0 = oldest observation, n_obs_steps-1 = most recent
        positions = torch.arange(self.n_obs_steps, device=device).unsqueeze(0).expand(B, -1)
        for key in all_keys:
            token_list.append(encoded[key])  # (B, T_obs, D)
            pos_list.append(positions)
            mod_list.append(
                torch.full((B, self.n_obs_steps), self.key_to_modality[key], dtype=torch.long, device=device)
            )
            range_list.append(
                torch.full((B, self.n_obs_steps), _RANGE_LONG, dtype=torch.long, device=device)
            )

        return {
            "tokens":    torch.cat(token_list, dim=1),   # (B, n_keys*T_obs, D)
            "positions": torch.cat(pos_list,   dim=1),   # (B, n_keys*T_obs)
            "modality":  torch.cat(mod_list,   dim=1),   # (B, n_keys*T_obs)
            "range":     torch.cat(range_list, dim=1),   # (B, n_keys*T_obs)
        }
