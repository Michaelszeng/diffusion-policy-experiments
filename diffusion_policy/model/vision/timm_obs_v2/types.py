"""Shared types for TimmObsEncoderV2 modular implementation."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


class ModalityType:
    """Constants for observation modality types."""

    NULL = 0
    RGB = 1
    DEPTH = 2
    LOW_DIM = 3
    FORCE = 4


class RangeType:
    """Constants for observation range types."""

    NULL = 0
    LONG = 1
    SHORT = 2


class VisualModality:
    """Constants for visual stream modality."""

    RGB = "rgb"
    DEPTH = "depth"


@dataclass
class VisualStreamFeatures:
    """Per-key visual stream features before downstream filtering/aggregation."""

    tokens: torch.Tensor
    modality: str
    num_prefix_tokens: int


VisualFeatureSet = Dict[str, VisualStreamFeatures]


@dataclass
class TokenBundle:
    """Canonical token representation used by output adapters.

    Shape legend:
    B = batch, T = horizon/time, N = tokens, D = embedding dim
    """

    tokens: torch.Tensor
    positions: Optional[torch.Tensor]
    modality: Optional[torch.Tensor]
    range: Optional[torch.Tensor]


@dataclass
class EncoderCoreOutputs:
    """Shared encoder outputs produced once and consumed by cat/dit adapters.

    Shape legend:
    B = batch, T = horizon/time, N = tokens, D = embedding dim
    """

    batch_size: int
    visual_features_by_key: VisualFeatureSet
    force_features_by_key: Dict[str, torch.Tensor]
    lowdim_features_by_key: Dict[str, torch.Tensor]
