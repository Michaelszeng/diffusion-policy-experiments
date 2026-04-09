import numpy as np
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer


def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    # -1, 1 normalization
    input_max = stat["max"]
    input_min = stat["min"]
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_image_passthrough_normalizer():
    """
    Converts uint8 images [0, 255] to float [0, 255] without any rescaling.

    Use this when the observation encoder handles all image preprocessing
    (channel ordering, normalization) internally (i.e. R3MObsEncoder). The dataset 
    returns raw uint8 HWC images; this normalizer simply casts to float32 and
    preserves the [0, 255] range so the encoder can apply its own pipeline.
    """
    scale = np.array([1.0], dtype=np.float32)
    offset = np.array([0.0], dtype=np.float32)
    stat = {
        "min": np.array([0.0], dtype=np.float32),
        "max": np.array([255.0], dtype=np.float32),
        "mean": np.array([127.5], dtype=np.float32),
        "std": np.array([73.5], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_image_range_normalizer():
    """
    Maps float images [0, 1] to [-1, 1] via x * 2 - 1.
    Use this for encoders that expect zero-centered inputs. The input_stats_dict
    reflects a uniform distribution over [0, 1] (mean=0.5, std=1/sqrt(12)).
    """
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1 / 12)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat["min"])
    offset = np.zeros_like(stat["min"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )