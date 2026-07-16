"""
Image dataset for the Franka Kitchen (relay-policy-learning) zarr format.

Expected zarr layout (per kitchen_demos.zarr):
    data/
        action            (T, 9)              float64   # 7 arm joints + 2 gripper
        state             (T, 60)             float64   # robot + env state
        scene             (T, 240, 320, 3)    uint8     # external scene camera
        wrist             (T, 240, 320, 3)    uint8     # wrist camera
    meta/
        episode_ends      (E,)                int64

Routing (handled by ImprovedDatasetSampler):
    * zarr "state"   -> obs["agent_pos"]
    * rgb keys       -> obs[key]            (uint8)
    * other obs keys -> obs[key]            (float32)
    * "action"       -> top-level "action"

Proprioception-only state:
    The zarr "state" is the D4RL Franka Kitchen observation, laid out as
    ``[qp(9), obj_qp(21), goal(30)]`` (see relay-policy-learning
    ``KitchenTaskRelaxV1._get_obs``). Only the leading ``qp(9)`` -- the robot's
    own joint positions (7 arm + 2 gripper) -- is proprioceptive; ``obj_qp``
    (object joint positions) and ``goal`` are privileged environment/goal
    information that a real robot could not observe. To keep the policy input
    free of privileged information, we expose only the ``qp(9)`` proprioceptive
    prefix as ``agent_pos`` (both in samples and when fitting the normalizer).

Unlike ManiskillZarrDataset, there is no "target" key (no goal conditioning).
"""

from typing import Dict, List

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseZarrImageDataset


class FrankaKitchenDataset(BaseZarrImageDataset):
    """
    Minimal zarr-backed dataset for Franka Kitchen data.

    Loads the standard ReplayBuffer zarr layout (``data/<key>`` arrays +
    ``meta/episode_ends``). Observation keys are determined entirely by
    ``shape_meta.obs``; this class only fixes the action key and restricts the
    low-dim state to its proprioceptive prefix.
    """

    # Number of leading dims of the 60-D zarr "state" that are proprioceptive
    # (robot joint positions qp: 7 arm + 2 gripper). The remaining dims are
    # privileged (object joint positions + goal) and are dropped.
    PROPRIO_DIM = 9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Guard against a config that declares agent_pos with a different width
        # than the proprioceptive slice we actually feed the model.
        ap = self.shape_meta["obs"].get("agent_pos")
        if ap is not None:
            declared = int(ap["shape"][0])
            assert declared == self.PROPRIO_DIM, (
                f"shape_meta.obs.agent_pos.shape=[{declared}] but this dataset "
                f"exposes only the proprioceptive prefix of the state "
                f"(PROPRIO_DIM={self.PROPRIO_DIM}). Set agent_pos shape to "
                f"{self.PROPRIO_DIM}."
            )

    def _get_buffer_keys(self) -> List[str]:
        # rgb_keys + lowdim_keys come from shape_meta.obs.
        # "state" is the buffer key behind the model-side "agent_pos";
        # add it explicitly when (and only when) the user asks for
        # "agent_pos" in shape_meta.
        keys = list(self.rgb_keys) + list(self.lowdim_keys) + ["action"]
        if "agent_pos" in self.lowdim_keys:
            keys.remove("agent_pos")
            keys.append("state")
        # dedup, preserve order
        seen = set()
        return [k for k in keys if not (k in seen or seen.add(k))]

    def _lowdim_key_map(self) -> Dict[str, str]:
        # Maps model-side normalizer keys to the buffer keys actually present.
        m: Dict[str, str] = {"action": "action"}
        for k in self.lowdim_keys:
            m[k] = "state" if k == "agent_pos" else k
        return m

    def _prepare_lowdim_for_norm(self, norm_key: str, arr):
        # Fit normalizer statistics on the same proprioceptive slice we feed the
        # model (see __getitem__), not the full 60-D privileged state.
        if norm_key == "agent_pos":
            return arr[:, : self.PROPRIO_DIM]
        return arr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.num_datasets == 1:
            data = self.samplers[0].sample_data(idx)
        else:
            i = int(np.random.choice(self.num_datasets, p=self.sample_probabilities))
            data = self.samplers[i].sample_data(idx % len(self.samplers[i]))

        if self.transforms is not None and self.rgb_keys:
            data = self._apply_color_jitter(data)

        # Drop the privileged (object + goal) tail of the kitchen state, keeping
        # only the proprioceptive qp(9) prefix routed into obs["agent_pos"].
        if "agent_pos" in data["obs"]:
            data["obs"]["agent_pos"] = data["obs"]["agent_pos"][..., : self.PROPRIO_DIM]

        return dict_apply(data, torch.from_numpy)
