"""
ZMQ inference server with observation-feature caching.

The server loads a trained TimmObsEncoder policy and exposes a ZMQ endpoint.
Each request supplies observations for all n_obs_steps positions in the obs window
(oldest → newest). Cached encoder features occupy the oldest positions; raw PNGs occupy
the newest. The server only runs the vision backbone on raw frames and returns those
features so the client can extend its cache.

Supported policies:
    DiffusionUnetTimmAttentionPolicy
    DiffusionUnetTimmFilmPolicy

Wire format (pickled via socket.send_pyobj / recv_pyobj):

    request = {
        # Per rgb key: cached features at oldest positions, raw PNG bytes at newest.
        # len(cached_long[k]) + len(rgb_raw[k]) must equal n_obs_steps.
        "rgb_raw":      { "<rgb_key>": [png_bytes, ...] },
        "cached_long":  { "<rgb_key>": [np.ndarray, ...] },
        "cached_short": { "<rgb_key>": [np.ndarray, ...] },  # dual-encoder only

        # Low-dim observations pre-batched across the obs window.
        "lowdim":       { "<low_dim_key>": np.ndarray of shape (n_obs_steps, dim) },
    }

    response = {
        "action": np.ndarray,                                          # (n_future_actions, action_dim)
        "new_features_long":  { "<rgb_key>": [np.ndarray, ...] },      # one per raw frame, oldest→newest
        "new_features_short": { "<rgb_key>": [np.ndarray, ...] },      # dual-encoder only
    }
    # On error: response is the traceback string instead of a dict.

NOTE: For dual-encoder policies, cached_short[k] covers positions [T - H, T - n_raw),
      where T = n_obs_steps, H = short_range_obs_horizon, n_raw = len(rgb_raw[k]).
"""

import os
import sys
import time
import traceback
from typing import Dict, List

import click
import cv2
import dill
import hydra
import numpy as np
import omegaconf
import torch
import zmq

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.diffusion_unet_timm_attention_policy import (
    DiffusionUnetTimmAttentionPolicy,
)
from diffusion_policy.policy.diffusion_unet_timm_film_policy import (
    DiffusionUnetTimmFilmPolicy,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace

SUPPORTED_POLICIES = (DiffusionUnetTimmAttentionPolicy, DiffusionUnetTimmFilmPolicy)


def echo_exception() -> str:
    exc_type, exc_value, exc_tb = sys.exc_info()
    # Extract unformatted traceback, print line of code where the exception occurred
    return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))


class PolicyInferenceNode:
    def __init__(
        self,
        ckpt_path: str,
        ip: str,
        port: int,
        device: str,
        num_ddim_inference_steps: int = 10,
        use_ddim: bool = True,
    ):
        # Load checkpoint (accept either a .ckpt or a run dir) and dump the cfg.
        self.ckpt_path = ckpt_path
        if not self.ckpt_path.endswith(".ckpt"):
            self.ckpt_path = os.path.join(self.ckpt_path, "checkpoints", "latest.ckpt")
        with open(self.ckpt_path, "rb") as f:
            payload = torch.load(f, map_location="cpu", pickle_module=dill)
        self.cfg = payload["cfg"]

        # Dump the loaded configuration to a YAML file for easy inspection.
        cfg_path = self.ckpt_path.replace(".ckpt", ".yaml")
        with open(cfg_path, "w") as f:
            f.write(omegaconf.OmegaConf.to_yaml(self.cfg))
        print(f"Loaded checkpoint: {self.ckpt_path}")
        print(f"Policy target: {self.cfg.policy._target_}")

        # Build the workspace via hydra and restore trained weights.
        cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace: BaseWorkspace = cls(self.cfg)
        self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # Prefer EMA copy at inference time when trained with EMA.
        self.policy: BaseImagePolicy = self.workspace.model
        if getattr(self.cfg.training, "use_ema", False):
            self.policy = self.workspace.ema_model
            print("Using EMA model")

        # Reject checkpoints whose policy class doesn't expose predict_action_cached.
        assert isinstance(self.policy, SUPPORTED_POLICIES), (
            f"Unsupported policy class {type(self.policy).__name__}. "
            f"policy_inference.py only supports {[c.__name__ for c in SUPPORTED_POLICIES]}. "
            f"Hybrid (Robomimic/R3M) and lowdim policies are not supported."
        )

        # Move policy to device and configure the inference sampler.
        self.device = torch.device(device)
        self.policy.eval().to(self.device)
        if hasattr(self.policy, "num_ddim_inference_steps"):
            self.policy.num_ddim_inference_steps = num_ddim_inference_steps
        self.use_ddim = use_ddim

        # Cache obs structure; sort-order must match TimmObsEncoder's internal ordering.
        obs_shape_meta = self.cfg.task.shape_meta.obs
        self.rgb_keys = sorted([k for k, v in obs_shape_meta.items() if v.get("type") == "rgb"])
        self.low_dim_keys = sorted([k for k, v in obs_shape_meta.items() if v.get("type", "low_dim") == "low_dim"])
        self.n_obs_steps = int(self.cfg.n_obs_steps)
        # None for FiLM / single-encoder attention; an int for dual-encoder attention.
        self.short_range_obs_horizon = getattr(self.policy, "short_range_obs_horizon", None)

        print(
            f"n_obs_steps={self.n_obs_steps} | rgb_keys={self.rgb_keys} | "
            f"low_dim_keys={self.low_dim_keys} | "
            f"short_range_obs_horizon={self.short_range_obs_horizon}"
        )

        self.ip = ip
        self.port = port

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _decode_rgb(self, png_bytes: bytes) -> torch.Tensor:
        """PNG bytes → (C, H, W) float32 tensor in [0, 1] on self.device."""
        img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None — bad PNG payload.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).contiguous().float().div(255.0).to(self.device)

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_action(self, request: dict) -> dict:
        # Window layout: positions [0, T - n_raw) are cached, [T - n_raw, T) are raw.
        T = self.n_obs_steps
        H = self.short_range_obs_horizon  # None for single-encoder policies

        # Decode raw RGB PNG bytes per key into per-key lists of (C,H,W) tensors on device.
        nobs_rgb_raw: Dict[str, List[torch.Tensor]] = {
            key: [self._decode_rgb(b) for b in request["rgb_raw"][key]]
            for key in self.rgb_keys
        }

        # Convert cached arrays to per-key lists of device tensors (aligned with oldest window positions).
        cached_long: Dict[str, List[torch.Tensor]] = {
            key: [torch.from_numpy(np.asarray(a)).float().to(self.device) for a in arr_list]
            for key, arr_list in request["cached_long"].items()
        }
        cached_short: Dict[str, List[torch.Tensor]] = {}
        if H is not None:
            cached_short = {
                key: [torch.from_numpy(np.asarray(a)).float().to(self.device) for a in arr_list]
                for key, arr_list in request.get("cached_short", {}).items()
            }
        nobs_lowdim_full: Dict[str, torch.Tensor] = {
            key: torch.from_numpy(np.asarray(request["lowdim"][key])).float().unsqueeze(0).to(self.device)
            for key in self.low_dim_keys
        }

        # Run the cache-aware policy: it normalizes obs, encodes only raw frames, then DDIM/DDPM denoising.
        raw_obs_dict = {**nobs_rgb_raw, **nobs_lowdim_full}
        with torch.inference_mode():
            result = self.policy.predict_action(
                raw_obs_dict,
                cached_long=cached_long,
                cached_short=cached_short,
                use_DDIM=self.use_ddim,
            )

        # Drop the first n_obs_steps - 1 entries (past-token predictions); return the future actions.
        action = result["action_pred"][0].detach().cpu().numpy()[T - 1:]

        # Convert newly-encoded features (per-key lists of tensors) to per-key lists of ndarrays.
        new_features_long = {key: [v.detach().cpu().numpy() for v in vs] for key, vs in result["new_features_long"].items()}
        new_features_short = {key: [v.detach().cpu().numpy() for v in vs] for key, vs in result["new_features_short"].items() if vs}

        return {
            "action": action,
            "new_features_long": new_features_long,
            "new_features_short": new_features_short,
        }

    def run_node(self):
        # Bind ZMQ REP and serve forever; exceptions are returned as the response body.
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{self.ip}:{self.port}")
        print(f"PolicyInferenceNode listening on tcp://{self.ip}:{self.port}")
        while True:
            request = socket.recv_pyobj()
            try:
                t0 = time.monotonic()
                response = self.predict_action(request)
                n_new_long = sum(len(v) for v in response["new_features_long"].values())
                print(
                    f"Inference time: {time.monotonic() - t0:.3f}s | "
                    f"new_features_long={n_new_long} entries | "
                    # f"action shape={response['action'].shape}"
                )
            except Exception:
                err = echo_exception()
                print(f"Error:\n{err}")
                response = err
            socket.send_pyobj(response)


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint (.ckpt or run dir)")
@click.option("--ip", default="0.0.0.0")
@click.option("--port", default=8766, type=int, help="Port to listen on")
@click.option("--device", default="cuda:0", help="Device to run on")
@click.option("--steps", default=10, type=int, help="Number of DDIM inference steps")
@click.option("--ddpm", is_flag=True, help="Use DDPM sampler instead of DDIM")
def main(input, ip, port, device, steps, ddpm):
    node = PolicyInferenceNode(
        ckpt_path=input,
        ip=ip,
        port=port,
        device=device,
        num_ddim_inference_steps=steps,
        use_ddim=not ddpm,
    )
    node.run_node()


if __name__ == "__main__":
    main()