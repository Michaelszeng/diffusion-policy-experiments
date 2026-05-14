"""
ZMQ inference server with per-frame observation-feature caching.

The server loads a trained TimmObsEncoder policy and exposes a ZMQ endpoint.
Each request contains n_obs_steps observations indexed by client-generated `frame_ids`.
The client may include cached encoder features for some frames; the server only runs vision
backbone forward passes on non-cached images.
The response returns the action sequence (with past prediction tokens sliced off) plus any
features the server just encoded so the client can extend its cache.

Supported policies:
    DiffusionUnetTimmAttentionPolicy
    DiffusionUnetTimmFilmPolicy

Wire format (pickled via socket.send_pyobj / recv_pyobj):

    request = {
        "frames": {                          # n_obs_steps entries, insertion order = oldest → newest
            <frame_id>: {
                "<rgb_key>":     {"raw": <png_bytes>}                               # raw frame
                                 | {"long": np.ndarray}                             # cached, long-range only
                                 | {"long": np.ndarray, "short": np.ndarray},       # cached, dual encoder
                "<low_dim_key>": np.ndarray,                                        # (dim,)
            },
            ...
        }
    }

    response = {
        "action": np.ndarray,                # (n_future_actions, action_dim)
        "new_features": {                    # everything the server encoded this turn
            <frame_id>: {
                "<rgb_key>": {"long": np.ndarray, "short": np.ndarray | omitted},
            },
            ...
        },
    }
    # On error: response is the traceback string instead of a dict.
"""

import os
import sys
import time
import traceback
from typing import Dict, Hashable, Tuple

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

    def _normalize_rgb_dict(self, key: str, frame_dict: Dict[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        """Apply the policy's normalizer to a per-frame dict of (C,H,W) tensors."""
        if len(frame_dict) == 0:
            return {}
        fids = list(frame_dict.keys())
        stacked = torch.stack([frame_dict[fid] for fid in fids], dim=0).unsqueeze(0)  # (1, T, C, H, W)
        normalized = self.policy.normalizer.normalize({key: stacked})[key]
        return {fid: normalized[0, i] for i, fid in enumerate(fids)}

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_action(self, request: dict) -> dict:
        # Parse request envelope; frame insertion order is oldest → newest.
        frames = request["frames"]
        if len(frames) != self.n_obs_steps:
            raise ValueError(f"request['frames'] has {len(frames)} entries, expected n_obs_steps={self.n_obs_steps}")
        frame_ids = list(frames.keys())
        expected_keys = set(self.rgb_keys) | set(self.low_dim_keys)

        # Walk every (fid, key) once and route to raw-RGB / cached / low-dim collections.
        raw_rgb_unnorm: Dict[str, Dict[Hashable, torch.Tensor]] = {k: {} for k in self.rgb_keys}
        cached_long: Dict[Tuple[str, Hashable], torch.Tensor] = {}
        cached_short: Dict[Tuple[str, Hashable], torch.Tensor] = {}
        lowdim_per_frame: Dict[str, Dict[Hashable, np.ndarray]] = {k: {} for k in self.low_dim_keys}

        for fid, frame in frames.items():
            provided = set(frame.keys())
            missing = expected_keys - provided
            if missing:
                raise KeyError(f"frame {fid!r} is missing keys: {sorted(missing)}")
            extra = provided - expected_keys
            if extra:
                raise KeyError(f"frame {fid!r} has unexpected keys: {sorted(extra)}")

            for key, val in frame.items():
                if key in self.rgb_keys:
                    self._parse_rgb_entry(fid, key, val, raw_rgb_unnorm, cached_long, cached_short)
                else:
                    lowdim_per_frame[key][fid] = np.asarray(val)

        # Decode + normalize raw RGB frames (per-key batched normalization).
        nobs_rgb_raw: Dict[str, Dict[Hashable, torch.Tensor]] = {
            key: self._normalize_rgb_dict(key, raw_rgb_unnorm[key]) for key in self.rgb_keys
        }

        # Stack per-frame low-dim slices in frame_ids order and normalize.
        nobs_lowdim_full: Dict[str, torch.Tensor] = {}
        for key in self.low_dim_keys:
            arrs = [lowdim_per_frame[key][fid] for fid in frame_ids]
            stacked = np.stack(arrs, axis=0)  # (n_obs_steps, dim)
            t = torch.from_numpy(stacked).float().unsqueeze(0).to(self.device)  # (1, T, dim)
            nobs_lowdim_full[key] = self.policy.normalizer.normalize({key: t})[key]

        # Run the cache-aware policy: encodes only un-cached frames, then DDIM/DDPM.
        # The encoder raises KeyError with frame/key context if a required entry is missing
        # (e.g., a short-range frame that has only 'long' cached and no raw).
        with torch.inference_mode():
            result = self.policy.predict_action_cached(
                nobs_rgb_raw=nobs_rgb_raw,
                nobs_lowdim_full=nobs_lowdim_full,
                cached_long=cached_long,
                cached_short=cached_short,
                frame_ids=frame_ids,
                use_DDIM=self.use_ddim,
            )

        # Drop the first n_obs_steps-1 entries (past_token_prediction outputs); return the rest.
        action_pred = result["action_pred"][0].detach().cpu().numpy()
        action = action_pred[self.n_obs_steps - 1 :]

        # Reshape new_features into nested dict[fid → dict[key → dict[tag → ndarray]]].
        new_features_nested: Dict[Hashable, Dict[str, Dict[str, np.ndarray]]] = {}
        for (key, fid, tag), tensor in result["new_features"].items():
            new_features_nested.setdefault(fid, {}).setdefault(key, {})[tag] = tensor.detach().cpu().numpy()
        return {"action": action, "new_features": new_features_nested}

    def _parse_rgb_entry(
        self,
        fid: Hashable,
        key: str,
        val: dict,
        raw_rgb_unnorm: Dict[str, Dict[Hashable, torch.Tensor]],
        cached_long: Dict[Tuple[str, Hashable], torch.Tensor],
        cached_short: Dict[Tuple[str, Hashable], torch.Tensor],
    ):
        """Dispatch one rgb-key entry: tagged union with either {'raw'} or {'long'[, 'short']}."""
        if not isinstance(val, dict):
            raise ValueError(f"frame {fid!r} key {key!r} must be a dict (raw/long/short); got {type(val).__name__}")
        keys = set(val.keys())
        if "raw" in keys:
            if keys != {"raw"}:
                raise ValueError(
                    f"frame {fid!r} key {key!r}: 'raw' is mutually exclusive with 'long'/'short' (got {sorted(keys)})"
                )
            raw_rgb_unnorm[key][fid] = self._decode_rgb(val["raw"])
            return
        unknown = keys - {"long", "short"}
        if unknown:
            raise ValueError(f"frame {fid!r} key {key!r} has unknown subkeys: {sorted(unknown)}")
        if "long" not in keys:
            raise ValueError(f"frame {fid!r} key {key!r}: must contain 'raw' or 'long' (got {sorted(keys)})")
        cached_long[(key, fid)] = torch.from_numpy(np.asarray(val["long"])).float().to(self.device)
        if "short" in keys:
            cached_short[(key, fid)] = torch.from_numpy(np.asarray(val["short"])).float().to(self.device)

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
                print(
                    f"Inference time: {time.monotonic() - t0:.3f}s | "
                    f"new_features={len(response['new_features'])} entries | "
                    f"action shape={response['action'].shape}"
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
