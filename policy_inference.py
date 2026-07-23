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

from diffusion_policy.common.inference_accel import apply_acceleration
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
        gripper_multiplier: float = 1.5,
        # Acceleration knobs — all default OFF so a bare run == original behavior.
        amp: str = "no",
        tf32: bool = False,
        camera_batch: bool = False,
        compile_backbone: bool = False,
        compile_unet: bool = False,
        compile_mode: str = "default",
        compile_unet_mode: str = "default",
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

        # Apply inference-acceleration knobs. With every knob at its default (all OFF), this is a
        # verified no-op — it only re-asserts PyTorch's defaults — so a bare run matches the
        # original behavior exactly. Acceleration engages only when a flag is turned on.
        # NOTE: --compile-unet with --unet-compile-mode reduce-overhead is NOT RTC-safe (the RTC
        # path backprops through the UNet) — see diffusion_policy/common/inference_accel.py.
        apply_acceleration(
            self.policy,
            amp=amp,
            tf32=tf32,
            camera_batch=camera_batch,
            compile_backbone=compile_backbone,
            compile_unet=compile_unet,
            compile_mode=compile_mode,
            compile_unet_mode=compile_unet_mode,
        )

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

        self.gripper_multiplier = float(gripper_multiplier)
        if self.gripper_multiplier != 1.0:
            print(f"gripper_multiplier={self.gripper_multiplier} (applied to action gripper columns)")

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

    def _build_rtc_kwargs(self, request: dict) -> dict:
        """Translate the optional ``request["rtc"]`` block into predict_action kwargs.

        The client sends ``prev_action_chunk`` in EEF execution space (7D one-arm or 14D
        bimanual: [pose(6), grip(1)] per arm). We map it into the model's action space:
          - gripper models (action_dim 7/14): reverse the post-inference gripper_multiplier;
          - no-gripper models (action_dim 6/12): drop the gripper column(s).
        Then normalize into the space the diffusion runs in. Returns {} when no rtc block.
        """
        rtc = request.get("rtc")
        if rtc is None:
            return {}

        prev = np.asarray(rtc["prev_action_chunk"], dtype=np.float32)  # (H, 7) or (H, 14)
        if prev.ndim != 2:
            raise ValueError(f"prev_action_chunk must be (H, D); got shape {prev.shape}.")
        A = int(self.policy.action_dim)
        g = self.gripper_multiplier

        if prev.shape[1] == 7:
            if A == 7:
                prev = prev.copy()
                if g != 1.0:
                    prev[:, 6] /= g
            elif A == 6:
                prev = prev[:, :6].copy()
            else:
                raise ValueError(f"7D prev_action_chunk incompatible with model action_dim={A}.")
        elif prev.shape[1] == 14:
            if A == 14:
                prev = prev.copy()
                if g != 1.0:
                    prev[:, 6] /= g
                    prev[:, 13] /= g
            elif A == 12:
                prev = np.concatenate([prev[:, :6], prev[:, 7:13]], axis=1)
            else:
                raise ValueError(f"14D prev_action_chunk incompatible with model action_dim={A}.")
        else:
            raise ValueError(f"prev_action_chunk dim {prev.shape[1]} not in (7, 14).")

        assert prev.shape[1] == A, (prev.shape, A)
        target = torch.from_numpy(prev).float().unsqueeze(0).to(self.device)  # (1, H, A)
        target = self.policy.normalizer["action"].normalize(target)
        return {
            "rtc_target": target,
            "rtc_inference_delay": int(rtc["inference_delay"]),
            "rtc_execution_horizon": int(rtc["execution_horizon"]),
            "rtc_schedule": str(rtc["prefix_attention_schedule"]),
            "rtc_max_guidance_weight": float(rtc["max_guidance_weight"]),
            "rtc_sigma_d": float(rtc.get("sigma_d", 1.0)),
        }

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
        rtc_kwargs = self._build_rtc_kwargs(request)
        # RTC needs autograd for the ΠGDM VJP, so it cannot run under inference_mode.
        grad_ctx = torch.enable_grad() if rtc_kwargs else torch.no_grad()
        with grad_ctx:
            result = self.policy.predict_action(
                raw_obs_dict,
                cached_long=cached_long,
                cached_short=cached_short,
                use_DDIM=self.use_ddim,
                **rtc_kwargs,
            )

        # Drop the first n_obs_steps - 1 entries (past-token predictions); return the future actions.
        action = result["action_pred"][0].detach().cpu().numpy()[T - 1:]

        # Apply gripper multiplier (action layout: 7D=[pose(6), grip(1)],
        # 14D=[poseL(6), gripL(1), poseR(6), gripR(1)]; 6D/12D are no-gripper).
        if self.gripper_multiplier != 1.0 and action.ndim == 2:
            action = action.copy()
            if action.shape[1] == 7:
                action[:, 6] *= self.gripper_multiplier
            elif action.shape[1] == 14:
                action[:, 6] *= self.gripper_multiplier
                action[:, 13] *= self.gripper_multiplier
            elif action.shape[1] in (6, 12):
                pass  # no-gripper action; nothing to scale
            else:
                print(f"WARNING: unrecognized action shape {action.shape}; gripper multiplier not applied.")

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
@click.option("--gripper-multiplier", default=1.5, type=float,
              help="Scale factor applied to gripper columns of returned action. Default: 1.5. Pass 1.0 to disable.")
# ── Acceleration (all OFF by default → bare run == original behavior) ──────────────────────
@click.option("--tf32/--no-tf32", default=False,
              help="TF32 matmul + cuDNN autotune. Default: OFF. ~1.8x (FiLM)/1.3x (attn), drift ~1e-4.")
@click.option("--camera-batch/--no-camera-batch", default=False,
              help="Batch all cameras into one backbone call. Default: OFF. Free, numerically exact, +7-9%.")
@click.option("--amp", type=click.Choice(["no", "bf16", "fp16"]), default="no",
              help="Autocast the backbone (+ FiLM UNet) to bf16/fp16. Default: no. Prefer fp16.")
@click.option("--compile-backbone", is_flag=True, help="torch.compile the ViT backbone(s). Small win.")
@click.option("--compile-unet", is_flag=True,
              help="torch.compile the denoising UNet (biggest lever). NOT RTC-safe with --unet-compile-mode reduce-overhead.")
@click.option("--compile-mode", type=click.Choice(["default", "reduce-overhead", "max-autotune"]),
              default="default", help="torch.compile mode for the BACKBONE (cudagraph modes auto-downgraded).")
@click.option("--unet-compile-mode", type=click.Choice(["default", "reduce-overhead", "max-autotune"]),
              default="default", help="torch.compile mode for the UNet. reduce-overhead is fastest but NOT RTC-safe.")
def main(input, ip, port, device, steps, ddpm, gripper_multiplier,
         tf32, camera_batch, amp, compile_backbone, compile_unet, compile_mode, unet_compile_mode):
    node = PolicyInferenceNode(
        ckpt_path=input,
        ip=ip,
        port=port,
        device=device,
        num_ddim_inference_steps=steps,
        use_ddim=not ddpm,
        gripper_multiplier=gripper_multiplier,
        amp=amp,
        tf32=tf32,
        camera_batch=camera_batch,
        compile_backbone=compile_backbone,
        compile_unet=compile_unet,
        compile_mode=compile_mode,
        compile_unet_mode=unet_compile_mode,
    )
    node.run_node()


if __name__ == "__main__":
    main()