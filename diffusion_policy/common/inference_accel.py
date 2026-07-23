"""
Inference acceleration knobs for the TimmObsEncoder diffusion policies.

This module centralises every speed/quality lever so the live server
(``policy_inference.py``) and the benchmark harness (``benchmark_inference.py``)
stay in sync. The levers, cheapest → heaviest:

  1. TF32 / cuDNN autotune  — global matmul/conv precision flags. Free, applies to
     the fp32 paths (the ViT backbone and, on the attention model, the UNet which
     self-pins to fp32). Negligible quality cost on Ampere+; large win on Blackwell.
  2. AMP autocast           — bf16/fp16 around ``predict_action`` via the policy's
     ``mixed_precision`` attr. Halves backbone cost. The attention UNet disables
     autocast internally (it NaNs in bf16), so AMP only touches the backbone there;
     on the FiLM model it also covers the (small) FiLM UNet.
  3. torch.compile          — inductor-compile the ViT backbone(s) and/or the UNet.
     The backbone is the dominant cost (3 encodes/call for FiLM, 6 for the dual
     encoder), so compiling it is the biggest single win available without TensorRT.
  4. DDIM step count        — handled at the call site (``policy.num_ddim_inference_steps``);
     latency is linear in the number of steps. See benchmark_inference.py.

Design notes / gotchas:
  * Compile is applied AFTER weights are loaded, so it does not interfere with
    checkpoint loading (no ``._orig_mod.`` key rewriting needed).
  * The RTC (ΠGDM) path backpropagates through the UNet. ``reduce-overhead``
    (CUDA graphs) does not support backward, so compiling the UNet with that mode
    is incompatible with RTC — we warn. The backbone is always safe to compile
    (its output is detached before the RTC VJP).
  * Every function is no-op-safe: passing the "off" value leaves the policy as-is.
"""

import logging
import os
import subprocess
import sys
import warnings
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

AMP_CHOICES = ("no", "bf16", "fp16")
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")


# ── 1. TF32 / cuDNN ───────────────────────────────────────────────────────────

def enable_tf32(enabled: bool = True, cudnn_benchmark: Optional[bool] = None) -> None:
    """Toggle the matmul-TF32 + cuDNN-autotune perf bundle (process-global).

    TF32 keeps fp32 storage but uses reduced-mantissa tensor-core matmuls.

    ``enabled=False`` **restores PyTorch's defaults** (matmul precision "highest",
    matmul TF32 off, cuDNN TF32 *on* — its default —, cuDNN benchmark off), so it
    exactly reproduces the model's pre-tuning behavior. It deliberately does NOT force
    cuDNN TF32 off: that was never disabled by default, and forcing it would make
    ``--no-tf32`` differ from the original. ``cudnn_benchmark`` defaults to follow
    ``enabled`` (on with TF32, off without); pass an explicit bool only to override.
    """
    bench = enabled if cudnn_benchmark is None else cudnn_benchmark
    if enabled:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True          # already the torch default
    else:
        torch.set_float32_matmul_precision("highest")   # ── torch defaults (== original) ──
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True           # torch default — do NOT force off
    torch.backends.cudnn.benchmark = bench
    logger.info(
        "TF32 %s | matmul_precision=%s cudnn.allow_tf32=%s cudnn.benchmark=%s",
        "ON" if enabled else "OFF (torch defaults)",
        torch.get_float32_matmul_precision(),
        torch.backends.cudnn.allow_tf32, bench,
    )


# ── 1b. Camera-batched encode ─────────────────────────────────────────────────

def set_camera_batch(policy: nn.Module, on: bool = True) -> None:
    """Toggle batching all rgb keys into one backbone call per encoder (inference-only).

    Free and numerically exact (the ViT has no cross-batch ops): replaces N per-camera
    batch-1 ViT calls with one batch-N call. Measured +7% (FiLM, 3 cams) / +9% (dual
    encoder, 6 calls → 2). See timm_obs_encoder.encode_with_cache.
    """
    enc = getattr(policy, "obs_encoder", None)
    if enc is not None:
        enc.batch_rgb_inference = on
    short = getattr(policy, "short_range_encoder", None)
    if short is not None:
        short.batch_rgb_inference = on
    logger.info("camera-batched encode: %s", on)


# ── 2. AMP autocast ───────────────────────────────────────────────────────────

def set_amp(policy: nn.Module, amp: str = "no") -> None:
    """Set the policy's ``mixed_precision`` attr, read by ``predict_action``.

    ``amp`` ∈ {"no", "bf16", "fp16"}. "bf16" is the safe default (wider dynamic
    range than fp16; CLIP backbones tolerate it well). The attention UNet ignores
    this and stays fp32 by design.
    """
    if amp not in AMP_CHOICES:
        raise ValueError(f"amp must be one of {AMP_CHOICES}, got {amp!r}")
    # The policies read getattr(self, "mixed_precision", None) or "no".
    policy.mixed_precision = None if amp == "no" else amp
    logger.info("AMP autocast: %s", amp)


# ── 3. torch.compile ──────────────────────────────────────────────────────────

def _ptxas_major_minor(path: str) -> Optional[tuple]:
    """Return (major, minor) CUDA version of a ptxas binary, or None if unreadable."""
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5).stdout
        for tok in out.replace(",", " ").split():
            if tok.count(".") == 1 and tok.replace(".", "").isdigit():
                maj, mn = tok.split(".")
                return int(maj), int(mn)
    except Exception:  # noqa: BLE001
        return None
    return None


def ensure_compile_toolchain() -> None:
    """Make sure Triton uses a ptxas new enough for this GPU before compiling.

    The classic Blackwell (sm_120) gotcha: if the conda env isn't activated on PATH,
    Triton falls back to an old system ``/usr/bin/ptxas`` that doesn't know
    ``sm_120a`` and torch.compile dies with a PTXAS codegen error. The torch cu12.x
    wheels ship a matching ptxas next to the Python executable; point Triton at it.

    Respects an already-set ``TRITON_PTXAS_PATH``. No-op if no local ptxas is found.
    """
    if os.environ.get("TRITON_PTXAS_PATH"):
        return
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    needed = cap[0] * 10 + cap[1]  # e.g. (12, 0) -> 120
    local = os.path.join(os.path.dirname(sys.executable), "ptxas")
    if not os.path.exists(local):
        return
    ver = _ptxas_major_minor(local)
    # cu12.8 ptxas (the first to support sm_120) reports 12.8; require >= that for sm120+.
    if ver is not None and (needed < 120 or ver >= (12, 8)):
        os.environ["TRITON_PTXAS_PATH"] = local
        logger.info("Set TRITON_PTXAS_PATH=%s (CUDA %d.%d) for sm_%d", local, ver[0], ver[1], needed)


def _backbone_safe_mode(mode: str) -> str:
    """Map a requested mode to a CUDA-graph-FREE equivalent for the backbone.

    The backbone's outputs (encoded features) escape the compiled region: they are
    cached and returned to the client (``new_features_long`` / ``_short``), and read
    after the UNet has run. CUDA graphs reuse static output buffers, so a graphed
    backbone hands back tensors that get overwritten on the next run
    ("accessing tensor output of CUDAGraphs that has been overwritten"). Both
    "reduce-overhead" and "max-autotune" enable CUDA graphs, so strip them here.
    """
    if mode == "reduce-overhead":
        return "default"
    if mode == "max-autotune":
        return "max-autotune-no-cudagraphs"
    return mode


def _compile_backbone(encoder: nn.Module, mode: str, dynamic: Optional[bool]) -> None:
    """Compile the timm ViT backbone of a TimmObsEncoder in place."""
    if encoder is None or not hasattr(encoder, "backbone"):
        return
    encoder.backbone = torch.compile(encoder.backbone, mode=mode, dynamic=dynamic)


def compile_policy(
    policy: nn.Module,
    *,
    backbone: bool = False,
    unet: bool = False,
    mode: str = "default",
    unet_mode: Optional[str] = None,
    dynamic: Optional[bool] = False,
) -> nn.Module:
    """torch.compile the requested sub-modules of a Timm diffusion policy, in place.

    Args:
        backbone  : compile the long-range obs_encoder backbone (and the
                    short_range_encoder backbone if the policy has one). CUDA-graph
                    modes are silently downgraded to a graph-free equivalent because
                    the backbone's outputs are cached/returned (see _backbone_safe_mode).
        unet      : compile the denoising UNet (``policy.model``).
        mode      : torch.compile mode for the backbone.
        unet_mode : torch.compile mode for the UNet; defaults to ``mode``.
                    "reduce-overhead" (CUDA graphs) is the fastest for the UNet but is
                    incompatible with the RTC backward pass — see module docstring.
        dynamic   : passed to torch.compile. ``False`` specialises on the exact input
                    shapes (fastest steady-state); the backbone batch dim varies
                    between cold-start (all frames raw) and steady-state (1 raw frame),
                    so a recompile happens for each distinct size.

    Returns the same policy object (mutated) for convenience.
    """
    if mode not in COMPILE_MODES:
        raise ValueError(f"mode must be one of {COMPILE_MODES}, got {mode!r}")
    unet_mode = unet_mode or mode
    if unet_mode not in COMPILE_MODES:
        raise ValueError(f"unet_mode must be one of {COMPILE_MODES}, got {unet_mode!r}")

    if backbone or unet:
        ensure_compile_toolchain()

    if backbone:
        safe = _backbone_safe_mode(mode)
        if safe != mode:
            warnings.warn(
                f"Backbone compile mode '{mode}' enables CUDA graphs, which corrupt the "
                f"cached features the encoder returns to the client. Using '{safe}' for the "
                f"backbone instead. (CUDA graphs are only safe on the UNet.)",
                RuntimeWarning,
            )
        _compile_backbone(getattr(policy, "obs_encoder", None), safe, dynamic)
        _compile_backbone(getattr(policy, "short_range_encoder", None), safe, dynamic)
        logger.info("torch.compile: backbone(s) (mode=%s, dynamic=%s)", safe, dynamic)

    if unet:
        if unet_mode == "reduce-overhead":
            warnings.warn(
                "Compiling the UNet with mode='reduce-overhead' (CUDA graphs) is "
                "incompatible with the RTC (ΠGDM) path, which backpropagates through "
                "the UNet. Use 'default'/'max-autotune' if you run RTC, or keep "
                "unet-compile off for RTC inference.",
                RuntimeWarning,
            )
        policy.model = torch.compile(policy.model, mode=unet_mode, dynamic=dynamic)
        logger.info("torch.compile: UNet (mode=%s, dynamic=%s)", unet_mode, dynamic)

    return policy


# ── Convenience wrapper ───────────────────────────────────────────────────────

def apply_acceleration(
    policy: nn.Module,
    *,
    amp: str = "no",
    tf32: bool = False,
    cudnn_benchmark: Optional[bool] = None,
    camera_batch: bool = False,
    compile_backbone: bool = False,
    compile_unet: bool = False,
    compile_mode: str = "default",
    compile_unet_mode: Optional[str] = None,
    compile_dynamic: Optional[bool] = False,
) -> nn.Module:
    """Apply all selected acceleration knobs to a loaded policy, in place.

    ``compile_mode`` controls the backbone; ``compile_unet_mode`` controls the UNet
    (defaults to ``compile_mode``). Use ``compile_unet_mode="reduce-overhead"`` for the
    fastest UNet when NOT serving RTC. ``camera_batch`` is free + numerically exact.
    Order matters only in that TF32/AMP must be set before the warmup calls that trigger
    compilation: call this once, after the policy is on-device and in eval mode, then run
    a few warmup inferences before timing.
    """
    enable_tf32(tf32, cudnn_benchmark=cudnn_benchmark)
    set_amp(policy, amp)
    set_camera_batch(policy, camera_batch)
    compile_policy(
        policy,
        backbone=compile_backbone,
        unet=compile_unet,
        mode=compile_mode,
        unet_mode=compile_unet_mode,
        dynamic=compile_dynamic,
    )
    return policy
