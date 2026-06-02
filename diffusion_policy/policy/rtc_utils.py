"""
Real-Time Chunking (RTC) via ΠGDM inpainting guidance for VP-SDE diffusion policies.

This is the server-side math for "Real-Time Action Chunking for Diffusion Policies"
(Part 5 / Algorithm 3 / Listing 1, plus the σ_d Cobot tweak of §6.7). It is used as a
drop-in correction inside the reverse-diffusion loop of an ε-prediction diffusion policy:
at every denoising step we nudge the model's clean estimate x̂₀ toward a previously
committed action chunk on a soft-masked prefix, then feed the algebraically-derived
ε_cond into the existing DDIM/DDPM scheduler step.

Two public helpers:
  - get_prefix_weights(start, end, total, schedule): the soft mask W (LeRobot-style
    signature; the ``exp`` curve follows the document's eq. (21)).
  - pigdm_eps_correction(...): one guidance step, returns ε_cond.
"""

import math
from typing import Literal

import torch


def get_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: Literal["zeros", "linear", "exp", "ones"],
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Soft prefix mask W of shape (total,), values in [0, 1].

    Indices [0, start)      -> 1.0          (frozen prefix; already committed)
    Indices [start, end)    -> taper 1 -> 0 (soft; previous chunk's best guess)
    Indices [end, total)    -> 0.0          (free; fully regenerated)

    ``start`` = inference delay ``d``; ``end`` = execution horizon (end of taper).
    ``linear``/``zeros``/``ones`` match LeRobot's get_prefix_weights exactly. ``exp``
    follows the document's eq. (21): W = (exp(c) - 1)/(e - 1) with the same linear ramp
    ``c`` LeRobot uses (NOT LeRobot's c*expm1(c)/(e-1) variant).
    """
    start = min(start, end)
    idx = torch.arange(total, device=device, dtype=dtype)
    if schedule == "ones":
        w = torch.ones(total, device=device, dtype=dtype)
    elif schedule == "zeros":
        w = (idx < start).to(dtype)
    elif schedule in ("linear", "exp"):
        # c is the clamped linear ramp == (end - i)/(end - start + 1) on the taper region,
        # clamped to 1 on the frozen prefix.
        c = ((start - 1 - idx) / (end - start + 1) + 1).clamp(0.0, 1.0)
        if schedule == "exp":
            w = torch.expm1(c) / (math.e - 1.0)
        else:
            w = c
    else:
        raise ValueError(f"Invalid prefix_attention_schedule: {schedule!r}")
    return torch.where(idx >= end, torch.zeros_like(w), w)


def build_rtc_tensors(
    rtc_target,
    batch_size: int,
    n_obs_steps: int,
    prediction_horizon: int,
    action_dim: int,
    inference_delay,
    execution_horizon,
    schedule: str,
    device,
    dtype,
):
    """Place a future-window RTC target + soft mask into the full prediction-horizon frame.

    ``rtc_target`` is the NORMALIZED previous-chunk target of shape (B, H, A) where
    H = prediction_horizon - (n_obs_steps - 1) is the future-action window the server returns
    (positions [n_obs_steps-1, prediction_horizon) of the denoised trajectory). The leading
    n_obs_steps-1 past-token positions get weight 0.

    Returns (target_full (B,P,A), weights_full (P,)) or (None, None) when rtc_target is None.
    """
    if rtc_target is None:
        return None, None

    P = int(prediction_horizon)
    start = int(n_obs_steps) - 1          # first future slot in the denoised trajectory
    H = P - start                          # future-window length == chunk_size
    if execution_horizon is None:
        execution_horizon = H
    if inference_delay is None:
        inference_delay = 0
    inference_delay = int(inference_delay)
    execution_horizon = int(execution_horizon)

    assert rtc_target.shape[1] == H and rtc_target.shape[2] == action_dim, (
        f"rtc_target shape {tuple(rtc_target.shape)} incompatible with future window "
        f"(H={H}, A={action_dim}); expected (B, {H}, {action_dim})."
    )
    assert 0 <= inference_delay <= execution_horizon <= H, (
        f"require 0 <= inference_delay ({inference_delay}) <= execution_horizon "
        f"({execution_horizon}) <= H ({H})."
    )

    weights_chunk = get_prefix_weights(
        inference_delay, execution_horizon, H, schedule, device=device, dtype=dtype
    )
    weights_full = torch.zeros(P, device=device, dtype=dtype)
    weights_full[start:P] = weights_chunk

    target_full = torch.zeros(batch_size, P, action_dim, device=device, dtype=dtype)
    target_full[:, start:P, :] = rtc_target.to(device=device, dtype=dtype)
    return target_full, weights_full


def pigdm_eps_correction(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    abar: torch.Tensor,
    target_full: torch.Tensor,
    weights_full: torch.Tensor,
    max_guidance_weight: float,
    sigma_d: float,
) -> torch.Tensor:
    """One ΠGDM guidance step → corrected noise estimate ε_cond.

    Follows Listing 1 / eq. (28),(31),(38) of the document, σ_d-generalized:

        x̂₀      = (x_t - √(1-ᾱ)·ε) / √ᾱ                         (Tweedie, eq. 25)
        e        = (target - x̂₀) · W                            (masked residual)
        g        = VJP  eᵀ (∂x̂₀/∂x_t)                            (one backward pass)
        w_t      = min(β, (ᾱσ_d² + (1-ᾱ)) / (√ᾱ σ_d²))          (eq. 38; → 1/√ᾱ at σ_d=1)
        x̂₀_cond  = x̂₀ + w_t · g                                  (eq. 28)
        ε_cond   = (x_t - √ᾱ·x̂₀_cond) / √(1-ᾱ)                   (eq. 31)

    Requirements:
      - ``x_t`` is a leaf with ``requires_grad_(True)``.
      - ``eps`` was produced by the model from ``x_t`` under an enabled-grad context, so the
        autograd graph from ``eps`` back to ``x_t`` exists.
      - ``weights_full`` has shape (P,), ``target_full`` has shape (B, P, A), both already
        placed in the full prediction-horizon frame (zeros outside the guided slice).

    Returns ε_cond detached (shape == eps.shape), safe to hand to ``scheduler.step``.
    """
    abar = abar.to(x_t.dtype)
    sqrt_abar = torch.sqrt(abar)
    sqrt_1m = torch.sqrt(1.0 - abar)

    x0_hat = (x_t - sqrt_1m * eps) / sqrt_abar

    # Masked residual (a_prev - x̂₀)·W. grad_outputs is the VJP seed → treat as a constant.
    e = ((target_full - x0_hat) * weights_full[None, :, None]).detach()
    (g,) = torch.autograd.grad(outputs=x0_hat, inputs=x_t, grad_outputs=e)

    sd2 = float(sigma_d) ** 2
    raw_w = (abar * sd2 + (1.0 - abar)) / (sqrt_abar * sd2)
    w_t = torch.clamp(raw_w, max=float(max_guidance_weight))

    x0_cond = (x0_hat + w_t * g).detach()
    eps_cond = (x_t.detach() - sqrt_abar * x0_cond) / sqrt_1m
    return eps_cond
