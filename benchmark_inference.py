"""
Benchmark + tune inference latency for the TimmObsEncoder diffusion policies.

Loads a checkpoint exactly like ``policy_inference.py`` (same EMA selection, same
class assertions), synthesises a realistic *steady-state* cache-mode request (one
new frame per camera at 10 Hz; older positions served from the feature cache), and
sweeps the acceleration knobs from ``diffusion_policy.common.inference_accel``:

    fp32 baseline → +TF32 → +bf16 AMP → +compile(backbone) → +compile(backbone+UNet)

For each config it reports end-to-end latency (mean / p50 / p95 over N timed calls)
and, crucially, the **action drift** versus the fp32 / 10-step reference so you can
judge the quality cost of bf16 and of fewer DDIM steps. Finally it sweeps the DDIM
step count on the fastest config to show the latency↔quality tradeoff.

Because torch.compile mutates the policy irreversibly, eager configs are measured
first, then the backbone/UNet are compiled once and the compiled configs measured.
Each config runs under try/except so one failure (e.g. a cudagraphs incompatibility
on bleeding-edge hardware) does not abort the whole sweep.

Usage:
    python benchmark_inference.py -i /path/to/latest.ckpt
    python benchmark_inference.py -i /path/to/latest.ckpt --compile-mode max-autotune \
        --unet-compile-mode reduce-overhead --iters 50

Run with the `umi` env: /home/ajay/miniforge3/envs/umi/bin/python benchmark_inference.py ...
"""

import statistics
import time
from typing import Dict, List, Optional

import click
import cv2
import numpy as np
import torch

from diffusion_policy.common.inference_accel import apply_acceleration, enable_tf32
from policy_inference import PolicyInferenceNode


# ── Synthetic request ─────────────────────────────────────────────────────────

def _png_bytes(h: int, w: int, rng: np.random.Generator) -> bytes:
    """Encode a random HxWx3 uint8 image to PNG bytes (matches the ZMQ wire format)."""
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok, "cv2.imencode failed"
    return buf.tobytes()


def build_request(node: PolicyInferenceNode, n_raw: int = 1, seed: int = 0) -> dict:
    """Build a cache-mode request with ``n_raw`` fresh frames per camera.

    Steady state (10 Hz, one new frame per step) is ``n_raw=1``: the oldest
    n_obs_steps-1 positions are served from the long cache, the newest from raw PNGs.
    Mirrors the layout documented at the top of policy_inference.py.
    """
    rng = np.random.default_rng(seed)
    T = node.n_obs_steps
    H = node.short_range_obs_horizon  # None for single-encoder policies
    D = node.policy.obs_encoder.feature_dim

    obs_meta = node.cfg.task.shape_meta.obs
    # RGB on-disk shape is (C, H, W); PNGs carry HxW.
    img_hw = {k: (int(obs_meta[k].shape[1]), int(obs_meta[k].shape[2])) for k in node.rgb_keys}

    n_raw = max(1, min(n_raw, T))
    rgb_raw: Dict[str, List[bytes]] = {}
    cached_long: Dict[str, List[np.ndarray]] = {}
    for k in node.rgb_keys:
        h, w = img_hw[k]
        rgb_raw[k] = [_png_bytes(h, w, rng) for _ in range(n_raw)]
        cached_long[k] = [rng.standard_normal(D).astype(np.float32) for _ in range(T - n_raw)]

    cached_short: Dict[str, List[np.ndarray]] = {}
    if H is not None:
        n_short_raw = min(n_raw, H)
        for k in node.rgb_keys:
            cached_short[k] = [rng.standard_normal(D).astype(np.float32) for _ in range(H - n_short_raw)]

    lowdim: Dict[str, np.ndarray] = {}
    for k in node.low_dim_keys:
        dim = int(np.prod(obs_meta[k].shape))
        lowdim[k] = rng.standard_normal((T, dim)).astype(np.float32)

    return {
        "rgb_raw": rgb_raw,
        "cached_long": cached_long,
        "cached_short": cached_short,
        "lowdim": lowdim,
    }


# ── Timing ────────────────────────────────────────────────────────────────────

def _time_call(node: PolicyInferenceNode, request: dict, seed: int) -> tuple:
    """One seeded inference. Returns (elapsed_ms, action ndarray).

    Seeding before the call fixes the initial diffusion noise (predict_action draws
    a single ``torch.randn`` from the global generator, since no generator is passed),
    so the action is comparable across configs and only differs by the knob effects.
    """
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    resp = node.predict_action(request)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), resp["action"]


def measure(
    node: PolicyInferenceNode,
    request: dict,
    steps: int,
    iters: int,
    warmup: int,
    seed: int = 0,
) -> Optional[dict]:
    """Warm up then time ``iters`` calls at the given DDIM step count.

    Returns dict(mean, p50, p95, action) in ms, or None if the config errored.
    """
    node.policy.num_ddim_inference_steps = steps
    try:
        for _ in range(warmup):
            _time_call(node, request, seed)
        times = []
        action = None
        for _ in range(iters):
            ms, action = _time_call(node, request, seed)
            times.append(ms)
    except Exception as e:  # noqa: BLE001 — one bad config shouldn't kill the sweep
        print(f"      !! errored: {type(e).__name__}: {str(e)[:200]}")
        return None
    times.sort()
    return {
        "mean": statistics.mean(times),
        "p50": times[len(times) // 2],
        "p95": times[min(len(times) - 1, int(0.95 * len(times)))],
        "action": action,
    }


def _drift(action: np.ndarray, ref: Optional[np.ndarray]) -> tuple:
    """(max-abs, rms) difference between an action chunk and the reference."""
    if ref is None or action is None or action.shape != ref.shape:
        return (float("nan"), float("nan"))
    d = (action.astype(np.float64) - ref.astype(np.float64))
    return float(np.abs(d).max()), float(np.sqrt((d ** 2).mean()))


# ── Sweep ─────────────────────────────────────────────────────────────────────

EAGER_CONFIGS = [
    # name,                 amp,    tf32,  note
    ("fp32 baseline",       "no",   False, "reference (current default)"),
    ("fp32 + TF32",         "no",   True,  "free"),
    ("bf16 AMP + TF32",     "bf16", True,  "backbone bf16; attn-UNet stays fp32"),
]


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint (.ckpt or run dir)")
@click.option("--device", default="cuda:0")
@click.option("--iters", default=30, type=int, help="Timed iterations per config")
@click.option("--warmup", default=8, type=int, help="Warmup iterations (compile/cudnn settle)")
@click.option("--n-raw", default=1, type=int, help="Fresh frames per camera (1 = 10Hz steady state)")
@click.option("--steps", default=10, type=int, help="DDIM steps for the head-to-head config table")
@click.option("--step-sweep", default="10,8,6,4,2", help="DDIM steps to sweep on the fastest config")
@click.option("--compile-mode", default="default", help="torch.compile mode for the backbone")
@click.option("--unet-compile-mode", default="default",
              help="torch.compile mode for the UNet (try reduce-overhead; not RTC-safe)")
@click.option("--no-compile", is_flag=True, help="Skip the torch.compile configs")
def main(input, device, iters, warmup, n_raw, steps, step_sweep, compile_mode, unet_compile_mode, no_compile):
    print("=" * 84)
    print("Loading checkpoint (EMA selection + class checks mirror policy_inference.py)…")
    node = PolicyInferenceNode(
        ckpt_path=input, ip="127.0.0.1", port=0, device=device,
        num_ddim_inference_steps=steps, use_ddim=True, gripper_multiplier=1.0,
    )
    is_dual = node.short_range_obs_horizon is not None
    n_enc = len(node.rgb_keys) * (2 if is_dual else 1)
    print("-" * 84)
    print(f"policy            : {type(node.policy).__name__}")
    print(f"backbone          : {node.cfg.policy.obs_encoder.model_name}")
    print(f"n_obs_steps       : {node.n_obs_steps}   action_dim: {node.policy.action_dim}   "
          f"horizon: {node.policy.prediction_horizon}")
    print(f"rgb keys          : {node.rgb_keys}")
    print(f"dual encoder      : {is_dual} (short_range_obs_horizon={node.short_range_obs_horizon})")
    print(f"backbone encodes  : {n_enc} ViT passes / call at n_raw={n_raw}")
    print(f"device            : {device}  ({torch.cuda.get_device_name(0)})")
    print("=" * 84)

    request = build_request(node, n_raw=n_raw, seed=0)

    results = []  # (name, note, stats)
    ref_action = None

    # ── Eager configs (no compile) ───────────────────────────────────────────
    for name, amp, tf32, note in EAGER_CONFIGS:
        enable_tf32(tf32)
        node.policy.mixed_precision = None if amp == "no" else amp
        print(f"\n>> {name}  (steps={steps})")
        stats = measure(node, request, steps, iters, warmup)
        if stats is None:
            results.append((name, note, None, (float("nan"), float("nan"))))
            continue
        if ref_action is None:
            ref_action = stats["action"]
        drift = _drift(stats["action"], ref_action)
        results.append((name, note, stats, drift))
        print(f"   mean {stats['mean']:7.2f} ms | p50 {stats['p50']:7.2f} | p95 {stats['p95']:7.2f} | "
              f"drift max {drift[0]:.2e} rms {drift[1]:.2e}")

    # ── Compiled configs ─────────────────────────────────────────────────────
    fastest_name, fastest_stats = None, None
    if not no_compile:
        # bf16 + TF32 stays on for the compiled configs.
        enable_tf32(True)
        node.policy.mixed_precision = "bf16"

        compiled_specs = [
            ("bf16 + compile(backbone)", dict(compile_backbone=True)),
            ("bf16 + compile(backbone+UNet)", dict(compile_backbone=True, compile_unet=True)),
        ]
        already_compiled_backbone = False
        for name, spec in compiled_specs:
            print(f"\n>> {name}  (steps={steps})  [first run compiles — may take a minute]")
            try:
                if spec.get("compile_backbone") and not already_compiled_backbone:
                    apply_acceleration(node.policy, amp="bf16", tf32=True,
                                       compile_backbone=True, compile_mode=compile_mode)
                    already_compiled_backbone = True
                if spec.get("compile_unet"):
                    from diffusion_policy.common.inference_accel import compile_policy
                    compile_policy(node.policy, unet=True, mode=unet_compile_mode)
            except Exception as e:  # noqa: BLE001
                print(f"   !! compile setup errored: {type(e).__name__}: {str(e)[:200]}")
                results.append((name, "", None, (float("nan"), float("nan"))))
                continue
            stats = measure(node, request, steps, iters, max(warmup, 12))
            if stats is None:
                results.append((name, "", None, (float("nan"), float("nan"))))
                continue
            drift = _drift(stats["action"], ref_action)
            results.append((name, "", stats, drift))
            print(f"   mean {stats['mean']:7.2f} ms | p50 {stats['p50']:7.2f} | p95 {stats['p95']:7.2f} | "
                  f"drift max {drift[0]:.2e} rms {drift[1]:.2e}")
            if stats and (fastest_stats is None or stats["mean"] < fastest_stats["mean"]):
                fastest_name, fastest_stats = name, stats

    # ── Summary table ─────────────────────────────────────────────────────────
    base = next((s for n, _, s, _ in results if n == "fp32 baseline" and s), None)
    base_ms = base["mean"] if base else None
    print("\n" + "=" * 84)
    print(f"SUMMARY  (steps={steps}, n_raw={n_raw}, {iters} iters)   "
          f"backbone={n_enc} ViT passes/call")
    print("-" * 84)
    print(f"{'config':<34}{'mean ms':>10}{'speedup':>9}{'drift(max)':>13}{'drift(rms)':>12}")
    print("-" * 84)
    for name, note, stats, drift in results:
        if stats is None:
            print(f"{name:<34}{'ERR':>10}")
            continue
        sp = f"{base_ms / stats['mean']:.2f}x" if base_ms else "-"
        print(f"{name:<34}{stats['mean']:>10.2f}{sp:>9}{drift[0]:>13.2e}{drift[1]:>12.2e}")
    print("=" * 84)

    # ── DDIM step sweep on the fastest config ─────────────────────────────────
    if fastest_stats is not None:
        try:
            sweep = [int(s) for s in step_sweep.split(",") if s.strip()]
        except ValueError:
            sweep = [10, 8, 6, 4, 2]
        print(f"\nDDIM step sweep on '{fastest_name}'  (drift vs fp32 baseline @ {steps} steps)")
        print("-" * 84)
        print(f"{'steps':>6}{'mean ms':>11}{'speedup':>9}{'drift(max)':>13}{'drift(rms)':>12}")
        print("-" * 84)
        for s in sweep:
            st = measure(node, request, s, iters, max(warmup, 12))
            if st is None:
                continue
            drift = _drift(st["action"], ref_action)
            sp = f"{base_ms / st['mean']:.2f}x" if base_ms else "-"
            print(f"{s:>6}{st['mean']:>11.2f}{sp:>9}{drift[0]:>13.2e}{drift[1]:>12.2e}")
        print("-" * 84)
        print("Pick the smallest step count whose drift is acceptable on-robot; combine with")
        print("the fastest config above. Wire the winners into policy_inference.py flags.")


if __name__ == "__main__":
    main()
