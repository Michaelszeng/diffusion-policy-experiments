"""
Comprehensive single-load acceleration sweep for one checkpoint.

Loads the policy once and measures a curated matrix of knobs — precision
(fp32/TF32/bf16/fp16), camera-batched encode, and torch.compile (backbone +
UNet) — plus a DDIM step sweep on the fully-accelerated config. Reports
end-to-end latency (mean ms) and action drift vs the fp32 baseline.

Reuses the request builder + timing from benchmark_inference.py. torch.compile is
one-way, so eager configs run first, then compiled configs are added incrementally.

    python benchmark_comprehensive.py -i <ckpt> --iters 30
"""
import click
import torch

from benchmark_inference import build_request, measure, _drift
from diffusion_policy.common.inference_accel import compile_policy, enable_tf32
from policy_inference import PolicyInferenceNode


def _set_cam_batch(node, on: bool):
    node.policy.obs_encoder.batch_rgb_inference = on
    if getattr(node.policy, "short_range_encoder", None) is not None:
        node.policy.short_range_encoder.batch_rgb_inference = on


def _set(node, *, amp, tf32, cam):
    enable_tf32(tf32)
    node.policy.mixed_precision = None if amp == "no" else amp
    _set_cam_batch(node, cam)


@click.command()
@click.option("--input", "-i", required=True)
@click.option("--device", default="cuda:0")
@click.option("--iters", default=30, type=int)
@click.option("--warmup", default=10, type=int)
@click.option("--steps", default=10, type=int)
def main(input, device, iters, warmup, steps):
    node = PolicyInferenceNode(input, "127.0.0.1", 0, device, num_ddim_inference_steps=steps,
                               use_ddim=True, gripper_multiplier=1.0)
    is_dual = node.short_range_obs_horizon is not None
    n_enc = len(node.rgb_keys) * (2 if is_dual else 1)
    print("=" * 92)
    print(f"{type(node.policy).__name__} | {node.cfg.policy.obs_encoder.model_name} | "
          f"n_obs={node.n_obs_steps} dual={is_dual} | {n_enc} ViT passes/call | {torch.cuda.get_device_name(0)}")
    print("=" * 92)
    req = build_request(node, n_raw=1, seed=0)
    rows = []
    ref = None

    # ── Eager configs (freely switchable) ────────────────────────────────────
    eager = [
        ("fp32 baseline",            dict(amp="no",   tf32=False, cam=False)),
        ("TF32",                     dict(amp="no",   tf32=True,  cam=False)),
        ("TF32 + camera-batch",      dict(amp="no",   tf32=True,  cam=True)),
        ("TF32 + bf16",              dict(amp="bf16", tf32=True,  cam=False)),
        ("TF32 + fp16",              dict(amp="fp16", tf32=True,  cam=False)),
        ("TF32 + bf16 + cam-batch",  dict(amp="bf16", tf32=True,  cam=True)),
    ]
    for name, st in eager:
        _set(node, **st)
        print(f">> {name}")
        s = measure(node, req, steps, iters, warmup)
        if s is None:
            rows.append((name, None, (float('nan'),)*2)); continue
        ref = ref if ref is not None else s["action"]
        d = _drift(s["action"], ref); rows.append((name, s, d))
        print(f"   {s['mean']:7.2f} ms | drift max {d[0]:.2e} rms {d[1]:.2e}")

    # ── Compiled configs (one-way), built on bf16 + TF32 + cam-batch ─────────
    _set(node, amp="bf16", tf32=True, cam=True)
    compiled = [
        ("+ compile backbone (MA-no-cudagraphs)", dict(backbone=True, mode="max-autotune")),
        ("+ compile UNet (reduce-overhead) = ALL", dict(unet=True, unet_mode="reduce-overhead")),
    ]
    for name, spec in compiled:
        print(f">> {name}  [compiling…]")
        try:
            compile_policy(node.policy, **spec)
            s = measure(node, req, steps, iters, max(warmup, 14))
        except Exception as e:  # noqa: BLE001
            print(f"   !! {type(e).__name__}: {str(e)[:160]}"); rows.append((name, None, (float('nan'),)*2)); continue
        if s is None:
            rows.append((name, None, (float('nan'),)*2)); continue
        d = _drift(s["action"], ref); rows.append((name, s, d))
        print(f"   {s['mean']:7.2f} ms | drift max {d[0]:.2e} rms {d[1]:.2e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    base = rows[0][1]["mean"] if rows and rows[0][1] else None
    print("\n" + "=" * 92)
    print(f"{'config':<42}{'mean ms':>10}{'speedup':>9}{'drift max':>13}{'drift rms':>12}")
    print("-" * 92)
    for name, s, d in rows:
        if s is None:
            print(f"{name:<42}{'ERR':>10}"); continue
        sp = f"{base/s['mean']:.2f}x" if base else "-"
        print(f"{name:<42}{s['mean']:>10.2f}{sp:>9}{d[0]:>13.2e}{d[1]:>12.2e}")
    print("=" * 92)

    # ── DDIM step sweep on the fully-accelerated config ──────────────────────
    print("\nDDIM step sweep on fully-accelerated config (drift vs fp32 baseline @ %d):" % steps)
    print(f"{'steps':>6}{'mean ms':>11}{'speedup':>9}{'drift rms':>12}")
    for st in (10, 8, 6, 4, 2):
        s = measure(node, req, st, iters, 14)
        if s is None: continue
        d = _drift(s["action"], ref); sp = f"{base/s['mean']:.2f}x" if base else "-"
        print(f"{st:>6}{s['mean']:>11.2f}{sp:>9}{d[1]:>12.2e}")


if __name__ == "__main__":
    main()
