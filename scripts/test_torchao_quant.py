"""
Test torchao quantization (int8 / fp8) on the ViT backbone linears.

Run in the clone env (has torchao):
    /home/ajay/miniforge3/envs/accel_test/bin/python scripts/test_torchao_quant.py -i <ckpt>

Quantizes the backbone's nn.Linear layers (attn qkv/proj, mlp fc1/fc2 — the bulk of the
FLOPs), then torch.compiles (quant kernels need compile to be fast), and reports isolated
backbone latency + cls-token drift vs fp32. fp8 is the Blackwell-native lever.
"""
import time
import click
import torch

from policy_inference import PolicyInferenceNode


def bench(fn, n=50, warmup=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


@click.command()
@click.option("--input", "-i", required=True)
@click.option("--batch", default=3, type=int)
def main(input, batch):
    from torchao.quantization import (
        quantize_,
        int8_weight_only,
        int8_dynamic_activation_int8_weight,
    )
    try:
        from torchao.quantization import float8_dynamic_activation_float8_weight
        HAS_FP8 = True
    except Exception:
        HAS_FP8 = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    node = PolicyInferenceNode(input, "127.0.0.1", 0, "cuda:0", num_ddim_inference_steps=10,
                               use_ddim=True, gripper_multiplier=1.0)
    crop = node.policy.obs_encoder.crop_size
    x = torch.randn(batch, 3, crop, crop, device="cuda")
    print(f"crop={crop} batch={batch}")

    import copy
    base = node.policy.obs_encoder.backbone.eval()
    with torch.no_grad():
        ref = base(x)[:, 0, :].float()
        t_fp32 = bench(lambda: base(x))
    print(f"\nfp32 + TF32 backbone: {t_fp32:.3f} ms  (1.00x)")

    recipes = [
        ("bf16 + compile",                      None),
        ("int8 weight-only + compile",          int8_weight_only()),
        ("int8 dyn-act int8-weight + compile",  int8_dynamic_activation_int8_weight()),
    ]
    if HAS_FP8:
        recipes.append(("fp8 dyn-act fp8-weight + compile", float8_dynamic_activation_float8_weight()))

    for name, recipe in recipes:
        try:
            m = copy.deepcopy(base).to(torch.bfloat16)
            xb = x.to(torch.bfloat16)
            if recipe is not None:
                quantize_(m, recipe)
            cm = torch.compile(m, mode="max-autotune-no-cudagraphs", dynamic=False)
            with torch.no_grad():
                cm(xb); torch.cuda.synchronize()
                t = bench(lambda: cm(xb))
                got = cm(xb)[:, 0, :].float()
            d = (got - ref).abs()
            print(f"{name:38s}: {t:6.3f} ms  ({t_fp32/t:.2f}x)  drift max {d.max():.2e} rms {d.pow(2).mean().sqrt():.2e}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:38s}: FAILED — {type(e).__name__}: {str(e)[:140]}")


if __name__ == "__main__":
    main()
