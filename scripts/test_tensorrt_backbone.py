"""
Test TensorRT (torch-tensorrt, dynamo, fp16) on the ViT backbone.

Run in the isolated clone env (has torch-tensorrt):
    /home/ajay/miniforge3/envs/accel_test/bin/python scripts/test_tensorrt_backbone.py -i <ckpt>

Two measurements:
  (1) ISOLATED backbone latency at the camera-batched steady-state batch size:
      fp32+TF32  vs  fp16 eager  vs  fp16 torch.compile  vs  fp16 TensorRT.
  (2) END-TO-END: swap the TRT engine into the policy and measure full predict_action
      latency + action drift vs the fp32 baseline.
"""
import copy
import time
import click
import numpy as np
import torch
import torch.nn as nn

from policy_inference import PolicyInferenceNode
from benchmark_inference import build_request


def bench(fn, n=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


class TRTBackbone(nn.Module):
    """Wrap a fp16 TRT engine so it drops into the fp32 encoder pipeline."""
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
    def forward(self, x):
        return self.engine(x.half()).float()


@click.command()
@click.option("--input", "-i", required=True)
@click.option("--batch", default=3, type=int, help="Camera-batched steady-state batch (3 cams)")
def main(input, batch):
    import torch_tensorrt
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    node = PolicyInferenceNode(input, "127.0.0.1", 0, "cuda:0", num_ddim_inference_steps=10,
                               use_ddim=True, gripper_multiplier=1.0)
    pol = node.policy
    bb = pol.obs_encoder.backbone.eval()
    crop = pol.obs_encoder.crop_size
    print(f"backbone crop={crop} | test batch={batch} | feature_dim={pol.obs_encoder.feature_dim}")

    x32 = torch.randn(batch, 3, crop, crop, device="cuda")
    x16 = x32.half()
    bb16 = copy.deepcopy(bb).eval().half()  # separate fp16 copy; bb stays fp32

    # ---- (1) isolated backbone latency ----
    with torch.no_grad():
        t_fp32 = bench(lambda: bb(x32))
        t_fp16 = bench(lambda: bb16(x16))
        cbb = torch.compile(bb16, mode="max-autotune-no-cudagraphs", dynamic=False)
        cbb(x16); torch.cuda.synchronize()
        t_comp = bench(lambda: cbb(x16))
        print("\nBuilding TensorRT engine (fp16, dynamo)…")
        trt = torch_tensorrt.compile(
            bb16, ir="dynamo", inputs=[x16],
            enabled_precisions={torch.half}, truncate_double=True,
        )
        trt(x16); torch.cuda.synchronize()
        t_trt = bench(lambda: trt(x16))

        # accuracy of TRT vs fp32 (cls token, which is what _process_rgb uses)
        ref = bb(x32)[:, 0, :].float()
        got = trt(x16)[:, 0, :].float()
        d = (got - ref).abs()
    print("\n=== ISOLATED BACKBONE (batch=%d, %dpx) ===" % (batch, crop))
    print(f"  fp32 + TF32        : {t_fp32:6.3f} ms  (1.00x)")
    print(f"  fp16 eager         : {t_fp16:6.3f} ms  ({t_fp32/t_fp16:.2f}x)")
    print(f"  fp16 torch.compile : {t_comp:6.3f} ms  ({t_fp32/t_comp:.2f}x)")
    print(f"  fp16 TensorRT      : {t_trt:6.3f} ms  ({t_fp32/t_trt:.2f}x)")
    print(f"  TRT cls-token drift vs fp32: max {d.max():.2e}  rms {d.pow(2).mean().sqrt():.2e}")

    # ---- (2) end-to-end swap ----
    pol.obs_encoder.batch_rgb_inference = True
    req = build_request(node, n_raw=1, seed=0)
    def e2e(seed=0):
        torch.manual_seed(seed); return node.predict_action(req)["action"]
    base = e2e()
    t_base = bench(lambda: e2e(), n=30, warmup=8)
    # swap TRT engine in
    pol.obs_encoder.backbone = TRTBackbone(trt)
    got = e2e()
    t_swap = bench(lambda: e2e(), n=30, warmup=8)
    dd = np.abs(got - base)
    print("\n=== END-TO-END (camera-batched, 10 DDIM steps) ===")
    print(f"  eager fp32+TF32 backbone : {t_base:6.2f} ms")
    print(f"  TensorRT fp16 backbone   : {t_swap:6.2f} ms  ({t_base/t_swap:.2f}x)")
    print(f"  action drift vs fp32: max {dd.max():.2e}  rms {np.sqrt((dd**2).mean()):.2e}")


if __name__ == "__main__":
    main()
