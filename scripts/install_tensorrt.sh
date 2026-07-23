#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Optional: install Torch-TensorRT into the `umi` conda env for the RTX 5090.
#
# Read this whole file before running. This is the ONE heavy/risky step — it pulls
# ~1–2 GB and must match your exact torch build. Only do this if the torch.compile
# numbers from benchmark_inference.py leave latency on the table you actually need.
#
# Hardware/software this targets (verified 2026-06):
#   GPU    : NVIDIA RTX 5090  (Blackwell, compute capability sm_120)
#   env    : runpod_remote_test  →  /home/ajay/miniforge3/envs/runpod_remote_test
#   torch  : 2.9.1   (CUDA 12.9, CUDA libs provided by conda — env/bin/ptxas is 12.9)
#
# Why this is finicky on this box:
#   * Blackwell (sm_120) needs TensorRT >= 10.7. Torch-TensorRT must be built for
#     torch 2.9. Your torch reports CUDA 12.9 (conda-provided) while the public
#     torch-tensorrt 2.9 wheels are built against cu128 — usually compatible across
#     12.x minor versions, but this mismatch is the #1 thing to watch if it errors.
#   * Torch-TensorRT brings its OWN `tensorrt` wheels as deps — do NOT also install
#     a system/standalone TensorRT, or you'll get two copies fighting on the path.
#   * If you hit ABI/symbol errors, the clean fallback is a fresh conda env with the
#     official cu128 torch wheel (pip install torch==2.9.1 --index-url .../cu128)
#     so torch + torch-tensorrt + tensorrt all agree on CUDA 12.8.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=/home/ajay/miniforge3/envs/runpod_remote_test/bin/python
PIP="$PY -m pip"

echo "== torch / CUDA in target env =="
$PY - <<'PYEOF'
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda,
      "| device:", torch.cuda.get_device_name(0),
      "| cap:", torch.cuda.get_device_capability(0))
assert torch.cuda.get_device_capability(0)[0] >= 9, "Expected Blackwell-class GPU"
PYEOF

echo
echo "== Installing torch-tensorrt matched to torch 2.9 (cu128) =="
# Torch-TensorRT 2.9.x is built against torch 2.9.x. It pulls the matching
# `tensorrt`/`tensorrt_cu12` wheels itself. The cu128 index serves the right libs.
$PIP install --no-deps torch-tensorrt==2.9.* \
  --extra-index-url https://download.pytorch.org/whl/cu128
# Its runtime deps (tensorrt 10.x bindings) — let pip resolve these normally:
$PIP install "tensorrt>=10.7" "tensorrt_cu12>=10.7" || true

echo
echo "== Smoke test: compile a tiny module through the Torch-TRT dynamo backend =="
$PY - <<'PYEOF'
import torch, torch_tensorrt
print("torch_tensorrt:", torch_tensorrt.__version__)
m = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.GELU()).eval().cuda().half()
ex = torch.randn(1, 64, device="cuda", dtype=torch.half)
trt = torch_tensorrt.compile(
    m, ir="dynamo", inputs=[ex],
    enabled_precisions={torch.half}, truncate_double=True,
)
out = trt(ex)
print("OK — TRT module ran, output:", tuple(out.shape), out.dtype)
PYEOF

echo
echo "Done. Next: see INFERENCE_OPTIMIZATION.md §TensorRT to compile the ViT backbone."
