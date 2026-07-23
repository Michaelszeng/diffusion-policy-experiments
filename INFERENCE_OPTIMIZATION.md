# Inference optimization — speeding up `policy_inference.py`

Knobs to cut inference latency for the two baggie checkpoints
(`2_obs_baggie_CLIP_film`, `8_obs_baggie_CLIP_attention_double_enc`) and any other
`DiffusionUnetTimmFilmPolicy` / `DiffusionUnetTimmAttentionPolicy` checkpoint.

> This is the **how-to-use-the-knobs** guide. For the full experiment log — every method
> tried (incl. TensorRT, quantization), the numbers, and *why* each did/didn't help — see
> [ACCELERATION_RESULTS.md](ACCELERATION_RESULTS.md).

**Env:** `runpod_remote_test` (`/home/ajay/miniforge3/envs/runpod_remote_test/bin/python`),
torch 2.9.1 / CUDA 12.9, on the RTX 5090 (Blackwell, sm_120).

```bash
conda activate runpod_remote_test          # puts the CUDA-12.9 ptxas on PATH (matters for compile)
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

---

## TL;DR — what to run

```bash
# 0. BASELINE = your original pre-optimization behavior. All acceleration is OFF by default,
#    so a bare run reproduces the original exactly (verified bit-for-bit vs PyTorch's defaults):
python policy_inference.py -i <ckpt>

# 1. Measure your checkpoint and see the speed/quality tradeoff of every knob:
python benchmark_inference.py -i /home/mnt/shared_models/baggie/8_obs_baggie_CLIP_attention_double_enc_v4/checkpoints/latest.ckpt

# 2. Opt in to acceleration. Start with the two FREE knobs (no meaningful quality cost):
python policy_inference.py -i <ckpt> --tf32 --camera-batch
#    Then add the rest for the full ~3x (fp16 + compile the UNet + fewer steps):
python policy_inference.py -i <ckpt> --tf32 --camera-batch --amp fp16 \
    --compile-unet --unet-compile-mode reduce-overhead --steps 6
#    Drop --unet-compile-mode reduce-overhead (or use 'default') if you serve RTC.
```

New `policy_inference.py` flags (all optional, all no-op-safe):

| flag | default | what it does |
|---|---|---|
| `--tf32 / --no-tf32` | **off** | TF32 tensor-core matmul + cuDNN autotune. Biggest *free* win — turn it on. |
| `--camera-batch / --no-camera-batch` | **off** | Batch all cameras into one backbone call. Free, numerically exact, +7–9% — turn it on. |
| `--amp {no,bf16,fp16}` | `no` | Autocast the ViT backbone (and FiLM UNet). **Prefer `fp16`** — faster-ish *and* more accurate than bf16 here. |
| `--compile-backbone` | off | `torch.compile` the ViT backbone(s). Small win — the backbone is only ~2 ms; the UNet loop is the bottleneck. |
| `--compile-unet` | off | `torch.compile` the denoising UNet. |
| `--compile-mode {default,reduce-overhead,max-autotune}` | `default` | mode for the **backbone**. CUDA-graph modes are auto-downgraded (backbone outputs escape — see knob 3). |
| `--unet-compile-mode {default,reduce-overhead,max-autotune}` | `default` | mode for the **UNet**. `reduce-overhead` is fastest but **not RTC-safe**. |
| `--steps N` | 10 | DDIM steps. Latency is ~linear in N. |

---

## Priority list — what to turn on, in order

Everything is OFF by default. Turn knobs on top-to-bottom: the first group costs no quality, the
second group trades quality for speed (increasing aggressiveness). Speeds are measured on the two
baggie checkpoints (FiLM / attention) at 10 steps unless noted.

### Group 1 — no quality cost (turn all of these on)
*(ranked by how much speed they buy)*

| # | Flag | Speed | Quality cost |
|---|---|---|---|
| 1 | `--compile-unet` (`--unet-compile-mode reduce-overhead`) | **biggest** — attention 1.27→**2.22×**, FiLM ~1.9→**2.33×** | none (same kernels, replayed). ⚠️ **not RTC-safe** → use `--unet-compile-mode default` if you serve RTC (still lossless, a hair slower) |
| 2 | `--tf32` | **1.80×** FiLM / **1.27×** attention | ~none (drift ~1e-4, 10× below the model's own error) |
| 3 | `--camera-batch` | +7% FiLM / +10% attention | **truly zero** (bit-identical output) |
| 4 | `--compile-backbone` | small (~2–5%) | ~none (fusion only) |

→ Group 1 alone ≈ **2.3× (FiLM) / 2.2× (attention)** at 10 steps, drift negligible.

### Group 2 — speed↑ for quality↓ (increasing aggressiveness)

| # | Flag | Speed | Quality cost (measured, action-MSE vs demos) |
|---|---|---|---|
| 5 | `--amp fp16` | small (~2%; backbone isn't the bottleneck) | tiny (drift rms ~2e-4). Prefer over `bf16` (more drift, same speed) |
| 6 | `--steps 6` | ~1.5× on top | **effectively none** — 0.00113 → 0.00109 (even slightly better) |
| 7 | `--steps 4` | another ~1.3× | still tiny — 0.00107 (flat) |
| 8 | `--steps 3` | a bit more | flat offline (0.00106) but near the edge — **verify in a rollout** |
| 9 | `--steps 2` | most | **real degradation** — 0.00135 (~25% worse); don't |

**The big surprise:** `--steps 10 → ~4` is essentially lossless *offline*, so it's nearly free too —
just confirm 3–4-step chunks in a rollout (low-step trajectories can be jerkier even at equal MSE).

### Recommended stacks
- **Safe max (no rollout needed):** `--tf32 --camera-batch --amp fp16 --compile-unet --unet-compile-mode reduce-overhead --steps 6` → **~3.3× / ~3.2×**. (Drop to `--unet-compile-mode default` for RTC.)
- **Push further (rollout-verified):** add `--steps 4` → **~4×**.

### Not worth it (proven no help — see ACCELERATION_RESULTS.md)
TensorRT on the backbone (2.8× isolated but 1.03× end-to-end), int8/fp8 quantization (slower than
fp16 here + less accurate), DPM-Solver++ (equivalent at best, worse at low steps), `channels_last`
(irrelevant for a ViT).

---

## Where the time goes

Both checkpoints share the **same `vit_base_patch16_clip_224.openai` ViT-B/16 backbone**,
and by default it runs in **fp32** (`mixed_precision: null` in the configs). With your
feature-caching, each 10 Hz step encodes only the newest frame per camera:

| model | n_obs | cameras | **ViT passes / call** | denoising UNet |
|---|---|---|---|---|
| FiLM (`2_obs`) | 2 | 3 | **3** (long enc only) | small FiLM 1D-UNet × `steps` |
| Attention double-enc (`8_obs`) | 8 | 3 | **6** (long + short enc) | cross-attention 1D-UNet × `steps`, **pinned to fp32** |

So the cost is **(N ViT passes) + (steps × one UNet forward) + fixed overhead**. Measured
decomposition (step-sweep of the accelerated config): the **DDIM UNet loop dominates** — per
step ≈ 2.3 ms (FiLM) / 2.7 ms (attention), while the whole ViT backbone is only ~2 ms (FiLM) /
~4 ms (attention). At 10 steps the UNet loop is ~75–80% of the time. So the big levers are
**compiling the UNet** and **fewer steps**; backbone tricks (fp16/compile/camera-batch) give
smaller absolute gains. (This is why TensorRT on the backbone — 2.8× in isolation — is only
1.03× end-to-end; see ACCELERATION_RESULTS.md.) The attention UNet is deliberately kept in fp32
(it NaNs in bf16 — see the comment in `conditional_unet_1d_static_attention.py`), so AMP there
only touches the backbone.

The benchmark decomposes this for you: it sweeps the DDIM step count, so the slope is
the per-step UNet cost and the intercept is "backbone + fixed overhead".

---

## What each knob actually does (plain English)

Quick mental model of a GPU: the **CPU** *launches* work ("kernels") onto the **GPU**, which runs
the math. Matrix multiplies/convs run on dedicated **tensor cores**. Two things cost time: the
actual math, and the *per-kernel launch overhead* (the CPU queueing each kernel). Different knobs
attack different parts of that.

### TF32 — cheaper fp32 matmuls
A normal fp32 number carries 23 bits of precision ("mantissa"). TF32 tells the tensor cores to
round each input to **10 mantissa bits** before multiplying (keeping fp32's full range and writing
an fp32 result). The matmul runs several× faster and, for a vision model, the precision drop is
invisible (we measured action drift ~1e-4). It's a global switch — the model itself is unchanged.
The companion **cuDNN autotune** (`cudnn.benchmark`) tries a few conv algorithms on the first call
and caches the fastest; free after warmup because our input shapes are fixed.

> **"Was TF32 already on?" — there are *two* TF32 switches with opposite PyTorch defaults, which is
> the usual confusion.** `cuda.matmul.allow_tf32` (matrix multiplies) is **OFF** by default — and
> matmuls are essentially all of this model's compute, so your original runs were full fp32 here;
> our `--tf32` knob *turns this on* (the ~1.8× win). `cudnn.allow_tf32` (convolutions only) is **ON**
> by default, but convs are a sliver of the work, so it never mattered. That's why `--no-tf32`
> leaves `cudnn.allow_tf32=True` — to *match* your original (matmul-TF32 off, conv-TF32 on), not to
> keep TF32 on. So: matmul-TF32 = the real lever and it was off before; you're enabling it now.

### AMP / autocast (fp16, bf16) — "mixed precision"
Runs the heavy ops in **16-bit** floats instead of 32-bit: half the memory traffic + faster
tensor-core paths. "Autocast" = PyTorch automatically uses 16-bit for the safe ops (matmul/conv)
and keeps 32-bit for sensitive ones (softmax, norms). The two 16-bit formats differ:
- **fp16**: more precision (10 mantissa bits), narrower range (can overflow to `inf`).
- **bf16**: full fp32 *range*, less precision (7 mantissa bits).
A CLIP backbone's activations are in-range, so **fp16 is both slightly faster and more accurate**
here — hence we prefer it. The attention UNet opts out (it overflows in 16-bit), so AMP only
speeds up the backbone on that model.

### Camera-batching — one GPU call instead of three
All cameras share one ViT, and a ViT processes each image independently (no cross-image mixing).
Originally we called the ViT once *per camera* (3 small GPU jobs). Camera-batching stacks the 3
images and calls the ViT **once** on a batch of 3 — identical math, but one bigger job instead of
three small ones, so less launch overhead and better GPU utilization. Free and bit-for-bit identical.

### torch.compile — fuse many ops into few fast kernels
Eager PyTorch runs one op at a time, each a separate GPU kernel. `torch.compile` traces the whole
forward once and (via its *Inductor*/*Triton* backend) **fuses** lots of small ops into a few big,
shape-specialized kernels. The first call is slow (it compiles); every call after is faster. The
**mode** sets how aggressive it is:

- **`default`** — trace + fuse + generate kernels. Solid; compiles in seconds.
- **`reduce-overhead`** = **CUDA graphs** *(the one you asked about)*. Even after fusing, the CPU
  still has to launch each remaining kernel one-by-one. For a *tiny* model like our 1D UNet, that
  launching is most of the wall-clock — the GPU finishes each kernel before the CPU can queue the
  next, so it idles ("launch-bound"). A **CUDA graph records the entire launch sequence once**, then
  every later call **replays the whole recording with a single command** — the per-kernel CPU cost
  nearly vanishes. Big win for the UNet (run 6–10× per inference). Two catches: it reuses fixed
  output buffers (so a graphed module's outputs get overwritten on the next run — why we *don't*
  graph the backbone, whose features we cache and return), and it can't run backprop (why it's not
  RTC-safe).
- **`max-autotune`** — before locking in kernels, it benchmarks several candidate implementations of
  each op and keeps the fastest. Best steady-state speed, slowest to compile (minutes). For the
  backbone we use `max-autotune-no-cudagraphs` (the autotuning without the buffer-reuse hazard).

### DDIM steps — fewer denoising passes
A diffusion policy makes an action by starting from random noise and **denoising** it over N passes,
running the UNet once per pass. **DDIM** is the math that lets you take big jumps (e.g. 10 passes
instead of the 100 used in training). Fewer steps = fewer UNet calls = proportionally faster, at
some quality cost. For these checkpoints DDIM stays accurate down to ~3–4 steps (see ACCELERATION_RESULTS.md),
so it's one of the biggest free levers — and it multiplies with the UNet-compile win above.

The next section gives the *when/why/how-much* for each, in cheapest→heaviest order.

---

## The knobs, cheapest → heaviest

### 1. TF32 + cuDNN autotune  —  free, turn this on first (`--tf32`)
`torch.set_float32_matmul_precision("high")` + `allow_tf32` + `cudnn.benchmark`. Keeps fp32
storage but uses reduced-mantissa tensor-core matmuls. Everything fp32 (the backbone always;
the attention UNet always) speeds up with **negligible** action drift. Off by default (so a bare
run == original); enable with `--tf32`. This is the single biggest free win — do it always.

### 2. AMP on the backbone (prefer fp16)  —  near-free
`--amp fp16` (or `bf16`) sets `policy.mixed_precision`, wrapping `predict_action` in autocast.
The attention UNet ignores it (stays fp32 by design); on FiLM the small FiLM UNet also runs in
the autocast dtype. At batch 1–3 the extra gain over TF32 is small (the ViT is launch-bound, not
bandwidth-bound). **Prefer `fp16` over `bf16`**: measured here fp16 is both slightly faster and
*more accurate* (drift 2e-4 vs 1.4e-3) — fp16 has 10 mantissa bits vs bf16's 7, and CLIP
activations sit comfortably in fp16's range.

### 3. torch.compile the backbone  —  small (the backbone isn't the bottleneck)
`--compile-backbone` inductor-compiles the ViT. First call pays a one-time compile cost
(seconds for `default`, a minute+ for `max-autotune`); after warmup it's fused + faster.
`--compile-mode` sets the backbone's mode. **CUDA-graph modes (`reduce-overhead`,
`max-autotune`) are auto-downgraded for the backbone**, because the backbone's encoded
features escape the compiled region (they're cached and returned to your client) and CUDA
graphs would hand back buffers that get overwritten on the next call. So the backbone
effectively runs `default` / `max-autotune-no-cudagraphs`. In practice backbone compile is a
small win at batch-1 (the ViT is launch-bound and TF32/bf16 already got most of it).

> **Blackwell ptxas gotcha (handled for you):** if the env isn't activated, Triton can grab
> the old system `/usr/bin/ptxas` (CUDA 12.0), which doesn't know `sm_120a`, and compile dies
> with `PTXAS error: Value 'sm_120a' is not defined`. `inference_accel.ensure_compile_toolchain()`
> auto-points `TRITON_PTXAS_PATH` at the CUDA-12.9 ptxas next to your Python. Activating the
> conda env also fixes it.

### 4. torch.compile the UNet  —  the big win for the attention model
`--compile-unet` compiles the denoising UNet, which runs `steps` times per inference, so the
fusion win is multiplied. This is the **dominant lever for the attention double-encoder**
(1.26× → 2.16×), whose fp32 cross-attention UNet is the bottleneck.
`--unet-compile-mode reduce-overhead` (CUDA graphs) is fastest because the 1D-UNet is tiny and
launch-bound (it adds ~10–15% over `default` on both models). The UNet's output is consumed by
the scheduler and never escapes, so CUDA graphs are safe here — **except with the RTC (ΠGDM)
path**, which backprops through the UNet. For RTC, use `--unet-compile-mode default` (supports
backward) or leave UNet compile off.

### 5. Fewer DDIM steps  —  free, but test quality
`--steps` cuts UNet forwards linearly. The benchmark's step-sweep reports the **action drift**
vs the 10-step reference at 8/6/4/2 steps. Drift is a proxy — confirm on-robot — but it lets
you pick the smallest step count that's still acceptable. Many policies are fine at 4–6.

### 6. DPM-Solver++ sampler  —  was mis-configured, not "broken"; fixable but no win here
The usual "fewer steps, same quality" trick is to swap DDIM for a 2nd-order DPM-Solver++. A
naive `DPMSolverMultistepScheduler` swap produced garbage (rms drift ~185 vs a DDIM@100 ref).
**That was a configuration bug, not a property of DPM++** — and the tell was in the data:

| sampler / config | @50 | @20 | @10 | @5 | reads as |
|---|--:|--:|--:|--:|---|
| DDIM (production) | 0.033 | 0.078 | 0.101 | 0.129 | consistent solver — error ↓ with steps |
| DPM++ vanilla | 185 | 244 | 264 | 274 | **flat & huge → wrong fixed point (bug)** |
| DPM++ `lambda_min_clipped=-5.1` | 0.87 | 0.88 | 0.90 | 1.29 | consistent again — error ↓ with steps |
| DPM++ `lambda_min_clipped=-5.1` + thresholding | 0.24 | 0.24 | 0.25 | 0.27 | stable; residual = clip-vs-no-clip gap |

(drift = rms of the normalized action trajectory vs DDIM@100.)

**Root cause — zero terminal SNR.** A flat error that doesn't shrink with steps can't be
discretization error (that → 0 as steps → ∞); it means the solver converges to the *wrong*
answer. Instrumentation cleared the obvious suspects: the scheduler's `alphas_cumprod` matches the
DDPM schedule exactly (maxdiff `0`) and its ε→x̂₀ conversion matches `(x−σ_t·ε)/α_t` to `4.7e-7`.
The real problem is the schedule: `squaredcos_cap_v2` gives **ᾱ₉₉ = 0** (zero terminal SNR), and
DPM-Solver++ (which works in x̂₀ space) starts its schedule at **t = 99**, where α_t = √ᾱ = 0, so
x̂₀ = (x − σ_t·ε)/α_t divides by ~0 and explodes. DDIM is immune because its schedule starts at
t = 90 (not 99) **and** `clip_sample=True` clamps x̂₀ — which is also why dynamic thresholding
"rescued but plateaued" DPM++ (clamping bounds the blow-up at the degenerate step without curing it).

**The fix** is the canonical one for zero-terminal-SNR (the same `lambda_min_clipped` Stable
Diffusion uses): clip the max log-SNR λ = log(α/σ) so α_t can't reach 0. Error collapses 185 → 0.87,
and the solver is *consistent* again (error shrinks with steps). Note: at the full 100-step
schedule DPM++ still NaNs at the exact-α=0 step (`t=99`), so it's only usable below ~50 steps.

**But is it actually *worse*? No — and "drift vs DDIM" can't answer that** (it measures agreement
with another approximate solver, not fidelity to the data). The right metric is **action MSE vs
held-out demonstrations** — the one the checkpoints were selected on (`val_ddim_mse`), reproduced
faithfully here (DDIM@10 = 0.00113 ≈ the topk value). On it, DDIM and fixed-DPM++ are
**statistically indistinguishable** on both models:

| sampler | FiLM MSE↓ | attention MSE↓ |
|---|--:|--:|
| DDIM @10 | 0.001126 | 0.000904 |
| DDIM @6 | 0.001091 | 0.000898 |
| DPM++ (fixed) @8 | 0.001147 | 0.000911 |
| DPM++ (fixed) @6 | 0.001156 | 0.000916 |

So fixed-DPM++ is **equivalent**, not worse. There's still **no reason to switch**: equal quality,
but it needs `lambda_min_clipped`, NaNs at the full schedule, and isn't the validated default — while
**DDIM@6 already matches DDIM@10** here. → **Just use DDIM at ~6 steps (knob 5) + compile.** Rank with
`scripts/eval_sampler_quality.py`; on-robot rollouts are the final word.

---

## Benchmark results (this repo, RTX 5090)

> Steady state = 1 fresh frame / camera (the 10 Hz case). Drift = action-chunk difference
> vs the fp32 / 10-step reference (same seed → same init noise). These are the **comprehensive
> sweep** (every knob, incl. camera-batch as its own row). Reproduce with
> `python benchmark_comprehensive.py -i <ckpt>`.

### FiLM `2_obs_baggie_CLIP_film` — 3 ViT passes/call, tiny FiLM UNet → **UNet-loop + overhead bound**

Every row is **fp32 baseline + exactly the knobs in its label** (each label is the full config, not a
delta from the row above — so `+ TF32 + fp16` has camera-batch *off*). Speedup is vs the fp32 baseline.

| config (10 DDIM steps) | mean ms | speedup | drift max / rms |
|---|--:|--:|--:|
| fp32 baseline (old default) | 65.99 | 1.00× | — |
| + TF32 | 36.65 | 1.80× | 4.3e-4 / 1.0e-4 |
| + TF32 + camera-batch | 34.37 | 1.92× | 4.3e-4 / 1.0e-4 *(free, bit-identical)* |
| + TF32 + fp16 | 35.97 | 1.83× | 9.9e-4 / 2.0e-4 |
| + TF32 + bf16 | 35.34 | 1.87× | 7.3e-3 / 1.4e-3 |
| + TF32 + bf16 + camera-batch | 32.60 | 2.02× | 6.9e-3 / 1.6e-3 |
| + TF32 + bf16 + camera-batch + compile(backbone) | 32.22 | 2.05× | 8.4e-3 / 1.7e-3 |
| + TF32 + bf16 + camera-batch + compile(backbone) + compile(UNet, `reduce-overhead`) | 28.32 | **2.33×** | 7.9e-3 / 1.7e-3 |

DDIM step sweep (on the last/full config): 10→28.9ms (2.3×) · 8→23.4 (2.8×) · **6→19.9 (3.3×)** ·
4→15.7 (4.2×) · 2→10.4 (6.4×).

→ **Recommendation:**
- RTC-safe: `--tf32 --camera-batch --amp fp16 --steps 6` ≈ 2.8× (backbone compile is marginal).
- Max speed (no RTC): `--amp fp16 --compile-unet --unet-compile-mode reduce-overhead --steps 6` ≈ **3.0–3.3×**.

<!-- RESULTS:ATTN -->
### Attention `8_obs_baggie_CLIP_attention_double_enc` — 6 ViT passes/call, heavy fp32 cross-attn UNet → **UNet-bound**

Same convention: every row = fp32 baseline + exactly the knobs in its label (not a delta from the row above).

| config (10 DDIM steps) | mean ms | speedup | drift max / rms |
|---|--:|--:|--:|
| fp32 baseline (old default) | 81.10 | 1.00× | — |
| + TF32 | 63.89 | 1.27× | 2.2e-4 / 4.1e-5 |
| + TF32 + camera-batch | 58.04 | 1.40× | 2.2e-4 / 4.1e-5 *(free, bit-identical; 6 ViT calls → 2)* |
| + TF32 + fp16 | 66.07 | 1.23× | 1.9e-4 / 4.0e-5 |
| + TF32 + bf16 | 65.13 | 1.25× | 7.5e-4 / 2.0e-4 |
| + TF32 + bf16 + camera-batch | 59.17 | 1.37× | 8.6e-4 / 1.9e-4 |
| + TF32 + bf16 + camera-batch + compile(backbone) | 58.93 | 1.38× | 1.0e-3 / 2.7e-4 |
| + TF32 + bf16 + camera-batch + compile(backbone) + compile(UNet, `reduce-overhead`) | 36.46 | **2.22×** | 1.0e-3 / 2.7e-4 |

DDIM step sweep (on the last/full config): 10→36.6ms (2.2×) · 8→31.0 (2.6×) · **6→25.5 (3.2×)** ·
4→19.7 (4.1×) · 2→14.7 (5.5×). (Camera-batch helps more here — 6 ViT calls collapse to 2.)

→ **Recommendation:** here the fp32 cross-attention UNet dominates, so **compiling the UNet
is the lever** (bf16 / backbone-compile barely move it).
- RTC-safe: `--compile-unet --steps 6` ≈ 2.5× (UNet compile mode defaults to `default`, which supports backward).
- Max speed (no RTC): `--compile-unet --unet-compile-mode reduce-overhead --steps 6` ≈ **3.0×**.

> Speedup ratios are within-run; the absolute baseline drifts a little run-to-run
> (sshfs/thermal), e.g. fp32 baseline measured 82–91 ms across runs.

**How to read it:** start from the fp32 baseline (a bare run = your old default); add `--tf32 --camera-batch`
(free); add `--amp fp16` if the drift is acceptable; then `--compile-unet` (`--unet-compile-mode reduce-overhead`
when not serving RTC). Finally pick a step count from the sweep.

---

## TensorRT (optional, advanced)

On this stack, **torch.compile already captures most of the achievable ViT speedup with zero
install risk and full RTC compatibility**, so try the knobs above first. Reach for TensorRT
only if the benchmark shows the backbone is still your bottleneck and you need more.

**Why it's the heavy option here:**
- Blackwell (sm_120) needs **TensorRT ≥ 10.7**, and torch-tensorrt must match torch 2.9. Your
  torch reports CUDA 12.9 (conda-provided) vs the public cu128 torch-tensorrt wheels — usually
  fine across 12.x minor versions, but it's the thing most likely to fight you.
- TensorRT engines are **inference-only** → they cannot accelerate the **RTC** UNet (it needs
  autograd). The backbone is fine (its output is detached before the RTC VJP), so the realistic
  TRT target is the **ViT backbone only**, with the diffusers loop staying in Python.

**Install (read it first — it's the one heavy step):**
```bash
bash scripts/install_tensorrt.sh
```

**Compile the backbone through Torch-TensorRT (dynamo backend, fp16):**
```python
import torch, torch_tensorrt
from diffusion_policy.common.inference_accel import enable_tf32
enable_tf32(True)

bb = policy.obs_encoder.backbone.eval().cuda().half()
example = torch.randn(1, 3, 192, 192, device="cuda", dtype=torch.half)  # crop = floor(224*0.9/16)*16 = 192
trt_bb = torch_tensorrt.compile(
    bb, ir="dynamo", inputs=[example],
    enabled_precisions={torch.half}, truncate_double=True,
)
policy.obs_encoder.backbone = trt_bb        # and policy.short_range_encoder.backbone for the dual encoder
```

Caveats: the batch dim varies (1 frame steady-state, more on cold start) — give
`torch_tensorrt.Input(min_shape/opt_shape/max_shape=...)` a dynamic batch range, or accept
one engine per batch size. Validate action drift against the fp32 baseline before trusting it.

---

## RTC interaction (important)

The RTC (real-time chunking / ΠGDM) path runs under `torch.enable_grad()` and backprops
through the UNet. Therefore:
- **Backbone** acceleration (TF32, bf16, compile, even TensorRT) — always safe; the encoder
  output is detached before the RTC guidance step.
- **UNet** `--compile-unet` with `reduce-overhead` (CUDA graphs) or a TensorRT UNet — **not
  compatible with RTC.** Use `--compile-mode default` (supports backward) or leave UNet compile
  off when serving RTC.
