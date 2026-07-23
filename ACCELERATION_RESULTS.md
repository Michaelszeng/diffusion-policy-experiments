# Inference acceleration — everything I tested, with results

Goal: make `policy_inference.py` faster for the two baggie checkpoints without retraining.
This is the full log — every method tried, whether it worked, the numbers, and *why*.
For the short "which knobs do I flip" version, see [INFERENCE_OPTIMIZATION.md](INFERENCE_OPTIMIZATION.md).

## Setup

- **GPU** RTX 5090 (Blackwell, sm_120) · **env** `runpod_remote_test` (torch 2.9.1, CUDA 12.9, diffusers 0.35.2)
- **Models** (both share a `vit_base_patch16_clip_224.openai` ViT-B/16 backbone, fp32 by default):

  | checkpoint | policy | n_obs | cameras | **ViT passes / call** | denoiser |
  |---|---|--:|--:|--:|---|
  | `2_obs_baggie_CLIP_film` | FiLM | 2 | 3 | **3** | small FiLM 1D-UNet |
  | `8_obs_baggie_CLIP_attention_double_enc` | attention dual-enc | 8 | 3 | **6** (long+short enc) | fp32 cross-attention 1D-UNet |

- **Measurement** steady state = 1 fresh frame/camera (the 10 Hz case), rest served from the feature cache.
  Latency = mean end-to-end `predict_action` (CUDA-event timed, 30 iters after warmup). **Drift** = action-chunk
  difference vs the fp32 / 10-step baseline, same seed → same init noise (max abs, and rms). Drift is a proxy —
  confirm on-robot — but it makes the quality cost of each knob visible.
- **Reproduce**: `python benchmark_comprehensive.py -i <ckpt>` and `python benchmark_inference.py -i <ckpt>`.

## TL;DR

- Your old default ran the ViT backbone **in fp32 with TF32 off** — the single biggest miss. Turning TF32 on is
  **1.8× (FiLM) / 1.3× (attention)** for free, with drift ~1e-4.
- Stacking the safe knobs (**TF32 + camera-batch + torch.compile + bf16**) reaches **2.3× / 2.2×** at 10 DDIM steps,
  with drift ~1e-3 (negligible). Dropping to **6 steps** → **~3.3× / ~3.2×**.
- **The bottleneck is the DDIM UNet loop, not the vision backbone** — for *both* models. Step-sweep
  decomposition of the accelerated config: per-DDIM-step ≈ 2.3 ms (FiLM) / 2.7 ms (attention); fixed
  cost (backbone + Python/PNG/transfer overhead) ≈ 5.7 ms / 9.3 ms. At 10 steps the UNet loop is ~80% /
  ~75% of the time; the ViT backbone is only ~2 ms (FiLM) / ~4 ms (attention). This is why **TensorRT on
  the backbone — 2.8× in isolation — is only 1.03× end-to-end** (Amdahl), and why compiling the UNet +
  cutting steps are the levers that actually move the needle.
- Things that did **not** help / not worth it: **TensorRT on the backbone** (1.03× e2e), bf16 over fp16
  (fp16 is faster-ish *and* more accurate here), `channels_last` (irrelevant for a ViT), backbone
  **quantization** (same Amdahl ceiling as TRT). **DPM-Solver++** is *equivalent* to DDIM on action-MSE once fixed
  (`lambda_min_clipped`), so there's no reason to switch — just use fewer DDIM steps (below).

---

## Full results

### FiLM `2_obs_baggie_CLIP_film` — UNet-loop + overhead bound (backbone ~2 ms)

Every row = fp32 baseline + exactly the knobs in its label (full config, not a delta from the row above).

| config (10 DDIM steps) | mean ms | speedup | drift max | drift rms | notes |
|---|--:|--:|--:|--:|---|
| fp32 baseline (old default) | 65.99 | 1.00× | — | — | TF32 off |
| + TF32 | 36.65 | 1.80× | 4.3e-4 | 1.0e-4 | **free** |
| + TF32 + camera-batch | 34.37 | 1.92× | 4.3e-4 | 1.0e-4 | free, numerically exact |
| + TF32 + bf16 | 35.34 | 1.87× | 7.3e-3 | 1.4e-3 | |
| + TF32 + fp16 | 35.97 | 1.83× | 9.9e-4 | **2.0e-4** | fp16 more accurate than bf16 |
| + TF32 + bf16 + camera-batch | 32.60 | 2.02× | 6.9e-3 | 1.6e-3 | |
| + TF32 + bf16 + camera-batch + compile(backbone) | 32.22 | 2.05× | 8.4e-3 | 1.7e-3 | small win (ViT launch-bound at B=3) |
| **+ TF32 + bf16 + camera-batch + compile(backbone) + compile(UNet, CUDA graphs)** | **28.32** | **2.33×** | 7.9e-3 | 1.7e-3 | full stack |

DDIM-step sweep on the last/full config: 10→28.9ms (2.3×) · 8→23.4 (2.8×) · **6→19.9 (3.3×)** · 4→15.7 (4.2×) · 2→10.4 (6.4×).

### Attention `8_obs_baggie_CLIP_attention_double_enc` — heavily UNet-bound (fp32 cross-attention)

Same convention: every row = fp32 baseline + exactly the knobs in its label (full config, not a delta).

| config (10 DDIM steps) | mean ms | speedup | drift max | drift rms | notes |
|---|--:|--:|--:|--:|---|
| fp32 baseline (old default) | 81.10 | 1.00× | — | — | TF32 off |
| + TF32 | 63.89 | 1.27× | 2.2e-4 | 4.1e-5 | **free** |
| + TF32 + camera-batch | 58.04 | 1.40× | 2.2e-4 | 4.1e-5 | free, exact — 6 ViT calls → 2 |
| + TF32 + bf16 | 65.13 | 1.25× | 7.5e-4 | 2.0e-4 | bf16 doesn't help (UNet-bound) |
| + TF32 + fp16 | 66.07 | 1.23× | 1.9e-4 | 4.0e-5 | |
| + TF32 + bf16 + camera-batch | 59.17 | 1.37× | 8.6e-4 | 1.9e-4 | |
| + TF32 + bf16 + camera-batch + compile(backbone) | 58.93 | 1.38× | 1.0e-3 | 2.7e-4 | backbone isn't the bottleneck here |
| **+ TF32 + bf16 + camera-batch + compile(backbone) + compile(UNet, CUDA graphs)** | **36.46** | **2.22×** | 1.0e-3 | 2.7e-4 | full stack; UNet compile is the lever |

DDIM-step sweep on the last/full config: 10→36.6ms (2.2×) · 8→31.0 (2.6×) · **6→25.5 (3.2×)** · 4→19.7 (4.1×) · 2→14.7 (5.5×).

---

## Per-method findings

### ✅ TF32 (tensor-core fp32 matmul) — biggest free win
Everything fp32 (the ViT always; the attention UNet always, since it self-pins to fp32) gets tensor-core matmuls
with ~10-bit mantissa. **1.80× / 1.27×**, drift ~1e-4 (negligible). Exposed as `--tf32` (off by default; opt in).

### ✅ Camera-batching — free, numerically exact, and bigger on the dual encoder
`encode_with_cache` ran one ViT call **per camera** (3 batch-1 calls; 6 for the dual encoder). Batching all cameras'
fresh frames into **one** batch-N ViT call (the ViT has no cross-batch ops) cut launch overhead: **1.80→1.92× (FiLM)**,
**1.27→1.40× (attention)**. Drift is *byte-identical* to the unbatched path — confirmed numerically equivalent.
Toggle: `policy.obs_encoder.batch_rgb_inference = True` (and `.short_range_encoder`). Implemented in
[timm_obs_encoder.py](diffusion_policy/model/vision/timm_obs_encoder.py).

### ◐ bf16 / fp16 autocast on the backbone — modest, and fp16 wins
At batch 1–3 the ViT is **launch-bound, not bandwidth-bound**, so half-precision adds little over TF32 (FiLM
1.80→1.87×; attention *slightly worse*). And **fp16 is more accurate than bf16 here** (drift 2e-4 vs 1.4e-3) — fp16
has 10 mantissa bits vs bf16's 7, and CLIP activations sit in fp16's range. The attention UNet ignores AMP (fp32-pinned
by design — it NaNs in bf16). Verdict: optional; if used, prefer **fp16**.

### ✅ torch.compile the UNet (CUDA graphs) — the big lever for the attention model
Compiling the denoiser (run `steps`× per inference) with `reduce-overhead` (CUDA graphs) kills per-step launch
overhead. **Attention: 1.40→2.22×.** FiLM: 1.92→2.33×. The UNet output is consumed by the scheduler and never escapes,
so CUDA graphs are safe — **except under RTC** (it backprops through the UNet; use `default` mode there).

### ◐ torch.compile the backbone — small
FiLM 2.02→2.05×, attention ~flat. At batch 1–3 the ViT is launch-bound and TF32/fp16 already captured most of it.
**Gotcha (handled):** CUDA-graph modes corrupt the backbone's cached/returned features, so the backbone is forced to a
graph-free mode (`max-autotune-no-cudagraphs`). Also, Triton needs a Blackwell-aware `ptxas` — auto-pointed at the
env's CUDA-12.9 ptxas (else it grabs the system 12.0 one and dies with `sm_120a` errors).

### ✅ Fewer DDIM steps — linear, the multiplier on everything
Latency is ~linear in step count; DDIM is a *consistent* solver so error grows smoothly as you coarsen
(rms 1.7e-3 @10 → 1.7e-2 @6 → 4.7e-2 @4 for FiLM). 6 steps looks like a sweet spot on drift; **confirm on-robot.**

### ◑ DPM-Solver++ — fixable, and *equivalent* to DDIM on the data-grounded metric (no reason to switch)
This one took three passes to get right; the methodology matters more than the verdict.

**Pass 1 (wrong metric).** A naive `DPMSolverMultistepScheduler` swap looked like it "diverged" — rms ~185 vs a
**DDIM@100 reference**, flat across step counts (185→244→264→274 @50/20/10/5).

**Pass 2 (find the bug).** A *flat* error that doesn't shrink with steps can't be discretization error (that → 0 as
steps → ∞) — it means convergence to the *wrong fixed point*: a config bug. Root cause: `squaredcos_cap_v2` has
**ᾱ₉₉ = 0** (zero terminal SNR); DPM++ works in x̂₀ space and starts at t=99, where x̂₀ = (x−σ_t·ε)/α_t divides by
α_t=√ᾱ=0 and explodes. (Verified the scheduler's `alphas_cumprod` matches DDPM exactly and its ε→x̂₀ conversion is
correct to 4.7e-7 — so it's the α→0 division, not a parameterization slip.) Fix = `lambda_min_clipped=-5.1` (the
standard SD knob). **Self-consistency test** (each sampler vs *its own* high-step limit) then shows the fix worked —
fixed-DPM++ converges, and *faster* than DDIM (the expected higher-order behavior):

| vs own limit | @4 | @8 | @16 | @32 |
|---|--:|--:|--:|--:|
| DDIM (limit @100) | 0.15 | 0.10 | 0.085 | 0.057 |
| **fixed DPM++** (limit @48) | 1.06 | 0.054 | 0.018 | **0.0037** |
| vanilla DPM++ (broken) | 94 | 85 | 68 | 33 |

**Pass 3 (right metric).** Convergence ≠ quality. Since fixed-DPM++ converges to a *different* point than DDIM,
"drift vs DDIM" can't rank them — only **action MSE vs held-out demonstrations** can (the metric the checkpoints were
*selected* on; faithfully reproduced — DDIM@10 below = 0.00113 ≈ the topk `val_ddim_mse`). On that metric the samplers
are **statistically indistinguishable** on *both* models:

| sampler | FiLM MSE↓ | attention MSE↓ |
|---|--:|--:|
| DDIM @10 | 0.001126 | 0.000904 |
| DDIM @6 | 0.001091 | 0.000898 |
| DPM++ (fixed) @8 | 0.001147 | 0.000911 |
| DPM++ (fixed) @6 | 0.001156 | 0.000916 |

(256 held-out windows from the 3 val episodes; spread ~2–6%, within noise — note DDIM@6 even edges DDIM@10.)

**Is DPM++ "more efficient" (fewer steps)?** As an integrator, yes — mid-range it reaches its *own* converged answer
in ~half DDIM's steps (self-consistency vs own limit: DPM++ 0.018 @16 / 0.0037 @32 vs DDIM 0.085 / 0.057). But that
advantage is unreachable here, and at the **low-step regime that actually buys speed it's the *opposite*** — DPM++ is
worse at every step count and collapses at 2–3 (FiLM action MSE):

| steps | DDIM MSE↓ | DPM++ MSE↓ |
|--:|--:|--:|
| 6 | 0.001091 | 0.001156 |
| 4 | 0.001071 | 0.001206 |
| 3 | **0.001055** | 0.001486 |
| 2 | 0.001349 | **0.008275** |

Why: a 2nd-order multistep solver needs warmup steps and is hurt most by the zero-SNR first step, while on a 100-step
schedule DDIM is *already converged by ~4 steps* (no gap to recover) and `clip_sample` keeps it robust. So **DDIM is the
more efficient sampler for this model.**

**Verdict:** fixed-DPM++ is **equivalent to DDIM at moderate steps and worse at low steps — never better here**. No reason
to switch (DPM++ also needs `lambda_min_clipped`, NaNs at the full schedule, isn't the validated default). The real lever
is the opposite of a fancier sampler: **DDIM holds quality down to ~3–4 steps** (0.00106 @3 ≈ 0.00113 @10, within noise)
where DPM++ can't — so push DDIM low for more speed. (Offline MSE ranks; very-low-step chunks can be jerkier on-robot
even at equal MSE — confirm in rollout before shipping 3 steps.) Repro: `python scripts/eval_sampler_quality.py -i <ckpt>`.

### ✗ channels_last memory format — N/A for a ViT
`channels_last` accelerates **conv-dominated** CNNs. A ViT-B is one conv stem then all attention/MLP matmuls, which are
layout-insensitive. No meaningful effect; not worth wiring the NHWC plumbing. (SDPA/flash attention is already used by
timm's ViT and the UNet's `MultiheadAttention`, and `torch.compile` fuses further — no separate knob needed.)

### ◐ TensorRT (torch-tensorrt, fp16) on the ViT backbone — fast in isolation, ~nothing end-to-end
Installed torch-tensorrt 2.9.0 + tensorrt 10.13 in an isolated env clone (the naive `pip install`
first pulled torch-tensorrt 2.12 / tensorrt 11 / CUDA-13 wheels — all incompatible with torch 2.9;
the matched set is `torch-tensorrt>=2.9,<2.10`). Built a dynamo fp16 engine for the backbone at the
camera-batched steady-state batch (3):

| backbone (batch 3, 192px) | latency | vs fp32 |
|---|--:|--:|
| fp32 + TF32 | 1.89 ms | 1.00× |
| fp16 eager | 1.20 ms | 1.57× |
| fp16 torch.compile | 0.87 ms | 2.17× |
| **fp16 TensorRT** | **0.67 ms** | **2.80×** |

TensorRT is genuinely the fastest backbone (2.8×, drift vs fp32 max 4.7e-2 / rms 4.3e-3).
**But swapped into the policy it's only 1.03× end-to-end** (34.5 → 33.4 ms, action drift 5e-5):

```
end-to-end (camera-batched, 10 steps):  eager fp32 backbone 34.50 ms → TRT fp16 backbone 33.35 ms (1.03×)
```

This is the headline lesson (Amdahl): the backbone is only ~2 ms of ~34 ms, so making it 2.8×
faster saves ~1 ms. **The backbone is not the bottleneck — the DDIM UNet loop is.** Not worth the
TensorRT install/maintenance for this workload; `torch.compile` fp16 (2.17× isolated, zero install)
already captures most of the backbone win, and the real gains are on the UNet + step count.

### ✗ Quantization (torchao int8 / fp8) on the ViT backbone — slower than bf16, less accurate
Tested int8 and fp8 on the backbone (isolated, batch 3). Two practical snags first: torchao's CUDA
kernels are **not built for torch 2.9.1** (needs ≥2.11, ao#2919) — so it falls back to inductor, which
*does* generate int8 matmuls (`_int_mm`) but with quant/dequant overhead — and the functional API is
deprecated in favour of `Int8WeightOnlyConfig`-style configs.

| backbone (batch 3, 192px) | latency | vs fp32 | drift rms |
|---|--:|--:|--:|
| fp32 + TF32 | 1.79 ms | 1.00× | — |
| **bf16 + compile** | **1.00 ms** | **1.79×** | 1.2e-2 |
| int8 weight-only + compile | 1.26 ms | 1.42× | 2.9e-2 |
| int8 dyn-act int8-weight + compile | 1.44 ms | 1.24× | 7.1e-2 |
| fp8 dyn-act fp8-weight + compile | 1.68 ms | 1.07× | 1.1e-1 |

**Every quantization recipe is slower than plain bf16+compile and less accurate.** At batch 3 the ViT is
launch/overhead-bound, not compute-bound, so the per-layer quantize→int8-matmul→dequantize overhead
exceeds the matmul savings; fp8 adds the most overhead and the worst accuracy (drift 1.1e-1). Quantization
pays off on *large* compute-bound matmuls (big LLM/SD layers), not a batch-3 ViT-B. And like TRT it targets
the backbone → Amdahl-capped at ~1.0× end-to-end regardless. **Not worth it here.** (Quantizing the actual
bottleneck — the UNet — is a non-starter: the attention UNet is fp32-pinned because it NaNs in low
precision, and the FiLM UNet's Conv1ds are too small to benefit.)

---

## Recommendations

All acceleration is **OFF by default** (a bare `policy_inference.py` run == original behavior); opt in per flag.

| | RTC-safe (backprops through UNet) | Max speed (no RTC) |
|---|---|---|
| **FiLM** | `--tf32 --camera-batch --amp fp16 --steps 6` | add `--compile-unet --unet-compile-mode reduce-overhead` → **~3.3×** |
| **Attention** | `--tf32 --camera-batch --compile-unet --steps 6` (UNet compile `default` supports backward) | set `--unet-compile-mode reduce-overhead` → **~3.2×** |

`--tf32` + `--camera-batch` are the safe, free wins — turn them on always. Pick the DDIM step count from the
sweep by on-robot quality. See INFERENCE_OPTIMIZATION.md for the full flag reference and the priority list.
