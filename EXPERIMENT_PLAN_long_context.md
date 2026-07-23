# Long-Context Reactivity Experiment — Runbook (baggie task)

## 0. Hypothesis

Framing:
- **Short context + short action horizon (AH):** reactive, but limited "context" → loops, oscillation, can't learn to pause.
- **Short context + long AH:** fixes loops/pausing, but poor reactivity.
- **Long context + short AH (the claim):** should get reactivity *and* the loop/pause fix.

**Primary hypothesis (H1):** success rate of `long-context @ AH2` ≥ the *best* `short-context` condition (over AH ∈ {2,4,8}).

**Secondary pattern of interest (H2):** short-context success degrades as AH shrinks; long-context stays flat or improves as AH shrinks. i.e. a **context × AH interaction**.

> Caveat locked in by the metric choice: we record **binary success only**. So the loop/pause *mechanism* is **inferred** from the success pattern, not measured. See §6 for a zero-cost optional annotation that recovers most of the mechanism story.

---

## 1. The 2×3 grid

| # | Context | Checkpoint / arch | AH | transition-latency-ms | viz-log-dir |
|---|---------|-------------------|----|-----------------------|-------------|
| 1 | short | `2_obs_baggie_DINOv3_film_baggie_v4` (FiLM, n_obs=2) | 2 | 150 | `model_eval_final/baggie_short2obs_film_ah2` |
| 2 | short | ″ | 4 | 200 | `model_eval_final/baggie_short2obs_film_ah4` |
| 3 | short | ″ | 8 | 200 | `model_eval_final/baggie_short2obs_film_ah8` |
| 4 | long | `8_obs_baggie_CLIP_attention_double_enc_v4` (DINOv3 attn double-enc, n_obs=8, short_range=2) | 2 | 150 | `model_eval_final/baggie_long8obs_attn_ah2` |
| 5 | long | ″ | 4 | 200 | `model_eval_final/baggie_long8obs_attn_ah4` |
| 6 | long | ″ | 8 | 200 | `model_eval_final/baggie_long8obs_attn_ah8` |

6 conditions × ~20 trials = **~120 robot trials**.

The AH→latency map (**150 / 200 / 200** for AH 2/4/8) is applied **identically to both context arms** — latency is a function of AH only, so context stays the clean variable. Confirm/adjust these three numbers once; then never change them mid-experiment.

### ⚠️ Confound (read before drawing conclusions)
> **Naming note:** the `..._CLIP_...` in the long checkpoint's folder name is a **misnomer** — verified from its dumped config, it uses the DINOv3 backbone (`vit_base_patch16_dinov3.lvd1689m`), same as the short model. Keep the on-disk path as-is (it's the real filename), but don't let the name mislead the writeup.

Backbone is **matched** (both DINOv3 `vit_base_patch16`, same baggie-v4 data). So the two checkpoints differ in only **two** ways: context length (n_obs 2 vs 8) and architecture (FiLM single-encoder vs attention double-encoder). Therefore:
- A win for the long model supports **"our long-context recipe ≥ the short baseline"** — clean at the system level.
- It still does **not** fully isolate **"context length alone causes it,"** because conditioning also changes (FiLM → cross-attention). But note the arch difference is *interpretable here*: the long model's short-range encoder uses `short_range_obs_horizon = 2`, which is exactly the short model's entire window (n_obs=2). So the long model effectively **contains the short model's reactive 2-frame window as one branch** and adds 8 frames of long-range cached context on top. That makes "the extra context is what helps" the natural reading of a long-model win — modulo FiLM-vs-attention conditioning.
- For a fully airtight context-only ablation you'd still want a matched-arch pair (e.g. a DINOv3-FiLM 8_obs, or a DINOv3-attn 2_obs). Not required for this run; just scope the claim accordingly.

---

## 2. Held constant across ALL 6 conditions
- DDIM `--steps 6`
- `--frequency 10` (10 Hz control)
- Acceleration flags: `--compile-unet --tf32 --camera-batch --compile-backbone`
- `--cached` on the eval (speed optimization; per-frame features are deterministic so it does not change outputs — but hold it constant regardless)
- Default `gripper_multiplier` (1.5)
- **Same physical initial-condition set** (see §4) — this is the biggest lever for statistical power on a binary metric.
- A single fixed, written **success criterion** for the baggie task (fill in below, decide once, don't renegotiate mid-run):
  - _Success = ______________________ (e.g. "baggie lifted clear of table and placed in target zone within 60 s, no human intervention")._
  - _Timeout = ____ s. A trial hitting timeout = failure._
  - _Any human contact / e-stop / it knocks the object off the workspace = failure._

---

## 3. Inference servers (start one at a time)

You only restart the server when switching **context**. Switching **AH** is just a different `arx_scheduled_eval` invocation against the *same* running server — free. So the expensive axis is context (2 servers total, plus recompile), the cheap axis is AH.

**SHORT-context server:**
```bash
python3 policy_inference.py \
  -i /home/mnt/ajaybati/Documents/diffusion-policy-experiments/data/outputs/mundane/2_obs_baggie_DINOv3_film_baggie_v4/checkpoints/latest.ckpt \
  --compile-unet --tf32 --camera-batch --compile-backbone --steps 6
```

**LONG-context server:**
```bash
python3 policy_inference.py \
  -i /home/mnt/shared_models/baggie/8_obs_baggie_CLIP_attention_double_enc_v4/checkpoints/epoch\=050-val_loss\=0.0340-val_ddim_mse\=0.001121.ckpt \
  --compile-unet --tf32 --camera-batch --compile-backbone --steps 6
```

> On startup the server dumps a `.yaml` next to the exact `.ckpt` it loaded. That dumped yaml is what `--training-config` should point to (§4), so the eval's `n_obs_steps`/shape_meta match the loaded weights exactly.

---

## 4. Eval commands (6)

Point `--training-config` at the yaml the server dumped for the loaded checkpoint. Keep the server for a context up while you sweep all 3 AHs for it.

**SHORT context** (server from §3 running):
```bash
# AH2
python arx_scheduled_eval.py \
  --training-config /home/mnt/ajaybati/Documents/diffusion-policy-experiments/data/outputs/mundane/2_obs_baggie_DINOv3_film_baggie_v4/checkpoints/latest.yaml \
  --cached --frequency 10 --action-horizon 2 --transition-latency-ms 150 \
  --viz-log-dir model_eval_final/baggie_short2obs_film_ah2

# AH4
python arx_scheduled_eval.py \
  --training-config /home/mnt/ajaybati/Documents/diffusion-policy-experiments/data/outputs/mundane/2_obs_baggie_DINOv3_film_baggie_v4/checkpoints/latest.yaml \
  --cached --frequency 10 --action-horizon 4 --transition-latency-ms 200 \
  --viz-log-dir model_eval_final/baggie_short2obs_film_ah4

# AH8
python arx_scheduled_eval.py \
  --training-config /home/mnt/ajaybati/Documents/diffusion-policy-experiments/data/outputs/mundane/2_obs_baggie_DINOv3_film_baggie_v4/checkpoints/latest.yaml \
  --cached --frequency 10 --action-horizon 8 --transition-latency-ms 200 \
  --viz-log-dir model_eval_final/baggie_short2obs_film_ah8
```

**LONG context** (restart server with the 8_obs checkpoint first):
```bash
# AH2
python arx_scheduled_eval.py \
  --training-config /home/mnt/shared_models/baggie/8_obs_baggie_CLIP_attention_double_enc_v4/checkpoints/epoch\=050-val_loss\=0.0340-val_ddim_mse\=0.001121.yaml \
  --cached --frequency 10 --action-horizon 2 --transition-latency-ms 150 \
  --viz-log-dir model_eval_final/baggie_long8obs_attn_ah2

# AH4
python arx_scheduled_eval.py \
  --training-config /home/mnt/shared_models/baggie/8_obs_baggie_CLIP_attention_double_enc_v4/checkpoints/epoch\=050-val_loss\=0.0340-val_ddim_mse\=0.001121.yaml \
  --cached --frequency 10 --action-horizon 4 --transition-latency-ms 200 \
  --viz-log-dir model_eval_final/baggie_long8obs_attn_ah4

# AH8
python arx_scheduled_eval.py \
  --training-config /home/mnt/shared_models/baggie/8_obs_baggie_CLIP_attention_double_enc_v4/checkpoints/epoch\=050-val_loss\=0.0340-val_ddim_mse\=0.001121.yaml \
  --cached --frequency 10 --action-horizon 8 --transition-latency-ms 200 \
  --viz-log-dir model_eval_final/baggie_long8obs_attn_ah8
```

---

## 5. Trial protocol (ordering matters for real-robot validity)

**Matched initial conditions (do this).** Predefine **20 reference start layouts** for the baggie (tape marks / a jig / reference photos). Number them 1–20. Run the **same 20** in **every** condition. This turns the study into a paired design: each layout gets 6 outcomes, so §7 can compare conditions *within* the same start state and cancel most scene-to-scene variance.

**Counterbalance against drift.** The robot/scene drifts over hours (lighting, wear, calibration). Don't run all 20 of a condition back-to-back at one time of day, or "condition" gets confounded with "time." Because switching AH is free but switching context needs a server restart, use:

- **Round structure per context block:** with a context's server up, cycle AH in a **randomized order** (e.g. shuffle {2,4,8}) doing **mini-blocks of ~5 trials** each, until you reach 20 per AH. Re-shuffle AH order between mini-blocks.
- **Split each context into two time-separated blocks** and **counterbalance which context goes first:**
  - Session A: SHORT (10 trials/AH) → LONG (10 trials/AH)
  - Session B (later / next day): LONG (10 trials/AH) → SHORT (10 trials/AH)
  - Totals: 20 trials per condition, each context sampled at two different times, order counterbalanced. Only 4 server starts total.
- Within a mini-block, run the fixed layouts in the **same layout order** across conditions so paired comparisons line up by layout id.

**Per trial:** reset to layout _i_ → start the trial → run until success criterion met or timeout/failure → record (§6) → reset.

---

## 6. What to record (tally)

**Primary (required): binary success**, one row per trial, keyed so it joins back to condition + layout.

CSV template — `results_long_context.csv`:
```
context,arch,action_horizon,transition_latency_ms,layout_id,trial_idx,success,failure_mode,notes
short,film,2,150,1,1,1,,
short,film,2,150,2,2,0,loop,"circled over baggie, never closed"
long,attn,2,150,1,1,1,,
...
```
- `success` ∈ {0,1}.
- `layout_id` ∈ 1..20 (the matched start state) — required for the paired analysis.

**Optional but strongly recommended — one word on *failures only* (`failure_mode`).** Costs ~5 s and is the *only* thing that lets a success-rate table speak to your actual hypothesis (was the short model failing from loops/no-pause, and did long context remove those specific failures?). Use a tiny fixed vocabulary so it tallies:
- `loop` — repeated/oscillating motion, never progressing
- `no_pause` — failed to stop/hold when it should have (barreled through)
- `reactivity` — too slow to react to a slip/perturbation/moved object
- `grasp` — grasp attempt failed (mechanical), not a policy-logic failure
- `drift` — hardware/calibration/lighting artifact, not the policy
- `other`

Tag `grasp`/`drift` so you can exclude non-policy failures from a sensitivity analysis.

---

## 7. Analysis

Per condition, compute **success rate + Wilson 95% CI** (Wilson, not normal — n=20 is small):

| Condition | n | successes | rate | 95% CI |
|-----------|---|-----------|------|--------|
| short AH2 |   |           |      |        |
| short AH4 |   |           |      |        |
| short AH8 |   |           |      |        |
| long AH2  |   |           |      |        |
| long AH4  |   |           |      |        |
| long AH8  |   |           |      |        |

**Primary test (H1):** `long AH2` vs the best short condition. Because layouts are matched, use **McNemar's test** on the paired per-layout outcomes (compares the two conditions on the same 20 start states) rather than an unpaired 2-proportion test — much more power.

**Pooled test (H2, interaction) — pools all 120 trials for power:** logistic regression
`success ~ context * action_horizon` (treat AH as categorical; add `(1|layout_id)` random intercept if you can, since layouts repeat). The **context×AH interaction** term is the statistical statement of "long context helps *more* at short AH."

**Plot:** success rate vs AH (x = 2,4,8), two lines (short, long), Wilson CI bars. The claim looks like: long line flat/high across AH; short line rising with AH but staying under long@AH2.

**Failure-mode readout (if you logged it):** stacked bar of failure_mode per condition. The hypothesis predicts `loop`/`no_pause` dominate short-context-low-AH and largely vanish in long-context.

**Power caveat:** at n=20/condition a single pairwise contrast only resolves large gaps (≈25–30+ percentage points) at p<0.05. The pooled logistic regression is where the real power is — lead with it. If early results are borderline, the cheapest fix is more trials on the two decisive conditions (`long AH2`, best short), not spreading thin across all six.

---

## 8. Execution checklist
- [ ] Fill in success criterion + timeout (§2)
- [ ] Mark & number 20 reference layouts (§5)
- [ ] Confirm AH→latency map: 150 / 200 / 200 (§1)
- [ ] Session A: SHORT server → sweep AH (10/AH); restart → LONG server → sweep AH (10/AH)
- [ ] Session B (later): LONG first → SHORT, remaining 10/AH each
- [ ] Log every trial to `results_long_context.csv` with `layout_id`
- [ ] Run §7 analysis; make the success-vs-AH plot
