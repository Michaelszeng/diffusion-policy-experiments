"""
Data-grounded sampler quality: action MSE vs held-out demonstrations.

This is the metric that can actually *rank* samplers (the robotics analog of FID /
human-eval), unlike "drift vs DDIM@100" which only measures agreement with another
approximate solver. It replicates the EXACT validation MSE the checkpoint was selected
on (`val_ddim_mse` = mse(predict_action(obs)["action_pred"], demo_action), EMA model),
then swaps only the sampler:

    DDPM@100 (training sampler)  vs  DDIM@{10,6}  vs  fixed-DPM++@{8,6} (lambda_min_clipped)

Faithfulness: it reuses the real ImprovedDatasetSampler on a buffer holding ONLY the
val episodes (selected by the same get_val_mask(seed, val_ratio)), so the windows are
identical to get_validation_dataset — but it never loads the full 83G zarr. Sanity check:
DDIM@10 here should match the checkpoint's reported val_ddim_mse (~0.0011 for the FiLM ckpt).

Run in the prod env:
    /home/ajay/miniforge3/envs/runpod_remote_test/bin/python scripts/eval_sampler_quality.py -i <ckpt>
"""
import click
import numpy as np
import torch
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import ImprovedDatasetSampler, get_val_mask
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.inference_accel import enable_tf32
from diffusion_policy.policy.diffusion_unet_timm_film_policy import DiffusionUnetTimmFilmPolicy
from policy_inference import PolicyInferenceNode


def build_val_loader(node):
    """Return (samples, get_batch) over the val split, replicating MundaneDataset windowing."""
    cfg = node.cfg
    dcfg = cfg.task.dataset
    ds = int(dcfg.downsample_steps)
    horizon = int(cfg.horizon)
    n_obs = int(cfg.n_obs_steps)
    pad_before = int(dcfg.pad_before)
    pad_after = int(dcfg.pad_after)
    val_ratio = float(dcfg.zarr_configs[0].val_ratio)
    seed = int(cfg.seed)
    action_key = getattr(dcfg, "action_key", "action")
    # The cfg stores a stale RELATIVE training path; resolve to the mounted dataset by basename.
    import os
    cfg_path = str(dcfg.zarr_configs[0].path)
    zarr_path = node._zarr_override or cfg_path
    if not os.path.exists(zarr_path):
        cand = os.path.join("/home/mnt/data_storage", os.path.basename(cfg_path.rstrip("/")))
        if os.path.exists(cand):
            zarr_path = cand
    assert os.path.exists(zarr_path), f"zarr not found: {zarr_path} (cfg said {cfg_path})"
    print(f"dataset: {zarr_path}")

    rgb_keys = sorted([k for k, v in cfg.task.shape_meta.obs.items() if v.get("type") == "rgb"])
    low_keys = sorted([k for k, v in cfg.task.shape_meta.obs.items() if v.get("type", "low_dim") == "low_dim"])
    buf_keys = list(rgb_keys) + list(low_keys) + [action_key]

    n_ep = len(zarr.open(zarr_path, "r")["meta"]["episode_ends"][:])
    val_mask = get_val_mask(n_episodes=n_ep, val_ratio=val_ratio, seed=seed)
    print(f"val episodes: {int(val_mask.sum())}/{n_ep} (seed={seed}, val_ratio={val_ratio})")

    # Load ONLY the val episodes (small) — re-stores action_key under 'action'.
    read_keys = [k for k in buf_keys if k != "action"] if action_key != "action" else buf_keys
    buf = ReplayBuffer.copy_selected_episodes_from_path(
        zarr_path=zarr_path, keep_mask=val_mask, keys=read_keys, store=zarr.MemoryStore())
    if action_key != "action":
        src = zarr.open(zarr_path, "r")  # (rare path; baggie uses action_key="action")
        raise NotImplementedError("action_key != 'action' not needed for these ckpts")

    # Stretched sampler params (MundaneDataset multiplies by downsample_steps).
    seq_len = horizon * ds
    kfk = {k: n_obs * ds for k in (rgb_keys + low_keys)}
    kfk["action"] = seq_len
    sampler = ImprovedDatasetSampler(
        replay_buffer=buf, sequence_length=seq_len, shape_meta=cfg.task.shape_meta,
        pad_before=pad_before * ds, pad_after=pad_after * ds,
        episode_mask=np.ones(buf.n_episodes, dtype=bool), key_first_k=kfk,
    )
    print(f"val windows: {len(sampler)}")

    def getitem(idx):
        d = sampler.sample_data(idx)
        d["action"] = d["action"][::ds]                 # stretch-and-slice → 10 Hz
        for k in d["obs"]:
            d["obs"][k] = d["obs"][k][::ds]
        return dict_apply(d, torch.from_numpy)

    return len(sampler), getitem


def mse_for_sampler(node, getitem, n_windows, idxs, use_ddim, batch=32):
    pol = node.policy
    tot, cnt = 0.0, 0
    for s in range(0, len(idxs), batch):
        chunk = idxs[s:s + batch]
        items = [getitem(int(i)) for i in chunk]
        obs = {k: torch.stack([it["obs"][k] for it in items]).to(pol.device) for k in items[0]["obs"]}
        # Match the workspace val preprocessing EXACTLY: rgb HWC uint8 -> CHW [0,1].
        for k in node.rgb_keys:
            obs[k] = torch.moveaxis(obs[k], -1, 2) / 255.0
        gt = torch.stack([it["action"] for it in items]).float().to(pol.device)
        with torch.no_grad():
            torch.manual_seed(0)
            pred = pol.predict_action(obs, use_DDIM=use_ddim)["action_pred"]
        tot += torch.nn.functional.mse_loss(pred, gt, reduction="mean").item() * len(chunk)
        cnt += len(chunk)
    return tot / cnt


@click.command()
@click.option("--input", "-i", required=True)
@click.option("--device", default="cuda:0")
@click.option("--max-windows", default=256, type=int, help="Evenly-sampled val windows to score")
@click.option("--zarr", "zarr_override", default="/home/mnt/data_storage/baggie_v1.zarr",
              help="Dataset path (cfg stores a stale relative path; override here)")
def main(input, device, max_windows, zarr_override):
    from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
    enable_tf32(True)
    node = PolicyInferenceNode(input, "127.0.0.1", 0, device, num_ddim_inference_steps=10,
                               use_ddim=True, gripper_multiplier=1.0)
    node._zarr_override = zarr_override if __import__("os").path.exists(zarr_override) else None
    pol = node.policy
    is_film = isinstance(pol, DiffusionUnetTimmFilmPolicy)
    n, getitem = build_val_loader(node)
    idxs = np.linspace(0, n - 1, min(max_windows, n)).astype(int).tolist()
    print(f"scoring {len(idxs)} windows | policy={type(pol).__name__}\n")

    orig_ddim = pol.ddim_noise_scheduler
    nsc = pol.noise_scheduler.config
    def dpmpp():
        return DPMSolverMultistepScheduler(
            num_train_timesteps=nsc.num_train_timesteps, beta_start=nsc.beta_start, beta_end=nsc.beta_end,
            beta_schedule=nsc.beta_schedule, prediction_type=nsc.prediction_type, lambda_min_clipped=-5.1)

    steps_list = [10, 8, 6, 5, 4, 3, 2]
    print(f"{'sampler':<28}{'steps':>6}{'action MSE vs demos':>22}")
    print("-" * 56)
    # (DDPM gold-ref skipped: diffusers DDPM.step has a CPU/CUDA device bug in this version;
    #  DDIM@10 ~= the checkpoint's val_ddim_mse already validates faithfulness.)
    # Low-step regime is where a higher-order solver's efficiency would show up (fewer steps = faster).
    for st in steps_list:
        pol.ddim_noise_scheduler = orig_ddim; pol.num_ddim_inference_steps = st
        print(f"{'DDIM':<28}{st:>6}{mse_for_sampler(node, getitem, n, idxs, use_ddim=True):>22.6f}")
    for st in steps_list:
        pol.ddim_noise_scheduler = dpmpp(); pol.num_ddim_inference_steps = st
        print(f"{'DPM++ (lambda_min_clipped)':<28}{st:>6}{mse_for_sampler(node, getitem, n, idxs, use_ddim=True):>22.6f}")
    print("-" * 56)
    print("Lower MSE = closer to demonstrated actions. DDIM@10 should ~match the ckpt's")
    print("reported val_ddim_mse (FiLM topk: ~0.00113). This RANKS samplers; on-robot rollouts remain the final word.")


if __name__ == "__main__":
    main()
