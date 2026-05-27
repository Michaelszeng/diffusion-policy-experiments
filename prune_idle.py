"""
Prune idle frames from zarr replay buffer datasets.

A frame at index i is classified as "idle" if the pusher states
s_i, s_{i+1}, ..., s_{i+obs_horizon} all fit within a hypersphere of radius
`idle_tolerance` — meaning the actions taken at frames i through i+obs_horizon-1
resulted in no meaningful motion (whether because no motion was commanded, or
because friction / contact forces resisted the command). Only those first
obs_horizon frames are pruned; s_{i+obs_horizon} serves as the witness state and
is handled by the next overlapping window. Episodes that are entirely idle are
removed.

`idle_tolerance` is in the units of the state vector. For the planar-push task
the state is [x, y, θ] in metres/radians, so 0.002 m ≈ 2 mm is a reasonable
starting point.

Expects the standard ReplayBuffer zarr layout (data/, meta/). For datasets with
non-standard layouts (e.g. FurnitureBench), first translate them using the
appropriate translation script (e.g. translate_furniture_bench.py).

Usage:
  python prune_idle.py data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large.zarr
  python prune_idle.py data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large.zarr --output data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large_pruned.zarr
  python prune_idle.py data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large.zarr data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large_pruned.zarr --output data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large_pruned_merged.zarr
  python prune_idle.py data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large.zarr --obs-horizon 3 --idle-tolerance 0.002
"""

import os
import sys
import argparse
import pathlib
import time
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from diffusion_policy.common.replay_buffer import ReplayBuffer


# ============================================================================
# Idle detection
# ============================================================================

def fits_within_radius(P, r):
    """
    Check if a set of points fits within a hypersphere of radius r (any center).

    Used to test whether a window of pusher states is "idle": if all states in
    the window cluster within a ball of radius r, the robot did not move
    meaningfully during that interval regardless of what was commanded.
    """
    P = np.asarray(P, dtype=np.float64)
    n = P.shape[0]

    if n <= 1:
        return True

    r2 = r ** 2
    max_pair_dist2 = (2 * r) ** 2

    # Compute all-pair squared distances
    diff = P[:, None, :] - P[None, :, :]
    dist2 = np.einsum("...i,...i->...", diff, diff)

    # 1. Necessary condition: diameter <= 2r
    if np.any(dist2 > max_pair_dist2):
        return False

    # 2. Sufficient condition: fits in ball around centroid?
    centroid = np.mean(P, axis=0)
    if np.all(np.sum((P - centroid) ** 2, axis=-1) <= r2):
        return True

    # 3. Sufficient condition: fits in ball around midpoint of widest pair?
    flat_idx = np.argmax(dist2)
    i, j = np.unravel_index(flat_idx, dist2.shape)
    midpoint = (P[i] + P[j]) * 0.5
    if np.all(np.sum((P - midpoint) ** 2, axis=-1) <= r2):
        return True

    return False


def prune_episode(
    episode_data: Dict[str, np.ndarray],
    obs_horizon: int,
    idle_tolerance: float,
    state_key: str = "state",
) -> Optional[Dict[str, np.ndarray]]:
    """
    Prune idle frames from a single episode.

    For each window of obs_horizon+1 consecutive pusher states, if all states fit
    within a hypersphere of radius idle_tolerance the robot did not move during
    that interval. The first obs_horizon frames of the window are pruned (their
    actions caused no motion). The last state in the window is NOT pruned here —
    it acts as the witness and will be handled by the next overlapping window if
    the robot remains idle.

    Args:
        episode_data: Dict of numpy arrays for the episode
        obs_horizon: Look-ahead window size for idle detection
        idle_tolerance: Maximum enclosing-ball radius (in state units) for idle detection
        state_key: Dataset key for the pusher state used for idle detection

    Returns:
        Pruned episode dict, or None if the entire episode is idle
    """
    states = episode_data[state_key]
    episode_length = len(states)

    # Initialize mask - keep all frames by default
    keep_mask = np.ones(episode_length, dtype=bool)

    for i in range(episode_length - obs_horizon):
        if fits_within_radius(states[i : i + obs_horizon + 1], idle_tolerance):
            # Prune frames i..i+obs_horizon-1; their actions caused no motion.
            # Frame i+obs_horizon is the witness state — do not prune it here.
            keep_mask[i : i + obs_horizon] = False

    # Check if entire episode is idle
    if not np.any(keep_mask):
        return None

    # Apply mask to all keys
    return {key: value[keep_mask] for key, value in episode_data.items()}


# ============================================================================
# Pruning pipeline
# ============================================================================

def prune_zarr(
    source_zarr_paths,
    output_zarr_path,
    obs_horizon: int = 2,
    idle_tolerance: float = 0.002,
    state_key: str = "state",
):
    start_time = time.time()

    output_zarr_path = pathlib.Path(output_zarr_path)
    output_buffer = ReplayBuffer.create_from_path(str(output_zarr_path), mode="w")

    total_episodes_in = 0
    total_episodes_out = 0
    total_frames_before = 0
    total_frames_after = 0
    removed_episodes = []

    for source_path in source_zarr_paths:
        source_path = pathlib.Path(source_path)
        print(f"\nProcessing: {source_path}")

        source_buffer = ReplayBuffer.create_from_path(str(source_path), mode="r")
        n_eps = source_buffer.n_episodes
        print(f"  Episodes: {n_eps}  |  Keys: {list(source_buffer.keys())}")

        if state_key not in source_buffer.keys():
            raise ValueError(
                f"State key '{state_key}' not found in {source_path}. "
                f"Available keys: {list(source_buffer.keys())}"
            )

        for ep_idx in tqdm(range(n_eps), desc="  Pruning"):
            ep_slice = source_buffer.get_episode_slice(ep_idx)
            episode_data = {key: source_buffer[key][ep_slice] for key in source_buffer.keys()}

            total_episodes_in += 1
            total_frames_before += len(episode_data[state_key])

            pruned = prune_episode(episode_data, obs_horizon, idle_tolerance, state_key)

            if pruned is None:
                removed_episodes.append((source_path.name, ep_idx))
                continue

            total_frames_after += len(pruned[state_key])
            total_episodes_out += 1

            compressors = {k: ("disk" if v.ndim == 4 else "default") for k, v in pruned.items()}
            output_buffer.add_episode(pruned, compressors=compressors)

    elapsed = time.time() - start_time

    print(f"\n{'='*55}")
    print("Pruning Statistics")
    print(f"{'='*55}")
    print(f"  Episodes in:                    {total_episodes_in}")
    print(f"  Episodes out:                   {total_episodes_out}")
    print(f"  Episodes removed (all-idle):    {len(removed_episodes)}")
    for name, idx in removed_episodes:
        print(f"    - {name} episode {idx}")
    print(f"  Frames before pruning:          {total_frames_before}")
    print(f"  Frames after pruning:           {total_frames_after}")
    if total_frames_before > 0:
        pct = 100 * (total_frames_before - total_frames_after) / total_frames_before
        print(f"  Frames pruned:                  {total_frames_before - total_frames_after} ({pct:.1f}%)")
    print(f"  Time elapsed:                   {elapsed:.1f}s")
    print(f"\n  Output: {output_zarr_path}")
    print(f"  Total episodes: {output_buffer.n_episodes}")
    print(f"  Total steps:    {output_buffer.n_steps}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prune idle (no-op) frames from zarr replay buffer datasets."
    )
    parser.add_argument(
        "source_zarr",
        type=str,
        nargs="+",
        help="Source zarr path(s). Multiple paths are merged into one output.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=(
            "Output zarr path. Defaults to <source_dir>/<source_stem>_pruned.zarr "
            "(uses the first source when multiple are given)."
        ),
    )
    parser.add_argument(
        "--obs-horizon",
        type=int,
        default=2,
        help="Observation horizon for idle detection (default: 2).",
    )
    parser.add_argument(
        "--idle-tolerance",
        type=float,
        default=0.002,
        help=(
            "Enclosing-ball radius for idle state detection in state units "
            "(default: 0.002 — roughly 2 mm for planar-push [x, y, θ] states)."
        ),
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default="state",
        help="Dataset key for the pusher state used for idle detection (default: state).",
    )

    args = parser.parse_args()

    source_paths = [pathlib.Path(p) for p in args.source_zarr]

    if args.output is not None:
        output_path = pathlib.Path(args.output)
    else:
        first = source_paths[0]
        output_path = first.parent / f"{first.stem}_pruned.zarr"

    print(f"obs_horizon:    {args.obs_horizon}")
    print(f"idle_tolerance: {args.idle_tolerance}")
    print(f"state_key:      {args.state_key}")

    prune_zarr(
        source_zarr_paths=source_paths,
        output_zarr_path=output_path,
        obs_horizon=args.obs_horizon,
        idle_tolerance=args.idle_tolerance,
        state_key=args.state_key,
    )


if __name__ == "__main__":
    main()
