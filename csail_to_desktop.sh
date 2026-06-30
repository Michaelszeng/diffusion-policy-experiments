#!/bin/bash

# Source directory on your computer
# SOURCE_DIR="data/outputs/furniture_bench/"
# SOURCE_DIR="data/outputs/furniture_bench_context_ablation/"
# SOURCE_DIR="data/outputs/robomimic/"
# SOURCE_DIR="data/outputs/maniskill/"
# SOURCE_DIR="data/outputs/isaac_sim/"
SOURCE_DIR="data/outputs/manifeel/"

# SOURCE_DIR="/home/michzeng/benchmark-furniturebench-juicer/dataset/raw/sim/one_leg/"
# TARGET_DIR="/data/locomotion/michzeng/benchmark-furniturebench-juicer/dataset/raw/sim/one_leg/dagger_iter2_ah1_nm"
# TARGET_DIR="/data/locomotion/michzeng/benchmark-furniturebench-juicer/dataset/raw/sim/one_leg/dagger_iter2_ah3_nm"
# TARGET_DIR="/data/locomotion/michzeng/benchmark-furniturebench-juicer/dataset/raw/sim/one_leg/dagger_iter2_ah6_nm"
# TARGET_DIR="/data/locomotion/michzeng/benchmark-furniturebench-juicer/dataset/raw/sim/one_leg/dagger_iter2_ah10_nm"
# TARGET_DIR="/data/locomotion/michzeng/benchmark-furniturebench-juicer/dataset/raw/sim/one_leg/dagger_iter2_ah15_nm"

# SOURCE_DIR="data/outputs/franka_kitchen/"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/franka_kitchen/2_obs_human_expert"

# SOURCE_DIR="data/diffusion_experiments/maniskill/"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/diffusion_experiments/maniskill/maniskill_planar_push_t_teleop_merged_pruned.zarr"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/diffusion_experiments/maniskill/maniskill_planar_push_t_teleop_merged_pruned_obs_horizon_3_idle_tol_0_0005.zarr"

# Target directory on the target computer
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop_r3m_test"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_non_markovian_200_eps"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_200_eps"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/lowdim_2_obs_one_leg_scripted"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_CLIP_film"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_CLIP_attention"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench"

# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_dart_0_200_eps"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_dart_0_0625_200_eps"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_dart_0_125_200_eps"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_dart_0_25_200_eps"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_dart_0_5_200_eps"

# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop_r3m"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/4_obs_one_leg_teleop"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/4_obs_one_leg_teleop_r3m"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/4_obs_one_leg_teleop_r3m_double_enc"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/8_obs_one_leg_teleop_r3m"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/16_obs_one_leg_teleop_r3m"

# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_non_markovian_dart_0_5_200_eps"

# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/robomimic/2_obs_tool_hang_film"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/robomimic/8_obs_tool_hang_double_enc"

# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/maniskill/2_obs_human_expert_copy2"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/maniskill/2_obs_markovian_expert_copy2"

# SOURCE_DIR="/home/michzeng/benchmark-furniturebench-juicer/dataset/processed/sim/one_leg/scripted/low/"
# TARGET_DIR="/data/locomotion/michzeng/benchmark-furniturebench-juicer/dataset/processed/sim/one_leg/scripted/low/success_translated.zarr"

# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench_context_ablation/2_obs_one_leg_teleop_r3m_cross_attn_scripted_non_markovian_100_eps"

# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/isaac_sim/2_obs_gear_assembly_human_expert"

# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/manifeel/2_obs_peg_in_hole_human_expert_"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/manifeel/2_obs_plug_insert_human_expert_"
TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/manifeel/2_obs_nut_bolt_human_expert_"

####################################################################################

TARGET_USER="michzeng"
TARGET_HOST="slurm-login.csail.mit.edu"

# Rsync command with -avh flags
mkdir -p "$SOURCE_DIR"
rsync -avh "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" "$SOURCE_DIR"

# Example usage:
# ./csail_to_desktop.sh