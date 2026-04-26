#!/bin/bash

# Source directory on your computer
SOURCE_DIR="data/outputs/furniture_bench/"

# Target directory on the target computer
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/lowdim_2_obs_one_leg_scripted"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_CLIP_film"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_CLIP_attention"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench"

# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_dart_0_200_eps"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_dart_0_03125_200_eps"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_dart_0_0625_200_eps"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_dart_0_125_200_eps"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_dart_0_25_200_eps"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_dart_0_5_200_eps"

# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop_r3m"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/4_obs_one_leg_teleop"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/4_obs_one_leg_teleop_r3m"
TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/8_obs_one_leg_teleop_r3m"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/16_obs_one_leg_teleop_r3m"

# SOURCE_DIR="/home/michzeng/benchmark-furniturebench-juicer/dataset/processed/sim/one_leg/scripted/low/"
# TARGET_DIR="/data/locomotion/michzeng/benchmark-furniturebench-juicer/dataset/processed/sim/one_leg/scripted/low/success_translated.zarr"

####################################################################################

TARGET_USER="michzeng"
TARGET_HOST="slurm-login.csail.mit.edu"

# Rsync command with -avh flags
mkdir -p "$SOURCE_DIR"
rsync -avh "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" "$SOURCE_DIR"

# Example usage:
# ./csail_to_desktop.sh