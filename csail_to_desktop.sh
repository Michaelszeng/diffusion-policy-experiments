#!/bin/bash

# Source directory on your computer
SOURCE_DIR="data/outputs/furniture_bench/"
# SOURCE_DIR="data/outputs/"

# Target directory on the target computer
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_v2_dataset"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_no_pressure"
TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_screw_z_pressure"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_CLIP_film"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_CLIP_attention"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop_CLIP_film"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop_CLIP_attention"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench"
# TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop_no_robust_rearrangement_data"

####################################################################################

TARGET_USER="michzeng"
TARGET_HOST="slurm-login.csail.mit.edu"

# Rsync command with -avh flags
mkdir -p "$SOURCE_DIR"
rsync -avh "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" "$SOURCE_DIR"

# Example usage:
# ./csail_to_desktop.sh