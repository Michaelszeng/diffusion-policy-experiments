#!/bin/bash

# Source directory on your computer
SOURCE_DIR="data/outputs/furniture_bench/2_obs_one_leg_scripted/checkpoints"

# Target directory on the target computer
TARGET_USER="michzeng"

TARGET_HOST="slurm-login.csail.mit.edu"
TARGET_DIR="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted/checkpoints/epoch=095-val_loss=0.0659-val_ddim_mse=0.015579.ckpt"

# Rsync command with -avh flags
mkdir -p "$SOURCE_DIR"
rsync -avh "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" "$SOURCE_DIR"

# Example usage:
# ./csail_to_desktop.sh