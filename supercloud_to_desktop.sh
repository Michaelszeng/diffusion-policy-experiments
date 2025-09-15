#!/bin/bash

# Source directory on your computer
SOURCE_DIR="data/outputs/grasp_two_bins_flat/same_middle_same_return/basic_training/2_obs/checkpoints/latest.ckpt"

# Target directory on the target computer
TARGET_USER="mzeng"

TARGET_HOST="txe1-login.mit.edu"
TARGET_DIR="/home/gridsan/mzeng/diffusion-search-learning/data/outputs/grasp_two_bins_flat/same_middle_same_return/basic_training/2_obs/checkpoints/latest.ckpt"

# Rsync command with -avh flags
mkdir -p "$SOURCE_DIR"
rsync -avh "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" "$SOURCE_DIR" 

# Example usage:
# ./rsync_script.sh