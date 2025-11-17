#!/bin/bash

# Source directory on your computer
SOURCE_DIR="data/outputs/planar_pushing/2_obs_idle_frames_pruned"

# Target directory on the target computer
TARGET_USER="mzeng"
TARGET_HOST="txe1-login.mit.edu"
TARGET_DIR="/home/gridsan/mzeng/diffusion-policy-experiments/data/outputs/planar_pushing"

# Rsync command with -avh flags
rsync -avh "$SOURCE_DIR" "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" 

# Example usage:
# ./desktop_to_supercloud.sh
