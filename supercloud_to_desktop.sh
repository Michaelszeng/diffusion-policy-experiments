#!/bin/bash

# Source directory on your computer
SOURCE_DIR="data/outputs/planar_pushing/2_obs"

# Target directory on the target computer
TARGET_USER="mzeng"

TARGET_HOST="txe1-login.mit.edu"
TARGET_DIR="/home/gridsan/mzeng/diffusion-policy-experiments/data/outputs/planar_pushing/2_obs/checkpoints"

# Rsync command with -avh flags
mkdir -p "$SOURCE_DIR"
rsync -avh "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" "$SOURCE_DIR"

# Example usage:
# ./rsync_script.sh