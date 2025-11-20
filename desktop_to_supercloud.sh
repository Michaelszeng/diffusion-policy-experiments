#!/bin/bash

# Source directory on your computer
# SOURCE_DIR="data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large_pruned.zarr"
SOURCE_DIR="/home/michzeng/.maniskill/demos/PushT-v1/"

# Target directory on the target computer
TARGET_USER="mzeng"
TARGET_HOST="txe1-login.mit.edu"
# TARGET_DIR="/home/gridsan/mzeng/diffusion-policy-experiments/data/diffusion_experiments/planar_pushing/"
TARGET_DIR="~/.maniskill/demos/PushT-v1/"

# Create target directory on remote if it doesn't exist
ssh "$TARGET_USER@$TARGET_HOST" "mkdir -p $TARGET_DIR"

# Rsync command with -avh flags
rsync -avh "$SOURCE_DIR" "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" 

# Example usage:
# ./desktop_to_supercloud.sh
