#!/bin/bash

# SOURCE_DIR="/home/michzeng/Downloads/IsaacGym_Preview_4_Package.tar.gz"
# TARGET_DIR="/data/locomotion/michzeng/"

# SOURCE_DIR="           /home/michzeng/benchmark-furniturebench-juicer/dataset/imitation-juicer-data-processed-001/processed/sim/one_leg/teleop/low/success_truncated_translated.zarr"
# TARGET_DIR="/data/locomotion/michzeng/benchmark-furniturebench-juicer/dataset/imitation-juicer-data-processed-001/processed/sim/one_leg/teleop/low/"

# SOURCE_DIR="           /home/michzeng/benchmark-furniturebench-juicer/dataset/processed/diffik/sim/one_leg/teleop/low/success_processed_translated.zarr"
# TARGET_DIR="/data/locomotion/michzeng/benchmark-furniturebench-juicer/dataset/processed/diffik/sim/one_leg/teleop/low/"

SOURCE_DIR="           /home/michzeng/benchmark-furniturebench-juicer/dataset/processed/sim/one_leg/scripted/low/success_translated.zarr"
TARGET_DIR="/data/locomotion/michzeng/benchmark-furniturebench-juicer/dataset/processed/sim/one_leg/scripted/low/"

####################################################################################

# Strip leading spaces added for readability
SOURCE_DIR=$(echo "$SOURCE_DIR" | sed -e 's/^[[:space:]]*//')
TARGET_DIR=$(echo "$TARGET_DIR" | sed -e 's/^[[:space:]]*//')

# Target directory on the target computer
TARGET_USER="michzeng"
TARGET_HOST="slurm-login.csail.mit.edu"

# Create target directory on remote if it doesn't exist
ssh "$TARGET_USER@$TARGET_HOST" "mkdir -p $TARGET_DIR"

# Rsync command with -avh flags
rsync -avh "$SOURCE_DIR" "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" 

# Example usage:
# ./csail_to_desktop.sh
