#!/bin/bash

# Source directory on your computer
# SOURCE_DIR="/home/michzeng/benchmark-furniturebench-juicer/dataset/imitation-juicer-data-processed-001"
SOURCE_DIR="/home/michzeng/benchmark-furniturebench-juicer/dataset/processed"

# Target directory on the target computer
TARGET_USER="michzeng"
TARGET_HOST="slurm-login.csail.mit.edu"

# TARGET_DIR="/data/locomotion/michzeng/benchmark-furniturebench-juicer/dataset/"
TARGET_DIR="/data/locomotion/michzeng/benchmark-furniturebench-juicer/dataset/"

# Create target directory on remote if it doesn't exist
ssh "$TARGET_USER@$TARGET_HOST" "mkdir -p $TARGET_DIR"

# Rsync command with -avh flags
rsync -avh "$SOURCE_DIR" "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" 

# Example usage:
# ./desktop_to_csail.sh
