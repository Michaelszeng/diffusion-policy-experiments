#!/bin/bash

# Source directory on your computer
SOURCE_DIR="data/diffusion_experiments/grasp_two_bins/two_bins_flat_data_2_left_same_return_same_center_1_second_total.zarr"
# SOURCE_DIR="data/diffusion_experiments/grasp_two_bins/two_bins_flat_data_2_right_same_return_same_center_1_second_total.zarr"

# Target directory on the target computer
TARGET_USER="mzeng"
TARGET_HOST="txe1-login.mit.edu"
TARGET_DIR="/home/gridsan/mzeng/diffusion-search-learning/data/diffusion_experiments/grasp_two_bins/"
# TARGET_DIR="/home/gridsan/mzeng/diffusion-search-learning/data/diffusion_experiments/grasp_two_bins/"

# Rsync command with -avh flags
rsync -avh "$SOURCE_DIR" "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" 

# Example usage:
# ./desktop_to_supercloud.sh
