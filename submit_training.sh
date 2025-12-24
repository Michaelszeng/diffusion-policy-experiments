#!/bin/bash

# Usage
# LLsub ./submit_training.sh -s 20 -g volta:1

# Initialize and Load Modules
echo "[submit_training.sh] Loading modules and virtual environment."
source /etc/profile
module load anaconda/2023b

# Assume current directory is diffusion-policy-experiments
# source .robodiff/bin/activate || echo "Training with anaconda/2023b module instead of venv"

# Set wandb to offline since Supercloud has no internet access
echo "[submit_training.sh] Setting wandb to offline."
wandb offline

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
export cHYDRA_FULL_ERROR=1

# EXPERIMENT_NAME=2_obs

# EXPERIMENT_NAME=2_obs_32_horizon
# EXPERIMENT_NAME=6_obs_32_horizon
# EXPERIMENT_NAME=10_obs_32_horizon
# EXPERIMENT_NAME=14_obs_32_horizon
# EXPERIMENT_NAME=18_obs_32_horizon
# EXPERIMENT_NAME=2_obs_32_horizon_idle_frames_pruned
# EXPERIMENT_NAME=6_obs_32_horizon_idle_frames_pruned
# EXPERIMENT_NAME=10_obs_32_horizon_idle_frames_pruned
# EXPERIMENT_NAME=14_obs_32_horizon_idle_frames_pruned
# EXPERIMENT_NAME=18_obs_32_horizon_idle_frames_pruned

CONFIG_DIR=config/planar_pushing
CONFIG_NAME=${EXPERIMENT_NAME}.yaml
HYDRA_RUN_DIR=data/outputs/planar_pushing/${EXPERIMENT_NAME}

# CONFIG_DIR=config/maniskill
# CONFIG_NAME=2_obs_state_based.yaml
# HYDRA_RUN_DIR=data/outputs/maniskill/2_obs_state_based

echo "[submit_training.sh] Config name: $CONFIG_NAME"
echo "[submit_training.sh] Hydra run dir: $HYDRA_RUN_DIR"
echo "[submit_training.sh] Running training code..."
echo "[submit_training.sh] Date: $DATE"
echo "[submit_training.sh] Time: $TIME"

python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME \
	hydra.run.dir=$HYDRA_RUN_DIR
