#!/bin/bash
#SBATCH --job-name=diffusion_policy_training
#SBATCH --account=locomotion
#SBATCH --partition=locomotion-h200
#SBATCH --qos=locomotion-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage: sbatch submit_training_csail.sh

# ============================================================================
# Training hyperparameters
# ============================================================================
BATCH_SIZE=512
NUM_WORKERS=16
LR=8e-4
# ============================================================================

DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
echo "DATE: $DATE"
echo "TIME: $TIME"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# --- Environment Setup ---
echo "Changing directory to $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }
# Load venv environment
source env/bin/activate

mkdir -p logs

# --- WANDB Authentication ---
export WANDB_USERNAME="michzeng"
export WANDB_API_KEY="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"
echo "WANDB_USERNAME and WANDB_API_KEY set for authentication."

export TORCH_LOGS="recompiles"
export HYDRA_FULL_ERROR=1

# ============================================================================
# Experiment configuration
# ============================================================================

# --- Planar Pushing ---
# EXPERIMENT_CLASS=planar_pushing
# EXPERIMENT_NAME=2_obs

# EXPERIMENT_NAME=2_obs_300_data
# EXPERIMENT_NAME=2_obs_200_data
# EXPERIMENT_NAME=2_obs_50_data
# EXPERIMENT_NAME=2_obs_25_data
# EXPERIMENT_NAME=2_obs_12_5_data
# EXPERIMENT_NAME=2_obs_6_25_data

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

# --- Furniture Bench ---
EXPERIMENT_CLASS=furniture_bench
# EXPERIMENT_NAME=2_obs_one_leg_teleop
EXPERIMENT_NAME=2_obs_one_leg_scripted

# --- Experiment Configuration ---
CONFIG_DIR=config/${EXPERIMENT_CLASS}
CONFIG_NAME=${EXPERIMENT_NAME}.yaml
HYDRA_RUN_DIR=data/outputs/${EXPERIMENT_CLASS}/${EXPERIMENT_NAME}
# ============================================================================

echo "Config dir:    $CONFIG_DIR"
echo "Config name:   $CONFIG_NAME"
echo "Hydra run dir: $HYDRA_RUN_DIR"
echo "BATCH_SIZE:    $BATCH_SIZE"
echo "NUM_WORKERS:   $NUM_WORKERS"
echo "LR:            $LR"
echo "=========================================="

export PYTHONPATH=$PWD:$PYTHONPATH

python train.py \
    --config-dir=$CONFIG_DIR \
    --config-name=$CONFIG_NAME \
    hydra.run.dir=$HYDRA_RUN_DIR \
    dataloader.batch_size=$BATCH_SIZE \
    dataloader.num_workers=$NUM_WORKERS \
    optimizer.lr=$LR \
