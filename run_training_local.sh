#!/bin/bash
set -euo pipefail

# Usage: ./run_training_local.sh

# ============================================================================
# Training hyperparameters
# ============================================================================
BATCH_SIZE=64
LR=1e-4

NUM_WORKERS=8
RESUME=False
CHECKPOINT_PATH=""  # path to .ckpt file (or empty for latest.ckpt)

USE_TORCH_COMPILE=false
TORCH_COMPILE_MODE='default'

# Number of GPUs to use locally. Set to 1 for single-GPU training.
NUM_GPUS=1
# ============================================================================

# ============================================================================
# Experiment configuration
# ============================================================================
EXPERIMENT_CLASS=mundane
EXPERIMENT_NAME=2_obs_pill_CLIP_film
# EXPERIMENT_NAME=8_obs_pill_CLIP_attention_double_enc

CONFIG_DIR=config/${EXPERIMENT_CLASS}
CONFIG_NAME=${EXPERIMENT_NAME}.yaml
HYDRA_RUN_DIR=data/outputs/${EXPERIMENT_CLASS}/${EXPERIMENT_NAME}
# ============================================================================

DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
echo "DATE: $DATE"
echo "TIME: $TIME"
echo "HOSTNAME: $HOSTNAME"
echo "PWD: $PWD"

# --- Environment Setup ---
source env/bin/activate

mkdir -p logs

# --- WANDB Authentication ---
export WANDB_USERNAME="michzeng"
SECRETS_FILE="$PWD/.secrets"
if [ -f "$SECRETS_FILE" ]; then
    source "$SECRETS_FILE"
    echo "Loaded secrets from $SECRETS_FILE"
else
    echo "ERROR: $SECRETS_FILE not found. Create it from .secrets.template before running." >&2
    exit 1
fi
echo "WANDB_USERNAME and WANDB_API_KEY set for authentication."

export TORCH_LOGS="recompiles"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PWD:${PYTHONPATH:-}

echo "Config dir:      $CONFIG_DIR"
echo "Config name:     $CONFIG_NAME"
echo "Hydra run dir:   $HYDRA_RUN_DIR"
echo "BATCH_SIZE:      $BATCH_SIZE"
echo "NUM_WORKERS:     $NUM_WORKERS"
echo "LR:              $LR"
echo "RESUME:          $RESUME"
echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "NUM_GPUS:        $NUM_GPUS"
echo "=========================================="

# When torch.compile is enabled, drop the last incomplete batch to prevent
# shape-change recompiles at epoch boundaries (e.g. einops SymInt hash errors).
COMPILE_ARGS=""
if [ "$USE_TORCH_COMPILE" = "true" ]; then
    COMPILE_ARGS="training.use_torch_compile=true training.torch_compile_mode=$TORCH_COMPILE_MODE +dataloader.drop_last=true"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    ACCELERATE_ARGS="--num_processes=$NUM_GPUS --multi_gpu --mixed_precision=bf16 --gpu_ids=all"
else
    ACCELERATE_ARGS="--num_processes=$NUM_GPUS --mixed_precision=bf16 --gpu_ids=all"
fi
echo "ACCELERATE_ARGS: $ACCELERATE_ARGS"

if [ -z "$CHECKPOINT_PATH" ] || [ "$CHECKPOINT_PATH" == "null" ]; then
    accelerate launch $ACCELERATE_ARGS train.py \
        --config-dir=$CONFIG_DIR \
        --config-name=$CONFIG_NAME \
        hydra.run.dir=$HYDRA_RUN_DIR \
        dataloader.batch_size=$BATCH_SIZE \
        dataloader.num_workers=$NUM_WORKERS \
        optimizer.lr=$LR \
        training.resume=$RESUME \
        training.mixed_precision=bf16 \
        $COMPILE_ARGS
else
    accelerate launch $ACCELERATE_ARGS train.py \
        --config-dir=$CONFIG_DIR \
        --config-name=$CONFIG_NAME \
        hydra.run.dir=$HYDRA_RUN_DIR \
        dataloader.batch_size=$BATCH_SIZE \
        dataloader.num_workers=$NUM_WORKERS \
        optimizer.lr=$LR \
        training.resume=$RESUME \
        training.mixed_precision=bf16 \
        $COMPILE_ARGS \
        training.checkpoint_path=\"$CHECKPOINT_PATH\"
fi
