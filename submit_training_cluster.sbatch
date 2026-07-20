#!/bin/bash
#SBATCH --job-name=diffusion_policy_training
#SBATCH --account=<>
#SBATCH --partition=<>
#SBATCH --qos=<>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage: sbatch submit_training_csail.sh

# ============================================================================
# Training hyperparameters
# ============================================================================
BATCH_SIZE=64
LR=1e-4

NUM_WORKERS=8
RESUME=True  # NOTE: always set to True so that requeueing will automatically resume from the last checkpoint
# Optional MANUAL checkpoint override. Leave empty to automatically select based on:
#   1. latest.ckpt in the run dir (if present) takes precedence, otherwise the
#   2. training.checkpoint_path from the yaml (e.g. a warm-start ckpt).
CHECKPOINT_PATH=""  # path to .ckpt file (or empty to use the automatic policy above)

USE_TORCH_COMPILE=false
TORCH_COMPILE_MODE='default'
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
SECRETS_FILE="$SLURM_SUBMIT_DIR/.secrets"
if [ -f "$SECRETS_FILE" ]; then
    source "$SECRETS_FILE"
    echo "Loaded secrets from $SECRETS_FILE"
else
    echo "ERROR: $SECRETS_FILE not found. Create it from .secrets.template before submitting." >&2
    exit 1
fi
echo "WANDB_USERNAME and WANDB_API_KEY set for authentication."

export TORCH_LOGS="recompiles"  # Print torch recompilation logs to stdout
export HYDRA_FULL_ERROR=1

# ============================================================================
# Experiment configuration
# ============================================================================
EXPERIMENT_CLASS=<>  # Set this to the config directory you are using (i.e. "planar_pushing")
EXPERIMENT_NAME=<>  # Set this to the config filename ("2_obs")

# ============================================================================

CONFIG_DIR=config/${EXPERIMENT_CLASS}
CONFIG_NAME=${EXPERIMENT_NAME}.yaml
HYDRA_RUN_DIR=data/outputs/${EXPERIMENT_CLASS}/${EXPERIMENT_NAME}

echo "Config dir:    $CONFIG_DIR"
echo "Config name:   $CONFIG_NAME"
echo "Hydra run dir: $HYDRA_RUN_DIR"
echo "BATCH_SIZE:    $BATCH_SIZE"
echo "NUM_WORKERS:   $NUM_WORKERS"
echo "LR:            $LR"
echo "RESUME:        $RESUME"
echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "=========================================="

export PYTHONPATH=$PWD:$PYTHONPATH

# When torch.compile is enabled, drop the last incomplete batch to prevent
# shape-change recompiles at epoch boundaries (e.g. einops SymInt hash errors).
COMPILE_ARGS=""
if [ "$USE_TORCH_COMPILE" = "true" ]; then
    COMPILE_ARGS="training.use_torch_compile=true training.torch_compile_mode=$TORCH_COMPILE_MODE +dataloader.drop_last=true"
fi

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "NUM_GPUS: $NUM_GPUS"

# Pick a unique rendezvous port per SLURM job to avoid EADDRINUSE when
# multiple torchrun jobs share a node on a shared partition (default is 29500).
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "MASTER_PORT: $MASTER_PORT"

# --- Checkpoint resolution -------------------------------------------------
# RESUME stays True so SLURM requeue works. We decide WHAT to resume from here,
# in priority order:
#   1. A manually-set CHECKPOINT_PATH above (explicit override).
#   2. $HYDRA_RUN_DIR/checkpoints/latest.ckpt if it exists (this run was already
#      partway through and got requeued/restarted) -> force checkpoint_path=null
#      so the workspace loads latest.ckpt instead of the yaml's warm-start ckpt.
#   3. Otherwise leave training.checkpoint_path from the yaml untouched (e.g. a
#      warm-start checkpoint for fine-tuning on the first launch).
CKPT_OVERRIDE=()
LATEST_CKPT="$HYDRA_RUN_DIR/checkpoints/latest.ckpt"
if [ -n "$CHECKPOINT_PATH" ] && [ "$CHECKPOINT_PATH" != "null" ]; then
    echo "Checkpoint policy: manual CHECKPOINT_PATH -> $CHECKPOINT_PATH"
    CKPT_OVERRIDE=(training.checkpoint_path="$CHECKPOINT_PATH")
elif [ -f "$LATEST_CKPT" ]; then
    echo "Checkpoint policy: found $LATEST_CKPT -> resuming from latest.ckpt (overriding yaml checkpoint_path)"
    CKPT_OVERRIDE=(training.checkpoint_path=null)
else
    echo "Checkpoint policy: no latest.ckpt -> using training.checkpoint_path from yaml (warm-start if set)"
fi

exec torchrun --nproc_per_node=$NUM_GPUS --master-port=$MASTER_PORT train.py \
    --config-dir=$CONFIG_DIR \
    --config-name=$CONFIG_NAME \
    hydra.run.dir=$HYDRA_RUN_DIR \
    dataloader.batch_size=$BATCH_SIZE \
    dataloader.num_workers=$NUM_WORKERS \
    optimizer.lr=$LR \
    training.resume=$RESUME \
    training.mixed_precision=bf16 \
    $COMPILE_ARGS \
    "${CKPT_OVERRIDE[@]}"
