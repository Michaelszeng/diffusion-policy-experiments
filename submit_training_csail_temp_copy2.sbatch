#!/bin/bash
#SBATCH --job-name=diffusion_policy_training
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-a100
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
CHECKPOINT_PATH=""  # path to .ckpt file (or empty for latest.ckpt)

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

# --- Isaac Sim ---
EXPERIMENT_CLASS=isaac_sim
# EXPERIMENT_NAME=2_obs_gear_assembly_human_expert  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_gear_assembly_human_expert_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=4_obs_gear_assembly_human_expert_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=8_obs_gear_assembly_human_expert_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=12_obs_gear_assembly_human_expert_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=16_obs_gear_assembly_human_expert_attention_double_enc  # batch size 64, LR 1e-4
EXPERIMENT_NAME=20_obs_gear_assembly_human_expert_attention_double_enc  # batch size 64, LR 1e-4

# --- Maniskill ---
# EXPERIMENT_CLASS=maniskill
# EXPERIMENT_NAME=2_obs_markovian_expert  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_human_expert  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_human_expert_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=4_obs_human_expert_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=8_obs_human_expert_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=12_obs_human_expert_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=16_obs_human_expert_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=20_obs_human_expert_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_human_expert_pruned_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=4_obs_human_expert_pruned_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=8_obs_human_expert_pruned_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=12_obs_human_expert_pruned_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=16_obs_human_expert_pruned_attention_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=20_obs_human_expert_pruned_attention_double_enc  # batch size 64, LR 1e-4

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

# --- Furniture Bench Context Ablation ---
# EXPERIMENT_CLASS=furniture_bench_context_ablation
# EXPERIMENT_NAME=2_obs_one_leg_teleop_r3m_cross_attn_scripted_non_markovian_100_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_teleop_r3m_cross_attn_scripted_non_markovian_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_teleop_r3m_cross_attn_scripted_non_markovian_400_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=20_obs_one_leg_teleop_r3m_cross_attn_scripted_non_markovian_100_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=20_obs_one_leg_teleop_r3m_cross_attn_scripted_non_markovian_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=20_obs_one_leg_teleop_r3m_cross_attn_scripted_non_markovian_400_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=20_obs_one_leg_scripted_r3m_non_markovian_100_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=20_obs_one_leg_scripted_r3m_non_markovian_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=20_obs_one_leg_scripted_r3m_non_markovian_400_eps  # batch size 64, LR 1e-4

# --- Furniture Bench ---
# EXPERIMENT_CLASS=furniture_bench
# EXPERIMENT_NAME=2_obs_one_leg_teleop_r3m  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_teleop_r3m_only_imitation_juicer_data  # batch size 64, LR 1e-4

# EXPERIMENT_NAME=2_obs_one_leg_teleop_r3m_cross_attn  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=4_obs_one_leg_teleop_r3m  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=8_obs_one_leg_teleop_r3m  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=12_obs_one_leg_teleop_r3m  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=16_obs_one_leg_teleop_r3m  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=20_obs_one_leg_teleop_r3m  # batch size 64, LR 1e-4

# EXPERIMENT_NAME=2_obs_one_leg_teleop_r3m_cross_attn_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=4_obs_one_leg_teleop_r3m_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=8_obs_one_leg_teleop_r3m_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=12_obs_one_leg_teleop_r3m_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=16_obs_one_leg_teleop_r3m_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=20_obs_one_leg_teleop_r3m_double_enc  # batch size 64, LR 1e-4

# EXPERIMENT_NAME=2_obs_one_leg_teleop_r3m_cross_attn_double_enc_scripted_non_markovian_1000_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=4_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=8_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=12_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=16_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=20_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps  # batch size 64, LR 1e-4

# EXPERIMENT_NAME=lowdim_2_obs_one_leg_scripted_r3m  # batch size 64, LR 1e-4

# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_100_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_400_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_1000_eps  # batch size 64, LR 1e-4

# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_03125_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_0625_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_125_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_25_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_5_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_75_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_1_0_200_eps  # batch size 64, LR 1e-4

# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_400_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_03125_400_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_0625_400_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_125_400_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_25_400_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_5_400_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_0_75_400_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_dart_1_0_400_eps  # batch size 64, LR 1e-4

# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_non_markovian_dart_0_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_non_markovian_dart_0_0625_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_non_markovian_dart_0_125_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_non_markovian_dart_0_25_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_non_markovian_dart_0_5_200_eps  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_one_leg_scripted_r3m_non_markovian_dart_1_0_200_eps  # batch size 64, LR 1e-4

# --- Robomimic ---
# EXPERIMENT_CLASS=robomimic
# EXPERIMENT_NAME=2_obs_tool_hang_film  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=2_obs_tool_hang_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=4_obs_tool_hang_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=8_obs_tool_hang_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=12_obs_tool_hang_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=16_obs_tool_hang_double_enc  # batch size 64, LR 1e-4
# EXPERIMENT_NAME=20_obs_tool_hang_double_enc  # batch size 64, LR 1e-4

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

if [ -z "$CHECKPOINT_PATH" ] || [ "$CHECKPOINT_PATH" == "null" ]; then
    exec python train.py \
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
    exec python train.py \
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
