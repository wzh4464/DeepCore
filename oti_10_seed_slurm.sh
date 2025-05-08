#!/bin/bash
#SBATCH --job-name=flip_OTI_Adult_multiseed
#SBATCH --output=logs/epochs_log/%x_%A_%a.log
#SBATCH --error=logs/epochs/%x_%A_%a_err.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=debug
#SBATCH --array=0-39  # 10 seeds × 4 flip percentages = 40 jobs

# Define seeds array (10 different seeds)
seeds=(42 123 456 789 101 202 303 404 505 606)

# Define flip percentages
flip_percentages=(40 30 20 10)

# Calculate seed index and flip percentage index from SLURM_ARRAY_TASK_ID
seed_index=$((SLURM_ARRAY_TASK_ID % 10))
flip_index=$((SLURM_ARRAY_TASK_ID / 10))

# Get current seed and flip percentage
current_seed=${seeds[$seed_index]}
current_flip=${flip_percentages[$flip_index]}

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using seed: $current_seed (Seed index: $seed_index)"
echo "Using flip percentage: $current_flip (Flip index: $flip_index)"

# Activate the virtual environment
PYTHON="/home/zihan/codes/DeepCore/.venv/bin/python"

# Change to the correct directory
cd /home/zihan/codes/DeepCore

# Set the SLURM experiment parameters
gpu_list="0"
experiment_name="flip_MNIST_${SLURM_JOB_ID}_flip${current_flip}_seed${current_seed}"
save_path="results/flip_OTI_${SLURM_JOB_ID}_${current_flip}_seed${current_seed}"

# Build the command
cmd=(
    "$PYTHON" "main.py"
    "--dataset" "MNIST"
    "--model" "LeNet"
    "--selection" "OTI"
    "--exp" "flip"
    "--workers" "4"
    "--num_exp" "1"
    "--epochs" "5"
    "--selection_epochs" "5"
    "--data_path" "./data"
    "--optimizer" "SGD"
    "--lr" "0.1"
    "--scheduler" "CosineAnnealingLR"
    "--save_path" "$save_path"
    "--num_gpus" "1"
    "--seed" "$current_seed"
    "--num_scores" "100"
    "--num_flip" "$current_flip"
    "--oti_mode" "full"
)

# Print configuration for logging
echo "Starting experiment at $(date)"
echo "Configuration:"
echo "Experiment name: $experiment_name"
echo "Number of flips: $current_flip"
echo "Seed: $current_seed"
echo "Command: ${cmd[*]}"

# Execute the experiment
"${cmd[@]}"

# Completion message
echo "Experiment finished at $(date)"