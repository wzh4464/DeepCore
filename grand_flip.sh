#!/bin/bash
#SBATCH --job-name=flip_MNIST
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j_err.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=debug

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--num_flip)
            numflip="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Check if numflip is set
if [ -z "$numflip" ]; then
    echo "Error: numflip not specified. Use -f or --num_flip to set the value."
    exit 1
fi

# Activate the virtual environment
PYTHON="/home/zihan/codes/DeepCore/.venv/bin/python"

# Change to the correct directory
cd /home/zihan/codes/DeepCore

# Set the SLURM experiment parameters
gpu_list="0"
experiment_name="flip_MNIST_${SLURM_JOB_ID}"
save_path="results/flip_GraNd_${SLURM_JOB_ID}_${numflip}"

# Build the command
cmd=(
    "$PYTHON" "main.py"
    "--dataset" "MNIST"
    "--model" "LeNet"
    "--selection" "GraNd"
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
    "--seed" "0"
    "--num_scores" "100"
    "--num_flip" "$numflip"
)

# Print configuration for logging
echo "Starting experiment at $(date)"
echo "Configuration:"
echo "Experiment name: $experiment_name"
echo "Number of flips: $numflip"
echo "Command: ${cmd[*]}"

# Execute the experiment
"${cmd[@]}"

# Completion message
echo "Experiment finished at $(date)"