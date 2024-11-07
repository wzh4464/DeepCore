#!/bin/bash
#SBATCH --job-name=OTI_experiment
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j_err.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=128:00:00
#SBATCH --partition=debug

# Activate the virtual environment
PYTHON="/home/zihan/codes/DeepCore/.venv/bin/python"

# Change to the correct directory
cd /home/zihan/codes/DeepCore

# Define parameter arrays with command line support
# Usage example: sbatch slurm_job.sh "0 1" "0 1"
# This will test regularization=[0,1] and learning_rate=[0,1] combinations
if [ $# -lt 2 ]; then
    # Default values if no arguments provided
    regularization_options=(1)
    learning_rate_options=(0)
    echo "No arguments provided, using default values:"
    echo "regularization_options=(1)"
    echo "learning_rate_options=(0)"
else
    # Convert space-separated command line args into arrays
    IFS=' ' read -r -a regularization_options <<< "$1"
    IFS=' ' read -r -a learning_rate_options <<< "$2"
    echo "Using provided values:"
    echo "regularization_options=(${regularization_options[*]})"
    echo "learning_rate_options=(${learning_rate_options[*]})"
fi

# Counter to track running processes
running_processes=0
max_processes=3

# Set batch size
batch=128

# Iterate through all combinations
for reg in "${regularization_options[@]}"
do
    for lr in "${learning_rate_options[@]}"
    do
        # Calculate GPU number: (reg + lr * 2) % 3
        gpu=0
        
        # Set experiment name
        experiment_name="1107_oti_reg_${reg}_lr_${lr}_batch_${batch}_gpu_${gpu}"
        
        # Start building the command with basic parameters
        cmd=(
            "$PYTHON" "main.py"
            "--dataset" "CIFAR10"
            "--model" "ResNet18"
            "--selection" "OTI"
            "--num_exp" "1"
            "--batch" "${batch}"
            "--epochs" "5"
            "--selection_epochs" "5"
            "--data_path" "./data"
            "--gpu" "${gpu}"
            "--optimizer" "SGD"
            "--lr" "0.1"
            "--momentum" "0.9"
            "--weight_decay" "5e-4"
            "--scheduler" "CosineAnnealingLR"
            "--save_path" "/backup/${experiment_name}_results_${SLURM_JOB_ID}_lr0_5"
            "--num_gpus" "1"
            "--oti_mode" "full"
            "--seed" "0"
        )

        random_cmd=(
            "$PYTHON" "main.py"
            "--dataset" "CIFAR10"
            "--model" "ResNet18"
            "--selection" "Uniform"
            "--num_exp" "1"
            "--batch" "${batch}"
            "--epochs" "150"
            "--selection_epochs" "150"
            "--data_path" "./data"
            "--fraction" "0.8"
            "--gpu" "${gpu}"
            "--optimizer" "SGD"
            "--lr" "0.01"
            "--momentum" "0.9"
            "--weight_decay" "5e-4"
            "--scheduler" "CosineAnnealingLR"
            "--save_path" "/backup/${experiment_name}_results_${SLURM_JOB_ID}"
            "--num_gpus" "1"
            "--oti_mode" "full"
            "--seed" "0"
        )

        cmd=("${random_cmd[@]}")

        # Add flag parameters as needed
        if [ "$reg" -eq 1 ]; then
            cmd+=("--oti_use_regularization")
        fi
        if [ "$lr" -eq 1 ]; then
            cmd+=("--oti_use_learning_rate")
        fi

        # Print current configuration
        echo "Starting: reg=$reg, lr=$lr on GPU $gpu"
        
        # Run command in the background, redirecting output
        (
            echo "Starting experiment at $(date)"
            echo "Configuration: reg=$reg, lr=$lr, gpu=$gpu"
            echo "Command: ${cmd[*]}"
            "${cmd[@]}"
            echo "Finished experiment at $(date)"
        ) &

        # Increment running processes count
        ((running_processes++))
        
        # Wait for any process to complete if maximum concurrent processes are reached
        if [ $running_processes -ge $max_processes ]; then
            wait -n
            ((running_processes--))
        fi
    done
done

# Wait for all remaining processes to complete
wait
echo "All experiments completed at $(date)"
