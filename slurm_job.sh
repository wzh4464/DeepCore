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

## ad_oti example
# sbatch slurm_job.sh --mode AD_OTI --regularization "1" --learning_rate "1" --adoti_mode full --delta_min 1 --delta_max 20 --delta_step 3 --eps_min 0.005 --eps_max 0.01

# Activate the virtual environment
PYTHON="/home/zihan/codes/DeepCore/.venv/bin/python"

# Change to the correct directory
cd /home/zihan/codes/DeepCore

# Initialize default values for AD_OTI parameters
adoti_mode=""
delta_min=1
delta_max=3
delta_step=1
eps_min=0.1
eps_max=0.05

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode)
            mode="$2"
            shift 2
            ;;
        --adoti_mode)
            adoti_mode="$2"
            shift 2
            ;;
        --delta_min)
            delta_min="$2"
            shift 2
            ;;
        --delta_max)
            delta_max="$2"
            shift 2
            ;;
        --delta_step)
            delta_step="$2"
            shift 2
            ;;
        --eps_min)
            eps_min="$2"
            shift 2
            ;;
        --eps_max)
            eps_max="$2"
            shift 2
            ;;
        --regularization)
            IFS=',' read -r -a regularization_options <<< "$2"
            shift 2
            ;;
        --learning_rate)
            IFS=',' read -r -a learning_rate_options <<< "$2"
            shift 2
            ;;
        *) 
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Define default values if no arguments provided
if [ -z "${regularization_options[*]}" ]; then
    regularization_options=(1)
    echo "No regularization options provided, using default values: (1)"
fi
if [ -z "${learning_rate_options[*]}" ]; then
    learning_rate_options=(0)
    echo "No learning rate options provided, using default values: (0)"
fi
if [ -z "${mode}" ]; then
    mode="Uniform"
    echo "No mode provided, using default mode: Uniform"
else
    echo "Mode set to: ${mode}"
fi

# Set selection method based on mode
case $mode in
    Uniform)
        selection_method="Uniform"
        ;;
    OTI)
        selection_method="OTI"
        ;;
    AD_OTI)
        selection_method="AD_OTI"
        ;;
    *)
        echo "Invalid mode: $mode. Supported modes are Uniform, OTI, AD_OTI."
        exit 1
        ;;
esac

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
        oti_cmd=(
            "$PYTHON" "main.py"
            "--dataset" "MNIST"
            "--model" "LeNet"
            "--selection" "${selection_method}"
            "--num_exp" "1"
            "--batch" "${batch}"
            "--epochs" "150"
            "--selection_epochs" "150"
            "--fraction" "0.5"
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
            "--fraction" "0.5"
            "--gpu" "${gpu}"
            "--optimizer" "SGD"
            "--lr" "0.01"
            "--momentum" "0.9"
            "--weight_decay" "5e-4"
            "--scheduler" "CosineAnnealingLR"
            "--save_path" "/backup/${experiment_name}_results_${SLURM_JOB_ID}"
            "--num_gpus" "1"
            "--oti_mode" "full"
            "--seed" "5"
        )

        # cmd=("${random_cmd[@]}")

        # Add flag parameters as needed
        if [ "$reg" -eq 1 ]; then
            oti_cmd+=("--oti_use_regularization")
        fi
        if [ "$lr" -eq 1 ]; then
            oti_cmd+=("--oti_use_learning_rate")
        fi

        # Print current configuration
        echo "Starting: reg=$reg, lr=$lr on GPU $gpu"
        
        # 在执行主要任务的部分添加条件判断
        if [ "$selection_method" = "AD_OTI" ]; then
            # 跳过主要任务
            echo "Skipping OTI task for AD_OTI mode"
        else
            # 原有的主要任务执行代码
            (
                echo "Starting experiment at $(date)"
                echo "Configuration: reg=$reg, lr=$lr, gpu=$gpu"
                echo "Command: ${oti_cmd[*]}"
                "${oti_cmd[@]}"
                echo "Finished experiment at $(date)"
            ) &
        fi

        # 如果启用了 AD_OTI 模式，则启动 AD_OTI 任务
        if [ "$selection_method" = "AD_OTI" ] && [ ! -z "$adoti_mode" ]; then
            # 定义 AD_OTI 命令
            adoti_cmd=(
                "$PYTHON" "main.py"
                "--dataset" "MNIST"
                "--model" "LeNet"
                "--selection" "AD_OTI"
                "--num_exp" "1"
                "--epochs" "5"
                "--selection_epochs" "5"
                "--fraction" "0.8"
                "--data_path" "./data"
                "--gpu" "${gpu}"
                "--optimizer" "SGD"
                "--lr" "0.1"
                "--momentum" "0.9"
                "--weight_decay" "5e-4"
                "--scheduler" "CosineAnnealingLR"
                "--save_path" "/backup/${experiment_name}_adoti_results_${SLURM_JOB_ID}_lr0_5"
                "--num_gpus" "1"
                "--seed" "0"
                "--delta_min" "${delta_min}"
                "--delta_max" "${delta_max}"
                "--delta_step" "${delta_step}"
                "--eps_min" "${eps_min}"
                "--eps_max" "${eps_max}"
                "--oti_mode" "${adoti_mode}"
                "--log_level" "DEBUG"
            )

            # 启动 AD_OTI 任务
            (
                echo "Starting AD_OTI experiment at $(date)"
                echo "Configuration: reg=$reg, lr=$lr, gpu=$gpu, delta_min=$delta_min, delta_max=$delta_max, delta_step=$delta_step, eps_min=$eps_min, eps_max=$eps_max"
                echo "Command: ${adoti_cmd[*]}"
                "${adoti_cmd[@]}"
                echo "Finished AD_OTI experiment at $(date)"
            ) &
        fi

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
