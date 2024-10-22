#!/bin/bash

#SBATCH --job-name=OTI_experiment
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j_err.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:3
#SBATCH --mem=64G
#SBATCH --time=128:00:00
#SBATCH --partition=debug

# 激活虚拟环境
PYTHON="/home/zihan/codes/DeepCore/.venv/bin/python"

# 切换到正确的目录
cd /home/zihan/codes/DeepCore

# 定义参数数组
regularization_options=(0 1)
learning_rate_options=(0 1)

# 创建日志目录
mkdir -p logs

# 计数器用于跟踪当前运行的进程数
running_processes=0
max_processes=3

# 循环遍历所有组合
for reg in "${regularization_options[@]}"
do
    for lr in "${learning_rate_options[@]}"
    do
        # 计算 GPU 编号: (reg + lr * 2) % 3
        gpu=$(( (reg + lr * 2) % 3 ))
        
        # 设置实验名称
        experiment_name="reg_${reg}_lr_${lr}"
        
        # 构建命令
        cmd="$PYTHON main.py \
            --dataset TinyImageNet \
            --model ResNet34 \
            --selection OTI \
            --num_exp 1 \
            --batch 128 \
            --epochs 20 \
            --selection_epochs 20 \
            --data_path ./data \
            --gpu ${gpu} \
            --optimizer SGD \
            --lr 0.5 \
            --scheduler CosineAnnealingLR \
            --save_path /backup/${experiment_name}_results_${SLURM_JOB_ID}_lr0_5 \
            --num_gpus 1 \
            --oti_mode full"

        # 添加标志参数
        if [ $reg -eq 1 ]; then
            cmd+=" --oti_use_regularization"
        fi
        if [ $lr -eq 1 ]; then
            cmd+=" --oti_use_learning_rate"
        fi

        # 创建独立的日志文件
        log_file="logs/${experiment_name}_${SLURM_JOB_ID}.log"
        err_file="logs/${experiment_name}_${SLURM_JOB_ID}.err"

        # 打印当前配置
        echo "Starting: reg=$reg, lr=$lr on GPU $gpu"
        echo "Log file: $log_file"
        
        # 在后台运行命令，并重定向输出到日志文件
        (
            echo "Starting experiment at $(date)" > "$log_file"
            echo "Configuration: reg=$reg, lr=$lr, gpu=$gpu" >> "$log_file"
            echo "Command: $cmd" >> "$log_file"
            eval "$cmd" >> "$log_file" 2>> "$err_file"
            echo "Finished experiment at $(date)" >> "$log_file"
        ) &

        # 增加运行进程计数
        ((running_processes++))

        # 如果达到最大进程数，等待任意一个进程完成
        if [ $running_processes -ge $max_processes ]; then
            wait -n  # 等待任意一个子进程完成
            ((running_processes--))
        fi
    done
done

# 等待所有剩余进程完成
wait

echo "All experiments completed at $(date)"