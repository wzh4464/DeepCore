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

# 循环遍历所有组合
for reg in "${regularization_options[@]}"
do
    for lr in "${learning_rate_options[@]}"
    do
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
            --selection_epochs 5 \
            --data_path ./data \
            --gpu 0 1 2 \
            --optimizer SGD \
            --lr 0.5 \
            --scheduler CosineAnnealingLR \
            --save_path /backup/${experiment_name}_results_${SLURM_JOB_ID}_lr0_5 \
            --num_gpus 3 \
            --oti_mode full"

        # 添加标志参数
        if [ $reg -eq 1 ]; then
            cmd+=" --oti_use_regularization"
        fi
        if [ $lr -eq 1 ]; then
            cmd+=" --oti_use_learning_rate"
        fi

        # 运行 Python 脚本
        echo "Running: $cmd"
        eval $cmd
        
        # 等待一段时间，以确保资源释放
        sleep 60
    done
done
