#!/bin/bash
#SBATCH --job-name=flip_MNIST
#SBATCH --output=logs/epochs_log/%x_%j.log
#SBATCH --error=logs/epochs/%x_%j_err.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=debug

# 默认参数值
method="OTI"
numflip="40"
seed="42"
epochs="5"
selection_epochs="5"
lr="0.001"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--method)
            method="$2"
            shift 2
            ;;
        -f|--num_flip)
            numflip="$2"
            shift 2
            ;;
        -s|--seed)
            seed="$2"
            shift 2
            ;;
        -e|--epochs)
            epochs="$2"
            selection_epochs="$2"
            shift 2
            ;;
        -se|--selection_epochs)
            selection_epochs="$2"
            shift 2
            ;;
        -l|--lr)
            lr="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 验证方法参数
valid_methods=("GraNd" "OTI" "influence_function")
if [[ ! " ${valid_methods[@]} " =~ " ${method} " ]]; then
    echo "错误: 无效的方法 '${method}'。请使用以下选项之一: ${valid_methods[*]}"
    exit 1
fi

# 激活虚拟环境
PYTHON="/home/zihan/codes/DeepCore/.venv/bin/python"

# 切换到正确的目录
cd /home/zihan/codes/DeepCore

# 设置SLURM实验参数
gpu_list="0"
experiment_name="flip_${method}_MNIST_${SLURM_JOB_ID}"

# 根据选择的方法设置保存路径
case $method in
    "GraNd")
        save_path="results/flip_GraNd_${SLURM_JOB_ID}_${numflip}_${epochs}_${lr}"
        ;;
    "OTI")
        save_path="results/flip_oti_${SLURM_JOB_ID}_${numflip}_${epochs}_${lr}"
        ;;
    "influence_function")
        save_path="results/flip_influence_function_${SLURM_JOB_ID}_${numflip}_${epochs}_${lr}"
        ;;
esac

# 构建命令
cmd=(
    "$PYTHON" "main.py"
    "--dataset" "MNIST"
    "--model" "LeNet"
    "--selection" "$method"
    "--exp" "flip"
    "--workers" "4"
    "--num_exp" "1"
    "--epochs" "$epochs"
    "--selection_epochs" "$epochs"
    "--data_path" "./data"
    "--optimizer" "SGD"
    "--lr" "$lr"
    "--selection_lr" "$lr"
    "--scheduler" "CosineAnnealingLR"
    "--save_path" "$save_path"
    "--num_gpus" "1"
    "--seed" "$seed"
    "--num_scores" "100"
    "--num_flip" "$numflip"
)

# 为OTI方法添加特定参数
if [ "$method" == "OTI" ]; then
    cmd+=("--oti_mode" "full")
fi

# 打印配置以记录
echo "开始实验 $(date)"
echo "配置:"
echo "方法: $method"
echo "实验名称: $experiment_name"
echo "翻转数量: $numflip"
echo "随机种子: $seed"
echo "训练轮数: $epochs"
echo "选择轮数: $selection_epochs"
echo "学习率: $lr"
echo "命令: ${cmd[*]}"

# 执行实验
"${cmd[@]}"

# 完成信息
echo "实验完成 $(date)" 