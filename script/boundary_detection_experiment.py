###
# File: ./script/boundary_detection_experiment.py
# Created Date: Monday, May 13th 2025
# Author: Your Name
# -----
# HISTORY:
# Date          By       Comments
# ----------    ------   ---------------------------------------------------------
###

import os
import time
import subprocess
import threading
from queue import Queue

# 配置参数（基于提供的debug配置）
gpus = [0, 1, 2]  # 可用GPU列表
lrs = [0.05]  # 学习率
num_boundary_values = [10, 20, 30, 40]  # 边界点数量
num_scores_values = [100]  # 样本总数
boundary_transform_intensities = [0.1,0.5,1.0]  # 边界变换强度
seeds = list(range(5))  # 0-4

# 实验配置
experiments = [
    {
        "name": "oti",
        "selection": "OTI",
        "save_path_base": "./results/boundary_detection/oti",
        "extra_args": "--oti_mode full --oti_use_regularization --oti_use_learning_rate"
    },
    {
        "name": "grand",
        "selection": "GraNd",
        "save_path_base": "./results/boundary_detection/grand",
        "extra_args": ""
    },
    {
        "name": "influence_function",
        "selection": "influence_function",
        "save_path_base": "./results/boundary_detection/influence_function",
        "extra_args": ""
    }
]

# 基础命令（基于提供的debug配置）
base_cmd = (
    "/home/zihan/codes/DeepCore/.venv/bin/python main.py --dataset MNIST --model LeNet --exp boundary_detection "
    "--num_exp 1 --epochs 1 --selection_epochs 1 --data_path ./data "
    "--optimizer SGD --scheduler CosineAnnealingLR --num_gpus 1 "
)

def run_task(task):
    gpu, lr, num_boundary, num_scores, transform_intensity, seed, experiment = task

    # 创建特定于此任务的保存路径
    save_path = (f"{experiment['save_path_base']}/nb{num_boundary}_ns{num_scores}_"
                f"ti{transform_intensity}_lr{lr}_seed{seed}")

    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)

    # 构建完整的命令
    cmd = (
        f"{base_cmd} --selection {experiment['selection']} {experiment['extra_args']} "
        f"--gpu {gpu} --lr {lr} --save_path {save_path} "
        f"--seed {seed} --num_boundary {num_boundary} --num_scores {num_scores} "
        f"--boundary_transform_intensity {transform_intensity}"
    )

    print(f"开始任务: {experiment['name']} - GPU {gpu}, LR {lr}, "
          f"Num Boundary {num_boundary}, Num Scores {num_scores}, "
          f"Transform Intensity {transform_intensity}, Seed {seed}")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"任务失败: {experiment['name']} - GPU {gpu}, LR {lr}, "
              f"Num Boundary {num_boundary}, Num Scores {num_scores}, "
              f"Transform Intensity {transform_intensity}, Seed {seed}")
    else:
        print(f"任务完成: {experiment['name']} - GPU {gpu}, LR {lr}, "
              f"Num Boundary {num_boundary}, Num Scores {num_scores}, "
              f"Transform Intensity {transform_intensity}, Seed {seed}")

def run_experiment(experiment):
    print(f"\n{'='*80}")
    print(f"开始运行 {experiment['name']} 边界点检测实验")
    print(f"{'='*80}\n")

    # 生成所有任务组合
    all_tasks = []
    for lr in lrs:
        for num_boundary in num_boundary_values:
            for num_scores in num_scores_values:
                for transform_intensity in boundary_transform_intensities:
                    for seed in seeds:
                        all_tasks.append((None, lr, num_boundary, num_scores, 
                                        transform_intensity, seed, experiment))

    # 动态分配GPU和任务
    task_queue = Queue()
    for task in all_tasks:
        task_queue.put(task)

    num_tasks = len(all_tasks)
    print(f"总任务数: {num_tasks}")
    
    # 任务执行工作线程
    def worker(gpu_id):
        while not task_queue.empty():
            try:
                _, lr, num_boundary, num_scores, transform_intensity, seed, exp = task_queue.get_nowait()
            except:
                break
            task = (gpu_id, lr, num_boundary, num_scores, transform_intensity, seed, exp)
            run_task(task)
            task_queue.task_done()

    # 启动工作线程
    threads = []
    for gpu in gpus:
        # 每个GPU启动 4 个线程
        for _ in range(4):
            t = threading.Thread(target=worker, args=(gpu,))
            t.start()
            threads.append(t)

    # 等待所有线程完成
    for t in threads:
        t.join()

    print(f"\n{experiment['name']} 边界点检测实验完成！")

def run_analysis():
    """实验完成后运行分析和可视化"""
    print("\n开始运行结果分析和可视化...")
    
    # 运行分析脚本
    cmd = "python main.py --exp plot_detection_rate_vs_epochs --save_path ./results/boundary_detection"
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print("分析脚本运行失败！")
    else:
        print("分析和可视化完成！")

if __name__ == "__main__":
    start_time = time.time()

    # 依次运行每个实验
    for experiment in experiments:
        run_experiment(experiment)

    # 运行分析和可视化
    run_analysis()

    # 计算总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "="*80)
    print(f"所有边界点检测实验完成！总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    print("=" * 80)
