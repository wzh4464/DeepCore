###
# File: ./script/exp_heatmap.py
# Created Date: Wednesday, May 21st 2025
# Author: Zihan
# -----
# Last Modified: Wednesday, 21st May 2025 10:29:10 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

#!/usr/bin/env python3
import os
import subprocess
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

# Base directory for saving results
BASE_SAVE_PATH = "./results/experiment_heatmap"

# Define the parameter configurations for each experiment type
# Experiment 1: Varying delta_min and delta_max
delta_configs = [
    # (delta_min, delta_max)
    (1, 3),
    (2, 3),
    (3, 3),
    (2, 2),
    (1, 4),
    (1, 5)
]

# Experiment 2: Varying eps_min and eps_max
eps_configs = [
    # (eps_min, eps_max)
    (0.05, 0.05),
    (0.1, 0.05),
    (0.15, 0.05),
    (0.2, 0.05),
    (0.1, 0.02),
    (0.1, 0.07),
    (0.1, 0.1)
]

# Seeds to use for each configuration
seeds = list(range(8))  # 0 to 7

# Available GPUs
gpus = [0, 1, 2, 3]

# Base command template
base_cmd_template = (
    "python main.py "
    "--dataset MNIST "
    "--model LeNet "
    "--selection AD_OTI "
    "--num_exp 1 "
    "--epochs 5 "
    "--selection_epochs 5 "
    "--data_path ./data "
    "--optimizer SGD "
    "--lr 0.1 "
    "--scheduler CosineAnnealingLR "
    "--num_gpus 1 "
    "--workers 4 "  # Set workers to 4 as requested
    "--oti_mode full "
    "--exp flip "
    "--num_flip 40 "
    "--delta_0 2 "
    "--delta_step 1 "
    "--gpu {gpu} "
    "--save_path {save_path} "
    "--delta_min {delta_min} "
    "--delta_max {delta_max} "
    "--eps_min {eps_min} "
    "--eps_max {eps_max} "
    "--seed {seed}"
)

def run_command(cmd):
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, result, ""
    except subprocess.CalledProcessError as e:
        error_msg = f"错误输出: {e.stderr.decode()}"
        return False, None, error_msg

def read_result(save_path):
    """Read the ratio value from the result.txt file."""
    result_file = os.path.join(save_path, "result.txt")
    try:
        with open(result_file, 'r') as f:
            ratio = float(f.read().strip())
        return ratio
    except (FileNotFoundError, ValueError) as e:
        print(f"读取结果文件 {result_file} 出错: {e}")
        return None

def run_single_task(task):
    """运行单个实验任务"""
    gpu = task['gpu']
    config_type = task['type']
    delta_min = task['delta_min']
    delta_max = task['delta_max']
    eps_min = task['eps_min']
    eps_max = task['eps_max']
    seed = task['seed']
    
    # 创建保存路径
    if config_type == 'delta':
        save_path = f"{BASE_SAVE_PATH}/delta_exp/delta_min_{delta_min}_delta_max_{delta_max}/seed_{seed}"
    else:  # eps
        save_path = f"{BASE_SAVE_PATH}/eps_exp/eps_min_{eps_min}_eps_max_{eps_max}/seed_{seed}"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 构建并执行命令
    cmd = base_cmd_template.format(
        gpu=gpu,
        save_path=save_path,
        delta_min=delta_min,
        delta_max=delta_max,
        eps_min=eps_min,
        eps_max=eps_max,
        seed=seed
    )
    
    print(f"开始执行: GPU {gpu}, {'delta' if config_type == 'delta' else 'eps'} 配置, seed {seed}")
    success, _, error_msg = run_command(cmd)
    
    # 读取结果
    if success:
        result = read_result(save_path)
        if result is not None:
            print(f"GPU {gpu}, {'delta_min=' + str(delta_min) + '_delta_max=' + str(delta_max) if config_type == 'delta' else 'eps_min=' + str(eps_min) + '_eps_max=' + str(eps_max)}, seed {seed}: ratio = {result}")
            return {
                'config_type': config_type,
                'delta_min': delta_min,
                'delta_max': delta_max,
                'eps_min': eps_min,
                'eps_max': eps_max,
                'seed': seed,
                'result': result
            }
    else:
        print(f"GPU {gpu}, {'delta' if config_type == 'delta' else 'eps'} 配置, seed {seed} 执行失败")
        print(f"完整命令: {cmd}")
        print(error_msg)
    
    return None

def create_all_tasks():
    """创建所有任务列表"""
    all_tasks = []

    # 创建delta配置的任务
    for delta_min, delta_max in delta_configs:
        for seed in seeds:
            all_tasks.append(
                {
                    "type": "delta",
                    "delta_min": delta_min,
                    "delta_max": delta_max,
                    "eps_min": 0.1,  # 默认值
                    "eps_max": 0.05,  # 默认值
                    "seed": seed,
                }
            )

    # 创建eps配置的任务
    for eps_min, eps_max in eps_configs:
        for seed in seeds:
            all_tasks.append({
                'type': 'eps',
                'delta_min': 1,   # 默认值
                'delta_max': 3,  # 默认值
                'eps_min': eps_min,
                'eps_max': eps_max,
                'seed': seed
            })

    return all_tasks

def run_gpu_tasks(gpu_id, tasks):
    """在指定GPU上运行一组任务，最多同时运行8个"""
    results = []
    
    # 为每个任务添加GPU ID
    for task in tasks:
        task['gpu'] = gpu_id
    
    # 使用线程池同时运行最多8个任务
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_single_task, task) for task in tasks]
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    
    return results

def run_all_experiments():
    """运行所有实验"""
    # 创建所有任务
    all_tasks = create_all_tasks()
    print(f"总共创建了 {len(all_tasks)} 个任务")

    # 将任务均匀分配到4个GPU
    gpu_tasks = [[] for _ in range(len(gpus))]
    for i, task in enumerate(all_tasks):
        gpu_idx = i % len(gpus)
        gpu_tasks[gpu_idx].append(task)

    # 统计每个GPU的任务数
    for i, tasks in enumerate(gpu_tasks):
        print(f"GPU {gpus[i]} 被分配了 {len(tasks)} 个任务")

    all_results = []

    # 在每个GPU上运行任务
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(run_gpu_tasks, gpus[i], tasks)
            for i, tasks in enumerate(gpu_tasks)
        ]

        # 收集所有结果
        for future in as_completed(futures):
            gpu_results = future.result()
            all_results.extend(gpu_results)

    # 整理结果
    organized_results = {}
    for result in all_results:
        config_type = result['config_type']
        if config_type == 'delta':
            key = f"delta_min_{result['delta_min']}_delta_max_{result['delta_max']}"
        else:  # eps
            key = f"eps_min_{result['eps_min']}_eps_max_{result['eps_max']}"

        if key not in organized_results:
            organized_results[key] = []

        organized_results[key].append(result['result'])

    return organized_results


def summarize_results(all_results):
    """汇总结果并保存到文件"""
    summary_file = f"{BASE_SAVE_PATH}/summary_results.txt"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    with open(summary_file, 'w') as f:
        # Delta实验结果
        f.write("Delta实验结果:\n")
        f.write("-------------------------\n")
        print("\nDelta实验结果:")
        print("-------------------------")

        for delta_min, delta_max in delta_configs:
            key = f"delta_min_{delta_min}_delta_max_{delta_max}"
            if key in all_results and all_results[key]:
                values = all_results[key]
                mean = np.mean(values)
                std = np.std(values)
                result_str = f"delta_min={delta_min}, delta_max={delta_max}: 平均值={mean:.4f}, 标准差={std:.4f}, 值={values}"
                f.write(result_str + "\n")
                print(result_str)

        # Epsilon实验结果
        f.write("\nEpsilon实验结果:\n")
        f.write("--------------------------\n")
        print("\nEpsilon实验结果:")
        print("--------------------------")

        for eps_min, eps_max in eps_configs:
            key = f"eps_min_{eps_min}_eps_max_{eps_max}"
            if key in all_results and all_results[key]:
                values = all_results[key]
                mean = np.mean(values)
                std = np.std(values)
                result_str = f"eps_min={eps_min}, eps_max={eps_max}: 平均值={mean:.4f}, 标准差={std:.4f}, 值={values}"
                f.write(result_str + "\n")
                print(result_str)

    print(f"\n结果汇总已保存到 {summary_file}")


def main():
    # 创建基础目录
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)
    
    print("开始在4个GPU上运行实验...")
    print(f"将运行 {len(delta_configs)} 个delta配置和 {len(eps_configs)} 个epsilon配置")
    print(f"每个配置将使用 {len(seeds)} 个不同的种子")
    print(f"每个GPU将同时运行8个任务")
    
    # 运行所有实验
    results = run_all_experiments()
    
    # 汇总结果
    summarize_results(results)
    
    print("所有实验已完成!")

if __name__ == "__main__":
    main()
