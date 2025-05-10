###
# File: ./script/run_all_experiments.py
# Created Date: Friday, May 9th 2025
# Author: Claude
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import time
import subprocess
import threading
from queue import Queue

# 配置参数
gpus = [0, 1, 2, 3]
lrs = [0.001, 0.005, 0.01, 0.02, 0.05]
num_flips = [10, 20, 30, 40]
seeds = list(range(16))

# 实验配置
experiments = [
    {
        "name": "influence_function",
        "selection": "influence_function",
        "save_path_base": "./results/influence_function_flip",
        "extra_args": ""
    },
    {
        "name": "loo",
        "selection": "loo",
        "save_path_base": "./results/loo_flip",
        "extra_args": ""
    },
    # {
    #     "name": "oti",
    #     "selection": "OTI",
    #     "save_path_base": "./results/oti_flip",
    #     "extra_args": "--oti_mode full"
    # },
    {
        "name": "tracin",
        "selection": "TracIn",
        "save_path_base": "./results/TracIn_flip",
        "extra_args": ""
    }
]

# 基础命令
base_cmd = (
    "/home/jie/DeepCore/.venv/bin/python -m main --dataset MNIST --model LeNet --exp flip "
    "--num_exp 1 --epochs 5 --selection_epochs 5 --data_path ./data "
    "--optimizer SGD --scheduler CosineAnnealingLR --num_gpus 1 --workers 2 "
)

def run_task(task):
    gpu, lr, num_flip, seed, experiment = task
    save_path = f"{experiment['save_path_base']}/flip_nf{num_flip}_lr{lr}_seed{seed}"
    cmd = (
        f"{base_cmd} --selection {experiment['selection']} {experiment['extra_args']} "
        f"--gpu {gpu} --lr {lr} --save_path {save_path} "
        f"--seed {seed} --num_flip {num_flip}"
    )
    print(f"开始任务: {experiment['name']} - GPU {gpu}, LR {lr}, Num Flip {num_flip}, Seed {seed}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"任务失败: {experiment['name']} - GPU {gpu}, LR {lr}, Num Flip {num_flip}, Seed {seed}")
    else:
        print(f"任务完成: {experiment['name']} - GPU {gpu}, LR {lr}, Num Flip {num_flip}, Seed {seed}")

def run_experiment(experiment):
    print(f"\n{'='*80}")
    print(f"开始运行 {experiment['name']} 实验")
    print(f"{'='*80}\n")
    
    # 生成所有任务
    all_tasks = []
    for lr in lrs:
        for num_flip in num_flips:
            for seed in seeds:
                all_tasks.append((None, lr, num_flip, seed, experiment))  # 先不分配 GPU
    
    # 动态分配 GPU
    task_queue = Queue()
    for task in all_tasks:
        task_queue.put(task)
    
    gpu_status = {gpu: None for gpu in gpus}  # None 表示空闲
    
    def worker(gpu_id):
        while not task_queue.empty():
            try:
                _, lr, num_flip, seed, exp = task_queue.get_nowait()
            except:
                break
            task = (gpu_id, lr, num_flip, seed, exp)
            run_task(task)
            task_queue.task_done()
    
    threads = []
    for gpu in gpus:
        t = threading.Thread(target=worker, args=(gpu,))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    print(f"\n{experiment['name']} 实验完成！")

if __name__ == "__main__":
    start_time = time.time()

    # 依次运行每个实验
    for experiment in experiments:
        run_experiment(experiment)

    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "="*80)
    print(f"所有实验完成！总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    print("=" * 80)
