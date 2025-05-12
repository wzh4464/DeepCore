###
# File: ./script/early_detection_experiment.py
# Created Date: Monday, May 12th 2025
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

# 配置参数
gpus = [0, 1, 2, 3]
lrs = [0.05]  # 只用一个学习率
num_flips = [10, 20, 30, 40]
seeds = list(range(8))  # 0-7

# 实验配置
experiments = [
    {
        "name": "grand",
        "selection": "GraNd",
        "save_path_base": "./results/early_detection/grand",
        "extra_args": ""
    },
    {
        "name": "influence_function",
        "selection": "influence_function",
        "save_path_base": "./results/early_detection/influence_function",
        "extra_args": ""
    },
    {
        "name": "oti",
        "selection": "OTI",
        "save_path_base": "./results/early_detection/oti",
        "extra_args": "--oti_mode full"
    }
]

# 基础命令
base_cmd = (
    "/home/jie/DeepCore/.venv/bin/python -m main --dataset MNIST --model LeNet --exp early_detection "
    "--num_exp 1 --num_eval 1 --epochs 5 --selection_epochs 5 --data_path ./data "
    "--optimizer SGD --scheduler CosineAnnealingLR --num_gpus 1 --workers 1 "
)

def run_task(task):
    gpu, lr, num_flip, seed, experiment = task
    save_path = f"{experiment['save_path_base']}/nf{num_flip}_lr{lr}_seed{seed}"
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
    print(f"开始运行 {experiment['name']} 早期检测实验")
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

    num_tasks = len(all_tasks)
    max_threads = num_tasks / len(gpus)

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
        for _ in range(max_threads):
            t = threading.Thread(target=worker, args=(gpu,))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    print(f"\n{experiment['name']} 早期检测实验完成！")

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
    print(f"所有早期检测实验完成！总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    print("=" * 80) 
