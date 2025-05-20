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
import argparse
from queue import Queue

def parse_args():
    parser = argparse.ArgumentParser(description='早期检测实验')
    parser.add_argument('--gpus', type=str, default='0,1,2',
                      help='要使用的GPU ID列表，用逗号分隔，例如 "0,1,2" 或 "0,3"')
    parser.add_argument('--lrs', type=str, default='0.0005',
                      help='学习率列表，用逗号分隔，例如 "0.0005" 或 "0.0001,0.0005"')
    parser.add_argument('--num_flips', type=str, default='10,20,30,40',
                      help='翻转数量列表，用逗号分隔，例如 "10,20,30,40"')
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4,5,6,7',
                      help='随机种子列表，用逗号分隔，例如 "0,1,2,3,4,5,6,7"')
    return parser.parse_args()

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
    "python -m main --dataset MNIST --model LeNet --exp early_detection "
    "--num_exp 1 --num_eval 1 --data_path ./data "
    "--optimizer SGD --scheduler CosineAnnealingLR --num_gpus 1 --workers 1 "
)

def run_task(task):
    gpu, lr, num_flip, seed, experiment = task
    save_path = f"{experiment['save_path_base']}/nf{num_flip}_lr{lr}_seed{seed}"
    cmd = (
        f"{base_cmd} --selection {experiment['selection']} {experiment['extra_args']} "
        f"--gpu {gpu} --lr {lr} --selection_lr {lr} --save_path {save_path} "
        f"--seed {seed} --num_flip {num_flip}"
    )
    print(f"开始任务: {experiment['name']} - GPU {gpu}, LR {lr}, Num Flip {num_flip}, Seed {seed}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"任务失败: {experiment['name']} - GPU {gpu}, LR {lr}, Num Flip {num_flip}, Seed {seed}")
    else:
        print(f"任务完成: {experiment['name']} - GPU {gpu}, LR {lr}, Num Flip {num_flip}, Seed {seed}")

def run_experiment(experiment, gpus, lrs, num_flips, seeds):
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
    max_threads = num_tasks // len(gpus)

    def worker(gpu_id):
        while not task_queue.empty():
            try:
                _, lr, num_flip, seed, exp = task_queue.get_nowait()
            except Exception:
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
    args = parse_args()
    
    # 解析命令行参数
    gpus = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    lrs = [float(lr.strip()) for lr in args.lrs.split(',')]
    num_flips = [int(nf.strip()) for nf in args.num_flips.split(',')]
    seeds = [int(seed.strip()) for seed in args.seeds.split(',')]
    
    print(f"使用 GPU: {gpus}")
    print(f"学习率: {lrs}")
    print(f"翻转数量: {num_flips}")
    print(f"随机种子: {seeds}")
    
    start_time = time.time()

    # 依次运行每个实验
    for experiment in experiments:
        run_experiment(experiment, gpus, lrs, num_flips, seeds)

    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "="*80)
    print(f"所有早期检测实验完成！总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    print("=" * 80) 
