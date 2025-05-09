###
# File: ./script/TracIn_flip.py
# Created Date: Friday, May 9th 2025
# Author: Zihan
# -----
# Last Modified: Friday, 9th May 2025 10:16:31 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import time
import subprocess
from multiprocessing import Pool

# 配置参数
gpus = [0, 1, 2, 3]
lrs = [0.001, 0.005, 0.01, 0.02, 0.05]
num_flips = [10, 20, 30, 40]
seeds = list(range(16))

base_cmd = (
    "/home/jie/DeepCore/.venv/bin/python -m main --dataset MNIST --model LeNet --exp flip --selection TracIn "
    "--num_exp 1 --epochs 5 --selection_epochs 5 --data_path ./data "
    "--optimizer SGD --scheduler CosineAnnealingLR --num_gpus 1 --workers 2 "
)

save_path_base = "./results/TracIn_flip"

def run_task(task):
    gpu, lr, num_flip, seed = task
    save_path = f"{save_path_base}/flip_nf{num_flip}_lr{lr}_seed{seed}"
    cmd = (
        f"{base_cmd} --gpu {gpu} --lr {lr} --save_path {save_path} "
        f"--seed {seed} --num_flip {num_flip}"
    )
    print(f"开始任务: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"任务失败: {cmd}")
    else:
        print(f"任务完成: {cmd}")

if __name__ == "__main__":
    # 生成所有任务
    all_tasks = []
    for lr in lrs:
        for num_flip in num_flips:
            for seed in seeds:
                all_tasks.append((None, lr, num_flip, seed))  # 先不分配 GPU

    # 动态分配 GPU
    from queue import Queue
    import threading

    task_queue = Queue()
    for task in all_tasks:
        task_queue.put(task)

    gpu_status = {gpu: None for gpu in gpus}  # None 表示空闲

    def worker(gpu_id):
        while not task_queue.empty():
            try:
                _, lr, num_flip, seed = task_queue.get_nowait()
            except:
                break
            task = (gpu_id, lr, num_flip, seed)
            run_task(task)
            task_queue.task_done()

    threads = []
    for gpu in gpus:
        t = threading.Thread(target=worker, args=(gpu,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("所有任务完成！")
