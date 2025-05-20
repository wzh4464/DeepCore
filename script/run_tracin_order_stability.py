###
# File: ./script/run_tracin_order_stability.py
# Created Date: [Current Date]
# Author: [Your Name/Gemini]
# -----
# HISTORY:
# Date      \t\tBy   \tComments
# ----------\t\t------\t---------------------------------------------------------
###

import os
import time
import subprocess
import threading
from queue import Queue
import itertools

# ======================================================================================
# 实验配置区域
# ======================================================================================

# 设备类型配置: 'cuda' 或 'mps'
# 如果使用 'cuda', gpus 列表中的 ID 将被使用。
# 如果使用 'mps', gpus 列表将被忽略，所有 MPS 任务在单个 MPS 设备上运行 (可以并发多个任务)。
# device_types = ['cuda']  # 例如: ['cuda', 'mps'] 或 ['mps']
device_types = ['mps']

# GPU列表 (仅当 device_types 包含 'cuda' 时相关)
gpus = [0, 1]  # 根据可用CUDA GPU进行修改

# 数据集和模型配置
datasets_models = [
    {"dataset": "MNIST", "model": "LeNet"},
    # {"dataset": "CIFAR10", "model": "ResNet18"}, # 可以添加更多组合
]

# 随机种子
seeds = [42, 123] # 可以根据需要添加更多种子

# TracIn 选择阶段的 Epochs
selection_epochs_list = [1,2,3,4,5]

# TracIn 使用的测试样本数量
num_scores_list = [100] # 通常固定，除非有特定测试需求

# 训练批次大小
batch_sizes = [256]

# 基础保存路径
base_save_dir = "./results/tracin_order_stability"

# Python解释器路径 (如果不在系统PATH或特定venv)
# python_executable = "/home/jie/DeepCore/.venv/bin/python" # 与 early_detection_experiment.py 保持一致
python_executable = "python" # 假设在正确的环境中运行，或python在PATH中

# 并发MPS任务的数量 (如果 device_types 包含 'mps')
# 注意：这仍然在单个MPS设备上运行，但允许同时启动多个Python进程进行实验。
# 过多并发可能导致资源竞争，根据机器性能调整。
max_concurrent_mps_tasks = 1

# ======================================================================================
# 基础命令构建
# ======================================================================================
# 注意: --exp tracin_order 表明我们要运行的是修改后的特定顺序稳定性实验
# 其他通用参数可以放在这里
base_cmd_template = (
    f"{python_executable} -m main --exp tracin_order "
    # "--num_gpus 1 " # 这个参数的含义可能需要根据main.py的修改来确定
    "--workers 2 " 
    "--data_path ./data " # 假设数据在 ./data 目录下
)

# ======================================================================================
# 任务执行函数
# ======================================================================================
def run_task(task_params):
    (
        device_type, # 'cuda' or 'mps'
        gpu_id_or_none, # cuda_id or None for mps/cpu
        dataset_config,
        seed,
        selection_epochs,
        num_scores,
        batch_size,
    ) = task_params

    dataset_name = dataset_config["dataset"]
    model_name = dataset_config["model"]

    # 构建保存路径，加入设备类型
    specific_save_path = os.path.join(
        base_save_dir,
        f"{dataset_name}_{model_name}_{device_type}", # 使用 device_type 而不是 device_str
        f"seed{seed}_e{selection_epochs}_b{batch_size}_s{num_scores}",
    )
    os.makedirs(specific_save_path, exist_ok=True)

    # 构建特定于设备的参数
    device_arg_str = f"--device {device_type}"
    if device_type == 'cuda' and gpu_id_or_none is not None:
        # 只有当是cuda设备且gpu_id有效时，才添加 --gpu 参数
        # 假设 main.py 的 --gpu 参数用于指定单个CUDA ID，如果支持多个，main.py的解析需要对应调整
        device_arg_str += f" --gpu {gpu_id_or_none}"
    # 对于 MPS 或 CPU，不传递 --gpu 参数，或者 main.py 应忽略它

    # 构建完整命令
    cmd = (
        f"{base_cmd_template} "
        f"--dataset {dataset_name} "
        f"--model {model_name} "
        f"--seed {seed} "
        f"--selection_epochs {selection_epochs} "
        f"--num_scores {num_scores} "
        f"--batch {batch_size} "
        f"--save_path {specific_save_path} "
        f"{device_arg_str} " # 替换旧的 --gpu device_str
    )

    actual_device_log = f"{device_type}:{gpu_id_or_none}" if device_type == 'cuda' else device_type
    log_prefix = f"Task (Device: {actual_device_log}, Dataset: {dataset_name}, Model: {model_name}, Seed: {seed}, Epochs: {selection_epochs}, Batch: {batch_size})"
    print(f"{log_prefix} | 开始. CMD: {cmd}")

    try:
        process = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"{log_prefix} | 完成. STDOUT:\n{process.stdout[-500:]}") # 打印最后500字符的stdout
        if process.stderr:
            print(f"{log_prefix} | STDERR:\n{process.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"{log_prefix} | 失败. Error Code: {e.returncode}")
        print(f"{log_prefix} | STDOUT:\n{e.stdout}")
        print(f"{log_prefix} | STDERR:\n{e.stderr}")
    except Exception as e:
        print(f"{log_prefix} | 执行时发生意外错误: {e}")

# ======================================================================================
# 主逻辑：任务生成与分配
# ======================================================================================
def main():
    print("=" * 80)
    print("开始运行 TracIn 顺序稳定性批量实验")
    print("=" * 80 + "\n")

    all_task_configs = []

    for device_type_config in device_types:
        for dataset_config in datasets_models:
            for seed_val in seeds:
                for sel_epochs in selection_epochs_list:
                    for n_scores in num_scores_list:
                        for b_size in batch_sizes:
                            if device_type_config == 'cuda':
                                for gpu_id in gpus:
                                    all_task_configs.append((
                                        device_type_config, # 'cuda'
                                        gpu_id,             # actual gpu_id
                                        dataset_config,
                                        seed_val,
                                        sel_epochs,
                                        n_scores,
                                        b_size,
                                    ))
                            elif device_type_config == 'mps':
                                all_task_configs.append((
                                    device_type_config, # 'mps'
                                    None,               # No specific gpu_id for mps
                                    dataset_config,
                                    seed_val,
                                    sel_epochs,
                                    n_scores,
                                    b_size,
                                ))
                            elif device_type_config == 'cpu': # 如果也想支持CPU
                                all_task_configs.append((
                                    device_type_config, # 'cpu'
                                    None,               # No specific gpu_id for cpu
                                    dataset_config,
                                    seed_val,
                                    sel_epochs,
                                    n_scores,
                                    b_size,
                                ))
                            else:
                                print(f"不支持的设备类型配置: {device_type_config}")

    if not all_task_configs:
        print("没有生成任何实验任务，请检查配置。")
        return

    print(f"总共生成 {len(all_task_configs)} 个实验任务配置。\n")

    task_queue = Queue()
    for config in all_task_configs:
        task_queue.put(config)

    threads = []
    num_worker_threads = 0

    # 根据配置的设备类型和数量确定worker数量
    # 对于CUDA，每个指定的GPU一个worker
    # 对于MPS，使用 max_concurrent_mps_tasks 作为worker数量
    # 对于CPU (如果添加)，可以类似MPS处理或指定一个并发数
    
    # 简单处理：总worker数是 （CUDA GPU数）+ （MPS并发数 if mps in types）
    if 'cuda' in device_types and gpus:
        num_worker_threads += len(gpus)
    if 'mps' in device_types:
        num_worker_threads = max(num_worker_threads, max_concurrent_mps_tasks) # 允许多个MPS任务，但不超过CUDA workers
        # 或者，如果希望MPS任务独立于CUDA worker计数：
        # num_worker_threads = 0 # 重置
        # if 'cuda' in device_types and gpus: num_worker_threads += len(gpus)
        # if 'mps' in device_types: num_worker_threads += max_concurrent_mps_tasks
    if 'cpu' in device_types: # 示例: 假设CPU任务也是并发的
        num_worker_threads = max(num_worker_threads, 1) # 至少一个worker给CPU, 或配置并发数

    if num_worker_threads == 0:
        # 如果队列中有任务，但没有配置worker，则默认至少一个worker顺序执行
        if not task_queue.empty():
            print("警告: 未明确配置worker线程数 (检查 GPUs, MPS, CPU 配置)，但队列中有任务。将使用1个worker顺序执行。")
            num_worker_threads = 1
        else:
            print("没有配置有效的worker线程，也没有任务在队列中。")
            return

    print(f"将启动 {num_worker_threads} 个 worker 线程.\n")

    def worker_thread_fn():
        while not task_queue.empty():
            try:
                task_config = task_queue.get_nowait()
                run_task(task_config) 
                task_queue.task_done()
            except Queue.Empty:
                break 
            except Exception as e:
                print(f"Worker线程出错: {e}")
                if not task_queue.empty():
                    try: 
                        task_queue.task_done() 
                    except ValueError: 
                        pass 
                break
    
    for _ in range(num_worker_threads):
        t = threading.Thread(target=worker_thread_fn)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\n" + "=" * 80)
    print("所有 TracIn 顺序稳定性实验完成！")
    print("=" * 80)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒") 
