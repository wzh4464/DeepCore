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
# 如果使用 'cuda', gpus 列表中的 ID 将被使用
# 如果使用 'mps', gpus 列表将被忽略，所有 MPS 任务在单个 MPS 设备上运行
device_types = ['cuda']  # 例如: ['cuda', 'mps'] 或 ['mps']

# GPU列表 (仅当 device_types 包含 'cuda' 时相关)
gpus = [0, 1, 2, 3]  # 使用4个GPU

# 数据集和模型配置
datasets_models = [
    {"dataset": "MNIST", "model": "LeNet"},
    # {"dataset": "CIFAR10", "model": "ResNet18"}, # 可以添加更多组合
]

# 随机种子 - 使用4个不同的种子
seeds = [42, 43, 44, 45]

# TracIn 选择阶段的 Epochs
selection_epochs_list = [1,3,5]

# TracIn 使用的测试样本数量
num_scores_list = [100] # 通常固定，除非有特定测试需求

# 训练批次大小
batch_sizes = [256]

# 基础保存路径
base_save_dir = "./results/tracin_order_stability"

# Python解释器路径 (如果不在系统PATH或特定venv)
python_executable = "python" # 假设在正确的环境中运行，或python在PATH中

# 并发MPS任务的数量 (如果 device_types 包含 'mps')
max_concurrent_mps_tasks = 1

# ======================================================================================
# 基础命令构建
# ======================================================================================
# 注意: --exp tracin_order 表明我们要运行的是修改后的特定顺序稳定性实验
base_cmd_template = (
    f"{python_executable} -m main --exp tracin_order "
    "--selection TracIn "
    "--workers 16 " # 每个GPU使用16个CPU
    "--data_path ./data " # 假设数据在 ./data 目录下
    "--fraction 0.1 "     # 必须的参数，虽然在此实验中不直接使用
)

# ======================================================================================
# 任务执行函数
# ======================================================================================
def run_task(task_params):
    (
        device_type,      # 'cuda' or 'mps'
        gpu_id_or_none,   # cuda_id or None for mps/cpu
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
        f"{dataset_name}_{model_name}_{device_type}",
        f"seed{seed}_e{selection_epochs}_b{batch_size}_s{num_scores}",
    )
    os.makedirs(specific_save_path, exist_ok=True)

    # 构建特定于设备的参数
    device_arg_str = f"--device {device_type}"
    if device_type == 'cuda' and gpu_id_or_none is not None:
        device_arg_str += f" --gpu {gpu_id_or_none}"

    # 构建完整命令
    cmd = (
        f"{base_cmd_template} "
        f"--dataset {dataset_name} "
        f"--model {model_name} "
        f"--seed {seed} "
        f"--selection_epochs {selection_epochs} "
        f"--num_scores {num_scores} " # 这里传入用于计算TracIn分数的测试样本数量
        f"--batch {batch_size} "
        f"--save_path {specific_save_path} "
        f"{device_arg_str} "
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
    print("开始运行 TracIn 顺序稳定性批量实验（修改版）")
    print("实验说明: 从训练集中选出200个点, 只计算选出的这200个点的TracIn score, 然后再从中选出40个点")
    print("分别将这40个点放在第一个epoch和最后一个epoch, 然后计算这40个点在200点中score的分布情况")
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
    if 'cuda' in device_types and gpus:
        num_worker_threads += len(gpus)
    if 'mps' in device_types:
        num_worker_threads = max(num_worker_threads, max_concurrent_mps_tasks)
    if 'cpu' in device_types:
        num_worker_threads = max(num_worker_threads, 1)

    if num_worker_threads == 0:
        # 如果队列中有任务，但没有配置worker，则默认至少一个worker顺序执行
        if not task_queue.empty():
            print("警告: 未明确配置worker线程数，但队列中有任务。将使用1个worker顺序执行。")
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
