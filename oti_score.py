###
 # File: /oti_score.py
 # Created Date: Monday, August 19th 2024
 # Author: Zihan
 # -----
 # Last Modified: Monday, 19th August 2024 9:27:08 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

# each GPU handles an epoch

import os
import pickle
import torch
import torch.multiprocessing as mp
import time
from tqdm import tqdm

def load_best_params(save_path):
    """加载最佳参数"""
    file_path = os.path.join(save_path, "best_params.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_initial_params(save_path):
    """加载初始参数"""
    file_path = os.path.join(save_path, "initial_params.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_single_epoch_data(file_path):
    """加载单个epoch的数据"""
    with open(file_path, "rb") as f:
        return pickle.load(f)

def calculate_l2_distance(params1, params2, device):
    """计算两个参数字典之间的L2距离"""
    return sum(torch.norm(params1[name].to(device) - params2[name].to(device)).item() for name in params1 if name in params2)

def gpu_worker(gpu_id, save_path, epochs, return_dict, device):
    """每个GPU负责计算分配到的epoch并汇总结果"""
    best_params = load_best_params(save_path)
    initial_params = load_initial_params(save_path)
    best_params = {k: v.to(device) for k, v in best_params.items()}
    initial_params = {k: v.to(device) for k, v in initial_params.items()}

    local_scores = {}

    for epoch in epochs:
        file_path = os.path.join(save_path, f"epoch_{epoch}_data.pkl")
        epoch_data = load_single_epoch_data(file_path)
        parameters = epoch_data["parameters"]
        data_order = epoch_data["data_order"]

        for i in tqdm(range(len(parameters)), desc=f"GPU {gpu_id} - Epoch {epoch}", position=gpu_id):
            data_idx = data_order[i]

            if i == 0 and epoch == min(epochs):
                prev_params = initial_params
            elif i == 0:
                prev_params = {k: v.to(device) for k, v in load_single_epoch_data(os.path.join(save_path, f"epoch_{epoch-1}_data.pkl"))["parameters"][-1]["params"].items()}
            else:
                prev_params = {k: v.to(device) for k, v in parameters[i-1]["params"].items()}

            current_params = {k: v.to(device) for k, v in parameters[i]["params"].items()}

            prev_distance = calculate_l2_distance(prev_params, best_params, device)
            current_distance = calculate_l2_distance(current_params, best_params, device)

            score = prev_distance - current_distance

            if data_idx not in local_scores:
                local_scores[data_idx] = score
            else:
                local_scores[data_idx] += score

        # 清理显存
        torch.cuda.empty_cache()

    return_dict[gpu_id] = local_scores

def main(save_path, num_epochs, num_gpus=3):
    # 配置共享字典用于存储各GPU计算的结果
    manager = mp.Manager()
    return_dict = manager.dict()

    # 分配每个GPU需要处理的epoch
    epochs_per_gpu = [list(range(i, num_epochs, num_gpus)) for i in range(num_gpus)]

    processes = []
    for gpu_id in range(num_gpus):
        device = torch.device(f"cuda:{gpu_id}")
        p = mp.Process(target=gpu_worker, args=(gpu_id, save_path, epochs_per_gpu[gpu_id], return_dict, device))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 汇总所有GPU的分数
    total_scores = {}
    for gpu_scores in return_dict.values():
        for idx, score in gpu_scores.items():
            if idx not in total_scores:
                total_scores[idx] = score
            else:
                total_scores[idx] += score

    # 保存最终得分
    with open(os.path.join(save_path, "oti_scores.pkl"), "wb") as f:
        pickle.dump(total_scores, f)
    print(f"Scores saved to {os.path.join(save_path, 'oti_scores.pkl')}")

if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间

    save_path = "results"  # 更新为实际的保存路径
    num_epochs = 5  # 设置实际的epoch数量
    num_gpus = 3  # 设置使用的GPU数量

    main(save_path, num_epochs, num_gpus)

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时间

    print(f"Total time taken: {total_time:.2f} seconds")
