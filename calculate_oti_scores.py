###
 # File: /calculate_oti_scores.py
 # Created Date: Friday, August 16th 2024
 # Author: Zihan
 # -----
 # Last Modified: Monday, 19th August 2024 9:27:26 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

# each GPU handles a part of an epoch

import os
import pickle
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time

def load_best_params(save_path):
    file_path = os.path.join(save_path, "best_params.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_initial_params(save_path):
    file_path = os.path.join(save_path, "initial_params.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_epoch_data_to_memory(save_path, epoch):
    """加载指定epoch的数据到CPU内存"""
    file_path = os.path.join(save_path, f"epoch_{epoch}_data.pkl")
    with open(file_path, "rb") as f:
        epoch_data = pickle.load(f)
    return epoch_data

def split_data_across_gpus(epoch_data, num_gpus):
    """将epoch数据分割成num_gpus份"""
    parameters = epoch_data["parameters"]
    data_order = epoch_data["data_order"]

    # 按照GPU数量分割数据
    chunk_size = len(parameters) // num_gpus
    splits = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_gpus - 1 else len(parameters)
        splits.append({
            "parameters": parameters[start_idx:end_idx],
            "data_order": data_order[start_idx:end_idx]
        })
    return splits

def calculate_l2_distance(params1, params2):
    """计算两个参数字典之间的L2距离"""
    return sum(torch.norm(params1[name] - params2[name]).item() for name in params1 if name in params2)

def gpu_worker(gpu_id, split_data, initial_params, best_params, return_dict, epoch):
    """每个GPU计算得分并汇总"""
    device = torch.device(f"cuda:{gpu_id}")
    best_params = {k: v.to(device) for k, v in best_params.items()}
    initial_params = {k: v.to(device) for k, v in initial_params.items()}

    local_scores = {}

    for i in tqdm(range(len(split_data["parameters"])), desc=f"GPU {gpu_id} - Epoch {epoch}", position=gpu_id):
        data_idx = split_data["data_order"][i]

        if i == 0:
            prev_params = {k: v.to(device) for k, v in initial_params.items()}
        else:
            prev_params = {k: v.to(device) for k, v in split_data["parameters"][i - 1]["params"].items()}

        current_params = {k: v.to(device) for k, v in split_data["parameters"][i]["params"].items()}

        prev_distance = calculate_l2_distance(prev_params, best_params)
        current_distance = calculate_l2_distance(current_params, best_params)

        score = prev_distance - current_distance

        if data_idx not in local_scores:
            local_scores[data_idx] = score
        else:
            local_scores[data_idx] += score

    # 将结果存储在共享字典中
    return_dict[gpu_id] = local_scores

def main(save_path, num_epochs, num_gpus=3):
    # 配置共享字典
    manager = mp.Manager()
    return_dict = manager.dict()

    best_params = load_best_params(save_path)
    initial_params = load_initial_params(save_path)

    total_scores = {}

    for epoch in range(num_epochs):
        # 1. 读取当前epoch的数据到CPU内存
        epoch_data = load_epoch_data_to_memory(save_path, epoch)
        
        # 2. 将数据分配到不同GPU
        splits = split_data_across_gpus(epoch_data, num_gpus)

        # 3. 启动多个进程，每个进程对应一个GPU进行计算
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(target=gpu_worker, args=(gpu_id, splits[gpu_id], initial_params, best_params, return_dict, epoch))
            processes.append(p)
            p.start()

        # 等待所有进程完成
        for p in processes:
            p.join()

        # 4. 汇总各个GPU计算得到的分数
        for gpu_scores in return_dict.values():
            for idx, score in gpu_scores.items():
                if idx not in total_scores:
                    total_scores[idx] = score
                else:
                    total_scores[idx] += score

        # 5. 更新initial_params为本epoch最后的参数
        initial_params = epoch_data["parameters"][-1]["params"]

        # 清理内存和显存
        torch.cuda.empty_cache()

    # 保存最终得分
    with open(os.path.join(save_path, "oti_scores.pkl"), "wb") as f:
        pickle.dump(total_scores, f)
    print(f"Scores saved to {os.path.join(save_path, 'oti_scores.pkl')}")

if __name__ == "__main__":
    begin = time.time()
    save_path = "results"  # 更新为实际的保存路径
    num_epochs = 5  # 设置实际的epoch数量
    main(save_path, num_epochs)
    end = time.time()
    print(f"Time elapsed: {end - begin} seconds")
