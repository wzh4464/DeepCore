import os
import pickle
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import argparse

def load_best_params(save_path):
    """Load the best model parameters from a file."""
    file_path = os.path.join(save_path, "best_params.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_initial_params(save_path):
    """Load the initial model parameters from a file."""
    file_path = os.path.join(save_path, "initial_params.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_epoch_data_to_memory(save_path, epoch):
    """Load data for a specific epoch into CPU memory."""
    file_path = os.path.join(save_path, f"epoch_{epoch}_data.pkl")
    with open(file_path, "rb") as f:
        epoch_data = pickle.load(f)
    return epoch_data

def split_data_across_gpus(epoch_data, num_gpus):
    """Split epoch data into chunks for each GPU."""
    parameters = epoch_data["parameters"]
    data_order = epoch_data["data_order"]

    # Split data according to the number of GPUs
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
    """Calculate the L2 distance between two sets of parameters."""
    return sum(torch.norm(params1[name] - params2[name]).item() for name in params1 if name in params2)

def gpu_worker(gpu_id, split_data, initial_params, best_params, return_dict, epoch):
    """Each GPU processes its assigned data and computes the scores."""
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

    # Store the results in the shared dictionary
    return_dict[gpu_id] = local_scores

def main(save_path, num_epochs, num_gpus, mode):
    # Setup shared dictionaries for inter-process communication
    manager = mp.Manager()
    return_dict = manager.dict()

    best_params = load_best_params(save_path)
    initial_params = load_initial_params(save_path)

    total_scores = {}

    for epoch in range(num_epochs):
        if mode == "gpu_split":
            # GPU split mode: Split epoch data across GPUs
            epoch_data = load_epoch_data_to_memory(save_path, epoch)
            splits = split_data_across_gpus(epoch_data, num_gpus)
            processes = []
            for gpu_id in range(num_gpus):
                p = mp.Process(target=gpu_worker, args=(gpu_id, splits[gpu_id], initial_params, best_params, return_dict, epoch))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            for gpu_scores in return_dict.values():
                for idx, score in gpu_scores.items():
                    if idx not in total_scores:
                        total_scores[idx] = score
                    else:
                        total_scores[idx] += score

            # Update initial_params with the last parameters from this epoch
            initial_params = epoch_data["parameters"][-1]["params"]

        elif mode == "epoch_split":
            # Epoch split mode: Assign different epochs to different GPUs
            epochs_per_gpu = [list(range(i, num_epochs, num_gpus)) for i in range(num_gpus)]
            processes = []
            for gpu_id in range(num_gpus):
                p = mp.Process(target=gpu_worker_epoch_split, args=(gpu_id, save_path, epochs_per_gpu[gpu_id], return_dict, initial_params, best_params))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            for gpu_scores in return_dict.values():
                for idx, score in gpu_scores.items():
                    if idx not in total_scores:
                        total_scores[idx] = score
                    else:
                        total_scores[idx] += score

        # Clean up memory and GPU cache
        torch.cuda.empty_cache()

    # Save final scores
    with open(os.path.join(save_path, "oti_scores.pkl"), "wb") as f:
        pickle.dump(total_scores, f)
    print(f"Scores saved to {os.path.join(save_path, 'oti_scores.pkl')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate OTI scores with different GPU processing modes.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to saved model parameters.")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs to process.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--mode", type=str, choices=["gpu_split", "epoch_split"], required=True, help="Mode of processing: 'gpu_split' or 'epoch_split'.")

    args = parser.parse_args()
    
    start_time = time.time()
    main(args.save_path, args.num_epochs, args.num_gpus, args.mode)
    end_time = time.time()

    print(f"Total time taken: {end_time - start_time:.2f} seconds")

# Example of how to run this script:
# python calculate_oti_scores.py --save_path /path/to/data --num_epochs 10 --num_gpus 3 --mode gpu_split
#
# Explanation:
# --save_path: Directory containing the saved model parameters.
# --num_epochs: The number of epochs to process (e.g., 10).
# --num_gpus: Number of GPUs to use (e.g., 3).
# --mode: The processing mode. "gpu_split" distributes data within a single epoch across GPUs, while "epoch_split" assigns different epochs to different GPUs.
