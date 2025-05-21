###
# File: ./script/exp_heatmap.py
# Created Date: Wednesday, May 21st 2025
# Author: Zihan
# -----
# Last Modified: Wednesday, 21st May 2025 7:37:00 pm
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
BASE_SAVE_PATH = "./results/experiment_runs"

# Define the parameter configurations for each experiment type
# Experiment 1: Varying delta_min and delta_max
delta_configs = [
    # (delta_min, delta_max)
    (4, 50),
    (6, 50),
    (8, 50),
    (10, 50),
    (12, 50),
    (8, 40),
    (8, 60),
    (8, 70)
]

# Experiment 2: Varying eps_min and eps_max
eps_configs = [
    # (eps_min, eps_max)
    (0.005, 0.1),
    (0.01, 0.1),
    (0.02, 0.1),
    (0.03, 0.1),
    (0.01, 0.05),
    (0.01, 0.15),
    (0.01, 0.2)
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
    "--delta_0 20 "
    "--delta_step 2 "
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
        return True, result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        print(f"stderr: {e.stderr.decode()}")
        return False, None

def read_result(save_path):
    """Read the ratio value from the result.txt file."""
    result_file = os.path.join(save_path, "result.txt")
    try:
        with open(result_file, 'r') as f:
            ratio = float(f.read().strip())
        return ratio
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading result from {result_file}: {e}")
        return None

def run_all_seeds_for_config(gpu, config):
    """Run all 8 seeds for a given configuration on a single GPU simultaneously."""
    config_type = config['type']
    delta_min = config['delta_min']
    delta_max = config['delta_max']
    eps_min = config['eps_min']
    eps_max = config['eps_max']
    
    # Create a unique identifier for this configuration
    if config_type == 'delta':
        config_id = f"delta_min_{delta_min}_delta_max_{delta_max}"
    else:  # eps
        config_id = f"eps_min_{eps_min}_eps_max_{eps_max}"
    
    print(f"GPU {gpu}: Starting all seeds for {config_id}")
    
    # Start all 8 seed processes simultaneously
    processes = []
    for seed in seeds:
        if config_type == 'delta':
            save_path = f"{BASE_SAVE_PATH}/delta_exp/delta_min_{delta_min}_delta_max_{delta_max}/seed_{seed}"
        else:  # eps
            save_path = f"{BASE_SAVE_PATH}/eps_exp/eps_min_{eps_min}_eps_max_{eps_max}/seed_{seed}"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        cmd = base_cmd_template.format(
            gpu=gpu,
            save_path=save_path,
            delta_min=delta_min,
            delta_max=delta_max,
            eps_min=eps_min,
            eps_max=eps_max,
            seed=seed
        )
        
        # Launch process in background
        print(f"Starting: {cmd}")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((seed, process))
    
    # Wait for all processes to complete and collect results
    results = []
    for seed, process in processes:
        stdout, stderr = process.communicate()
        exit_code = process.returncode
        
        if exit_code == 0:
            if config_type == 'delta':
                save_path = f"{BASE_SAVE_PATH}/delta_exp/delta_min_{delta_min}_delta_max_{delta_max}/seed_{seed}"
            else:  # eps
                save_path = f"{BASE_SAVE_PATH}/eps_exp/eps_min_{eps_min}_eps_max_{eps_max}/seed_{seed}"
                
            result = read_result(save_path)
            if result is not None:
                results.append((seed, result))
                print(f"GPU {gpu}, {config_id}, seed {seed}: ratio = {result}")
        else:
            print(f"Process for seed {seed} failed with exit code {exit_code}")
            print(f"stderr: {stderr.decode()}")
    
    print(f"GPU {gpu}: Completed all seeds for {config_id}")
    return config_id, results

def distribute_configs_to_gpus():
    """Distribute configurations across GPUs."""
    # Combine all configurations
    all_configs = []

    all_configs.extend(
        {
            'type': 'delta',
            'delta_min': delta_min,
            'delta_max': delta_max,
            'eps_min': 0.01,  # Default
            'eps_max': 0.1,  # Default
        }
        for delta_min, delta_max in delta_configs
    )
    all_configs.extend(
        {
            'type': 'eps',
            'delta_min': 8,  # Default
            'delta_max': 50,  # Default
            'eps_min': eps_min,
            'eps_max': eps_max,
        }
        for eps_min, eps_max in eps_configs
    )
    # Distribute configurations to GPUs
    gpu_assignments = {gpu: [] for gpu in gpus}
    for i, config in enumerate(all_configs):
        gpu_idx = i % len(gpus)
        gpu_assignments[gpus[gpu_idx]].append(config)

    return gpu_assignments

def run_all_experiments():
    """Run all experiments for all configurations."""
    # Get configuration assignments for each GPU
    gpu_assignments = distribute_configs_to_gpus()

    all_results = {}

    # Run configurations on each GPU
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []

        for gpu, configs in gpu_assignments.items():
            futures.extend(
                executor.submit(run_all_seeds_for_config, gpu, config)
                for config in configs
            )
        # Collect results as they complete
        for future in as_completed(futures):
            config_id, seed_results = future.result()
            # Extract just the ratio values
            ratios = [r[1] for r in seed_results]
            all_results[config_id] = ratios

    return all_results

def summarize_results(all_results):
    """Summarize the results and save to file."""
    summary_file = f"{BASE_SAVE_PATH}/summary_results.txt"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    with open(summary_file, 'w') as f:
        # Delta experiment results
        f.write("Delta Experiment Results:\n")
        f.write("-------------------------\n")
        print("\nDelta Experiment Results:")
        print("-------------------------")
        
        for delta_min, delta_max in delta_configs:
            key = f"delta_min_{delta_min}_delta_max_{delta_max}"
            if key in all_results and all_results[key]:
                values = all_results[key]
                mean = np.mean(values)
                std = np.std(values)
                result_str = f"delta_min={delta_min}, delta_max={delta_max}: Mean={mean:.4f}, Std={std:.4f}, Values={values}"
                f.write(result_str + "\n")
                print(result_str)
        
        # Epsilon experiment results
        f.write("\nEpsilon Experiment Results:\n")
        f.write("--------------------------\n")
        print("\nEpsilon Experiment Results:")
        print("--------------------------")
        
        for eps_min, eps_max in eps_configs:
            key = f"eps_min_{eps_min}_eps_max_{eps_max}"
            if key in all_results and all_results[key]:
                values = all_results[key]
                mean = np.mean(values)
                std = np.std(values)
                result_str = f"eps_min={eps_min}, eps_max={eps_max}: Mean={mean:.4f}, Std={std:.4f}, Values={values}"
                f.write(result_str + "\n")
                print(result_str)
    
    print(f"\nResults summary saved to {summary_file}")

def main():
    # Create the base directory if it doesn't exist
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)
    
    print("Starting experiments across 4 GPUs...")
    print(f"Will run {len(delta_configs)} delta configurations and {len(eps_configs)} epsilon configurations")
    print(f"Each configuration will be tested with {len(seeds)} seeds running simultaneously on one GPU")
    
    # Run all experiments
    results = run_all_experiments()
    
    # Summarize results
    summarize_results(results)
    
    print("All experiments completed!")

if __name__ == "__main__":
    main()
