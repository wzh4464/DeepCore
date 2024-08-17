import pickle
import torch
import os
import numpy as np

def load_epoch_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def find_data_index(data_order, target_index):
    return np.where(np.array(data_order) == target_index)[0][0]

def get_params_for_data_point(epoch_data, target_index):
    data_order = epoch_data['data_order']
    parameters = epoch_data['parameters']
    
    step = find_data_index(data_order, target_index)
    
    params_after = parameters[step]['params']
    
    if step == 0:
        print("Warning: This is the first step in this epoch, there are no 'before' parameters.")
        params_before = None
    else:
        params_before = parameters[step - 1]['params']
    
    return params_before, params_after, step

def compare_params(params_before, params_after):
    if params_before is None:
        print("Cannot compare parameters: 'before' parameters are not available.")
        return

    print("\nParameter differences:")
    for key in params_before.keys():
        if torch.is_tensor(params_before[key]) and torch.is_tensor(params_after[key]):
            diff = torch.norm(params_after[key] - params_before[key]).item()
            print(f"{key}: L2 norm of difference = {diff}")
        else:
            print(f"{key}: Unable to compute difference (non-tensor data)")

def main():
    file_path = 'results/epoch_0_data_0.pkl'
    target_index = 99  # Python uses 0-based indexing, so 100th data point is at index 99

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        epoch_data = load_epoch_data(file_path)
        params_before, params_after, step = get_params_for_data_point(epoch_data, target_index)

        print(f"Data point {target_index + 1} was processed at step {step + 1} in this epoch")
        
        if params_before:
            print("\nParameters before update:")
            for key, value in params_before.items():
                print(f"{key}: {type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
        else:
            print("\nNo 'before' parameters available (this is the first step in this epoch)")

        print("\nParameters after update:")
        for key, value in params_after.items():
            print(f"{key}: {type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")

        compare_params(params_before, params_after)

        # Calculate and print the score for this data point
        if params_before:
            score = sum(torch.norm(params_after[key] - params_before[key]).item() 
                        for key in params_before.keys() 
                        if torch.is_tensor(params_before[key]) and torch.is_tensor(params_after[key]))
            print(f"\nScore for data point {target_index + 1}: {score}")
        else:
            print("\nCannot calculate score: 'before' parameters are not available.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
