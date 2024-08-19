import os
import pickle
import torch

def load_epoch_data(save_path, epoch):
    file_path = os.path.join(save_path, f"epoch_{epoch}_data.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_best_params(save_path):
    file_path = os.path.join(save_path, "best_params.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_initial_params(save_path):
    file_path = os.path.join(save_path, "initial_params.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def calculate_l2_distance(params1, params2):
    return sum(torch.norm(params1[name] - params2[name]).item() for name in params1 if name in params2)

def calculate_scores(save_path, num_epochs):
    best_params = load_best_params(save_path)
    initial_params = load_initial_params(save_path)
    scores = {}

    for epoch in range(num_epochs):
        epoch_data = load_epoch_data(save_path, epoch)
        epoch_parameters = epoch_data["parameters"]
        data_order = epoch_data["data_order"]

        for i in range(len(epoch_parameters)):
            data_idx = data_order[i]
            
            if i == 0 and epoch == 0:
                # For the first data point in the first epoch, use initial parameters as previous
                prev_params = initial_params
            elif i == 0:
                # For the first data point in other epochs, use the last parameters from the previous epoch
                prev_params = load_epoch_data(save_path, epoch-1)["parameters"][-1]["params"]
            else:
                prev_params = epoch_parameters[i-1]["params"]
            
            current_params = epoch_parameters[i]["params"]

            prev_distance = calculate_l2_distance(prev_params, best_params)
            current_distance = calculate_l2_distance(current_params, best_params)

            score = prev_distance - current_distance

            if data_idx not in scores:
                scores[data_idx] = score
            else:
                scores[data_idx] += score

    return scores

if __name__ == "__main__":
    save_path = "results"  # Update this to your actual save path
    num_epochs = 5  # Update this to your actual number of epochs

    scores = calculate_scores(save_path, num_epochs)

    # Sort scores and get indices of top k elements
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Print top 10 scores as an example
    print("Top 10 scores:")
    for idx, score in sorted_scores[:10]:
        print(f"Data index: {idx}, Score: {score}")

    # Save scores to a file
    with open(os.path.join(save_path, "oti_scores.pkl"), "wb") as f:
        pickle.dump(scores, f)
    print(f"Scores saved to {os.path.join(save_path, 'oti_scores.pkl')}")
