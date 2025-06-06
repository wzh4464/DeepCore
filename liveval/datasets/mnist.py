from torchvision import datasets, transforms
import numpy as np
import torch

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        if hasattr(dataset, "classes"):
            self.classes = dataset.classes

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target, idx

    def __len__(self):
        return len(self.dataset)

def MNIST(data_path, permuted=False, permutation_seed=None):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    if permuted:
        np.random.seed(permutation_seed)
        pixel_permutation = np.random.permutation(28 * 28)
        transform = transforms.Compose(
            [transform, transforms.Lambda(lambda x: x.view(-1, 1)[pixel_permutation].view(1, 28, 28))])

    dst_train = IndexedDataset(datasets.MNIST(data_path, train=True, download=True, transform=transform))
    dst_test = IndexedDataset(datasets.MNIST(data_path, train=False, download=True, transform=transform))
    class_names = [str(c) for c in range(num_classes)]
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

def permutedMNIST(data_path, permutation_seed=None):
    return MNIST(data_path, True, permutation_seed)
