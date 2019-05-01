import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10

from skeletal.constants import DATA_ROOT_FOLDER


def get_loaders(data_dir=DATA_ROOT_FOLDER, split='train', train_split=0.9, batch_size=64):
    if split == 'test':
        test_dataset = CIFAR10(data_dir, train=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return test_loader

    dataset = CIFAR10(data_dir, train=True, download=True)
    train_len = len(dataset) * train_split
    valid_len = len(dataset) * (1-train_split) 
    train_dataset, valid_dataset = random_split(dataset,(train_len, valid_len))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader
