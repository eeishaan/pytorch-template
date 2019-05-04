# Copyright (C) 2019 Ishaan Kumar
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

from skeletal.constants import DATA_ROOT_FOLDER


def get_loaders(
        data_dir=DATA_ROOT_FOLDER, split="train",
        train_split=0.9, batch_size=64):
    transform = transforms.ToTensor()
    if split == "test":
        test_dataset = CIFAR10(data_dir, train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return test_loader

    dataset = CIFAR10(data_dir, train=True, download=True, transform=transform)
    train_len = int(len(dataset) * train_split)
    valid_len = len(dataset) - train_len
    train_dataset, valid_dataset = random_split(
        dataset, (train_len, valid_len))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader
