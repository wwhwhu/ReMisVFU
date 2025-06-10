import os
from art.utils import load_mnist, preprocess, to_categorical
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
# 需要export KERAS_HOME=/your/custom/cache
os.environ['KERAS_HOME'] = '/data/wwh/'
# def split_mnist_vfl(num_parties: int = 2,
#                         batch_size: int = 128,
#                         test_batch_size: int = 1024):
#     """
#     按列维度把 MNIST 垂直切成 num_parties 份，返回对应 DataLoader 列表。

#     Returns
#     -------
#     trainloaders : List[DataLoader]
#         第 p 个元素是参与方 p 的训练集 DataLoader（仅含其列切片 + 标签）
#     testloaders  : List[DataLoader]
#         第 p 个元素是参与方 p 的测试集 DataLoader
#     """
#     # 确定所有random数生成器的种子
#     np.random.seed(42)
#     torch.manual_seed(42)

#     # 1. 载入并预处理 MNIST
#     (x_raw, y_raw), (x_raw_test, y_raw_test), _, _ = load_mnist(raw=True)
#     X_train, y_train = preprocess(x_raw, y_raw)           # [N,28,28], one-hot
#     X_test,  y_test  = preprocess(x_raw_test, y_raw_test)
    
#     # 2. 打乱训练集
#     idx = np.random.permutation(X_train.shape[0])
#     X_train, y_train = X_train[idx], y_train[idx]
#     y_train_cls = np.argmax(y_train, axis=1).astype(int)
#     y_test_cls  = np.argmax(y_test,  axis=1).astype(int)
#     print(f"MNIST train set: {X_train.shape}, test set: {X_test.shape}")
#     # 3. 计算每方应拿多少列
#     cols_total   = 28
#     slice_size   = cols_total // num_parties
#     col_ranges = [(p * slice_size,
#                    cols_total if p == num_parties - 1 else (p + 1) * slice_size)
#                   for p in range(num_parties)]

#     def build_loader(X, y_cls, col_start, col_end, bs):
#         """裁剪列并封装成 DataLoader"""
#         X_slice = X[:, :, col_start:col_end]          # [N,28,cols_per_party]
#         X_slice = np.expand_dims(X_slice, axis=1)     # [N,1,28,cols]
#         ds = TensorDataset(torch.tensor(X_slice, dtype=torch.float32),
#                            torch.tensor(y_cls,    dtype=torch.long))
#         return DataLoader(ds, batch_size=bs, shuffle=False)

#     # 4. 为每个参与方构建 train & test loader
#     trainloaders, testloaders = [], []
#     for (c0, c1) in col_ranges:
#         trainloaders.append(build_loader(X_train, y_train_cls, c0, c1, batch_size)) # 包含num_parties个 DataLoader
#         testloaders.append(build_loader(X_test,  y_test_cls,  c0, c1, test_batch_size)) # 包含num_parties个 DataLoader

#     return trainloaders, testloaders

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import List, Tuple

class VerticalSplitDataset(Dataset):
    """
    Wraps a torchvision dataset to return a vertical slice of the image.
    """
    def __init__(self, base_dataset: Dataset, col_start: int, col_end: int):
        self.base = base_dataset
        self.col_start = col_start
        self.col_end = col_end

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        # img: Tensor[C, H, W]
        slice_img = img[:, :, self.col_start:self.col_end]
        return slice_img, label

def _split_dataset_vfl(
    dataset_train: Dataset,
    dataset_test: Dataset,
    num_parties: int,
    batch_size: int,
    test_batch_size: int
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Generic split for any dataset with PIL images converted to tensors.
    """
    # assume all images have same width
    _, H, W = dataset_train[0][0].shape
    slice_size = W // num_parties
    ranges = [
        (p * slice_size, W if p == num_parties - 1 else (p + 1) * slice_size)
        for p in range(num_parties)
    ]

    train_loaders = []
    test_loaders = []
    for (c0, c1) in ranges:
        train_ds = VerticalSplitDataset(dataset_train, c0, c1)
        test_ds  = VerticalSplitDataset(dataset_test,  c0, c1)
        train_loaders.append(DataLoader(train_ds, batch_size=batch_size, shuffle=False))
        test_loaders.append(DataLoader(test_ds,  batch_size=test_batch_size, shuffle=False))
    return train_loaders, test_loaders

def split_mnist_vfl(
    num_parties: int = 2,
    batch_size: int = 128,
    test_batch_size: int = 1024,
    data_path: str = './data'
):
    """
    Split MNIST vertically across num_parties.
    """
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    return _split_dataset_vfl(train_ds, test_ds, num_parties, batch_size, test_batch_size)

def split_fashionmnist_vfl(
    num_parties: int = 2,
    batch_size: int = 128,
    test_batch_size: int = 1024,
    data_path: str = './data'
):
    """
    Split FashionMNIST vertically across num_parties.
    """
    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
    test_ds  = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)
    return _split_dataset_vfl(train_ds, test_ds, num_parties, batch_size, test_batch_size)

def split_svhn_vfl(
    num_parties: int = 2,
    batch_size: int = 128,
    test_batch_size: int = 1024,
    data_path: str = './data'
):
    """
    Split SVHN vertically across num_parties.
    """
    transform = transforms.ToTensor()
    train_ds = datasets.SVHN(root=data_path, split='train', download=True, transform=transform)
    test_ds  = datasets.SVHN(root=data_path, split='test',  download=True, transform=transform)
    return _split_dataset_vfl(train_ds, test_ds, num_parties, batch_size, test_batch_size)

def split_cifar10_vfl(
    num_parties: int = 2,
    batch_size: int = 128,
    test_batch_size: int = 1024,
    data_path: str = './data'
):
    """
    Split CIFAR10 vertically across num_parties.
    """
    transform = transforms.ToTensor()
    train_ds = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    return _split_dataset_vfl(train_ds, test_ds, num_parties, batch_size, test_batch_size)

def split_cifar100_vfl(
    num_parties: int = 2,
    batch_size: int = 128,
    test_batch_size: int = 1024,
    data_path: str = './data'
):
    """
    Split CIFAR100 vertically across num_parties.
    """
    transform = transforms.ToTensor()
    train_ds = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
    return _split_dataset_vfl(train_ds, test_ds, num_parties, batch_size, test_batch_size)

def remove_from_index(index_list, original_list):
    # 删除 original_list 中索引为 index_list 的元素
    return [item for i, item in enumerate(original_list) if i not in index_list]


def load_and_split_data(dataset_name: str, num_parties: int = 2) -> Tuple[List[DataLoader], List[DataLoader], int, int]:
    # Load & split data among parties
    if dataset_name == "MNIST":
        trainloaders, testloaders = split_mnist_vfl(num_parties=num_parties)
        in_channel = 1  # MNIST is grayscale
        out_dim = 10  # MNIST has 10 classes
    elif dataset_name == "FashionMNIST":
        trainloaders, testloaders = split_fashionmnist_vfl(num_parties=num_parties)
        in_channel = 1  # FashionMNIST is grayscale
        out_dim = 10  # FashionMNIST has 10 classes
    elif dataset_name == "SVHN":
        trainloaders, testloaders = split_svhn_vfl(num_parties=num_parties)
        in_channel = 3  # SVHN is RGB
        out_dim = 10  # SVHN has 10 classes
    elif dataset_name == "CIFAR10":
        trainloaders, testloaders = split_cifar10_vfl(num_parties=num_parties)
        in_channel = 3  # CIFAR-10 is RGB
        out_dim = 10  # CIFAR-10 has 10 classes
    elif dataset_name == "CIFAR100":
        trainloaders, testloaders = split_cifar100_vfl(num_parties=num_parties)
        in_channel = 3  # CIFAR-100 is RGB
        out_dim = 100  # CIFAR-100 has 100 classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return trainloaders, testloaders, in_channel, out_dim