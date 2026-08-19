from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


DATA_DIR = Path("data")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_cifar10_datasets(data_dir: Path = DATA_DIR, pretrained_model: bool = False):
    """Membuat train/validation/test CIFAR-10 dengan preprocessing konsisten.

    Untuk CNN baseline, citra tetap 32x32 dan memakai normalisasi CIFAR-10.
    Untuk model pretrained ImageNet seperti ResNet-18, citra diubah ke 224x224
    dan memakai normalisasi ImageNet agar sesuai dengan bobot pretrained.
    """
    if pretrained_model:
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=8),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    train = datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=train_transform
    )
    test = datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=eval_transform
    )

    # CIFAR-10 tidak menyediakan validation split bawaan; gunakan 10% train sebagai validation.
    total = len(train)
    val_size = total // 10
    train_size = total - val_size
    train_subset, val_subset = _split_dataset(train, train_size, val_size)

    # Validation harus bebas augmentation.
    val_base = datasets.CIFAR10(
        root=str(data_dir), train=True, download=False, transform=eval_transform
    )
    val_subset = _index_subset(val_base, val_subset.indices)

    return train_subset, val_subset, test


def _split_dataset(dataset: Dataset, train_size: int, val_size: int):
    generator = torch.Generator().manual_seed(42)
    return torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )


def _index_subset(dataset: Dataset, indices):
    return Subset(dataset, indices)


def class_names() -> Tuple[str, ...]:
    return (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
