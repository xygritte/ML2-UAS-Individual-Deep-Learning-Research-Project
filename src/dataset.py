from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import Dataset


DATA_DIR = Path("data")


def get_cifar10_datasets(data_dir: Path = DATA_DIR):
    """Membuat dataset CIFAR-10 dengan preprocessing yang konsisten."""
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_transform)
    test = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=eval_transform)

    # CIFAR-10 tidak menyediakan validation split bawaan; gunakan 10% dari train sebagai validation.
    total = len(train)
    val_size = total // 10
    train_size = total - val_size
    train_subset, val_subset = __split_dataset(train, train_size, val_size)

    # Validation harus bebas augmentation.
    val_base = datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=eval_transform)
    val_subset = __index_subset(val_base, val_subset.indices)

    return train_subset, val_subset, test


def __split_dataset(dataset: Dataset, train_size: int, val_size: int):
    import torch
    generator = torch.Generator().manual_seed(42)
    return torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)


def __index_subset(dataset: Dataset, indices):
    from torch.utils.data import Subset
    return Subset(dataset, indices)


def class_names() -> Tuple[str, ...]:
    return (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )
