"""
data_loader_alzheimer.py
────────────────────────
DataLoader for the Alzheimer MRI dataset.
Structure expected:
    <DATA_ROOT>/
        train/
            NonDemented/
            VeryMildDemented/
            MildDemented/
            ModerateDemented/
        test/
            ...same subfolders...
"""

import os
import copy
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ── ImageNet statistics (standard for transfer learning) ──
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224):
    """
    Returns (train_transforms, val_test_transforms).

    Train  → aggressive augmentation to avoid overfitting on synthetic samples.
    Val/Test → only resize + normalize (no data leakage into evaluation).
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 20, img_size + 20)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return train_tf, eval_tf


def get_dataloaders(
    data_root: str,
    img_size:  int = 224,
    batch_size: int = 32,
    val_split:  float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
):
    """
    Build train / val / test DataLoaders for the Alzheimer dataset.

    Args:
        data_root   : path to folder containing train/ and test/ subfolders.
        img_size    : input image size (default 224 for ImageNet-pretrained models).
        batch_size  : samples per batch.
        val_split   : fraction of training data used for validation.
        num_workers : DataLoader worker processes.
        seed        : random seed for reproducible split.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_tf, eval_tf = get_transforms(img_size)

    # ── Full train dataset with augmentation ──
    full_train = datasets.ImageFolder(
        os.path.join(data_root, 'train'), transform=train_tf
    )
    test_ds = datasets.ImageFolder(
        os.path.join(data_root, 'test'), transform=eval_tf
    )

    # ── 80/20 train-val split ──
    n_val   = int(val_split * len(full_train))
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val],
                                    generator=generator)

    # Apply eval transforms to validation subset (no augmentation)
    val_ds.dataset = copy.deepcopy(full_train)
    val_ds.dataset.transform = eval_tf

    def _loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True
        )

    train_loader = _loader(train_ds, shuffle=True)
    val_loader   = _loader(val_ds,   shuffle=False)
    test_loader  = _loader(test_ds,  shuffle=False)

    class_names = full_train.classes
    print(f"[AlzheimerLoader] Train={len(train_ds)} | "
          f"Val={len(val_ds)} | Test={len(test_ds)}")
    print(f"[AlzheimerLoader] Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names


def denormalize(tensor):
    """Reverse ImageNet normalisation for visualisation."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)
