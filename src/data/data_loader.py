# ─── Imports ────────────────────────────────────────────
import os
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms

# ─── Constantes ─────────────────────────────────────────
SEED = 42

IMG_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BINARY_CLASSES = ['notumor', 'tumor']
TUMOR_CLASSES  = ['glioma', 'meningioma', 'pituitary']

DATASET_ROOT = Path('src/data/DATASET/classification/Training')
TEST_ROOT    = Path('src/data/DATASET/classification/Testing')

# ─── Transforms ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─── Dataset classes ─────────────────────────────────────
class BrainMRIDataset(Dataset):
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    def __init__(self, root, dataset_type, transform=None):
        self.root         = Path(root)
        self.dataset_type = dataset_type
        self.transform    = transform
        self.samples      = []
        self._load_samples()

    def _load_samples(self):
        if self.dataset_type == 'binary':
            for folder in self.root.iterdir():
                if not folder.is_dir():
                    continue
                label = 0 if folder.name.lower() == 'notumor' else 1
                for img_path in folder.iterdir():
                    if img_path.suffix.lower() in self.EXTENSIONS:
                        self.samples.append((img_path, label))
        else:  # '3class'
            for idx, class_name in enumerate(TUMOR_CLASSES):
                folder = self.root / class_name
                for img_path in folder.iterdir():
                    if img_path.suffix.lower() in self.EXTENSIONS:
                        self.samples.append((img_path, idx))
        random.seed(SEED)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class BrainMRIDatasetFixed(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.transform    = transform
        self.class_to_idx = class_to_idx
        self.samples      = []
        for folder_name, label in class_to_idx.items():
            folder_path = Path(root) / folder_name
            if not folder_path.exists():
                print(f"  Folder not found: {folder_path}")
                continue
            for img_path in folder_path.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ─── DataLoader factory ──────────────────────────────────
def get_loaders(dataset_type):
    full_train = BrainMRIDataset(DATASET_ROOT, dataset_type, transform=train_transform)
    full_val   = BrainMRIDataset(DATASET_ROOT, dataset_type, transform=val_transform)

    n_total = len(full_train)
    n_val   = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(SEED)
    split = random_split(range(n_total), [n_train, n_val], generator=generator)
    train_idx, val_idx = split[0].indices, split[1].indices

    train_loader = DataLoader(
        Subset(full_train, train_idx),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(full_val, val_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        BrainMRIDataset(TEST_ROOT, dataset_type, transform=val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f'[Dataset: {dataset_type}]')
    print(f'  Train : {len(train_idx):>5} images')
    print(f'  Val   : {len(val_idx):>5} images')
    print(f'  Test  : {len(BrainMRIDataset(TEST_ROOT, dataset_type)):>5} images')

    return train_loader, val_loader, test_loader