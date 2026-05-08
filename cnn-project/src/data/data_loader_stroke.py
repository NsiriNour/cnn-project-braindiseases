# ─── Imports ────────────────────────────────────────────
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms

# ─── Constantes ─────────────────────────────────────────
SEED       = 42
IMG_SIZE   = 224
BATCH_SIZE = 32
VAL_SPLIT  = 0.15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BINARY_CLASSES = ['Normal', 'Stroke']
STROKE_CLASSES = ['Bleeding', 'Ischemia']

DATA_ROOT = Path(__file__).resolve().parents[3] / 'data' / 'DATASET'/ 'Brain_Stroke_CT_Dataset'
TEST_ROOT = DATA_ROOT / 'External_Test'

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

# ─── Dataset ─────────────────────────────────────────────
class StrokeDataset(Dataset):
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

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
                if folder.name == 'External_Test':
                    continue
                label = 0 if folder.name == 'Normal' else 1
                png_folder = folder / 'PNG'
                if not png_folder.exists():
                    continue
                for img_path in png_folder.iterdir():
                    if img_path.suffix.lower() in self.EXTENSIONS:
                        self.samples.append((img_path, label))

        else:  # 'multiclass' → Bleeding vs Ischemia
            for idx, class_name in enumerate(STROKE_CLASSES):
                folder = self.root / class_name / 'PNG'
                if not folder.exists():
                    continue
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


# ─── DataLoader factory ──────────────────────────────────
def get_loaders(dataset_type):
    full_dataset     = StrokeDataset(DATA_ROOT, dataset_type, transform=train_transform)
    full_val_dataset = StrokeDataset(DATA_ROOT, dataset_type, transform=val_transform)

    n_total = len(full_dataset)
    n_test  = int(n_total * 0.15)
    n_val   = int(n_total * 0.15)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(SEED)
    split = random_split(range(n_total), [n_train, n_val, n_test], generator=generator)
    train_idx = split[0].indices
    val_idx   = split[1].indices
    test_idx  = split[2].indices

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        Subset(full_val_dataset, val_idx),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        Subset(full_val_dataset, test_idx),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )

    print(f'[Dataset: {dataset_type}]')
    print(f'  Train : {len(train_idx):>5} images')
    print(f'  Val   : {len(val_idx):>5} images')
    print(f'  Test  : {len(test_idx):>5} images')

    return train_loader, val_loader, test_loader