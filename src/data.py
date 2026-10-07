"""Dataset dan dataloader yang dibaca dari manifest.csv."""

import csv

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LeafDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def train_transform(size):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(size):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_loaders(manifest_csv, image_size, batch_size, num_workers=0):
    """Bangun dataloader train, val, dan test dari manifest.

    Augmentasi hanya dipakai di train. num_workers dibiarkan 0 secara
    default karena multiprocessing DataLoader sering bermasalah di Windows.
    """
    buckets = {"train": [], "val": [], "test": []}
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            buckets[row["split"]].append((row["path"], int(row["label"])))

    loaders = {}
    for split, rows in buckets.items():
        tf = train_transform(image_size) if split == "train" else eval_transform(image_size)
        ds = LeafDataset(rows, tf)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
    return loaders
