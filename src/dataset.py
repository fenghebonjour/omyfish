from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class FishDataset(Dataset):
    """
    Accepts two folder layouts:

    Auto-split (stratified 80/20 by default):
        data/raw/<class_name>/*.jpg

    Pre-split:
        data/raw/train/<class_name>/*.jpg
        data/raw/val/<class_name>/*.jpg
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        root_dir: str,
        transform=None,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        self.transform = transform
        root = Path(root_dir)

        if (root / "train").exists() and (root / "val").exists():
            data_dir = root / split
            self.classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.samples = self._scan(data_dir)
        else:
            self.classes = sorted(d.name for d in root.iterdir() if d.is_dir())
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            all_samples = self._scan(root)
            labels = [s[1] for s in all_samples]
            train_s, val_s = train_test_split(
                all_samples, test_size=val_ratio, random_state=seed, stratify=labels
            )
            self.samples = train_s if split == "train" else val_s

    def _scan(self, directory: Path) -> List[Tuple[Path, int]]:
        samples = []
        for cls_dir in sorted(directory.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in self.class_to_idx:
                continue
            idx = self.class_to_idx[cls_dir.name]
            for p in cls_dir.iterdir():
                if p.suffix.lower() in self.IMG_EXTS:
                    samples.append((p, idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

    def sample_weights(self) -> torch.Tensor:
        counts = np.zeros(len(self.classes))
        for _, lbl in self.samples:
            counts[lbl] += 1
        w = 1.0 / np.maximum(counts, 1)
        return torch.tensor([w[lbl] for _, lbl in self.samples], dtype=torch.float32)


def build_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, List[str]]:
    from src.transforms import get_train_transforms, get_val_transforms

    img_size = config["data"]["image_size"]
    val_ratio = 1.0 - config["data"]["train_split"]
    bs = config["training"]["batch_size"]
    nw = config["data"]["num_workers"]

    train_ds = FishDataset(config["data"]["data_dir"], get_train_transforms(img_size), "train", val_ratio)
    val_ds = FishDataset(config["data"]["data_dir"], get_val_transforms(img_size), "val", val_ratio)

    sampler = WeightedRandomSampler(train_ds.sample_weights(), len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    return train_loader, val_loader, train_ds.classes
