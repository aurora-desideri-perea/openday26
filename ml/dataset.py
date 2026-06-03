"""
QuickDraw dataset for inhibitor classification.

Classes:
  0 - selective  (square)
  1 - toxic      (triangle)
  2 - inactive   (circle, star, zigzag, line, cloud, hexagon)

QuickDraw numpy bitmaps are stored as (N, 784) uint8 arrays,
white strokes on black background — matching the thresholded
webcam output in app_webcam.html.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


SELECTIVE_CATEGORIES = ["square"]
INACTIVE_CATEGORIES  = ["circle", "star", "zigzag", "line", "cloud", "hexagon"]
SAMPLES_PER_CLASS = 10_000


class GaussianNoise:
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, tensor):
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)


def build_transforms(augment: bool) -> T.Compose:
    ops = []
    if augment:
        ops += [
            T.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            T.RandomPerspective(distortion_scale=0.3, p=0.5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            GaussianNoise(std=0.04),
        ]
    return T.Compose(ops) if ops else None


class QuickDrawDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        train_ratio: float = 0.85,
        samples_per_class: int = SAMPLES_PER_CLASS,
        seed: int = 42,
    ):
        data_dir = Path(data_dir)
        rng = np.random.default_rng(seed)

        images, labels = [], []

        # Pool selective categories (square + hexagon) then sample
        selective_parts = [
            np.load(data_dir / f"{cat}.npy") for cat in SELECTIVE_CATEGORIES
        ]
        selective_all = np.concatenate(selective_parts, axis=0)

        # Pool inactive categories then sample
        inactive_parts = [
            np.load(data_dir / f"{cat}.npy") for cat in INACTIVE_CATEGORIES
        ]
        inactive_all = np.concatenate(inactive_parts, axis=0)

        sources = [
            (selective_all,            0),
            (data_dir / "triangle.npy", 1),
            (inactive_all,             2),
        ]

        for source, label in sources:
            data = np.load(source) if isinstance(source, Path) else source
            idx = rng.choice(len(data), samples_per_class, replace=False)
            images.append(data[idx])
            labels.extend([label] * samples_per_class)

        images = np.concatenate(images, axis=0)
        labels = np.array(labels, dtype=np.int64)

        perm = rng.permutation(len(labels))
        images, labels = images[perm], labels[perm]

        n_train = int(len(labels) * train_ratio)
        if split == "train":
            self.images = images[:n_train]
            self.labels = labels[:n_train]
        else:
            self.images = images[n_train:]
            self.labels = labels[n_train:]

        self.transforms = build_transforms(augment=(split == "train"))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx].reshape(28, 28).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 28, 28)

        if self.transforms:
            tensor = self.transforms(tensor)

        return tensor, self.labels[idx]
