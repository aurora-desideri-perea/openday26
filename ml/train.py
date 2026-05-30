"""
Train InhibitorCNN on QuickDraw data.

Usage:
    python train.py

Saves the best model checkpoint to: ml/checkpoints/best_model.pt
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import QuickDrawDataset
from model import InhibitorCNN

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
CKPT_DIR = Path(__file__).parent / "checkpoints"
SAMPLES_PER_CLASS = 10_000
BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
# ──────────────────────────────────────────────────────────────────────────────


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = len(labels)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, labels) * bs
            n += bs

    return total_loss / n, total_acc / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = QuickDrawDataset(DATA_DIR, split="train", samples_per_class=SAMPLES_PER_CLASS)
    val_ds = QuickDrawDataset(DATA_DIR, split="val", samples_per_class=SAMPLES_PER_CLASS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    model = InhibitorCNN(num_classes=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pt")
            flag = "  ← best"

        print(
            f"Epoch {epoch:>2}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}{flag}"
        )

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to: {CKPT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
