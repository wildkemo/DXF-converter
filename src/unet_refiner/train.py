"""Training script for the U-Net DXF image refiner."""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .model import UNet
from .dataset import DXFImageDataset


# ── Loss ─────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth)


class CombinedLoss(nn.Module):
    """BCE + Dice — handles class imbalance and produces sharp edges."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return (self.bce_weight * self.bce(pred, target)
                + self.dice_weight * self.dice(pred, target))


# ── Training ─────────────────────────────────────────────────────────
def train(num_epochs=100, batch_size=4, learning_rate=1e-3,
          patience=15, image_size=512):
    """Train the U-Net refiner.

    Args:
        num_epochs:    Maximum training epochs.
        batch_size:    Mini-batch size.
        learning_rate: Initial learning rate.
        patience:      Early-stopping patience (epochs without val improvement).
        image_size:    Rendered image resolution.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'=' * 60}")
    print(f"  U-Net DXF Refiner — Training")
    print(f"{'=' * 60}")
    print(f"  Device:         {device}")
    print(f"  Image size:     {image_size}×{image_size}")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Early stopping: {patience} epochs patience")
    print(f"{'=' * 60}\n")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # ── Datasets ──
    print("Preparing datasets (rendering DXFs to images — first run only)...")
    train_ds = DXFImageDataset(root, 'train', image_size, augment=True)
    val_ds   = DXFImageDataset(root, 'val',   image_size, augment=False)

    # Pre-render so the progress is visible
    train_ds.process_all()
    val_ds.process_all()

    print(f"  Train pairs: {len(train_ds)}")
    print(f"  Val pairs:   {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # ── Model ──
    model = UNet(in_channels=1, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    ckpt_dir = os.path.join(root, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = float('inf')
    stale = 0

    header = (f"{'Epoch':>7} | {'Train Loss':>11} | {'Val Loss':>11} "
              f"| {'LR':>10} | {'Time':>7} | Status")
    print(header)
    print('-' * len(header))

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        for corrupted, perfect in train_loader:
            corrupted = corrupted.to(device)
            perfect = perfect.to(device)

            pred = model(corrupted)
            loss = criterion(pred, perfect)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * corrupted.size(0)

        train_loss /= len(train_ds)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for corrupted, perfect in val_loader:
                corrupted = corrupted.to(device)
                perfect = perfect.to(device)
                pred = model(corrupted)
                val_loss += criterion(pred, perfect).item() * corrupted.size(0)
        val_loss /= len(val_ds)

        lr_now = optimizer.param_groups[0]['lr']
        scheduler.step()
        elapsed = time.time() - t0

        # ── Checkpoint / early stopping ──
        if val_loss < best_val:
            improvement = best_val - val_loss
            best_val = val_loss
            stale = 0
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, 'best_unet.pt'))
            if improvement < float('inf'):
                status = f"✓ saved (↓{improvement:.5f})"
            else:
                status = "✓ saved (first)"
        else:
            stale += 1
            status = f"no improvement ({stale}/{patience})"

        print(f"{epoch:>4}/{num_epochs:<3}| {train_loss:>11.6f} | "
              f"{val_loss:>11.6f} | {lr_now:>10.6f} | "
              f"{elapsed:>5.1f}s | {status}")

        if stale >= patience:
            print(f"\n{'=' * 60}")
            print(f"  Early stopping after {epoch} epochs.")
            print(f"  Best val loss: {best_val:.6f}")
            print(f"{'=' * 60}")
            break
    else:
        print(f"\n{'=' * 60}")
        print(f"  Training completed ({num_epochs} epochs).")
        print(f"  Best val loss: {best_val:.6f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
