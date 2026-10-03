import argparse
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, random_split

import config
from datasets import FracAtlasSegDataset, is_holdout
from models import build_segmenter


def main(epochs: int, lr: float):
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Two instances so train gets augmentation (matters more here than for the
    # classifier — only 263 fractured images total) and val stays clean.
    # fracatlas_leg_nohw excludes hardware-positive fracture masks too, so the
    # segmenter isn't trained to trace plate/screw outlines instead of bone.
    train_ds_full = FracAtlasSegDataset(config.FRACATLAS_LEG_NOHW_DIR, img_size=config.IMG_SIZE, augment=True)
    eval_ds_full = FracAtlasSegDataset(config.FRACATLAS_LEG_NOHW_DIR, img_size=config.IMG_SIZE, augment=False)

    holdout = [is_holdout(train_ds_full._resolve_image_path(
                   train_ds_full.images_by_id[iid]["file_name"])) for iid in train_ds_full.image_ids]
    val_indices = [i for i, h in enumerate(holdout) if h]
    train_indices = [i for i, h in enumerate(holdout) if not h]
    n_train = len(train_indices)

    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(eval_ds_full, val_indices)

    train_loader = DataLoader(train_ds, batch_size=max(1, config.BATCH_SIZE // 2), shuffle=True,
                               num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=max(1, config.BATCH_SIZE // 2), shuffle=False,
                             num_workers=config.NUM_WORKERS)

    model = build_segmenter(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_iou = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)["out"]
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        model.eval()
        intersection, union = 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)["out"].argmax(dim=1)
                intersection += ((preds == 1) & (masks == 1)).sum().item()
                union += ((preds == 1) | (masks == 1)).sum().item()
        val_iou = intersection / union if union else 0.0
        print(f"epoch {epoch+1}/{epochs}  train_loss={running_loss/n_train:.4f}  val_iou={val_iou:.4f}")

        if val_iou >= best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), config.SEGMENTATION_CKPT)

    print(f"Best val_iou={best_val_iou:.4f}, saved to {config.SEGMENTATION_CKPT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args.epochs, args.lr)
