import argparse
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, random_split

import config
from datasets import CombinedClassifierDataset, is_holdout
from models import build_classifier
from preprocessing import classifier_train_transform, classifier_eval_transform


def main(epochs: int, lr: float):
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Two dataset instances over the same folders so train/val can use different
    # transforms (random_split alone would share one `.transform` between them).
    # CombinedClassifierDataset pools the tibia/fibula set with FracAtlas's leg
    # non-fractured images as extra negatives — see datasets.py for why.
    # fracatlas_leg_nohw (not fracatlas_leg): excludes hardware-positive images,
    # which Grad-CAM revealed the model was using as a shortcut instead of the
    # actual fracture line (69/263 fractured leg images vs 2/2010 non-fractured
    # had visible hardware — see filter_leg_images.py).
    train_ds_full = CombinedClassifierDataset(config.TIBIA_FIBULA_DIR, config.FRACATLAS_LEG_NOHW_DIR,
                                               transform=classifier_train_transform)
    eval_ds_full = CombinedClassifierDataset(config.TIBIA_FIBULA_DIR, config.FRACATLAS_LEG_NOHW_DIR,
                                              transform=classifier_eval_transform)

    holdout = [is_holdout(p) for p, _, _ in train_ds_full.samples]
    val_indices = [i for i, h in enumerate(holdout) if h]
    train_indices = [i for i, h in enumerate(holdout) if not h]
    n_train = len(train_indices)
    with open(config.OUTPUT_DIR / "holdout_files.json", "w") as f:
        json.dump([str(train_ds_full.samples[i][0]) for i in val_indices], f)

    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(eval_ds_full, val_indices)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS)

    # The Mendeley tibia/fibula set is heavily imbalanced.
    #  inverse-frequency weighting stops the model from just
    # always predicting "fractured" to game plain accuracy.
    not_fx, fx = train_ds_full.class_counts()
    total = not_fx + fx
    class_weights = torch.tensor([total / (2 * not_fx), total / (2 * fx)], dtype=torch.float32).to(device)
    print(f"Class counts (not_fractured, fractured) = ({not_fx}, {fx})  weights={class_weights.tolist()}")

    model = build_classifier(num_classes=2).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels, _sources in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        model.eval()
        correct, total = 0, 0
        class_correct, class_total = [0, 0], [0, 0]
        per_source = {}
        with torch.no_grad():
            for imgs, labels, sources in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                for c in (0, 1):
                    mask = labels == c
                    class_total[c] += mask.sum().item()
                    class_correct[c] += (preds[mask] == c).sum().item()
                for pred, label, source in zip(preds.tolist(), labels.tolist(), sources):
                    key = (source, label)
                    stats = per_source.setdefault(key, [0, 0])
                    stats[1] += 1
                    stats[0] += int(pred == label)

        val_acc = correct / total if total else 0.0
        recall_not_fx = class_correct[0] / class_total[0] if class_total[0] else float("nan")
        recall_fx = class_correct[1] / class_total[1] if class_total[1] else float("nan")
        balanced_acc = (recall_not_fx + recall_fx) / 2 if total else 0.0
        per_source_str = "  ".join(
            f"{src}/{'fx' if lbl else 'not_fx'}={c}/{t}" for (src, lbl), (c, t) in sorted(per_source.items())
        )
        print(f"epoch {epoch+1}/{epochs}  train_loss={running_loss/n_train:.4f}  "
              f"val_acc={val_acc:.4f}  balanced_acc={balanced_acc:.4f}  "
              f"recall(not_fx)={recall_not_fx:.3f}  recall(fx)={recall_fx:.3f}  |  {per_source_str}")

        if balanced_acc >= best_val_acc:
            best_val_acc = balanced_acc
            torch.save(model.state_dict(), config.CLASSIFIER_CKPT)

    print(f"Best balanced_acc={best_val_acc:.4f}, saved to {config.CLASSIFIER_CKPT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args.epochs, args.lr)
