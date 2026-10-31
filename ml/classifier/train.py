"""Training loop for the glitch/signal classifier.

Class-weighted cross-entropy (Gravity Spy classes are heavily imbalanced),
random time-shift augmentation, best checkpoint chosen by macro-F1 on val.
After training, the best checkpoint is scored on the held-out test split:
per-class report, full confusion matrix, and the headline number for this
project -- real-signal vs. glitch precision/recall.

    python -m ml.classifier.train --epochs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.classifier.dataset import NOISE_LABEL, REAL_SIGNAL_LABEL, Sample, class_names, load_dataset, split_dataset
from ml.classifier.model import build_model


def _to_tensors(samples: list[Sample], label_to_idx: dict[str, int]) -> TensorDataset:
    images = torch.from_numpy(np.stack([s.image for s in samples])).unsqueeze(1)
    labels = torch.tensor([label_to_idx[s.label] for s in samples])
    return TensorDataset(images, labels)


def _balanced_class_weights(samples: list[Sample], classes: list[str]) -> torch.Tensor:
    counts = np.array([sum(s.label == c for s in samples) for c in classes], dtype=np.float64)
    return torch.tensor(len(samples) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)


def _random_time_shift(images: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Rolls each image along the time axis. Frequency flips/shifts are never
    applied -- frequency position and chirp direction are physical features.
    """
    shifts = torch.randint(-max_shift, max_shift + 1, (images.shape[0],)).tolist()
    return torch.stack([torch.roll(img, s, dims=-1) for img, s in zip(images, shifts)])


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        preds.append(model(images.to(device)).argmax(dim=1).cpu())
        targets.append(labels)
    return torch.cat(targets).numpy(), torch.cat(preds).numpy()


def _save_confusion_matrix(cm: np.ndarray, classes: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if row_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Test confusion matrix (color = row-normalized recall)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_time_shift: int = 15,
    seed: int = 0,
    checkpoint_dir: str = "data/checkpoints",
    report_dir: str = "data/reports",
) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_dataset()
    classes = class_names(samples)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    train_samples, val_samples, test_samples = split_dataset(samples, seed=seed)
    print(f"{len(samples)} samples, {len(classes)} classes, "
          f"split {len(train_samples)}/{len(val_samples)}/{len(test_samples)}, device {device}")

    train_loader = DataLoader(_to_tensors(train_samples, label_to_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_to_tensors(val_samples, label_to_idx), batch_size=256)
    test_loader = DataLoader(_to_tensors(test_samples, label_to_idx), batch_size=256)

    model = build_model(len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=_balanced_class_weights(train_samples, classes).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_path = Path(checkpoint_dir) / "classifier_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = _random_time_shift(images, max_time_shift).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        scheduler.step()

        y_true, y_pred = _predict(model, val_loader, device)
        val_f1 = f1_score(y_true, y_pred, average="macro", labels=range(len(classes)), zero_division=0)
        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "image_size": tuple(samples[0].image.shape), "epoch": epoch}, checkpoint_path)
            marker = "  *saved"
        print(f"epoch {epoch:3d}  loss {total_loss / len(train_samples):.4f}  "
              f"val macro-F1 {val_f1:.4f}  val acc {(y_true == y_pred).mean():.4f}{marker}")

    best = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best["state_dict"])
    y_true, y_pred = _predict(model, test_loader, device)

    report = classification_report(y_true, y_pred, labels=range(len(classes)), target_names=classes,
                                   digits=3, zero_division=0)
    real = label_to_idx[REAL_SIGNAL_LABEL]
    is_real, pred_real = y_true == real, y_pred == real
    tp = int((is_real & pred_real).sum())
    signal_summary = (
        f"real_signal vs. glitch on test (best epoch {best['epoch']}):\n"
        f"  recall    {tp / max(is_real.sum(), 1):.3f}  ({tp}/{int(is_real.sum())} real signals caught)\n"
        f"  precision {tp / max(pred_real.sum(), 1):.3f}  ({int(pred_real.sum()) - tp} glitches called real)\n"
        "called real_signal, by image source -- if my-pipeline noise is called real\n"
        "much more often than Gravity Spy noise, the model is keying on the pipeline:\n"
    )
    for name, label, source in [
        ("real events (my Q-scan)", REAL_SIGNAL_LABEL, "qscan"),
        ("hardware-injected chirps (Gravity Spy)", REAL_SIGNAL_LABEL, "gravityspy"),
        ("off-source noise (my Q-scan)", NOISE_LABEL, "qscan"),
        ("No_Glitch noise (Gravity Spy)", NOISE_LABEL, "gravityspy"),
    ]:
        mask = np.array([s.label == label and s.source == source for s in test_samples])
        signal_summary += f"  {name:40s} {int(pred_real[mask].sum()):4d}/{int(mask.sum())}\n"
    print("\n" + report + "\n" + signal_summary)

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classifier_test_report.txt").write_text(report + "\n" + signal_summary)
    _save_confusion_matrix(confusion_matrix(y_true, y_pred, labels=range(len(classes))), classes,
                           out / "classifier_confusion_matrix.png")
    print(f"Saved {checkpoint_path}, {out / 'classifier_test_report.txt'}, {out / 'classifier_confusion_matrix.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
