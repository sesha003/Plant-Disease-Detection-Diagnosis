"""Evaluate a checkpoint on the held-out test split (or val).

    python -m plantdx.evaluate --checkpoint runs/resnet18_xxx/best.pt
    python -m plantdx.evaluate --checkpoint runs/resnet18_xxx/best.pt --split val

Writes to <out-dir>:
    metrics.json                  accuracy, top-3/top-5, macro & weighted F1
    classification_report.txt     precision / recall / F1 for each of the 38 classes
    confusion_matrix.png          row-normalised confusion matrix
    per_crop_accuracy.csv         accuracy grouped by crop
    top_confusions.csv            the most frequent (true -> predicted) mistakes
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from .models import build_model
from .utils import get_device, parse_overrides, deep_merge, pretty_label, save_json, setup_logging, split_label

LOG = logging.getLogger("plantdx")


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    mcfg = ckpt["config"]["model"]
    model = build_model(
        ckpt["model_name"],
        num_classes=len(ckpt["class_names"]),
        pretrained=False,  # weights come from the checkpoint
        dropout=float(mcfg.get("dropout", 0.3)),
        widths=mcfg.get("widths", (32, 64, 128, 256)),
    )
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    probs, targets = [], []
    for x, y in tqdm(loader, desc="evaluate", leave=False):
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def topk_accuracy(probs: np.ndarray, targets: np.ndarray, k: int) -> float:
    k = min(k, probs.shape[1])
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == targets[:, None]).any(axis=1).mean())


def plot_confusion(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    k = len(class_names)
    size = max(6, 0.42 * k)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    labels = [pretty_label(c) for c in class_names]
    ax.set_xticks(range(k), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(k), labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (each row sums to 1)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def compute_report(probs, targets, class_names, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = probs.argmax(1)
    labels = list(range(len(class_names)))
    metrics = {
        "n_images": int(len(targets)),
        "accuracy": float((preds == targets).mean()),
        "top3_accuracy": topk_accuracy(probs, targets, 3),
        "top5_accuracy": topk_accuracy(probs, targets, 5),
        "macro_f1": float(f1_score(targets, preds, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, labels=labels, average="weighted", zero_division=0)),
    }
    save_json(metrics, out_dir / "metrics.json")

    report = classification_report(
        targets, preds, labels=labels, target_names=class_names, digits=4, zero_division=0
    )
    (out_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(targets, preds, labels=labels)
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    # accuracy per crop
    per_crop = defaultdict(lambda: [0, 0])
    for t, p in zip(targets, preds):
        crop = split_label(class_names[t])[0]
        per_crop[crop][0] += int(t == p)
        per_crop[crop][1] += 1
    with open(out_dir / "per_crop_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop", "correct", "total", "accuracy"])
        for crop, (c, n) in sorted(per_crop.items()):
            w.writerow([crop, c, n, f"{c / n:.4f}"])

    # most common mistakes
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(
        ((int(off[i, j]), class_names[i], class_names[j]) for i, j in zip(*np.nonzero(off))),
        reverse=True,
    )[:20]
    with open(out_dir / "top_confusions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["count", "true", "predicted"])
        w.writerows(pairs)

    LOG.info(
        "Accuracy %.4f | top-3 %.4f | macro-F1 %.4f | weighted-F1 %.4f  (n=%d) -> %s",
        metrics["accuracy"], metrics["top3_accuracy"], metrics["macro_f1"],
        metrics["weighted_f1"], metrics["n_images"], out_dir,
    )
    return metrics


def evaluate_checkpoint(
    ckpt_path, out_dir=None, loader=None, split: str = "test", device=None, overrides: dict | None = None
) -> dict:
    device = device or get_device()
    model, ckpt = load_checkpoint(ckpt_path, device)
    if loader is None:
        from .data import build_dataloaders

        cfg = deep_merge(ckpt["config"], overrides or {})
        data = build_dataloaders(cfg, device)
        if data["class_names"] != ckpt["class_names"]:
            raise RuntimeError("Class list of the data differs from the one the model was trained on")
        loader = data["loaders"][split]
    out_dir = Path(out_dir or Path(ckpt_path).parent / f"eval_{split}")
    probs, targets = predict_loader(model, loader, device)
    return compute_report(probs, targets, ckpt["class_names"], out_dir)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.data_dir=/content/pv")
    args = ap.parse_args(argv)
    setup_logging()
    evaluate_checkpoint(args.checkpoint, args.out_dir, split=args.split, overrides=parse_overrides(args.set))


if __name__ == "__main__":
    main()
