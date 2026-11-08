"""Train a model.

    python -m plantdx.train --config configs/simple_cnn.yaml
    python -m plantdx.train --config configs/resnet18.yaml --set train.epochs=5 data.fraction=0.2
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from .data import build_dataloaders
from .models import build_model, count_parameters, set_backbone_trainable
from .utils import (
    get_device,
    load_config,
    parse_overrides,
    save_json,
    save_yaml,
    set_seed,
    setup_logging,
)

LOG = logging.getLogger("plantdx")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False, desc=""):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    preds, targets = [], []

    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()  # backprop: d(loss)/d(every weight)
                scaler.step(optimizer)  # gradient-descent update
                scaler.update()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())

    if n == 0:
        return {"loss": float("nan"), "acc": float("nan"), "macro_f1": float("nan")}
    p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": float(total_loss / n),
        "acc": float((p == t).mean()),
        "macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
    }


def plot_history(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("Loss")
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train acc")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val acc")
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], label="val macro-F1", ls="--")
    ax[1].set_title("Accuracy / F1")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(cfg: dict) -> Path:
    """Full training run. Returns the path of the best checkpoint."""
    mcfg, tcfg = cfg["model"], cfg["train"]
    run_dir = Path(cfg.get("output_dir", "runs")) / f"{mcfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    save_yaml(cfg, run_dir / "config.yaml")

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    LOG.info("Run dir: %s | device: %s", run_dir, device)

    # ---- data ------------------------------------------------------------- #
    data = build_dataloaders(cfg, device)
    loaders, class_names = data["loaders"], data["class_names"]
    save_json(class_names, run_dir / "class_names.json")

    # ---- model ------------------------------------------------------------ #
    model = build_model(
        mcfg["name"],
        num_classes=len(class_names),
        pretrained=bool(mcfg.get("pretrained", True)),
        dropout=float(mcfg.get("dropout", 0.3)),
        widths=mcfg.get("widths", (32, 64, 128, 256)),
    ).to(device)
    LOG.info("Model %s: %s parameters", mcfg["name"], f"{count_parameters(model):,}")

    freeze_epochs = int(tcfg.get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0 and mcfg["name"] != "simple_cnn":
        set_backbone_trainable(model, mcfg["name"], False)
        LOG.info(
            "Backbone frozen for %d epoch(s): training only the new head (%s params)",
            freeze_epochs,
            f"{count_parameters(model, trainable_only=True):,}",
        )

    weight = data["class_weights"].to(device) if cfg["data"].get("class_weighting") else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=float(tcfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 1e-4))
    )
    epochs = int(tcfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)  # mixed precision only on GPU

    # ---- loop ------------------------------------------------------------- #
    best_f1, best_epoch, patience = -1.0, 0, int(tcfg.get("early_stopping_patience", 5))
    history: list[dict] = []
    best_path = run_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1 and mcfg["name"] != "simple_cnn":
            set_backbone_trainable(model, mcfg["name"], True)
            LOG.info("Backbone unfrozen: fine-tuning the whole network")

        t0 = time.time()
        tr = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler, amp, f"train {epoch}")
        va = run_epoch(model, loaders["val"], criterion, device, amp=amp, desc=f"val {epoch}")
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        LOG.info(
            "epoch %3d | train loss %.4f acc %.4f | val loss %.4f acc %.4f F1 %.4f | %.0fs",
            epoch, tr["loss"], tr["acc"], va["loss"], va["acc"], va["macro_f1"], row["seconds"],
        )

        ckpt = {
            "model_name": mcfg["name"],
            "model_state": model.state_dict(),
            "class_names": class_names,
            "config": cfg,
            "epoch": epoch,
            "val_metrics": va,
        }
        torch.save(ckpt, run_dir / "last.pt")
        score = va["macro_f1"] if not np.isnan(va["macro_f1"]) else tr["macro_f1"]
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            torch.save(ckpt, best_path)
            LOG.info("  new best (macro-F1 %.4f) -> %s", best_f1, best_path.name)
        elif epoch - best_epoch >= patience:
            LOG.info("Early stopping: no improvement for %d epochs", patience)
            break

    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    plot_history(history, run_dir / "curves.png")
    LOG.info("Best epoch %d, val macro-F1 %.4f", best_epoch, best_f1)

    if tcfg.get("evaluate_after_train", True):
        from .evaluate import evaluate_checkpoint

        evaluate_checkpoint(best_path, run_dir / "eval_test", loader=loaders["test"], device=device)
    return best_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/simple_cnn.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5 model.name=resnet18")
    args = ap.parse_args(argv)
    setup_logging()
    train(load_config(args.config, parse_overrides(args.set)))


if __name__ == "__main__":
    main()
