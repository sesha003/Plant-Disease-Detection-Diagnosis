"""Predict the plant and disease for one image or a folder of images.

    python -m plantdx.predict --checkpoint runs/x/best.pt --image leaf.jpg
    python -m plantdx.predict --checkpoint runs/x/best.pt --image my_photos/ --topk 3 --csv preds.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from .data import build_transforms
from .evaluate import load_checkpoint
from .utils import get_device, split_label

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Predictor:
    """Reusable wrapper (also used by the Streamlit demo)."""

    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.device = get_device(device)
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.class_names = ckpt["class_names"]
        self.model_name = ckpt["model_name"]
        self.transform = build_transforms(int(ckpt["config"]["data"].get("img_size", 224)), train=False)

    @torch.no_grad()
    def __call__(self, image: Image.Image, topk: int = 3) -> list[dict]:
        x = self.transform(image.convert("RGB"))[None].to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        p, idx = probs.topk(min(topk, len(self.class_names)))
        results = []
        for prob, i in zip(p.tolist(), idx.tolist()):
            crop, disease = split_label(self.class_names[i])
            results.append({"label": self.class_names[i], "crop": crop, "disease": disease, "prob": prob})
        return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True, help="image file or folder")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    args = ap.parse_args(argv)

    pred = Predictor(args.checkpoint)
    src = Path(args.image)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]

    rows = []
    for f in files:
        res = pred(Image.open(f), args.topk)
        best = res[0]
        status = "healthy" if best["disease"].lower() == "healthy" else "DISEASED"
        print(f"{f.name}: {best['crop']} - {best['disease']} ({best['prob']:.1%}) [{status}]")
        for r in res[1:]:
            print(f"    also possible: {r['crop']} - {r['disease']} ({r['prob']:.1%})")
        rows.append({"file": str(f), "crop": best["crop"], "disease": best["disease"], "prob": round(best["prob"], 4)})

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "crop", "disease", "prob"])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
