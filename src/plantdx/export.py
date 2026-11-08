"""Export a trained model for deployment.

    python -m plantdx.export --checkpoint runs/resnet18_xxx/best.pt --format all

Outputs in <out-dir>:
    model.onnx        ONNX (ONNX Runtime, ONNX Runtime Mobile, TensorRT, ...)
    model.ts          TorchScript (C++ / Python without the source code)
    model_mobile.ptl  TorchScript optimised for the PyTorch Lite interpreter (Android/iOS)
    labels.json       class names in output order + preprocessing to apply in the app
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from .data import IMAGENET_MEAN, IMAGENET_STD
from .evaluate import load_checkpoint
from .models import count_parameters
from .utils import save_json, setup_logging

LOG = logging.getLogger("plantdx")


def export_onnx(model, example, path: Path) -> None:
    kwargs = dict(
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    try:  # torch >= 2.5: use the classic exporter (no onnxscript dependency)
        torch.onnx.export(model, example, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch has no `dynamo` argument
        torch.onnx.export(model, example, str(path), **kwargs)


def export_torchscript(model, example, out_dir: Path) -> None:
    traced = torch.jit.trace(model, example)
    traced.save(str(out_dir / "model.ts"))
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile

        optimize_for_mobile(traced)._save_for_lite_interpreter(str(out_dir / "model_mobile.ptl"))
    except Exception as e:  # mobile optimiser is not built into every wheel
        LOG.warning("Mobile (.ptl) export skipped: %s", e)


@torch.no_grad()
def benchmark(model, example, runs: int = 20) -> float:
    for _ in range(3):
        model(example)
    t0 = time.perf_counter()
    for _ in range(runs):
        model(example)
    return (time.perf_counter() - t0) / runs * 1000


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--format", default="all", choices=["onnx", "torchscript", "all"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    setup_logging()

    model, ckpt = load_checkpoint(args.checkpoint, "cpu")
    img_size = int(ckpt["config"]["data"].get("img_size", 224))
    example = torch.randn(1, 3, img_size, img_size)
    out_dir = Path(args.out_dir or Path(args.checkpoint).parent / "export")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("onnx", "all"):
        try:
            export_onnx(model, example, out_dir / "model.onnx")
            LOG.info("ONNX -> %s", out_dir / "model.onnx")
        except Exception as e:
            LOG.error("ONNX export failed (pip install onnx): %s", e)
    if args.format in ("torchscript", "all"):
        export_torchscript(model, example, out_dir)
        LOG.info("TorchScript -> %s", out_dir)

    save_json(
        {
            "model": ckpt["model_name"],
            "class_names": ckpt["class_names"],
            "input_size": [3, img_size, img_size],
            "preprocess": {
                "resize": int(round(img_size * 256 / 224)),
                "center_crop": img_size,
                "scale": "pixel / 255",
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            },
        },
        out_dir / "labels.json",
    )
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    LOG.info(
        "%s: %s params, %.1f MB (fp32), %.1f ms / image on this CPU",
        ckpt["model_name"], f"{count_parameters(model):,}", size_mb, benchmark(model, example),
    )


if __name__ == "__main__":
    main()
