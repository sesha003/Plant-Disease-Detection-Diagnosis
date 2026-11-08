"""See what the CNN does. Four sub-commands:

    # hand-made kernels (blur, sharpen, Sobel, Laplacian) applied to a leaf
    python -m plantdx.visualize kernels --image leaf.jpg

    # the learned first-layer filters of a trained model
    python -m plantdx.visualize filters --checkpoint runs/x/best.pt

    # feature maps produced by the first conv layers for one image
    python -m plantdx.visualize featuremaps --checkpoint runs/x/best.pt --image leaf.jpg

    # Grad-CAM heat map: which leaf regions drove the prediction
    python -m plantdx.visualize gradcam --checkpoint runs/x/best.pt --image leaf.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from .data import build_transforms, denormalize  # noqa: E402
from .models import gradcam_layer  # noqa: E402
from .utils import pretty_label  # noqa: E402

# --------------------------------------------------------------------------- #
# Classic kernels - the CNN learns kernels like these by itself
# --------------------------------------------------------------------------- #
CLASSIC_KERNELS = {
    "identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    "box blur": [[1 / 9] * 3] * 3,
    "gaussian blur": [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]],
    "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    "sobel x (vertical edges)": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    "sobel y (horizontal edges)": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    "laplacian (spots/blobs)": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
}


def _load_gray(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((size, size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None, None]  # (1,1,H,W)


def show_kernels(image: str, out: Path) -> Path:
    x = _load_gray(image)
    n = len(CLASSIC_KERNELS) + 1
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = axes.ravel()
    for ax, (name, k) in zip(axes, CLASSIC_KERNELS.items()):
        kernel = torch.tensor(k, dtype=torch.float32)[None, None]  # (out=1, in=1, 3, 3)
        y = F.conv2d(x, kernel, padding=1)[0, 0]  # the exact operation a Conv2d layer does
        if "sobel" in name or "laplacian" in name:
            y = y.abs()
        ax.imshow(y.numpy(), cmap="gray")
        ax.set_title(name, fontsize=10)
    # edge magnitude = sqrt(Gx^2 + Gy^2)
    gx = F.conv2d(x, torch.tensor(CLASSIC_KERNELS["sobel x (vertical edges)"], dtype=torch.float32)[None, None], padding=1)
    gy = F.conv2d(x, torch.tensor(CLASSIC_KERNELS["sobel y (horizontal edges)"], dtype=torch.float32)[None, None], padding=1)
    axes[len(CLASSIC_KERNELS)].imshow(torch.sqrt(gx**2 + gy**2)[0, 0].numpy(), cmap="gray")
    axes[len(CLASSIC_KERNELS)].set_title("edge magnitude sqrt(Gx²+Gy²)", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Hand-made 3x3 kernels applied to a leaf", fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Learned filters & feature maps
# --------------------------------------------------------------------------- #
def first_conv(model: nn.Module) -> nn.Conv2d:
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            return m
    raise RuntimeError("No conv layer with 3 input channels found")


def show_filters(model: nn.Module, out: Path, max_filters: int = 64) -> Path:
    w = first_conv(model).weight.detach().cpu()[:max_filters]  # (n, 3, k, k)
    n = w.shape[0]
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.3))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n:
            f = w[i]
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)  # scale to [0,1] to display as RGB
            ax.imshow(f.permute(1, 2, 0).numpy(), interpolation="nearest")
    k = w.shape[-1]
    fig.suptitle(f"First-layer learned filters ({n} shown, each {k}x{k}x3)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def show_feature_maps(model: nn.Module, x: torch.Tensor, out: Path, n_layers: int = 3, n_maps: int = 8) -> Path:
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)][:n_layers]
    acts = []
    hooks = [c.register_forward_hook(lambda m, i, o: acts.append(o.detach().cpu())) for c in convs]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()

    fig, axes = plt.subplots(len(acts) + 1, n_maps, figsize=(n_maps * 1.6, (len(acts) + 1) * 1.7))
    axes[0, 0].imshow(denormalize(x[0].cpu()).permute(1, 2, 0).numpy())
    axes[0, 0].set_title("input", fontsize=8)
    for r, a in enumerate(acts, start=1):
        for c in range(n_maps):
            if c < a.shape[1]:
                axes[r, c].imshow(a[0, c].numpy(), cmap="viridis")
            axes[r, c].set_title(f"conv{r} ch{c} {tuple(a.shape[2:])}", fontsize=6)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Feature maps: early layers respond to colour/edges, deeper ones to patterns")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017).

    weights a_k = mean over (i,j) of d y_c / d A_k(i,j)      (one number per channel k)
    heatmap    = ReLU( sum_k a_k * A_k )                       (H' x W'), upsampled to image size
    """

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.acts = self.grads = None
        self._h = layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.acts = out.detach()
        out.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)  # (1,K,1,1)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))  # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx, torch.softmax(logits, 1)[0, class_idx].item()

    def remove(self):
        self._h.remove()


def show_gradcam(model, model_name, x, class_names, out: Path) -> Path:
    cam = GradCAM(model, gradcam_layer(model, model_name))
    heat, idx, prob = cam(x)
    cam.remove()
    img = denormalize(x[0].detach().cpu()).permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(img)
    ax[0].set_title("input")
    ax[1].imshow(img)
    ax[1].imshow(heat, cmap="jet", alpha=0.45)
    ax[1].set_title(f"Grad-CAM: {pretty_label(class_names[idx])} ({prob:.1%})", fontsize=10)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kernels")
    k.add_argument("--image", required=True)
    for name in ("filters", "featuremaps", "gradcam"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        if name != "filters":
            p.add_argument("--image", required=True)
    for p in sub.choices.values():
        p.add_argument("--out-dir", default="outputs/visualize")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "kernels":
        print(show_kernels(args.image, out_dir / "classic_kernels.png"))
        return

    from .evaluate import load_checkpoint

    model, ckpt = load_checkpoint(args.checkpoint, "cpu")
    if args.cmd == "filters":
        print(show_filters(model, out_dir / f"filters_{ckpt['model_name']}.png"))
        return

    img_size = int(ckpt["config"]["data"].get("img_size", 224))
    x = build_transforms(img_size, train=False)(Image.open(args.image).convert("RGB"))[None]
    stem = Path(args.image).stem
    if args.cmd == "featuremaps":
        print(show_feature_maps(model, x, out_dir / f"featuremaps_{stem}.png"))
    else:
        print(show_gradcam(model, ckpt["model_name"], x, ckpt["class_names"], out_dir / f"gradcam_{stem}.png"))


if __name__ == "__main__":
    main()
