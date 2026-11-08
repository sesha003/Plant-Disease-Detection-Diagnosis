"""Data layer: download PlantVillage from Hugging Face, build leak-free splits,
and serve images to PyTorch.

Why not `datasets.load_dataset("mohanty/PlantVillage", "color")`?
The Hub repo uses a Python loading script (plant_village.py). `datasets` >= 4.0
no longer runs loading scripts, so that call fails on current installs. Instead
we fetch the same raw files the script uses and reproduce its logic here:

    data.zip                      -> the images (raw/<variant>/<class>/<file>)
    splits/<variant>_train.txt    -> official 80 % split (leaf-grouped)
    splits/<variant>_test.txt     -> official 20 % split (leaf-grouped)
    leaf_grouping/leaf-map.json   -> maps an image to the physical leaf it shows

The leaf id matters: one physical leaf was photographed several times. If two
photos of the same leaf land in train and test, the model is graded on images
it has effectively already seen ("cheating"). Every split we make is grouped by
leaf id, and we check for overlap before training.
"""

from __future__ import annotations

import json
import logging
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

LOG = logging.getLogger("plantdx")

HF_REPO = "mohanty/PlantVillage"
VARIANTS = ("color", "grayscale", "segmented")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    rel_path: str  # raw/<variant>/<class>/<file>, relative to the image root
    label: str  # e.g. "Tomato___Early_blight"
    leaf_id: str


# --------------------------------------------------------------------------- #
# 1. Download + extract
# --------------------------------------------------------------------------- #
def _find_image_root(images_dir: Path, variant: str) -> Path | None:
    """Return the folder that contains raw/<variant>/ (zip may add a top folder)."""
    if not images_dir.exists():
        return None
    candidates = [images_dir]
    candidates += [p for p in images_dir.iterdir() if p.is_dir()]
    candidates += [q for p in candidates[1:] for q in p.iterdir() if q.is_dir()]
    for c in candidates:
        if (c / "raw" / variant).is_dir():
            return c
    return None


def _extract_variant(zip_path: Path, out_dir: Path, variant: str) -> None:
    """Extract only raw/<variant>/... from data.zip (saves ~2/3 of disk space)."""
    from tqdm import tqdm

    needle = f"raw/{variant}/"
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m
            for m in zf.namelist()
            if needle in m and not m.endswith("/") and "__MACOSX" not in m
        ]
        if not members:
            raise RuntimeError(f"No '{needle}' entries found inside {zip_path}")
        LOG.info("Extracting %d '%s' images to %s ...", len(members), variant, out_dir)
        for m in tqdm(members, desc=f"unzip {variant}", unit="img"):
            zf.extract(m, out_dir)  # zipfile strips '..' and absolute paths


def download_plantvillage(
    data_dir: str | Path, variant: str = "color", delete_zip: bool = False
) -> Path:
    """Download split files, leaf map and images. Safe to call repeatedly."""
    from huggingface_hub import hf_hub_download

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        f"splits/{variant}_train.txt",
        f"splits/{variant}_test.txt",
        "leaf_grouping/leaf-map.json",
    ):
        if not (data_dir / name).exists():
            LOG.info("Downloading %s", name)
            hf_hub_download(HF_REPO, name, repo_type="dataset", local_dir=data_dir)

    marker = data_dir / f".extracted_{variant}"
    if not marker.exists():
        zip_path = data_dir / "data.zip"
        if not zip_path.exists():
            LOG.info("Downloading data.zip (~2.2 GB, one time only) ...")
            hf_hub_download(HF_REPO, "data.zip", repo_type="dataset", local_dir=data_dir)
        _extract_variant(zip_path, data_dir / "images", variant)
        marker.touch()
        if delete_zip:
            zip_path.unlink(missing_ok=True)
    return data_dir


# --------------------------------------------------------------------------- #
# 2. Index: split file + leaf map -> list[Sample]
# --------------------------------------------------------------------------- #
def leaf_id_for(file_name: str, class_name: str, leaf_map: dict) -> str:
    """Same rules as the Hub loading script, so our leaf ids match theirs."""
    ident = file_name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        ident = ident.replace(ext, "")
    ident = ident.strip()

    suggestions = leaf_map.get(ident.lower().strip())
    if not suggestions:
        return f"fallback_{ident}"
    if len(suggestions) == 1:
        return str(suggestions[0])
    for s in suggestions:
        if class_name in str(s):
            return str(s)
    return f"fallback_{ident}"


def read_split(split_file: Path, leaf_map: dict) -> list[Sample]:
    samples = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        parts = rel.split("/")
        if len(parts) < 4:  # expected raw/<variant>/<class>/<file>
            continue
        class_name, file_name = parts[2], parts[3]
        samples.append(Sample(rel, class_name, leaf_id_for(file_name, class_name, leaf_map)))
    return samples


def _drop_missing(samples: list[Sample], image_root: Path, split: str) -> list[Sample]:
    kept = [s for s in samples if (image_root / s.rel_path).is_file()]
    if len(kept) < len(samples):
        LOG.warning("%s: %d listed images not found on disk, skipped", split, len(samples) - len(kept))
    return kept


# --------------------------------------------------------------------------- #
# 3. Leaf-grouped splitting helpers
# --------------------------------------------------------------------------- #
def group_split(
    samples: list[Sample], fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Split by leaf id: returns (rest, picked) with ~`fraction` of leaves picked."""
    leaves = np.array(sorted({s.leaf_id for s in samples}))
    rng = np.random.default_rng(seed)
    rng.shuffle(leaves)
    n_pick = int(round(len(leaves) * fraction))
    picked = set(leaves[:n_pick].tolist())
    rest = [s for s in samples if s.leaf_id not in picked]
    chosen = [s for s in samples if s.leaf_id in picked]
    return rest, chosen


def check_no_leakage(splits: dict[str, list[Sample]]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = {s.leaf_id for s in splits[a]} & {s.leaf_id for s in splits[b]}
            if overlap:
                raise RuntimeError(
                    f"Leaf leakage: {len(overlap)} leaves appear in both '{a}' and '{b}'"
                )
    LOG.info("Leakage check passed: no leaf appears in more than one split")


def prepare_splits(cfg: dict) -> dict:
    """Returns {"train", "val", "test": list[Sample], "image_root", "class_names"}."""
    dcfg = cfg["data"]
    data_dir = Path(dcfg["data_dir"])
    variant = dcfg.get("variant", "color")
    seed = int(cfg.get("seed", 42))

    if dcfg.get("download", True):
        download_plantvillage(data_dir, variant, dcfg.get("delete_zip_after_extract", False))

    image_root = _find_image_root(data_dir / "images", variant)
    if image_root is None:
        raise FileNotFoundError(
            f"Could not find raw/{variant}/ under {data_dir / 'images'}. "
            "Set data.download=true or check data.data_dir."
        )

    leaf_map_path = data_dir / "leaf_grouping" / "leaf-map.json"
    leaf_map = json.loads(leaf_map_path.read_text()) if leaf_map_path.exists() else {}
    if not leaf_map:
        LOG.warning("leaf-map.json missing/empty: every image is treated as its own leaf")

    train_all = read_split(data_dir / "splits" / f"{variant}_train.txt", leaf_map)
    test = read_split(data_dir / "splits" / f"{variant}_test.txt", leaf_map)
    train_all = _drop_missing(train_all, image_root, "train")
    test = _drop_missing(test, image_root, "test")

    test_frac = dcfg.get("test_fraction")
    if test_frac is not None:
        # Custom 3-way split (e.g. 80% train / 10% val / 10% test) by leaf id
        test_frac = float(test_frac)
        val_frac = float(dcfg.get("val_fraction", 0.1))
        all_samples = train_all + test

        frac = float(dcfg.get("fraction", 1.0))
        if frac < 1.0:
            _, all_samples = group_split(all_samples, frac, seed)
            LOG.info("Using fraction=%.2f of the data for a quick run", frac)

        leaves = np.array(sorted({s.leaf_id for s in all_samples}))
        rng = np.random.default_rng(seed)
        rng.shuffle(leaves)
        n_test = int(round(len(leaves) * test_frac))
        n_val = int(round(len(leaves) * val_frac))
        test_leaves = set(leaves[:n_test].tolist())
        val_leaves = set(leaves[n_test : n_test + n_val].tolist())
        train_leaves = set(leaves[n_test + n_val :].tolist())

        train = [s for s in all_samples if s.leaf_id in train_leaves]
        val = [s for s in all_samples if s.leaf_id in val_leaves]
        test = [s for s in all_samples if s.leaf_id in test_leaves]
    else:
        # Optional: use only a fraction of leaves for quick experiments.
        frac = float(dcfg.get("fraction", 1.0))
        if frac < 1.0:
            _, train_all = group_split(train_all, frac, seed)
            _, test = group_split(test, frac, seed)
            LOG.info("Using fraction=%.2f of the data for a quick run", frac)

        # Validation set is carved out of the official train split, grouped by leaf.
        train, val = group_split(train_all, float(dcfg.get("val_fraction", 0.1)), seed)

    class_names = sorted({s.label for s in train})
    known = set(class_names)
    val = [s for s in val if s.label in known]
    test = [s for s in test if s.label in known]

    splits = {"train": train, "val": val, "test": test}
    check_no_leakage(splits)
    for k, v in splits.items():
        LOG.info("%-5s: %6d images, %5d leaves", k, len(v), len({s.leaf_id for s in v}))
    LOG.info("Classes: %d", len(class_names))
    return {**splits, "image_root": image_root, "class_names": class_names}


# --------------------------------------------------------------------------- #
# 4. Transforms
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int, train: bool, aug: dict | None = None) -> T.Compose:
    """Train: resize + random flips/rotation/brightness. Eval: deterministic resize."""
    aug = aug or {}
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return T.Compose(
            [
                T.Resize(int(round(img_size * 256 / 224))),
                T.CenterCrop(img_size),
                T.ToTensor(),
                normalize,
            ]
        )
    ops: list = [
        T.RandomResizedCrop(img_size, scale=tuple(aug.get("random_resized_crop_scale", (0.8, 1.0))))
    ]
    if aug.get("hflip", True):
        ops.append(T.RandomHorizontalFlip())
    if aug.get("vflip", True):
        ops.append(T.RandomVerticalFlip())
    if aug.get("rotation_deg", 0):
        ops.append(T.RandomRotation(aug["rotation_deg"]))
    if any(aug.get(k, 0) for k in ("brightness", "contrast", "saturation", "hue")):
        ops.append(
            T.ColorJitter(
                brightness=aug.get("brightness", 0),
                contrast=aug.get("contrast", 0),
                saturation=aug.get("saturation", 0),
                hue=aug.get("hue", 0),
            )
        )
    ops += [T.ToTensor(), normalize]
    return T.Compose(ops)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation for plotting. x: (3,H,W) or (N,3,H,W)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


# --------------------------------------------------------------------------- #
# 5. Dataset + loaders
# --------------------------------------------------------------------------- #
class PlantVillageDataset(Dataset):
    def __init__(self, samples, image_root, class_names, transform=None):
        self.samples = list(samples)
        self.image_root = Path(image_root)
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        # convert("RGB") also turns grayscale images into 3 identical channels,
        # so pretrained ImageNet models work with every variant.
        img = Image.open(self.image_root / s.rel_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[s.label]


def class_weights(samples: list[Sample], class_names: list[str]) -> torch.Tensor:
    """Inverse-frequency weights w_c = N / (K * n_c). Rare classes count more."""
    counts = Counter(s.label for s in samples)
    n, k = len(samples), len(class_names)
    return torch.tensor([n / (k * max(counts[c], 1)) for c in class_names], dtype=torch.float32)


def build_dataloaders(cfg: dict, device: torch.device | None = None) -> dict:
    info = prepare_splits(cfg)
    dcfg = cfg["data"]
    img_size = int(dcfg.get("img_size", 224))
    names = info["class_names"]
    tf_train = build_transforms(img_size, True, cfg.get("augment"))
    tf_eval = build_transforms(img_size, False)

    nw = int(dcfg.get("num_workers", 4))
    common = dict(
        batch_size=int(dcfg.get("batch_size", 64)),
        num_workers=nw,
        pin_memory=bool(device is not None and device.type == "cuda"),
        persistent_workers=nw > 0,
    )
    loaders = {
        "train": DataLoader(
            PlantVillageDataset(info["train"], info["image_root"], names, tf_train),
            shuffle=True,
            drop_last=len(info["train"]) > common["batch_size"],
            **common,
        ),
        "val": DataLoader(
            PlantVillageDataset(info["val"], info["image_root"], names, tf_eval), **common
        ),
        "test": DataLoader(
            PlantVillageDataset(info["test"], info["image_root"], names, tf_eval), **common
        ),
    }
    return {
        "loaders": loaders,
        "class_names": names,
        "class_weights": class_weights(info["train"], names),
        "splits": {k: info[k] for k in ("train", "val", "test")},
    }
