"""Small helpers shared by every script: config, seeding, device, logging."""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

LOG = logging.getLogger("plantdx")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


# --------------------------------------------------------------------------- #
# Reproducibility & device
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load a YAML config. A config may contain `base: other.yaml` to inherit."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base_cfg = load_config(path.parent / cfg.pop("base"))
        cfg = deep_merge(base_cfg, cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def parse_overrides(pairs: list[str] | None) -> dict:
    """Turn ["train.epochs=5", "model.name=resnet18"] into a nested dict."""
    result: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must look like key.subkey=value")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)  # "5" -> 5, "true" -> True, "[1,2]" -> [1, 2]
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def save_yaml(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def split_label(label: str) -> tuple[str, str]:
    """'Tomato___Early_blight' -> ('Tomato', 'Early blight')."""
    crop, _, disease = label.partition("___")
    clean = lambda s: s.replace("_", " ").replace(",", "").strip()  # noqa: E731
    return clean(crop), clean(disease) if disease else "unknown"


def pretty_label(label: str) -> str:
    crop, disease = split_label(label)
    return f"{crop} | {disease}"
