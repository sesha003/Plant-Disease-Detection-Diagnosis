"""End-to-end smoke test on a tiny synthetic PlantVillage-shaped dataset.

Runs in ~1 minute on CPU without internet:  pytest -q tests/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plantdx.data import Sample, check_no_leakage, group_split, leaf_id_for, prepare_splits  # noqa: E402
from plantdx.models import SUPPORTED_MODELS, SimpleCNN, build_model  # noqa: E402

CLASSES = ["Tomato___Early_blight", "Tomato___healthy", "Potato___Late_blight"]


def make_fake_dataset(root: Path, n_leaves: int = 8, photos_per_leaf: int = 2) -> Path:
    """Mimic data.zip + splits/ + leaf_grouping/ exactly as the Hub repo lays them out."""
    rng = np.random.default_rng(0)
    leaf_map, train_lines, test_lines = {}, [], []
    for ci, cls in enumerate(CLASSES):
        d = root / "images" / "raw" / "color" / cls
        d.mkdir(parents=True, exist_ok=True)
        for leaf in range(n_leaves):
            leaf_key = f"{cls}_leaf{leaf}"
            for k in range(photos_per_leaf):
                ident = f"img{ci}_{leaf}_{k}"
                arr = np.zeros((64, 64, 3), np.uint8)
                arr[..., ci] = 180 + rng.integers(0, 60)  # class-specific colour channel
                arr += rng.integers(0, 30, arr.shape, dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{ident}.JPG")
                leaf_map[ident.lower()] = [leaf_key]
                line = f"raw/color/{cls}/{ident}.JPG"
                (test_lines if leaf >= n_leaves - 2 else train_lines).append(line)
    (root / "splits").mkdir(parents=True)
    (root / "splits" / "color_train.txt").write_text("\n".join(train_lines))
    (root / "splits" / "color_test.txt").write_text("\n".join(test_lines))
    (root / "leaf_grouping").mkdir()
    (root / "leaf_grouping" / "leaf-map.json").write_text(json.dumps(leaf_map))
    (root / ".extracted_color").touch()
    return root


@pytest.fixture()
def cfg(tmp_path):
    data_dir = make_fake_dataset(tmp_path / "pv")
    return {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "runs"),
        "data": {
            "data_dir": str(data_dir), "variant": "color", "download": False, "val_fraction": 0.2,
            "img_size": 64, "batch_size": 8, "num_workers": 0,
        },
        "augment": {"hflip": True, "vflip": True, "rotation_deg": 10, "brightness": 0.2},
        "model": {"name": "simple_cnn", "pretrained": False, "dropout": 0.1, "widths": [8, 16]},
        "train": {"epochs": 2, "lr": 0.01, "early_stopping_patience": 5, "evaluate_after_train": True},
    }


def test_leaf_id_rules():
    lm = {"abc": ["leaf_1"], "dup": ["Tomato___x/leaf_9", "Potato___y/leaf_9"]}
    assert leaf_id_for("abc.JPG", "Tomato___x", lm) == "leaf_1"
    assert leaf_id_for("uuid___abc_final_masked.jpg", "Tomato___x", lm) == "leaf_1"
    assert leaf_id_for("dup.jpg", "Potato___y", lm) == "Potato___y/leaf_9"
    assert leaf_id_for("zzz.jpg", "Tomato___x", lm) == "fallback_zzz"


def test_leaf_id_on_real_hub_formats():
    # file name and leaf-map entries exactly as they appear in the Hub repo
    lm = {
        "rs_erly.b 9400": ["Tomato___Early_blight:::12.0"],
        "rs_hl 6251": ["Potato___healthy:::7.0", "Tomato___healthy:::31.0"],
    }
    f = "8714986a-ae97-489a-8259-67411bfa2379___RS_Erly.B 9400.JPG"
    assert leaf_id_for(f, "Tomato___Early_blight", lm) == "Tomato___Early_blight:::12.0"
    assert leaf_id_for("abc___RS_HL 6251.JPG", "Tomato___healthy", lm) == "Tomato___healthy:::31.0"
    # segmented variant adds '_final_masked'
    assert leaf_id_for("abc___RS_HL 6251_final_masked.jpg", "Potato___healthy", lm) == "Potato___healthy:::7.0"


def test_group_split_never_splits_a_leaf():
    samples = [Sample(f"raw/color/A___b/{i}.jpg", "A___b", f"leaf{i // 3}") for i in range(60)]
    rest, picked = group_split(samples, 0.25, seed=1)
    check_no_leakage({"a": rest, "b": picked})
    assert len(rest) + len(picked) == 60


def test_prepare_splits(cfg):
    info = prepare_splits(cfg)
    assert info["class_names"] == sorted(CLASSES)
    assert len(info["test"]) == len(CLASSES) * 2 * 2


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_every_model_outputs_one_logit_per_class(name):
    m = build_model(name, num_classes=38, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, 224, 224)).shape == (2, 38)


def test_simple_cnn_shapes():
    m = SimpleCNN(38)
    x = torch.randn(1, 3, 224, 224)
    shapes = []
    for block in m.features:
        x = block(x)
        shapes.append(tuple(x.shape[1:]))
    assert shapes == [(32, 112, 112), (64, 56, 56), (128, 28, 28), (256, 14, 14)]


def test_full_pipeline(cfg, tmp_path):
    from plantdx import export, predict, visualize
    from plantdx.train import train

    best = train(cfg)
    assert best.exists()
    run_dir = best.parent
    metrics = json.loads((run_dir / "eval_test" / "metrics.json").read_text())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert (run_dir / "eval_test" / "confusion_matrix.png").exists()
    assert (run_dir / "curves.png").exists()

    img = next((Path(cfg["data"]["data_dir"]) / "images").rglob("*.JPG"))
    predict.main(["--checkpoint", str(best), "--image", str(img)])
    out = tmp_path / "vis"
    visualize.main(["kernels", "--image", str(img), "--out-dir", str(out)])
    visualize.main(["filters", "--checkpoint", str(best), "--out-dir", str(out)])
    visualize.main(["featuremaps", "--checkpoint", str(best), "--image", str(img), "--out-dir", str(out)])
    visualize.main(["gradcam", "--checkpoint", str(best), "--image", str(img), "--out-dir", str(out)])
    assert len(list(out.glob("*.png"))) == 4

    export.main(["--checkpoint", str(best), "--format", "torchscript", "--out-dir", str(tmp_path / "exp")])
    assert (tmp_path / "exp" / "model.ts").exists()
