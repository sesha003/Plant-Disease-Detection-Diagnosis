"""Download PlantVillage once, before training (training also does this automatically).

    python scripts/download_data.py --variant color
    python scripts/download_data.py --variant segmented --data-dir /content/plantvillage
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plantdx.data import VARIANTS, download_plantvillage  # noqa: E402
from plantdx.utils import setup_logging  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/plantvillage")
    ap.add_argument("--variant", default="color", choices=VARIANTS)
    ap.add_argument("--delete-zip", action="store_true", help="remove data.zip after extracting")
    args = ap.parse_args()
    setup_logging()
    print("Data ready in", download_plantvillage(args.data_dir, args.variant, args.delete_zip))
