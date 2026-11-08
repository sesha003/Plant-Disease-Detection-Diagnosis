"""Collect test metrics of every run into one table (runs/comparison.csv + printed markdown).

    python scripts/compare_runs.py
"""

import csv
import json
import sys
from pathlib import Path

import yaml

runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
rows = []
for m in sorted(runs.glob("*/eval_test/metrics.json")):
    run = m.parents[1]
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    met = json.loads(m.read_text())
    rows.append(
        {
            "run": run.name,
            "model": cfg["model"]["name"],
            "variant": cfg["data"]["variant"],
            "accuracy": f"{met['accuracy']:.4f}",
            "top3": f"{met['top3_accuracy']:.4f}",
            "macro_f1": f"{met['macro_f1']:.4f}",
            "weighted_f1": f"{met['weighted_f1']:.4f}",
        }
    )

if not rows:
    sys.exit(f"No finished runs found in {runs}/")

with open(runs / "comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print("| " + " | ".join(rows[0]) + " |")
print("|" + "---|" * len(rows[0]))
for r in rows:
    print("| " + " | ".join(r.values()) + " |")
