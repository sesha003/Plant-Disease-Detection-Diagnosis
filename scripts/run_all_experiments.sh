#!/usr/bin/env bash
# Train and evaluate all three architectures from the project proposal.
# Results land in runs/<model>_<timestamp>/eval_test/metrics.json
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/download_data.py --variant color

python -m plantdx.train --config configs/simple_cnn.yaml       # 1) own CNN from scratch
python -m plantdx.train --config configs/resnet18.yaml         # 2) transfer learning
python -m plantdx.train --config configs/efficientnet_b0.yaml  # 2) transfer learning
python -m plantdx.train --config configs/mobilenet_v3.yaml     # 3) phone-sized model

# Optional extra experiment (from the original paper): does the model need the background?
# python -m plantdx.train --config configs/resnet18.yaml --set data.variant=segmented

python scripts/compare_runs.py
