#!/usr/bin/env bash
# Official quantization entry (pseudo-quant Phase-1).
# Usage (from repo root, conda env prism):
#   bash scripts/quant.sh
#   bash scripts/quant.sh configs/default.yaml
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-configs/default.yaml}"

echo "[PRISM] quant: config=${CONFIG}"
python main_quant.py --config "$CONFIG"
