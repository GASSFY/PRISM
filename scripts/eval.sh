#!/usr/bin/env bash
# Official evaluation entry (lmms-eval).
# Usage (from repo root, conda env prism):
#   bash scripts/eval.sh
#   bash scripts/eval.sh configs/default.yaml
# Optional: RESULTS_MD=path/to.md bash scripts/eval.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-configs/default.yaml}"

echo "[PRISM] eval: config=${CONFIG}"
if [[ -n "${RESULTS_MD:-}" ]]; then
  python main_eval.py --config "$CONFIG" --results_md "$RESULTS_MD"
else
  python main_eval.py --config "$CONFIG"
fi
