#!/usr/bin/env bash
# Compare offline (one-shot) vs sequential (MSE + error propagation).
# Usage (from repo root, conda env prism):
#   bash scripts/run_offline_vs_sequential.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/default.yaml}"
OUT_ROOT="${OUT_ROOT:-compare_offline_vs_sequential}"
TASKS="${TASKS:-mmmu_val,realworldqa,ocrbench,ai2d}"
RESULTS_MD="${OUT_ROOT}/compare_results.md"

mkdir -p "$OUT_ROOT/scale_cache" "$OUT_ROOT/logs"

cat > "$RESULTS_MD" <<EOF
# Offline vs Sequential (MSE + error prop)

Config: \`${CONFIG}\`
Tasks: ${TASKS}
Fixed: θ from yaml, target_bit from yaml

EOF

run_one() {
  local mode="$1"
  local scale="${OUT_ROOT}/scale_cache/prism_${mode}.pt"
  local log="${OUT_ROOT}/logs/${mode}.log"

  python3 - <<PY
import yaml
with open("${CONFIG}", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["quant_mode"] = "${mode}"
cfg["scale_path"] = "${scale}"
cfg["run_process"] = True
cfg["pseudo_quant"] = True
cfg["tasks"] = "${TASKS}"
cfg["output_path"] = "${OUT_ROOT}/${mode}"
with open("${CONFIG}", "w", encoding="utf-8") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print("[compare] quant_mode=${mode}, scale=${scale}")
PY

  echo "======== QUANT mode=${mode} ========"
  /usr/bin/time -f "WALL_SECONDS %e" python main_quant.py --config "$CONFIG" 2>&1 | tee "$log"

  echo "======== EVAL mode=${mode} ========"
  python main_eval.py --config "$CONFIG" --results_md "$RESULTS_MD" --output_path "${OUT_ROOT}/${mode}" 2>&1 | tee -a "$log"

  # append timing line if present
  if grep -q "Quant wall time" "$log"; then
    echo "" >> "$RESULTS_MD"
    echo "> mode=\`${mode}\`: $(grep 'Quant wall time' "$log" | tail -1)" >> "$RESULTS_MD"
    echo "" >> "$RESULTS_MD"
  fi
}

run_one offline
run_one sequential

echo "Done. Results: ${RESULTS_MD}"
