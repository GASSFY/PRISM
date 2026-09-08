#!/bin/bash
# PRISM multi-model ablation: same theta1/theta2/ratio grid as run_ablation.sh, per model under eval_new_results/<model_dir>/
# Usage: bash scripts/run_more_model.sh   (from any cwd — script cds to repo root)

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/default.yaml"
ROOT_OUT="eval_new_results"

mkdir -p "$ROOT_OUT"

# ---------- Parallel arrays: same length as configs/default.yaml model / model_args ----------
MODEL_TYPES=(
  "llava_onevision"
)
MODEL_ARGSS=(
  "pretrained=your/model-name"
)

if [ "${#MODEL_TYPES[@]}" -ne "${#MODEL_ARGSS[@]}" ]; then
  echo "Error: MODEL_TYPES and MODEL_ARGSS must have the same length."
  exit 1
fi

# Derive directory name from pretrained=... in model_args (up to next comma); unsafe chars -> _
derive_model_dir() {
  python3 -c "
import re, sys
s = sys.argv[1]
m = re.search(r'pretrained=([^,]+)', s)
val = (m.group(1).strip() if m else 'model')
for c in ['/', '\\\\', ':', '*', '?', '\"', '<', '>', '|', ' ']:
    val = val.replace(c, '_')
print(val or 'model')
" "$1"
}

# Disambiguate duplicate pretrained slugs with _<index>
resolve_model_dir() {
  local idx="$1"
  local base
  base="$(derive_model_dir "${MODEL_ARGSS[$idx]}")"
  local j
  for ((j = 0; j < idx; j++)); do
    if [ "$(derive_model_dir "${MODEL_ARGSS[$j]}")" = "$base" ]; then
      echo "${base}_${idx}"
      return
    fi
  done
  echo "$base"
}

# ---------- Evaluation tasks (customize here or via ABLATION_TASKS env) ----------
if [ -n "${ABLATION_TASKS}" ]; then
  IFS=',' read -ra TASKS <<< "$ABLATION_TASKS"
else
  TASKS=( "mmmu_val" "realworldqa" "ocrbench" "ai2d" "chartqa" )
fi

# ---------- Experiment configurations ----------
# Format: "theta1 theta2 ratio" — leading (0,0,0) for ratio=0
EXPERIMENTS=(
  "0.0  0.0  0.0"
  "0.0  1.0  0.01"
  "0.1  0.9  0.01"
  "0.3  0.7  0.01"
  "0.5  0.5  0.01"
  "0.7  0.3  0.01"
  "0.9  0.1  0.01"
  "1.0  0.0  0.01"
)

NUM_EXPERIMENTS=${#EXPERIMENTS[@]}
NUM_TASKS=${#TASKS[@]}
NUM_MODELS=${#MODEL_TYPES[@]}

for MI in "${!MODEL_TYPES[@]}"; do
  MODEL_TYPE="${MODEL_TYPES[$MI]}"
  MODEL_ARGS="${MODEL_ARGSS[$MI]}"
  MODEL_DIR="$(resolve_model_dir "$MI")"
  MODEL_ROOT="${ROOT_OUT}/${MODEL_DIR}"
  LOG_DIR="${MODEL_ROOT}/logs"
  SCALE_PATH="${MODEL_ROOT}/scale_cache/PRISM_w4.pt"
  MODEL_OUT="${MODEL_ROOT}"

  mkdir -p "$LOG_DIR" "${MODEL_ROOT}/scale_cache"

  echo ""
  echo "################################################################"
  echo "# Model index ${MI}/${NUM_MODELS}: ${MODEL_TYPE}"
  echo "# model_args: ${MODEL_ARGS}"
  echo "# Output dir: ${MODEL_ROOT}"
  echo "################################################################"

  python3 -c "
import yaml, sys
model_type, model_args = sys.argv[1], sys.argv[2]
with open('${CONFIG}', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg['model'] = model_type
cfg['model_args'] = model_args
with open('${CONFIG}', 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print('[multi-model] Set model and model_args in config')
" "$MODEL_TYPE" "$MODEL_ARGS"

  for TASK in "${TASKS[@]}"; do
    echo ""
    echo "========================================================"
    echo " Model: ${MODEL_DIR} | Task: ${TASK}"
    echo "========================================================"
    echo ""

    RESULTS_MD="${MODEL_ROOT}/${TASK}_ablation_results.md"
    cat > "$RESULTS_MD" << HEADER
# PRISM Ablation Study Results: ${TASK}

Model: ${MODEL_TYPE} | ${MODEL_ARGS} | W4 quantization | SpQR-style mixed precision

HEADER

    python3 -c "
import yaml
with open('${CONFIG}', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg['tasks'] = '${TASK}'
with open('${CONFIG}', 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f'[Ablation] Set config tasks to ${TASK}')
"

    python3 -c "
import yaml
with open('${CONFIG}', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg['scale_path'] = ''
with open('${CONFIG}', 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print('[Ablation] Set config scale_path to empty for FP16 baseline')
"

    echo "[Ablation] Running FP16 baseline for ${TASK}..."
    LOG_FP16="${LOG_DIR}/${TASK}_fp16_baseline.log"
    python3 main_eval.py --config "$CONFIG" --results_md "$RESULTS_MD" --output_path "$MODEL_OUT" 2>&1 | tee "$LOG_FP16"
    echo "[Ablation] FP16 baseline for ${TASK} complete."
    echo ""

    python3 -c "
import yaml, sys
scale_path = sys.argv[1]
with open('${CONFIG}', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg['scale_path'] = scale_path
with open('${CONFIG}', 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print('[Ablation] Restored config scale_path for ablation experiments')
" "$SCALE_PATH"

    IDX=0
    for EXP in "${EXPERIMENTS[@]}"; do
      read -r THETA1 THETA2 RATIO <<< "$EXP"
      IDX=$((IDX + 1))
      TAG="t1_${THETA1}_t2_${THETA2}_r_${RATIO}"
      LOG_FILE="${LOG_DIR}/${TASK}_${TAG}.log"

      echo ""
      echo "========================================================"
      echo " Model: ${MODEL_DIR} | Task: ${TASK} | Experiment ${IDX}/${NUM_EXPERIMENTS}: theta1=${THETA1}, theta2=${THETA2}, ratio=${RATIO}"
      echo "========================================================"
      echo ""

      python3 -c "
import yaml
with open('${CONFIG}', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg['asd_theta1'] = ${THETA1}
cfg['asd_theta2'] = ${THETA2}
cfg['asd_high_precision_ratio'] = ${RATIO}
with open('${CONFIG}', 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f'[Ablation] Updated config: theta1=${THETA1}, theta2=${THETA2}, ratio=${RATIO}')
"

      QUANT_TIME=0
      QUANT_MIN=0
      QUANT_SEC=0
      QUANT_RAN=0
      if [ -f "$SCALE_PATH" ]; then
        echo "[Ablation] Found existing scale file: ${SCALE_PATH}"
        echo "[Ablation] Skip quantization and run evaluation directly."
      else
        echo "[Ablation] Running quantization..."
        QUANT_START=$(date +%s)
        python3 main_quant.py --config "$CONFIG" 2>&1 | tee "${LOG_FILE}"
        QUANT_END=$(date +%s)
        QUANT_TIME=$((QUANT_END - QUANT_START))
        QUANT_MIN=$((QUANT_TIME / 60))
        QUANT_SEC=$((QUANT_TIME % 60))
        QUANT_RAN=1
        echo "[Ablation] Quantization finished in ${QUANT_MIN}m ${QUANT_SEC}s (${QUANT_TIME}s total)"
      fi

      echo "[Ablation] Running evaluation..."
      python3 main_eval.py --config "$CONFIG" --results_md "$RESULTS_MD" --output_path "$MODEL_OUT" 2>&1 | tee -a "${LOG_FILE}"

      echo "" >> "$RESULTS_MD"
      echo "> Quantization time: **${QUANT_MIN}m ${QUANT_SEC}s** (${QUANT_TIME}s)" >> "$RESULTS_MD"
      echo "" >> "$RESULTS_MD"

      if [ "$QUANT_RAN" -eq 1 ] && [ -f "$SCALE_PATH" ]; then
        rm -f "$SCALE_PATH"
        echo "[Ablation] Deleted ${SCALE_PATH} to save disk space."
      fi

      echo ""
      echo "[Ablation] Model ${MODEL_DIR} | Task ${TASK} | experiment ${IDX}/${NUM_EXPERIMENTS} complete."
      echo ""
    done

    echo "[Ablation] Model ${MODEL_DIR} | Task ${TASK} complete (1 FP16 + ${NUM_EXPERIMENTS} ablation runs)."
    echo ""
  done

  echo "[Ablation] Model ${MODEL_DIR} (all tasks) complete."
  echo ""
done

echo "========================================================"
echo " All models and tasks complete!"
echo " Models: ${NUM_MODELS}"
echo " Tasks: ${NUM_TASKS} (${TASKS[*]})"
echo " Per task per model: 1 FP16 baseline + ${NUM_EXPERIMENTS} ablation experiments"
echo " Results: ${ROOT_OUT}/<model_dir>/<task>_ablation_results.md"
echo " Logs:    ${ROOT_OUT}/<model_dir>/logs/"
echo " Scales:  ${ROOT_OUT}/<model_dir>/scale_cache/"
echo "========================================================"
