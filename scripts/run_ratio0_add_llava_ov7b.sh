#!/bin/bash
# Add-on PRISM experiment for llava-onevision-qwen2-7b-ov with (theta1,theta2,ratio)=(0,0,0)
# Usage: bash scripts/run_ratio0_add_llava_ov7b.sh

set -e
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/default.yaml"
MODEL_TYPE="llava_onevision"
MODEL_ARGS="pretrained=lmms-lab/llava-onevision-qwen2-7b-ov"
MODEL_DIR="llava-onevision-qwen2-7b-ov"

ROOT_OUT="result_add"
MODEL_ROOT="${ROOT_OUT}/${MODEL_DIR}"
LOG_DIR="${MODEL_ROOT}/logs"
SCALE_PATH="${MODEL_ROOT}/scale_cache/PRISM_w4.pt"
MODEL_OUT="${MODEL_ROOT}"

THETA1="0.0"
THETA2="0.0"
RATIO="0.0"
TAG="t1_${THETA1}_t2_${THETA2}_r_${RATIO}"
export CUDA_VISIBLE_DEVICES=0

TASKS=( "mmmu_val" "realworldqa" "ocrbench" "ai2d" "chartqa" )
NUM_TASKS=${#TASKS[@]}

mkdir -p "$LOG_DIR" "${MODEL_ROOT}/scale_cache"

echo "################################################################"
echo "# Add-on experiment: ratio=0"
echo "# Model: ${MODEL_TYPE}"
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
print('[ratio0-add] Set model and model_args in config')
" "$MODEL_TYPE" "$MODEL_ARGS"

for TASK in "${TASKS[@]}"; do
  echo ""
  echo "========================================================"
  echo " Task: ${TASK} | ratio=0 add-on"
  echo "========================================================"
  echo ""

  RESULTS_MD="${MODEL_ROOT}/${TASK}_ablation_results.md"
  cat > "$RESULTS_MD" << HEADER
# PRISM Add-on Results: ${TASK}

Model: ${MODEL_TYPE} | ${MODEL_ARGS} | W4 quantization | SpQR-style mixed precision
Experiment: theta1=${THETA1}, theta2=${THETA2}, ratio=${RATIO}

HEADER

  python3 -c "
import yaml, sys
task, scale_path, theta1, theta2, ratio = sys.argv[1:6]
with open('${CONFIG}', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg['tasks'] = task
cfg['scale_path'] = scale_path
cfg['asd_theta1'] = float(theta1)
cfg['asd_theta2'] = float(theta2)
cfg['asd_high_precision_ratio'] = float(ratio)
with open('${CONFIG}', 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f'[ratio0-add] Updated config for task={task}, theta1={theta1}, theta2={theta2}, ratio={ratio}')
" "$TASK" "$SCALE_PATH" "$THETA1" "$THETA2" "$RATIO"

  LOG_FILE="${LOG_DIR}/${TASK}_${TAG}.log"

  echo "[ratio0-add] Running quantization..."
  QUANT_START=$(date +%s)
  python3 main_quant.py --config "$CONFIG" 2>&1 | tee "${LOG_FILE}"
  QUANT_END=$(date +%s)
  QUANT_TIME=$((QUANT_END - QUANT_START))
  QUANT_MIN=$((QUANT_TIME / 60))
  QUANT_SEC=$((QUANT_TIME % 60))
  echo "[ratio0-add] Quantization finished in ${QUANT_MIN}m ${QUANT_SEC}s (${QUANT_TIME}s total)"

  echo "[ratio0-add] Running evaluation..."
  mkdir -p "$MODEL_OUT"
  python3 main_eval.py --config "$CONFIG" --results_md "$RESULTS_MD" --output_path "$MODEL_OUT" 2>&1 | tee -a "${LOG_FILE}"

  echo "" >> "$RESULTS_MD"
  echo "> Quantization time: **${QUANT_MIN}m ${QUANT_SEC}s** (${QUANT_TIME}s)" >> "$RESULTS_MD"
  echo "" >> "$RESULTS_MD"

  if [ -f "$SCALE_PATH" ]; then
    rm -f "$SCALE_PATH"
    echo "[ratio0-add] Deleted ${SCALE_PATH} to save disk space."
  fi

  echo "[ratio0-add] Task ${TASK} complete."
done

echo ""
echo "========================================================"
echo " ratio=0 add-on complete!"
echo " Tasks: ${NUM_TASKS} (${TASKS[*]})"
echo " Results: ${MODEL_ROOT}/<task>_ablation_results.md"
echo " Logs:    ${LOG_DIR}/<task>_${TAG}.log"
echo " Scales:  ${MODEL_ROOT}/scale_cache/"
echo "========================================================"
