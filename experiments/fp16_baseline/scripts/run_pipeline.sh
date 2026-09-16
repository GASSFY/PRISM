#!/usr/bin/env bash
# Detach-safe FP16 baseline: 2 models × {realworldqa,mmmu_val,ocrbench,ai2d}
# Prefer: screen -dmS prism_fp16 bash this_script
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=/root/autodl-tmp/hf_home
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true

OUT=experiments/fp16_baseline
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR" "$OUT/results"
MASTER_LOG="$LOGDIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
RESULTS_MD="$OUT/results/baseline.md"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "===== FP16 BASELINE START $(date) ====="
echo "REPO=$REPO_ROOT"
df -h /root/autodl-tmp | tail -1

echo "# FP16 baseline (same env as Ψ ablation)" > "$RESULTS_MD"
echo "" >> "$RESULTS_MD"
echo "Generated: $(date -Iseconds)" >> "$RESULTS_MD"
echo "" >> "$RESULTS_MD"
echo "Load mode must print \`[PRISM] Eval load mode: fp16\` for every cell." >> "$RESULTS_MD"
echo "" >> "$RESULTS_MD"

MODELS=(
  "llava_ov:experiments/fp16_baseline/configs/llava_ov.yaml"
  "internvl2_8b:experiments/fp16_baseline/configs/internvl2_8b.yaml"
)
TASKS=(realworldqa mmmu_val ocrbench ai2d)

# Hardcode autodl prism python so login-shell PATH cannot pick the other conda.
PYTHON=/root/autodl-tmp/miniconda3/envs/prism/bin/python
export PATH="/root/autodl-tmp/miniconda3/envs/prism/bin:$PATH"
echo "PYTHON=$PYTHON"
"$PYTHON" -c "import lmms_eval,sys; print('lmms_eval ok', sys.executable)"

for entry in "${MODELS[@]}"; do
  TAG="${entry%%:*}"
  CONFIG="${entry#*:}"
  echo ""
  echo "######## MODEL ${TAG} FP16 $(date) ########"

  for TASK in "${TASKS[@]}"; do
    echo "---- EVAL ${TAG} fp16 task=${TASK} $(date) ----"
    OUT_PATH="$OUT/results/${TAG}_fp16/${TASK}"
    mkdir -p "$OUT_PATH"
    "$PYTHON" main_eval.py \
      --config "$CONFIG" \
      --tasks "$TASK" \
      --output_path "$OUT_PATH" \
      --results_md "$RESULTS_MD" \
      2>&1 | tee "$LOGDIR/eval_${TAG}_fp16_${TASK}.log"
  done
done

echo "---- WRITE comparison.md $(date) ----"
"$PYTHON" "$OUT/scripts/build_comparison.py"

echo "===== FP16 BASELINE DONE $(date) ====="
echo "PIPELINE_EXIT:0" | tee -a "$OUT/logs/screen_exit.txt"
