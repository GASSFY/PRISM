#!/usr/bin/env bash
# Detach-safe Ψ ablation pipeline. Prefer: screen -dmS prism_psi bash this_script
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=/root/autodl-tmp/hf_home
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true

OUT=experiments/psi_ablation
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR" "$OUT/scale_cache" "$OUT/results" "$OUT/metrics"
MASTER_LOG="$LOGDIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
RESULTS_MD="$OUT/results/downstream.md"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "===== PSI ABLATION START $(date) ====="
echo "REPO=$REPO_ROOT"
df -h /root/autodl-tmp | tail -1

echo "# Ψ ablation downstream" > "$RESULTS_MD"
echo "" >> "$RESULTS_MD"
echo "Generated: $(date -Iseconds)" >> "$RESULTS_MD"
echo "" >> "$RESULTS_MD"

# tag config_yaml
MODELS=(
  "llava_ov:experiments/psi_ablation/configs/llava_ov.yaml"
  "internvl2_8b:experiments/psi_ablation/configs/internvl2_8b.yaml"
)
ALPHAS=(0.0 0.2)
TASKS=(realworldqa mmmu_val)

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate prism

for entry in "${MODELS[@]}"; do
  TAG="${entry%%:*}"
  CONFIG="${entry#*:}"
  HESS="$OUT/scale_cache/${TAG}_hessian.pt"
  echo ""
  echo "######## MODEL ${TAG} $(date) ########"

  for ALPHA in "${ALPHAS[@]}"; do
    ATAG=$(printf 'a%.2f' "$ALPHA")
    SCALE="$OUT/scale_cache/${TAG}_${ATAG}.pt"
    METRICS="$OUT/metrics/${TAG}_${ATAG}.json"
    COLS="$OUT/metrics/${TAG}_${ATAG}_cols.json"

    echo "---- BUILD ${TAG} alpha=${ALPHA} $(date) ----"
    python experiments/psi_ablation/scripts/build_one.py \
      --config "$CONFIG" \
      --alpha "$ALPHA" \
      --scale_path "$SCALE" \
      --metrics_json "$METRICS" \
      --cols_json "$COLS" \
      --hessian_cache "$HESS"

    # write eval yaml for this cell
    EVAL_YAML="$OUT/configs/eval_${TAG}_${ATAG}.yaml"
    python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${CONFIG}").read_text())
cfg["asd_theta1"] = 1.0 - float("${ALPHA}")
cfg["asd_theta2"] = float("${ALPHA}")
cfg["scale_path"] = "${SCALE}"
cfg["pseudo_quant"] = True
cfg["tasks"] = "realworldqa"
cfg["output_path"] = "experiments/psi_ablation/results/${TAG}_${ATAG}"
Path("${EVAL_YAML}").write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False))
print("wrote", "${EVAL_YAML}")
PY

    for TASK in "${TASKS[@]}"; do
      echo "---- EVAL ${TAG} alpha=${ALPHA} task=${TASK} $(date) ----"
      OUT_PATH="$OUT/results/${TAG}_${ATAG}/${TASK}"
      mkdir -p "$OUT_PATH"
      python main_eval.py \
        --config "$EVAL_YAML" \
        --tasks "$TASK" \
        --output_path "$OUT_PATH" \
        --results_md "$RESULTS_MD" \
        2>&1 | tee "$LOGDIR/eval_${TAG}_${ATAG}_${TASK}.log"
    done

    # free ~15G before next alpha/model
    echo "---- DELETE ckpt ${SCALE} to free disk ----"
    rm -f "$SCALE"
    df -h /root/autodl-tmp | tail -1
  done

  # hessian cache is small/medium; keep for reuse within model, delete after model done if large
  if [[ -f "$HESS" ]]; then
    HESS_SZ=$(stat -c%s "$HESS" 2>/dev/null || echo 0)
    if (( HESS_SZ > 2000000000 )); then
      echo "---- DELETE large hessian cache ----"
      rm -f "$HESS"
    fi
  fi
done

# summarize Jaccard of kept cols vs alpha=0.0 within each model
python - <<'PY'
import json
from pathlib import Path
root = Path('experiments/psi_ablation/metrics')
summary = {}
for tag in ['llava_ov', 'internvl2_8b']:
    p0 = root / f'{tag}_a0.00_cols.json'
    if not p0.exists():
        continue
    base = {tuple(x) for x in json.loads(p0.read_text())}
    summary[tag] = {}
    for alpha in ['0.00', '0.20']:
        p = root / f'{tag}_a{alpha}_cols.json'
        m = root / f'{tag}_a{alpha}.json'
        if not p.exists():
            continue
        cur = {tuple(x) for x in json.loads(p.read_text())}
        inter = len(base & cur)
        union = len(base | cur) or 1
        row = json.loads(m.read_text()) if m.exists() else {}
        summary[tag][alpha] = {
            'delta_ce': row.get('delta_ce'),
            'n_kept': row.get('n_kept'),
            'jaccard_vs_a0': inter / union,
            'overlap_vs_a0': inter / max(len(base), 1),
        }
out = Path('experiments/psi_ablation/results/summary_metrics.json')
out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('SUMMARY', json.dumps(summary, indent=2))
PY

echo "===== PSI ABLATION DONE $(date) ====="
echo "Log: $MASTER_LOG"
echo "Results md: $RESULTS_MD"
