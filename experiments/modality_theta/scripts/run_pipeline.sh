#!/usr/bin/env bash
# Detach-safe modality ablation: θ ∈ {0.0, 0.5, 1.0} × 2 models × RWQA/MMMU.
# Prefer: screen -dmS modality_ablation bash this_script
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=/root/autodl-tmp/hf_home
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true

OUT=experiments/modality_theta
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR" "$OUT/scale_cache" "$OUT/results" "$OUT/metrics" "$OUT/configs"
MASTER_LOG="$LOGDIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
RESULTS_MD="$OUT/results/downstream.md"
DOCS_MD="$REPO_ROOT/docs/exp_modality_fusion_ablation.md"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "===== MODALITY FUSION ABLATION START $(date) ====="
echo "REPO=$REPO_ROOT"
echo "Formula: K = θ·norm(K^T) + (1-θ)·norm(K^V)  (per-modality global-max norm)"
echo "θ grid (fixed, no search): 0.0 / 0.5 / 1.0"
df -h /root/autodl-tmp | tail -1

echo "# Modality fusion ablation downstream" > "$RESULTS_MD"
echo "" >> "$RESULTS_MD"
echo "Generated: $(date -Iseconds)" >> "$RESULTS_MD"
echo "" >> "$RESULTS_MD"
echo "Formula: \`K = θ·norm(K^T) + (1-θ)·norm(K^V)\`" >> "$RESULTS_MD"
echo "" >> "$RESULTS_MD"
echo "- θ=0.0 vision-only" >> "$RESULTS_MD"
echo "- θ=0.5 equal multimodal fusion" >> "$RESULTS_MD"
echo "- θ=1.0 text-only" >> "$RESULTS_MD"
echo "- score = fused K only (no Ψ), ratio=0.01, w_bit=4" >> "$RESULTS_MD"
echo "" >> "$RESULTS_MD"

MODELS=(
  "llava_ov:experiments/modality_theta/configs/llava_ov.yaml"
  "internvl2_8b:experiments/modality_theta/configs/internvl2_8b.yaml"
)
THETAS=(0.0 0.5 1.0)
TASKS=(realworldqa mmmu_val)

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate prism
echo "python=$(which python)"
python -c "import lmms_eval; print('lmms_eval', lmms_eval.__file__)"

for entry in "${MODELS[@]}"; do
  TAG="${entry%%:*}"
  CONFIG="${entry#*:}"
  ENERGY="$OUT/scale_cache/${TAG}_modality_energy.pt"
  echo ""
  echo "######## MODEL ${TAG} $(date) ########"

  for THETA in "${THETAS[@]}"; do
    TTAG=$(printf 't%.2f' "$THETA")
    SCALE="$OUT/scale_cache/${TAG}_${TTAG}.pt"
    METRICS="$OUT/metrics/${TAG}_${TTAG}.json"
    COLS="$OUT/metrics/${TAG}_${TTAG}_cols.json"

    echo "---- BUILD ${TAG} θ=${THETA} $(date) ----"
    python experiments/modality_theta/scripts/build_one.py \
      --config "$CONFIG" \
      --modality_theta "$THETA" \
      --scale_path "$SCALE" \
      --metrics_json "$METRICS" \
      --cols_json "$COLS" \
      --energy_cache "$ENERGY"

    EVAL_YAML="$OUT/configs/eval_${TAG}_${TTAG}.yaml"
    python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${CONFIG}").read_text())
cfg["split_modality"] = True
cfg["modality_theta"] = float("${THETA}")
cfg["scale_path"] = "${SCALE}"
cfg["pseudo_quant"] = True
cfg["tasks"] = "realworldqa"
cfg["output_path"] = "experiments/modality_theta/results/${TAG}_${TTAG}"
Path("${EVAL_YAML}").write_text(
    yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
)
print("wrote", "${EVAL_YAML}")
PY

    for TASK in "${TASKS[@]}"; do
      echo "---- EVAL ${TAG} θ=${THETA} task=${TASK} $(date) ----"
      OUT_PATH="$OUT/results/${TAG}_${TTAG}/${TASK}"
      mkdir -p "$OUT_PATH"
      python main_eval.py \
        --config "$EVAL_YAML" \
        --tasks "$TASK" \
        --output_path "$OUT_PATH" \
        --results_md "$RESULTS_MD" \
        2>&1 | tee "$LOGDIR/eval_${TAG}_${TTAG}_${TASK}.log"
    done

    echo "---- DELETE ckpt ${SCALE} to free disk ----"
    rm -f "$SCALE"
    df -h /root/autodl-tmp | tail -1
  done

  if [[ -f "$ENERGY" ]]; then
    ENERGY_SZ=$(stat -c%s "$ENERGY" 2>/dev/null || echo 0)
    if (( ENERGY_SZ > 2000000000 )); then
      echo "---- DELETE large energy cache ----"
      rm -f "$ENERGY"
    fi
  fi
done

# Write docs summary from metrics + downstream.md
python - <<'PY'
import json
import re
from pathlib import Path

root = Path("experiments/modality_theta")
metrics_dir = root / "metrics"
results_md = (root / "results" / "downstream.md").read_text(encoding="utf-8")
docs = Path("docs/exp_modality_fusion_ablation.md")

# Parse lmms-eval blocks roughly: keep last Value per (context header, task)
# Headers in results_md include theta via preceding BUILD echo? main_eval appends tables.
# Prefer metrics json + eval result jsons if present.

rows = []
for tag in ["llava_ov", "internvl2_8b"]:
    for theta in ["0.00", "0.50", "1.00"]:
        mpath = metrics_dir / f"{tag}_t{theta}.json"
        m = json.loads(mpath.read_text()) if mpath.exists() else {}
        scores = {"realworldqa": None, "mmmu_val": None}
        for task, key in [("realworldqa", "exact_match"), ("mmmu_val", "mmmu_acc")]:
            # look under results/<tag>_tθ/<task>/
            cand = list((root / "results" / f"{tag}_t{theta}" / task).rglob("results.json"))
            if not cand:
                cand = list((root / "results" / f"{tag}_t{theta}").rglob("**/results.json"))
            for p in cand:
                try:
                    data = json.loads(p.read_text())
                except Exception:
                    continue
                # lmms-eval format varies; try common paths
                res = data.get("results") or data
                if isinstance(res, dict) and task in res:
                    cell = res[task]
                    if isinstance(cell, dict):
                        for k, v in cell.items():
                            if key in k and isinstance(v, (int, float)):
                                scores[task] = float(v)
                                break
        label = {
            "0.00": "vision-only (θ=0)",
            "0.50": "fusion (θ=0.5)",
            "1.00": "text-only (θ=1)",
        }[theta]
        rows.append({
            "model": tag,
            "theta": theta,
            "label": label,
            "delta_ce": m.get("delta_ce"),
            "n_kept": m.get("n_kept"),
            "rwqa": scores["realworldqa"],
            "mmmu": scores["mmmu_val"],
        })

def fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)

lines = []
lines.append("# 实验：跨模态列重要度融合消融（θ∈{0, 0.5, 1}）")
lines.append("")
lines.append("> 阶段 1：验证拆分模态后的等权融合是否优于单模态。不做 θ 网格搜索。")
lines.append("")
lines.append("## 设定")
lines.append("")
lines.append("| 项 | 值 |")
lines.append("|----|-----|")
lines.append("| 模型 | LLaVA-OneVision-Qwen2-7B-OV、InternVL2-8B |")
lines.append("| 校准 | COCO ShareGPT4V，`n_samples=128` |")
lines.append("| 公式 | `K = θ·norm(K^T) + (1-θ)·norm(K^V)` |")
lines.append("| 归一化 | 各模态 importance 先做**全局 max 归一化**，再融合 |")
lines.append("| θ | `0.0` 纯视觉 / `0.5` 等权混模态 / `1.0` 纯文本 |")
lines.append("| 选列分数 | 纯融合 K（已移除 Ψ） |")
lines.append("| keep ratio | `0.01`；`w_bit=4`, `w_group=128` |")
lines.append("| 评测 | RealWorldQA + MMMU-val（下游真分数） |")
lines.append("")
lines.append("## 结果")
lines.append("")
lines.append("| 模型 | 设置 | θ | RWQA exact_match | MMMU mmmu_acc | ΔCE（校准，参考） | kept |")
lines.append("|------|------|---:|-----------------:|-------------:|-----------------:|-----:|")
for r in rows:
    lines.append(
        f"| {r['model']} | {r['label']} | {r['theta']} | "
        f"{fmt(r['rwqa'])} | {fmt(r['mmmu'])} | {fmt(r['delta_ce'])} | {fmt(r['n_kept'])} |"
    )
lines.append("")
lines.append("## 结论")
lines.append("")
lines.append("（流水线结束后根据上表填写：θ=0.5 是否同时不低于两侧单模态，以及相对旧 mixed 基线的对照结论。）")
lines.append("")
lines.append("## 复现")
lines.append("")
lines.append("```bash")
lines.append("source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh")
lines.append("conda activate prism")
lines.append("cd /root/autodl-tmp/quantization/PRISM")
lines.append("screen -dmS modality_ablation bash experiments/modality_theta/scripts/run_pipeline.sh")
lines.append("```")
lines.append("")
lines.append("本地产物：`experiments/modality_theta/`（configs / logs / metrics / results）。")
lines.append("")

docs.write_text("\n".join(lines), encoding="utf-8")
print("wrote", docs)

summary_path = root / "results" / "summary_metrics.json"
summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("wrote", summary_path)
PY

echo "===== MODALITY FUSION ABLATION DONE $(date) ====="
echo "Log: $MASTER_LOG"
echo "Results md: $RESULTS_MD"
echo "Docs: $DOCS_MD"
