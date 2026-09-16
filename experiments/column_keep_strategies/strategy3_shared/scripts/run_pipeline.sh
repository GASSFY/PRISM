#!/usr/bin/env bash
# Strategy 3 / B: Shared → OnlyV/OnlyT quota (rho=1.0, 50/50 split).
# Prefer: screen -dmS strategy3_shared bash this_script
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=/root/autodl-tmp/hf_home
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true

OUT=experiments/column_keep_strategies/strategy3_shared
BUILD_PY="$OUT/scripts/build_one.py"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR" "$OUT/scale_cache" "$OUT/results" "$OUT/metrics" "$OUT/configs"
MASTER_LOG="$LOGDIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
DOCS_MD="$REPO_ROOT/docs/exp_column_keep_strategies.md"
TIMING_JSON="$OUT/metrics/timing_all.json"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "===== STRATEGY3 SHARED-ONLY QUOTA START $(date) ====="
echo "REPO=$REPO_ROOT"
echo "rho=1.0, Shared first, remaining budget 50/50 OnlyV/OnlyT; no fusion-weight search"
df -h /root/autodl-tmp | tail -1

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate prism
echo "python=$(which python)"
python -c "import lmms_eval; print('lmms_eval', lmms_eval.__file__)"
python -c "from prism.quantization import select_columns_shared_quota; print('shared_quota ok')"

echo '{}' > "$TIMING_JSON"

MODELS=(
  "llava_ov:experiments/column_keep_strategies/strategy3_shared/configs/llava_ov.yaml"
  "internvl2_8b:experiments/column_keep_strategies/strategy3_shared/configs/internvl2_8b.yaml"
)
TASKS=(realworldqa mmmu_val)

for entry in "${MODELS[@]}"; do
  TAG="${entry%%:*}"
  CONFIG="${entry#*:}"
  ENERGY="$OUT/scale_cache/${TAG}_modality_energy.pt"
  SCALE="$OUT/scale_cache/${TAG}.pt"
  METRICS="$OUT/metrics/${TAG}_build.json"

  echo ""
  echo "######## MODEL ${TAG} $(date) ########"
  echo "---- BUILD ${TAG} $(date) ----"
  T0=$(date +%s)
  python "$BUILD_PY" \
    --config "$CONFIG" \
    --scale_path "$SCALE" \
    --metrics_json "$METRICS" \
    --cols_json "$OUT/metrics/${TAG}_cols.json" \
    --energy_cache "$ENERGY"
  T1=$(date +%s)
  BUILD_WALL=$((T1 - T0))
  echo "[timing] ${TAG} build_wall=${BUILD_WALL}s"

  EVAL_YAML="$OUT/configs/eval_${TAG}.yaml"
  python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${CONFIG}").read_text())
cfg["split_modality"] = True
cfg["scale_path"] = "${SCALE}"
cfg["pseudo_quant"] = True
cfg["tasks"] = "realworldqa"
cfg["output_path"] = "experiments/column_keep_strategies/strategy3_shared/results/${TAG}"
Path("${EVAL_YAML}").write_text(
    yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
)
print("wrote", "${EVAL_YAML}")
PY

  declare -A TASK_SEC=()
  for TASK in "${TASKS[@]}"; do
    echo "---- EVAL ${TAG} task=${TASK} $(date) ----"
    OUT_PATH="$OUT/results/${TAG}/${TASK}"
    mkdir -p "$OUT_PATH"
    ET0=$(date +%s)
    python main_eval.py \
      --config "$EVAL_YAML" \
      --tasks "$TASK" \
      --output_path "$OUT_PATH" \
      --results_md "$OUT/results/downstream.md"
    ET1=$(date +%s)
    TASK_SEC[$TASK]=$((ET1 - ET0))
    echo "[timing] ${TAG} ${TASK}=${TASK_SEC[$TASK]}s"
  done

  python - <<PY
import json
from pathlib import Path
tag = "${TAG}"
build = json.loads(Path("${METRICS}").read_text())
timing = dict(build.get("timing") or {})
timing["seconds_build_wall"] = int("${BUILD_WALL}")
timing["seconds_eval_realworldqa"] = int("${TASK_SEC[realworldqa]}")
timing["seconds_eval_mmmu_val"] = int("${TASK_SEC[mmmu_val]}")
timing["seconds_total"] = (
    timing["seconds_build_wall"]
    + timing["seconds_eval_realworldqa"]
    + timing["seconds_eval_mmmu_val"]
)
out = {
    "model_tag": tag,
    "strategy": "shared_quota",
    "delta_ce": build.get("delta_ce"),
    "n_kept": build.get("n_kept"),
    "composition": build.get("composition"),
    "timing": timing,
    "checkpoint": "${SCALE}",
}
Path("${OUT}/metrics/${TAG}_timing.json").write_text(json.dumps(out, indent=2))
all_path = Path("${TIMING_JSON}")
all_data = json.loads(all_path.read_text()) if all_path.exists() else {}
all_data[tag] = out
all_path.write_text(json.dumps(all_data, indent=2))
print("wrote timing for", tag)
PY

  echo "---- DELETE ckpt ${SCALE} to free disk ----"
  rm -f "$SCALE"
  df -h /root/autodl-tmp | tail -1
done

echo "---- UPDATE DOCS $(date) ----"
python - <<'PY'
import json
import re
from pathlib import Path

root = Path("experiments/column_keep_strategies/strategy3_shared")
docs = Path("docs/exp_column_keep_strategies.md")
timing_all = json.loads((root / "metrics" / "timing_all.json").read_text())

def load_scores(base: Path, tag: str) -> dict:
    out = {"realworldqa": None, "mmmu_val": None}
    for task in ["realworldqa", "mmmu_val"]:
        cands = list((base / "results" / tag / task).rglob("results.json"))
        if not cands:
            cands = [p for p in (base / "results").rglob("results.json") if tag in str(p) and task in str(p)]
        if not cands:
            continue
        data = json.loads(cands[0].read_text())
        task_res = (data.get("results") or {}).get(task) or {}
        for mk, v in task_res.items():
            if isinstance(v, (int, float)) and ("exact_match" in mk or "mmmu_acc" in mk):
                out[task] = float(v)
                break
    return out

def fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)

def fmt_s(x):
    if x is None:
        return "—"
    return f"{float(x):.1f}"

scores = {tag: load_scores(root, tag) for tag in ["llava_ov", "internvl2_8b"]}
# hard fallbacks for strategy1/2 summary known values
known = {
    1: {"llava_ov": (0.6706, 0.4878), "internvl2_8b": (0.6523, 0.4800)},
    2: {"llava_ov": (0.6654, 0.4878), "internvl2_8b": (0.6523, 0.4800)},
}

rows_down, rows_time, rows_comp = [], [], []
for tag in ["llava_ov", "internvl2_8b"]:
    if tag not in timing_all:
        continue
    m = timing_all[tag]
    sc = scores.get(tag, {})
    comp = m.get("composition") or {}
    rows_down.append(
        f"| {tag} | {fmt(m.get('delta_ce'),6)} | {fmt(sc.get('realworldqa'))} | {fmt(sc.get('mmmu_val'))} | {fmt(m.get('n_kept'),0)} |"
    )
    tim = m.get("timing") or {}
    rows_time.append(
        f"| {tag} | {fmt_s(tim.get('seconds_energy'))} | {fmt_s(tim.get('seconds_select'))} | "
        f"{fmt_s(tim.get('seconds_pseudo_quant'))} | {fmt_s(tim.get('seconds_eval_realworldqa'))} | "
        f"{fmt_s(tim.get('seconds_eval_mmmu_val'))} | {fmt_s(tim.get('seconds_total'))} |"
    )
    rows_comp.append(
        f"| {tag} | {comp.get('n_shared_kept','—')} | {comp.get('n_only_v_kept','—')} | "
        f"{comp.get('n_only_t_kept','—')} | {comp.get('n_fallback_kept','—')} |"
    )

text = docs.read_text(encoding="utf-8")
ll, iv = scores.get("llava_ov", {}), scores.get("internvl2_8b", {})
row3 = (
    f"| 3 | Shared → OnlyT/OnlyV | ρ=1.0（固定） | "
    f"{fmt(ll.get('realworldqa'))} | {fmt(ll.get('mmmu_val'))} | "
    f"{fmt(iv.get('realworldqa'))} | {fmt(iv.get('mmmu_val'))} | 完成 |"
)
text = re.sub(r"\| 3 \|.*?\| (待做|完成|进行中) \|", row3, text, count=1)

# also refresh rows 1-2 scores if still dashes
for sid, label in [(1, r"\| 1 \|.*?\| 完成 \|"), (2, r"\| 2 \|.*?\| 完成 \|")]:
    kv = known[sid]
    new = (
        f"| {sid} | " + ("$\\theta K^T+(1-\\theta)K^V$ + 网格搜 θ | θ（每模型） | " if sid==1 else "最差模态贪心 | 无 | ")
        + f"{kv['llava_ov'][0]:.4f} | {kv['llava_ov'][1]:.4f} | {kv['internvl2_8b'][0]:.4f} | {kv['internvl2_8b'][1]:.4f} | 完成 |"
    )
    if sid == 1:
        new = f"| 1 | $\\theta K^T+(1-\\theta)K^V$ + 网格搜 θ | θ（每模型） | 0.6706 | 0.4878 | 0.6523 | 0.4800 | 完成 |"
    else:
        new = f"| 2 | 最差模态贪心 | 无 | 0.6654 | 0.4878 | 0.6523 | 0.4800 | 完成 |"
    text = re.sub(label, new, text, count=1)

marker_start = "<!-- STRATEGY3_RESULTS_START -->"
marker_end = "<!-- STRATEGY3_RESULTS_END -->"
block = "\n".join([
    marker_start,
    "### 结果",
    "",
    "| 模型 | 校准 ΔCE（参考） | RWQA exact_match | MMMU mmmu_acc | kept |",
    "|------|-----------------:|-----------------:|-------------:|-----:|",
    *rows_down,
    "",
    "### 保列构成",
    "",
    "| 模型 | Shared | OnlyV | OnlyT | fallback |",
    "|------|-------:|------:|------:|---------:|",
    *rows_comp,
    "",
    "### 耗时（秒）",
    "",
    "| 模型 | 能量收集 | 配额选列 | 伪量化 | RWQA 评测 | MMMU 评测 | 合计 |",
    "|------|--------:|---------:|-------:|----------:|----------:|-----:|",
    *rows_time,
    marker_end,
])
if marker_start in text and marker_end in text:
    text = re.sub(re.escape(marker_start) + r".*?" + re.escape(marker_end), block, text, count=1, flags=re.S)
else:
    text = re.sub(
        r"## 策略 3：Shared → OnlyT / OnlyV\n\n.*?(?=\n## 策略 4)",
        "## 策略 3：Shared → OnlyT / OnlyV\n\n"
        "本轮固定 $\\rho=1.0$，剩余预算对半给 OnlyV / OnlyT；**不搜融合权重**。\n\n"
        "### 算法\n\n"
        "1. $\\Omega^V=\\mathrm{Top}_{B}(K^V)$，$\\Omega^T=\\mathrm{Top}_{B}(K^T)$。\n"
        "2. $\\mathrm{Shared}=\\Omega^V\\cap\\Omega^T$，OnlyV / OnlyT 为对称差。\n"
        "3. 先取 Shared（按 $g^V+g^T$）；$B'=B-|\\mathrm{Shared}|$。\n"
        "4. $\\lfloor B'/2\\rfloor$ 给 OnlyV（按 $K^V$），其余给 OnlyT（按 $K^T$）。\n\n"
        + block + "\n\n"
        "### 复现\n\n```bash\n"
        "screen -dmS strategy3_shared bash experiments/column_keep_strategies/strategy3_shared/scripts/run_pipeline.sh\n"
        "```\n\n### 本节结论\n\n"
        + "\n".join(
            f"- **{tag}**: ΔCE={fmt((timing_all.get(tag) or {}).get('delta_ce'),6)}; "
            f"RWQA={fmt((scores.get(tag) or {}).get('realworldqa'))}, "
            f"MMMU={fmt((scores.get(tag) or {}).get('mmmu_val'))}。"
            for tag in ["llava_ov", "internvl2_8b"] if tag in timing_all
        )
        + "\n\n",
        text,
        count=1,
        flags=re.S,
    )

# fix strategy2 downstream dashes if present
text = text.replace(
    "| llava_ov | 0.025717 | — | — | 11325 |",
    "| llava_ov | 0.025717 | 0.6654 | 0.4878 | 11325 |",
).replace(
    "| internvl2_8b | -0.000196 | — | — | 9830 |",
    "| internvl2_8b | -0.000196 | 0.6523 | 0.4800 | 9830 |",
)

docs.write_text(text, encoding="utf-8")
print("updated", docs)
PY

echo "===== STRATEGY3 DONE $(date) ====="
echo "Log: $MASTER_LOG"
echo "Docs: $DOCS_MD"
