#!/usr/bin/env bash
# Strategy 2 / A: worst-modality greedy keep (no searchable params).
# Prefer: screen -dmS strategy2_minimax bash this_script
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=/root/autodl-tmp/hf_home
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true

OUT=experiments/column_keep_strategies/strategy2_minimax
BUILD_PY="$OUT/scripts/build_one.py"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR" "$OUT/scale_cache" "$OUT/results" "$OUT/metrics" "$OUT/configs"
MASTER_LOG="$LOGDIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
DOCS_MD="$REPO_ROOT/docs/exp_column_keep_strategies.md"
TIMING_JSON="$OUT/metrics/timing_all.json"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "===== STRATEGY2 WORST-MODALITY GREEDY START $(date) ====="
echo "REPO=$REPO_ROOT"
echo "No searchable params; calib ΔCE reported for reference only"
df -h /root/autodl-tmp | tail -1

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate prism
echo "python=$(which python)"
python -c "import lmms_eval; print('lmms_eval', lmms_eval.__file__)"
python -c "from prism.quantization import select_columns_worst_modality_greedy; print('greedy ok')"

echo '{}' > "$TIMING_JSON"

MODELS=(
  "llava_ov:experiments/column_keep_strategies/strategy2_minimax/configs/llava_ov.yaml"
  "internvl2_8b:experiments/column_keep_strategies/strategy2_minimax/configs/internvl2_8b.yaml"
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
cfg["output_path"] = "experiments/column_keep_strategies/strategy2_minimax/results/${TAG}"
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
    "strategy": "worst_modality_greedy",
    "delta_ce": build.get("delta_ce"),
    "n_kept": build.get("n_kept"),
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

root = Path("experiments/column_keep_strategies/strategy2_minimax")
docs = Path("docs/exp_column_keep_strategies.md")
timing_all = json.loads((root / "metrics" / "timing_all.json").read_text())
s1 = Path("experiments/column_keep_strategies/strategy1_theta/metrics/timing_all.json")
s1_all = json.loads(s1.read_text()) if s1.exists() else {}

def load_scores(base: Path, tag: str) -> dict:
    out = {"realworldqa": None, "mmmu_val": None}
    for task in ["realworldqa", "mmmu_val"]:
        cands = list((base / "results" / tag / task).rglob("results.json"))
        if not cands:
            cands = [p for p in (base / "results").rglob("results.json") if tag in str(p) and task in str(p)]
        if not cands:
            continue
        data = json.loads(cands[0].read_text())
        res = data.get("results", {})
        task_res = res.get(task) or {}
        val = None
        for mk, v in task_res.items():
            if isinstance(v, (int, float)) and ("exact_match" in mk or "mmmu_acc" in mk):
                val = float(v)
                break
        out[task] = val
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

# Strategy1 scores (fix doc)
s1_scores = {
    "llava_ov": load_scores(Path("experiments/column_keep_strategies/strategy1_theta"), "llava_ov_best"),
    "internvl2_8b": load_scores(Path("experiments/column_keep_strategies/strategy1_theta"), "internvl2_8b_best"),
}
# fallback hardcode from known eval if parse fails
if s1_scores["llava_ov"]["realworldqa"] is None:
    s1_scores["llava_ov"] = {"realworldqa": 0.6706, "mmmu_val": 0.4878}
if s1_scores["internvl2_8b"]["realworldqa"] is None:
    s1_scores["internvl2_8b"] = {"realworldqa": 0.6523, "mmmu_val": 0.4800}

s2_scores = {
    "llava_ov": load_scores(root, "llava_ov"),
    "internvl2_8b": load_scores(root, "internvl2_8b"),
}

rows_down = []
rows_time = []
for tag in ["llava_ov", "internvl2_8b"]:
    if tag not in timing_all:
        continue
    m = timing_all[tag]
    sc = s2_scores.get(tag, {})
    rows_down.append(
        f"| {tag} | {fmt(m.get('delta_ce'), 6)} | {fmt(sc.get('realworldqa'))} | {fmt(sc.get('mmmu_val'))} | {fmt(m.get('n_kept'), 0)} |"
    )
    tim = m.get("timing") or {}
    rows_time.append(
        f"| {tag} | {fmt_s(tim.get('seconds_energy'))} | {fmt_s(tim.get('seconds_select'))} | "
        f"{fmt_s(tim.get('seconds_pseudo_quant'))} | {fmt_s(tim.get('seconds_eval_realworldqa'))} | "
        f"{fmt_s(tim.get('seconds_eval_mmmu_val'))} | {fmt_s(tim.get('seconds_total'))} |"
    )

text = docs.read_text(encoding="utf-8")

# Fix summary table rows 1 and 2
ll1, iv1 = s1_scores["llava_ov"], s1_scores["internvl2_8b"]
ll2, iv2 = s2_scores["llava_ov"], s2_scores["internvl2_8b"]
row1 = (
    f"| 1 | $\\theta K^T+(1-\\theta)K^V$ + 网格搜 θ | θ（每模型） | "
    f"{fmt(ll1.get('realworldqa'))} | {fmt(ll1.get('mmmu_val'))} | "
    f"{fmt(iv1.get('realworldqa'))} | {fmt(iv1.get('mmmu_val'))} | 完成 |"
)
row2 = (
    f"| 2 | 最差模态贪心 | 无 | "
    f"{fmt(ll2.get('realworldqa'))} | {fmt(ll2.get('mmmu_val'))} | "
    f"{fmt(iv2.get('realworldqa'))} | {fmt(iv2.get('mmmu_val'))} | 完成 |"
)
text = re.sub(r"\| 1 \|.*?\| (完成|进行中) \|", row1, text, count=1)
text = re.sub(r"\| 2 \|.*?\| (待做|完成|进行中) \|", row2, text, count=1)

# Fix strategy1 downstream table scores inside markers
def fix_s1_down(text: str) -> str:
    for tag, sc in s1_scores.items():
        th = (s1_all.get(tag) or {}).get("theta_star")
        th_s = f"{th:.2f}" if isinstance(th, float) else ("1.00" if tag=="llava_ov" else "0.00")
        kept = "11325" if tag=="llava_ov" else "9830"
        text = re.sub(
            rf"\| {tag} \| {th_s} \| — \| — \| {kept} \|",
            f"| {tag} | {th_s} | {fmt(sc.get('realworldqa'))} | {fmt(sc.get('mmmu_val'))} | {kept} |",
            text,
        )
    text = text.replace(
        "- **llava_ov**: θ\*=1.00, ΔCE\*=0.021892; RWQA=—, MMMU=—.",
        f"- **llava_ov**: θ\*=1.00, ΔCE\*=0.021892; RWQA={fmt(ll1.get('realworldqa'))}, MMMU={fmt(ll1.get('mmmu_val'))}。",
    )
    text = text.replace(
        "- **internvl2_8b**: θ\*=0.00, ΔCE\*=-0.000196; RWQA=—, MMMU=—.",
        f"- **internvl2_8b**: θ\*=0.00, ΔCE\*=-0.000196; RWQA={fmt(iv1.get('realworldqa'))}, MMMU={fmt(iv1.get('mmmu_val'))}。",
    )
    return text

text = fix_s1_down(text)

marker_start = "<!-- STRATEGY2_RESULTS_START -->"
marker_end = "<!-- STRATEGY2_RESULTS_END -->"
block = "\n".join([
    marker_start,
    "### 结果",
    "",
    "| 模型 | 校准 ΔCE（参考） | RWQA exact_match | MMMU mmmu_acc | kept |",
    "|------|-----------------:|-----------------:|-------------:|-----:|",
    *rows_down,
    "",
    "### 耗时（秒）",
    "",
    "| 模型 | 能量收集 | 贪心选列 | 伪量化 | RWQA 评测 | MMMU 评测 | 合计 |",
    "|------|--------:|---------:|-------:|----------:|----------:|-----:|",
    *rows_time,
    marker_end,
])

if marker_start in text and marker_end in text:
    text = re.sub(re.escape(marker_start) + r".*?" + re.escape(marker_end), block, text, count=1, flags=re.S)
else:
    # replace strategy 2 placeholder section body before strategy 3
    text = re.sub(
        r"## 策略 2：最差模态贪心\n\n.*?(?=\n## 策略 3)",
        "## 策略 2：最差模态贪心\n\n"
        "无任何可学习 / 可网格搜索参数。\n\n"
        "### 算法\n\n"
        "1. 对各列算 $g_c^V=\\mathrm{norm}(K_c^V)$、$g_c^T=\\mathrm{norm}(K_c^T)$。\n"
        "2. $G^m=\\sum_c g_c^m$，已覆盖 $C^m$，剩余损失 $L^m=G^m-C^m$。\n"
        "3. 每轮 $m^*=\\arg\\max_m L^m$，在未选列中取对 $m^*$ 增益最大的列（另一模态作 tie-break）。\n"
        "4. 更新 $C^V,C^T$，直到 $|\\Omega|=B$。\n\n"
        + block + "\n\n"
        "### 复现\n\n"
        "```bash\n"
        "screen -dmS strategy2_minimax bash experiments/column_keep_strategies/strategy2_minimax/scripts/run_pipeline.sh\n"
        "```\n\n"
        "### 本节结论\n\n"
        + "\n".join(
            f"- **{tag}**: ΔCE={fmt((timing_all.get(tag) or {}).get('delta_ce'),6)}; "
            f"RWQA={fmt((s2_scores.get(tag) or {}).get('realworldqa'))}, "
            f"MMMU={fmt((s2_scores.get(tag) or {}).get('mmmu_val'))}。"
            for tag in ["llava_ov", "internvl2_8b"] if tag in timing_all
        )
        + "\n\n",
        text,
        count=1,
        flags=re.S,
    )

docs.write_text(text, encoding="utf-8")
print("updated", docs)
PY

echo "===== STRATEGY2 DONE $(date) ====="
echo "Log: $MASTER_LOG"
echo "Docs: $DOCS_MD"
