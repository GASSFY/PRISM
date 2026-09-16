#!/usr/bin/env bash
# Strategy 1: per-model θ grid search (calib ΔCE) → downstream RWQA + MMMU.
# Prefer: screen -dmS strategy1_theta bash this_script
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=/root/autodl-tmp/hf_home
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true

OUT=experiments/column_keep_strategies/strategy1_theta
SEARCH_PY=experiments/modality_theta/scripts/search_modality_theta.py
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR" "$OUT/scale_cache" "$OUT/results" "$OUT/metrics" "$OUT/configs"
MASTER_LOG="$LOGDIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
DOCS_MD="$REPO_ROOT/docs/exp_column_keep_strategies.md"
TIMING_JSON="$OUT/metrics/timing_all.json"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "===== STRATEGY1 θ SEARCH START $(date) ====="
echo "REPO=$REPO_ROOT"
echo "Protocol: 8-point calib ΔCE search per model; downstream never used for θ*"
df -h /root/autodl-tmp | tail -1

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate prism
echo "python=$(which python)"
python -c "import lmms_eval; print('lmms_eval', lmms_eval.__file__)"

MODELS=(
  "llava_ov:experiments/column_keep_strategies/strategy1_theta/configs/llava_ov.yaml"
  "internvl2_8b:experiments/column_keep_strategies/strategy1_theta/configs/internvl2_8b.yaml"
)
TASKS=(realworldqa mmmu_val)

python - <<'PY'
import json
from pathlib import Path
Path("experiments/column_keep_strategies/strategy1_theta/metrics/timing_all.json").write_text("{}")
PY

for entry in "${MODELS[@]}"; do
  TAG="${entry%%:*}"
  CONFIG="${entry#*:}"
  MODEL_OUT="$OUT"
  ENERGY="$OUT/scale_cache/${TAG}_modality_energy.pt"
  SEARCH_METRICS="$OUT/metrics/${TAG}_search.json"
  SCALE="$OUT/scale_cache/${TAG}_best.pt"

  echo ""
  echo "######## MODEL ${TAG} $(date) ########"

  # Per-model search writes into a temp out_dir layout expected by search script
  SEARCH_DIR="$OUT/work_${TAG}"
  mkdir -p "$SEARCH_DIR/scale_cache" "$SEARCH_DIR/logs" "$SEARCH_DIR/metrics"

  # Prefer shared energy cache path via symlink already in scale_cache
  if [[ -e "$ENERGY" ]]; then
    ln -sfn "$(readlink -f "$ENERGY")" "$SEARCH_DIR/scale_cache/modality_energy.pt"
  fi

  echo "---- SEARCH ${TAG} $(date) ----"
  T0=$(date +%s)
  python "$SEARCH_PY" \
    --config "$CONFIG" \
    --out_dir "$SEARCH_DIR" \
    --energy_cache "$SEARCH_DIR/scale_cache/modality_energy.pt"
  T1=$(date +%s)
  SEARCH_WALL=$((T1 - T0))
  echo "[timing] ${TAG} search_wall=${SEARCH_WALL}s"

  # Promote artifacts
  cp -f "$SEARCH_DIR/metrics/search_modality_theta.json" "$SEARCH_METRICS"
  cp -f "$SEARCH_DIR/scale_cache/prism_modality_best.pt" "$SCALE"
  # keep energy in shared location if newly created
  if [[ -f "$SEARCH_DIR/scale_cache/modality_energy.pt" && ! -e "$ENERGY" ]]; then
    cp -f "$SEARCH_DIR/scale_cache/modality_energy.pt" "$ENERGY"
  fi

  THETA=$(python - <<PY
import json
m=json.load(open("${SEARCH_METRICS}"))
print(f"{m['best']['modality_theta']:.2f}")
PY
)
  echo "[result] ${TAG} θ*=${THETA}"

  EVAL_YAML="$OUT/configs/eval_${TAG}_best.yaml"
  python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${CONFIG}").read_text())
cfg["split_modality"] = True
cfg["modality_theta"] = float("${THETA}")
cfg["scale_path"] = "${SCALE}"
cfg["pseudo_quant"] = True
cfg["tasks"] = "realworldqa"
cfg["output_path"] = "experiments/column_keep_strategies/strategy1_theta/results/${TAG}_best"
Path("${EVAL_YAML}").write_text(
    yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
)
print("wrote", "${EVAL_YAML}")
PY

  declare -A TASK_SEC=()
  for TASK in "${TASKS[@]}"; do
    echo "---- EVAL ${TAG} θ*=${THETA} task=${TASK} $(date) ----"
    OUT_PATH="$OUT/results/${TAG}_best/${TASK}"
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

  # Merge timing into timing_all.json + per-model timing
  python - <<PY
import json
from pathlib import Path
tag = "${TAG}"
search = json.loads(Path("${SEARCH_METRICS}").read_text())
timing = dict(search.get("timing") or {})
timing["seconds_search_pipeline_wall"] = int("${SEARCH_WALL}")
timing["seconds_eval_realworldqa"] = int("${TASK_SEC[realworldqa]}")
timing["seconds_eval_mmmu_val"] = int("${TASK_SEC[mmmu_val]}")
timing["seconds_total"] = (
    timing.get("seconds_search_pipeline_wall", 0)
    + timing["seconds_eval_realworldqa"]
    + timing["seconds_eval_mmmu_val"]
)
out = {
    "model_tag": tag,
    "theta_star": search["best"]["modality_theta"],
    "delta_ce_star": search["best"]["delta_ce"],
    "n_kept": search["best"].get("n_kept"),
    "coarse_best": search.get("coarse_best"),
    "trials": search.get("trials"),
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

  # Free disk: drop work dir ckpt copy is enough; remove large work tree weights if any leftover
  rm -f "$SEARCH_DIR/scale_cache/prism_modality_best.pt" || true
  df -h /root/autodl-tmp | tail -1
done

echo "---- UPDATE DOCS $(date) ----"
python - <<'PY'
import json
import re
from pathlib import Path

root = Path("experiments/column_keep_strategies/strategy1_theta")
docs = Path("docs/exp_column_keep_strategies.md")
timing_all = json.loads((root / "metrics" / "timing_all.json").read_text())

# Parse downstream scores from lmms-eval results.json if present
def load_scores(tag: str) -> dict:
    out = {"realworldqa": None, "mmmu_val": None}
    base = root / "results" / f"{tag}_best"
    for task, key, metric in [
        ("realworldqa", "realworldqa", "exact_match,flexible-extract"),
        ("mmmu_val", "mmmu_val", "mmmu_acc,none"),
    ]:
        # find results.json under task dir
        cands = list((base / task).rglob("results.json"))
        if not cands:
            # sometimes lmms-eval nests deeper
            cands = list(base.rglob("results.json"))
            cands = [p for p in cands if task in str(p)]
        if not cands:
            continue
        data = json.loads(cands[0].read_text())
        res = data.get("results", {})
        # try common keys
        task_res = res.get(task) or res.get(key) or {}
        val = None
        for mk in [metric, "exact_match,flexible-extract", "mmmu_acc,none", "exact_match", "mmmu_acc"]:
            if mk in task_res:
                val = task_res[mk]
                break
        # fallback: first numeric-looking metric
        if val is None:
            for k, v in task_res.items():
                if isinstance(v, (int, float)) and ("exact_match" in k or "mmmu_acc" in k or k.endswith(",none")):
                    val = v
                    break
        out[task] = float(val) if val is not None else None
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

rows_search = []
rows_down = []
rows_time = []
summary_cells = {}

for tag in ["llava_ov", "internvl2_8b"]:
    if tag not in timing_all:
        continue
    m = timing_all[tag]
    scores = load_scores(tag)
    th = m["theta_star"]
    dce = m["delta_ce_star"]
    coarse = (m.get("coarse_best") or {}).get("modality_theta")
    trials = m.get("trials") or []
    trial_str = "; ".join(
        f"θ={t['modality_theta']:.2f}/ΔCE={t['delta_ce']:.6f}/{t['seconds']:.0f}s"
        for t in trials
    )
    rows_search.append(
        f"| {tag} | {fmt(coarse,2)} | {fmt(th,2)} | {fmt(dce,6)} | `{trial_str}` |"
    )
    rows_down.append(
        f"| {tag} | {fmt(th,2)} | {fmt(scores['realworldqa'])} | {fmt(scores['mmmu_val'])} | {fmt(m.get('n_kept'),0)} |"
    )
    tim = m.get("timing") or {}
    energy = tim.get("seconds_energy")
    search = tim.get("seconds_search_trials")
    if search is None:
        search = tim.get("seconds_search_pipeline_wall")
    final = tim.get("seconds_final_ckpt")
    rwqa_t = tim.get("seconds_eval_realworldqa")
    mmmu_t = tim.get("seconds_eval_mmmu_val")
    total = tim.get("seconds_total")
    rows_time.append(
        f"| {tag} | {fmt_s(energy)} | {fmt_s(search)} | {fmt_s(final)} | {fmt_s(rwqa_t)} | {fmt_s(mmmu_t)} | {fmt_s(total)} |"
    )
    summary_cells[tag] = scores

# Rebuild strategy-1 sections in the master doc while preserving placeholders for 2–4
text = docs.read_text(encoding="utf-8")

# Update summary table row for strategy 1
ll = summary_cells.get("llava_ov", {})
iv = summary_cells.get("internvl2_8b", {})
new_sum_row = (
    f"| 1 | \(\\theta K^T+(1-\\theta)K^V\) + 网格搜 θ | θ（每模型） | "
    f"{fmt(ll.get('realworldqa'))} | {fmt(ll.get('mmmu_val'))} | "
    f"{fmt(iv.get('realworldqa'))} | {fmt(iv.get('mmmu_val'))} | 完成 |"
)
text = re.sub(
    r"\| 1 \|.*?\| 进行中 \|",
    new_sum_row,
    text,
    count=1,
    flags=re.S,
)
# also replace if already "完成" from rerun
text = re.sub(
    r"\| 1 \|.*?\| 完成 \|",
    new_sum_row,
    text,
    count=1,
    flags=re.S,
)

def replace_table_after(header: str, new_rows: list[str], text: str) -> str:
    # Find header line then replace subsequent data rows until blank or next ##
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        out.append(lines[i])
        if lines[i].strip() == header:
            # keep next separator / header row(s) until we hit a data row starting with | and not --- 
            i += 1
            while i < len(lines) and lines[i].startswith("|"):
                out.append(lines[i])
                # stop copying after the markdown separator line
                if re.match(r"^\|\s*[-:]+", lines[i]):
                    i += 1
                    break
                i += 1
            # skip old data rows
            while i < len(lines) and lines[i].startswith("|"):
                i += 1
            for r in new_rows:
                out.append(r)
            continue
        i += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")

text = replace_table_after(
    "### θ 搜索轨迹",
    rows_search,
    text,
)
# The header for search table is the markdown table header under that section — replace by unique first col pattern
# Simpler: rewrite whole strategy 1 result blocks with markers

marker_start = "<!-- STRATEGY1_RESULTS_START -->"
marker_end = "<!-- STRATEGY1_RESULTS_END -->"
block = "\n".join(
    [
        marker_start,
        "### θ 搜索轨迹",
        "",
        "| 模型 | 粗扫最优 θ | θ\*（最终） | ΔCE\* | 8 次 trial 明细 |",
        "|------|-----------:|----------:|------:|----------------|",
        *rows_search,
        "",
        "### 下游结果（仅 θ\*）",
        "",
        "| 模型 | θ\* | RWQA exact_match | MMMU mmmu_acc | kept |",
        "|------|----:|-----------------:|-------------:|-----:|",
        *rows_down,
        "",
        "### 耗时（秒）",
        "",
        "分段计时，便于后续部署/推理优化对照。",
        "",
        "| 模型 | 能量收集 | θ 搜索（含试探伪量化+CE） | 最终 ckpt | RWQA 评测 | MMMU 评测 | 合计 |",
        "|------|--------:|--------------------------:|----------:|----------:|----------:|-----:|",
        *rows_time,
        marker_end,
    ]
)

if marker_start in text and marker_end in text:
    text = re.sub(
        re.escape(marker_start) + r".*?" + re.escape(marker_end),
        block,
        text,
        count=1,
        flags=re.S,
    )
else:
    # insert after strategy 1 formula section's "### θ 搜索轨迹" old content — replace from that header through 耗时 table
    pattern = r"### θ 搜索轨迹\n.*?\n### 复现"
    text = re.sub(pattern, block + "\n\n### 复现", text, count=1, flags=re.S)

# conclusion stub
concl = []
for tag, m in timing_all.items():
    concl.append(
        f"- **{tag}**: θ\*={m['theta_star']:.2f}, ΔCE\*={m['delta_ce_star']:.6f}; "
        f"RWQA={fmt(summary_cells.get(tag, {}).get('realworldqa'))}, "
        f"MMMU={fmt(summary_cells.get(tag, {}).get('mmmu_val'))}."
    )
concl_text = "\n".join(concl) if concl else "（跑完后填写）"
text = re.sub(
    r"### 本节结论\n\n.*?(?=\n---|\n## 策略 2)",
    f"### 本节结论\n\n{concl_text}\n\n",
    text,
    count=1,
    flags=re.S,
)

docs.write_text(text, encoding="utf-8")
print("updated", docs)
PY

echo "===== STRATEGY1 θ SEARCH DONE $(date) ====="
echo "Log: $MASTER_LOG"
echo "Docs: $DOCS_MD"
echo "Timing: $TIMING_JSON"
