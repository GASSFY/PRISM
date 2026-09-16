#!/usr/bin/env python3
"""Build FP16 vs W4 comparison table for quick lookup."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FP16 = REPO / "experiments" / "fp16_baseline"
PSI = REPO / "experiments" / "psi_ablation"

TASK_METRIC = {
    "realworldqa": ("exact_match,flexible-extract", "exact_match"),
    "mmmu_val": ("mmmu_acc,none", "mmmu_acc"),
    "ocrbench": ("ocrbench_accuracy,none", "ocrbench_accuracy"),
    "ai2d": ("exact_match,flexible-extract", "exact_match"),
}


def _pick_metric(task: str, metrics: dict) -> float | None:
    preferred_key, substr = TASK_METRIC[task]
    if preferred_key in metrics and isinstance(metrics[preferred_key], (int, float)):
        return float(metrics[preferred_key])
    for k, v in metrics.items():
        if substr in k and isinstance(v, (int, float)) and "stderr" not in k:
            return float(v)
    return None


def load_score(results_json: Path, task: str) -> float | None:
    candidates = [results_json]
    # psi pipeline overwrote parent results.json with the last task only
    if results_json.parent.name == task:
        candidates.append(results_json.parent.parent / "results.json")
    for path in candidates:
        score = _load_score_file(path, task)
        if score is not None:
            return score
    return None


def _load_score_file(results_json: Path, task: str) -> float | None:
    if not results_json.exists():
        return None
    data = json.loads(results_json.read_text())
    # lmms-eval nests under results[task]
    block = data.get("results", data)
    task_block = block.get(task)
    if not isinstance(task_block, dict):
        # sometimes keys are aliased
        for k, v in block.items():
            if isinstance(v, dict) and (k == task or task in k):
                task_block = v
                break
    if not isinstance(task_block, dict):
        return None
    return _pick_metric(task, task_block)


def parse_downstream_md() -> dict[tuple[str, str, str], float]:
    """W4 scores from psi downstream.md (parent JSON only keeps last task)."""
    import re

    md = PSI / "results" / "downstream.md"
    if not md.exists():
        return {}
    order = [
        ("llava_ov", "w4_a0.00", "realworldqa"),
        ("llava_ov", "w4_a0.00", "mmmu_val"),
        ("llava_ov", "w4_a0.20", "realworldqa"),
        ("llava_ov", "w4_a0.20", "mmmu_val"),
        ("internvl2_8b", "w4_a0.00", "realworldqa"),
        ("internvl2_8b", "w4_a0.00", "mmmu_val"),
        ("internvl2_8b", "w4_a0.20", "realworldqa"),
        ("internvl2_8b", "w4_a0.20", "mmmu_val"),
    ]
    vals = re.findall(
        r"\|(?:realworldqa|mmmu_val)\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-9.]+)",
        md.read_text(),
    )
    return {key: float(raw) for key, raw in zip(order, vals)}


def main() -> None:
    models = [
        ("llava_ov", "LLaVA-OV-7B"),
        ("internvl2_8b", "InternVL2-8B"),
    ]
    tasks = ["realworldqa", "mmmu_val", "ocrbench", "ai2d"]
    variants = [
        ("fp16", FP16 / "results" / "{tag}_fp16" / "{task}" / "results.json"),
        ("w4_a0.00", PSI / "results" / "{tag}_a0.00" / "{task}" / "results.json"),
        ("w4_a0.20", PSI / "results" / "{tag}_a0.20" / "{task}" / "results.json"),
    ]

    lines: list[str] = []
    lines.append("# FP16 vs W4 comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("Same env: `HF_HOME`, lmms-eval tasks, seed `0,1234,1234,1234`.")
    lines.append("W4 cells from `experiments/psi_ablation/` (offline, r=0.01). Empty = not run yet.")
    lines.append("")

    md_scores = parse_downstream_md()
    for tag, pretty in models:
        lines.append(f"## {pretty}")
        lines.append("")
        header = "| Task | FP16 | W4 α=0.0 (K-only) | W4 α=0.2 (+Ψ) | FP16−W4@0 |"
        lines.append(header)
        lines.append("|------|------|-------------------|---------------|-----------|")
        for task in tasks:
            scores: dict[str, float | None] = {}
            for name, pattern in variants:
                path = Path(str(pattern).format(tag=tag, task=task))
                scores[name] = load_score(path, task)
                if scores[name] is None:
                    scores[name] = md_scores.get((tag, name, task))
            fp = scores["fp16"]
            a0 = scores["w4_a0.00"]
            a2 = scores["w4_a0.20"]
            gap = ""
            if fp is not None and a0 is not None:
                gap = f"{fp - a0:+.4f}"
            def fmt(x: float | None) -> str:
                return f"{x:.4f}" if x is not None else "—"
            lines.append(
                f"| {task} | {fmt(fp)} | {fmt(a0)} | {fmt(a2)} | {gap or '—'} |"
            )
        lines.append("")

    lines.append("## Paths")
    lines.append("")
    lines.append("- FP16 raw: `experiments/fp16_baseline/results/<model>_fp16/<task>/`")
    lines.append("- FP16 markdown dump: `experiments/fp16_baseline/results/baseline.md`")
    lines.append("- W4 Ψ ablation: `experiments/psi_ablation/results/downstream.md`")
    lines.append("")

    out = FP16 / "results" / "comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
