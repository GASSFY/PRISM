# FP16 vs W4 comparison

Generated: 2026-09-14T18:08:09

Same env: `HF_HOME`, lmms-eval tasks, seed `0,1234,1234,1234`.
W4 cells from `experiments/psi_ablation/` (offline, r=0.01). Empty = not run yet.

## LLaVA-OV-7B

| Task | FP16 | W4 α=0.0 (K-only) | W4 α=0.2 (+Ψ) | FP16−W4@0 |
|------|------|-------------------|---------------|-----------|
| realworldqa | — | 0.6667 | 0.6601 | — |
| mmmu_val | — | 0.4878 | 0.4944 | — |
| ocrbench | — | — | — | — |
| ai2d | 0.8141 | — | — | — |

## InternVL2-8B

| Task | FP16 | W4 α=0.0 (K-only) | W4 α=0.2 (+Ψ) | FP16−W4@0 |
|------|------|-------------------|---------------|-----------|
| realworldqa | — | 0.6431 | 0.6523 | — |
| mmmu_val | — | 0.4789 | 0.4789 | — |
| ocrbench | — | — | — | — |
| ai2d | 0.8229 | — | — | — |

## Paths

- FP16 raw: `experiments/fp16_baseline/results/<model>_fp16/<task>/`
- FP16 markdown dump: `experiments/fp16_baseline/results/baseline.md`
- W4 Ψ ablation: `experiments/psi_ablation/results/downstream.md`

