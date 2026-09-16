# FP16 baseline (same env as Ψ ablation)

## Goal
Measure unquantized FP16 accuracy for LLaVA-OV-7B and InternVL2-8B under the same lmms-eval / HF cache / seed setup used by `experiments/psi_ablation/`.

## Matrix
| Dim | Choice |
|-----|--------|
| Models | LLaVA-OV-7B, InternVL2-8B |
| Precision | FP16 (no `scale_path`) |
| Tasks | realworldqa, mmmu_val, ocrbench, ai2d |
| Seed | `0,1234,1234,1234` (same as ψ run) |

## Where to look
- Per-run tables: `results/baseline.md`
- Flat comparison (FP16 vs W4 α=0/0.2): `results/comparison.md`
- Raw JSON: `results/<model>_fp16/<task>/results.json`
- Logs: `logs/`
