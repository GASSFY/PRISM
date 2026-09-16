# Ψ ablation — lean credible design

## Goal
Test whether Ψ (theta2>0) beats K-only (theta2=0) across models.

## Matrix (time-boxed)
| Dim | Choice | Why |
|-----|--------|-----|
| Models | LLaVA-OV-7B + InternVL2-8B | Cross-model; both cached |
| α | {0.0, 0.2} only | Drop 0.3 unless borderline later |
| Tasks | realworldqa + mmmu_val | Enough for gate; 3rd task deferred |
| Protocol | offline, r=0.01, 128 calib | Match main narrative |

## Future ablation protocol (reuse)
1. **Gate** (~per cell): calib ΔCE + RWQA
2. **Confirm** (only if gate non-null): +MMMU
3. **Full** (paper only): +ai2d/ocrbench/...

This run does Gate+Confirm together for Ψ (one-shot project gate).
