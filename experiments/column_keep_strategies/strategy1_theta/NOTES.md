# NOTES — Strategy 1 (θ linear fusion + grid search)

Part of the four-strategy column-keep comparison.
Master write-up: `docs/exp_column_keep_strategies.md`.

- Select θ* **per model** by calib ΔCE only (8-point coarse+refine).
- Downstream RWQA / MMMU evaluate θ* only — never enter the search loop.
- Energy caches may be reused from `experiments/modality_theta/scale_cache/`.
