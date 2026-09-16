# NOTES — Strategy 2 / A (worst-modality greedy)

Master write-up: `docs/exp_column_keep_strategies.md`.

- **No searchable parameters** (no θ / μ / β).
- Each round: rescue the modality with larger remaining uncovered gain
  \(L^m = G^m - C^m\) until keep-budget \(B\) is filled.
- Calib ΔCE is reported for reference only; downstream never enters selection.
