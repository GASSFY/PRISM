# NOTES — Column-keep strategy comparison

Local root for the four-strategy study documented in
`docs/exp_column_keep_strategies.md`.

| Subdir | Strategy |
|--------|----------|
| `strategy1_theta/` | \(\theta K^T+(1-\theta)K^V\) + grid search |
| `strategy2_minimax/` | worst-modality greedy (no searchable params) |
| `strategy3_shared/` | Shared→OnlyV/OnlyT quota (ρ=1.0) |

Strategy 4 (floor + disagreement bonus) was **dropped**: too complex for the
Occam's-razor story, and its \(\beta\) has no natural upper bound.

