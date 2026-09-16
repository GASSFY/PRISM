# 实验目录

本页只做索引。详细设定与结果见对应笔记；原始日志 / checkpoint / 实验脚本在本地 `experiments/`（默认不推远程）。

| 实验 | 笔记 | 本地产物 | 是否进论文消融 |
|------|------|----------|----------------|
| Offline vs Sequential | [exp_offline_vs_sequential.md](exp_offline_vs_sequential.md) | `experiments/offline_vs_sequential/` | 否（证据：删 sequential） |
| Holdout vs Full | [exp_holdout_vs_full.md](exp_holdout_vs_full.md) | `experiments/holdout_vs_full/` | 否（证据：删 hold-out 协议） |
| Ψ ablation | [exp_psi_ablation.md](exp_psi_ablation.md) | `experiments/psi_ablation/` | 否（证据：删 Ψ） |
| 跨模态融合 θ∈{0,0.5,1} | [exp_modality_fusion_ablation.md](exp_modality_fusion_ablation.md) | `experiments/modality_theta/` | 阶段 1 固定网格探索 |
| **列选择策略对比（主对照）** | [exp_column_keep_strategies.md](exp_column_keep_strategies.md) | `experiments/column_keep_strategies/` | **是（核心）** |

每个本地产物目录有 `NOTES.md`，说明「证明了什么 → 因此删了什么」。

正式入口：`bash scripts/quant.sh` / `bash scripts/eval.sh`。算法与代码地图见 [ALGORITHM.md](ALGORITHM.md)。
