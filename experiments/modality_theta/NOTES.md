# NOTES — Modality fusion θ ablation

## 本实验在测什么

拆分视觉 / 文本二阶能量后，用固定网格 `θ ∈ {0, 0.5, 1}` 融合列重要度，并看下游 RealWorldQA + MMMU。

公式：`K = θ·norm(K^T) + (1-θ)·norm(K^V)`（各模态先做全局 max 归一化）。

## 当前结论（阶段 1）

θ=0.5 **没有**稳定优于单模态端点；差距多在 ~0.5–0.7 个点内，接近噪声。详见 `docs/exp_modality_fusion_ablation.md`。

## 与主代码的关系

跨模态融合路径保留在主代码（`--split_modality` + `--modality_theta`）。本目录是该能力的实验流水线与产物，不是已证伪分支。
