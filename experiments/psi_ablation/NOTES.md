# NOTES — Ψ ablation

## 本实验证明了什么

在 LLaVA-OV-7B 与 InternVL2-8B 上，加入相对显著性 Ψ（`asd_theta2>0`）相对纯 \(K\)（`asd_theta2=0`）**没有稳定收益**。选列回到全局归一化后的 \(K\) 即可。

## 因此从主代码删掉了什么

- `compute_Psi` / `compute_ASD` / `asd_theta1` / `asd_theta2`
- `ASD = θ1·K + θ2·Ψ` 组合公式

当前主路径分数为：

- 混合校准：`K = norm(||W||² · E[x²])`
- 跨模态：`K = θ·norm(K^T) + (1-θ)·norm(K^V)`

本目录保留 DESIGN / metrics / results，仅作证据存档，**不计入论文正式消融**。公开笔记：`docs/exp_psi_ablation.md`。
