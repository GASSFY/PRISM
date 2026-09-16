# NOTES — Offline vs Sequential

## 本实验证明了什么

**Offline 全局选列**与 **Sequential 逐层选列 + 误差下传**在 RealWorldQA 上基本打平（差约 0.001，stderr 内），但 offline 更快、实现更简单。Phase-1 主路径只保留 offline。

## 因此从主代码删掉了什么

- `quant_mode=sequential` / `--mse_epochs` / `--mse_lr`
- `prism/quantization/sequential_pseudo_quant.py`
- 层内 `select_high_precision_columns_local`

本目录保留 configs / logs / results，仅作证据存档，**不计入论文正式消融**。

详见公开笔记：`docs/exp_offline_vs_sequential.md`。
