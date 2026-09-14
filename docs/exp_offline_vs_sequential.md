# 实验：Offline vs Sequential

## 设定

| 项 | 值 |
|----|-----|
| 模型 | `lmms-lab/llava-onevision-qwen2-7b-ov` |
| 校准 | COCO ShareGPT4V，`n_samples=128` |
| θ1 / θ2 | 0.8 / 0.2 |
| keep ratio | 0.01（`target_bit: null`） |
| w_bit / w_group | 4 / 128 |
| Offline | 全局 ASD → 一次伪量化 |
| Sequential | 层内 ASD → 伪量化 → 误差下传，`mse_epochs=0` |
| 评测 | RealWorldQA |

## 结果

| 模式 | Quant wall time | kept 列 | RWQA exact_match |
|------|-----------------|---------|------------------|
| Offline | 71.9 s | 11325 | **0.6601** ±0.0171 |
| Sequential | 121.4 s | 11340 | **0.6588** ±0.0172 |

## 结论

Offline 与历史 ASDQ 参考（~0.66）对齐；Sequential 在本设定下与 Offline 基本打平（差约 0.001，落在 stderr 内）。Phase-1 主叙事继续以 **offline 全局列分配** 为主，sequential 作对照。

## 复现

本地产物：`experiments/offline_vs_sequential/`（configs / logs / results；scale 可能已清理）。  
正式入口仍用仓库根目录 `scripts/quant.sh` / `scripts/eval.sh`，配置见该实验目录下的 `configs/`。
