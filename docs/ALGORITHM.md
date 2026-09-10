# PRISM Phase-1 算法说明（伪量化）

> 本文只描述 **Phase-1**：PRISM 分组混合精度伪量化 + ASD 选列，用于测精度。  
> **不含** real-int4 packing、CUDA dequant、推理加速等 AI Infra；那是后续阶段。

对照实现：`prism/metrics/asd.py`、`prism/quantization/mixed_precision.py`、`prism/quantization/quant_funcs.py`。  
方法近邻：OWQ（敏感列）、SpQR（分组 + outlier 处理）。本地可参考 `../owq`、`../SpQR-main`。

---

## 分组

一个 group = **一行 × `w_group` 列**（如 128）。权重 `(out, in)` 经 `reshape(-1, w_group)` 后，每一行是一个 group，各有 scale/zero。

## ASD

\[
\mathrm{ASD}_c = \theta_1 K_c^{\mathrm{norm}} + \theta_2 \Psi_c^{\mathrm{norm}}
\]

- \(K\)：\(\mathrm{importance}_c = \|W_{:,c}\|^2 \cdot E[x_c^2]\)，全局 max 归一化  
- \(\Psi\)：层内 z-score（负值置 0），再全局归一化  
- 全局排序取 top 保列比例 \(r\) 的 **列** 保留 float；\(r\) 由平均比特预算换算：

\[
\mathrm{target\_bit}=(1-r)\cdot\mathrm{low}+r\cdot\mathrm{high}
\quad\Rightarrow\quad
r=\frac{\mathrm{target\_bit}-\mathrm{low}}{\mathrm{high}-\mathrm{low}}
\]

默认 `target_bit=4.01`，`low=4`，`high=16`（保列按 FP16 计）。

## 伪量化流程

### 路线 A — offline（一口气）

1. 校准收集全部层 `diag(H) ≈ E[x_c²]`（全程 FP）  
2. **全局** ASD 选高精度列  
3. 一口气对所有 Linear 做分组混合精度伪量化  
4. 保存 float state_dict  

### 路线 B — sequential（逐层 MSE + 误差传递）

1. Catcher 缓存第 0 块输入  
2. 对每个 transformer block（在**当前**激活上）：  
   - 收集本层 `E[x²]`，**层内** ASD 选列 → 伪量化  
   - 短 AdamW：仅保列可训，层输出拟合 FP 输出（MSE）  
   - 用量化后权重 forward，输出作为下一块输入（误差下传）  
3. 保存 float state_dict  

配置：`quant_mode: offline | sequential`，`mse_epochs` / `mse_lr`（仅 sequential）。  
墙钟时间在 `main_quant.py` 日志中打印 `Quant wall time`。

端到端：`main_quant.py` → `main_eval.py`。对比脚本：`scripts/run_offline_vs_sequential.sh`。  
实验路线见 [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md)。
