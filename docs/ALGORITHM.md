# PRISM Phase-1 算法说明（伪量化）

> 本文只描述 **Phase-1**：SpQR 风格伪量化 + ASD 选列，用于测精度。  
> **不含** real-int4 packing、CUDA dequant、推理加速等 AI Infra；那是后续阶段。

对照实现：`asdq/metrics/asd.py`、`asdq/quantization/mixed_precision.py`、`asdq/quantization/quant_funcs.py`。  
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
- 全局排序取 top `asd_high_precision_ratio` 的 **列** 保留 float

## 伪量化流程

1. 校准收集 `diag(H) ≈ E[x_c²]`  
2. 算 ASD，选高精度列  
3. 组内：高精度位用行均值填充 → 拟合 scale/zero → 量化反量化 → 写回原 float  
4. 保存 **float state_dict**（假量化后的权重）；评估时加载覆盖 FP 模型

端到端：`main_quant.py` → `main_eval.py`。实验路线见 [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md)。
