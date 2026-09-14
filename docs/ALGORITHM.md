# PRISM Phase-1 算法说明（伪量化）

> 本文描述 **Phase-1**：PRISM 分组混合精度伪量化 + ASD 选列，用于测多模态任务精度。  
> **不含** real-int4 packing、CUDA dequant、推理加速（后续阶段）。

**PRISM** = Precision Retention by Importance Scoring for Multimodal；包名 **`prism`**。

方法近邻：OWQ（敏感列）、SpQR（分组 + outlier）。本地可对照 `../owq`、`../SpQR-main`。

---

## 阶段边界

| 阶段 | 目标 | 做 | 不做 |
|------|------|----|------|
| **Phase-1（当前）** | 伪量化后的任务精度 | Hessian → ASD 选列 → 分组混合精度 **pseudo-quant** → lmms-eval | real-int4、CUDA kernel、推理 bench |
| **Phase-2（以后）** | 真量化部署 | packing / kernel / 显存延迟 | — |

Novelty 提醒：\(K\) 与 OWQ 同族，不宜当唯一卖点；\(\Psi\) 需消融。更稳叙事在全局列预算 / VLM 特化上。

---

## 分组

一个 group = **一行 × `w_group` 列**（如 128）。权重 `(out, in)` 经 `reshape(-1, w_group)` 后，每一行是一个 group，各有 scale/zero。

## ASD

\[
\mathrm{ASD}_c = \theta_1 K_c^{\mathrm{norm}} + \theta_2 \Psi_c^{\mathrm{norm}}
\]

- \(K\)：\(\mathrm{importance}_c = \|W_{:,c}\|^2 \cdot E[x_c^2]\)，全局 max 归一化  
- \(\Psi\)：层内 z-score（负值置 0），再全局归一化  
- 全局排序取 top 保列比例 \(r\) 的 **列** 保留 float；\(r\) 可由平均比特预算换算：

\[
\mathrm{target\_bit}=(1-r)\cdot\mathrm{low}+r\cdot\mathrm{high}
\quad\Rightarrow\quad
r=\frac{\mathrm{target\_bit}-\mathrm{low}}{\mathrm{high}-\mathrm{low}}
\]

也可用 `target_bit: null` + `asd_high_precision_ratio` 直接指定 \(r\)（如 `0.01`）。

---

## 伪量化流程

### 路线 A — offline（一口气）

1. 校准收集全部层 `diag(H) ≈ E[x_c²]`（全程 FP）  
2. **全局** ASD 选高精度列  
3. 一口气对所有 Linear 做分组混合精度伪量化  
4. 保存 float state_dict  

### 路线 B — sequential（逐层 MSE + 误差传递）

1. Catcher 缓存第 0 块输入  
2. 对每个 transformer block（在当前激活上）：层内 ASD → 伪量化 → 短 AdamW（仅保列）→ 量化权重 forward 下传  
3. 保存 float state_dict  

配置：`quant_mode: offline | sequential`，`mse_epochs` / `mse_lr`（仅 sequential）。

**当前主叙事：offline 全局列分配**；sequential 作对照 / 可选增强。跨模态应落在列重要度或层间预算上，避免做成 MBQ 式 scale 搜索续集。调参只用校准 CE / 小验证集，**测试集不调参**。

---

## 核心代码地图

```text
main_quant.py  (--quant_mode offline|sequential)
  offline:
    → prism/calibration/hessian_collector.py
    → prism/quantization/mixed_precision.py + prism/metrics/asd.py
    → prism/quantization/quantize.py
  sequential:
    → prism/quantization/sequential_pseudo_quant.py
  → prism/quantization/checkpoint.py   # float state_dict only

main_eval.py
  → prism/quantization/eval_load.py
  → lmms-eval
```

正式入口脚本：

```bash
bash scripts/quant.sh [configs/default.yaml]
bash scripts/eval.sh  [configs/default.yaml]
```

实验产物与实验专用脚本在本地 `experiments/`（不进远程仓库）。公开实验笔记见 [EXPERIMENTS.md](EXPERIMENTS.md)。
