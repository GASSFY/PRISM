# PRISM Phase-1 算法说明（伪量化）

> 本文描述 **Phase-1**：PRISM 分组混合精度伪量化 + 全局 \(K\) 选列，用于测多模态任务精度。  
> **不含** real-int4 packing、CUDA dequant、推理加速（后续阶段）。

**PRISM** = Precision Retention by Importance Scoring for Multimodal；包名 **`prism`**。

方法近邻：OWQ（敏感列）、SpQR（分组 + outlier）。本地可对照 `../owq`、`../SpQR-main`。

---

## 阶段边界

| 阶段 | 目标 | 做 | 不做 |
|------|------|----|------|
| **Phase-1（当前）** | 伪量化后的任务精度 | 激活能量 → \(K\) 选列 → 分组混合精度 **pseudo-quant** → lmms-eval | real-int4、CUDA kernel、推理 bench |
| **Phase-2（以后）** | 真量化部署 | packing / kernel / 显存延迟 | — |

Novelty 提醒：\(K\) 与 OWQ 同族，不宜当唯一卖点。已证伪并删除：逐层 sequential、相对显著性 \(\Psi\)、校准 hold-out 搜参协议（证据见 `experiments/*/NOTES.md`）。

---

## 分组

一个 group = **一行 × `w_group` 列**（如 128）。权重 `(out, in)` 经 `reshape(-1, w_group)` 后，每一行是一个 group，各有 scale/zero。

## 列重要度 \(K\)

\[
K_c = \|W_{:,c}\|^2 \cdot E[x_c^2]
\]

全局 max 归一化后排序，按保列比例 \(r\) 保留 float 列。\(r\) 可由平均比特预算换算：

\[
\mathrm{target\_bit}=(1-r)\cdot\mathrm{low}+r\cdot\mathrm{high}
\quad\Rightarrow\quad
r=\frac{\mathrm{target\_bit}-\mathrm{low}}{\mathrm{high}-\mathrm{low}}
\]

也可用 `target_bit: null` + `asd_high_precision_ratio` 直接指定 \(r\)（如 `0.01`）。

### 可选：跨模态融合

分别收集视觉 / 文本 token 的激活能量，再：

\[
K = \theta\cdot\mathrm{norm}(K^T) + (1-\theta)\cdot\mathrm{norm}(K^V)
\]

配置：`split_modality: true`，`modality_theta ∈ [0,1]`（1=纯文本，0=纯视觉）。

---

## 伪量化流程（offline only）

1. 校准收集全部层 `E[x_c²]`（全程 FP；可选按 `vision_mask` 拆模态）  
2. **全局** \(K\)（或融合 \(K\)）选高精度列  
3. 一口气对所有 Linear 做分组混合精度伪量化  
4. 保存 float state_dict  

调参只用校准 CE / 小验证集，**测试集不调参**。

---

## 核心代码地图

```text
main_quant.py
  → prism/calibration/hessian_collector.py
  → prism/quantization/mixed_precision.py + prism/metrics/asd.py
  → prism/quantization/quantize.py
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
