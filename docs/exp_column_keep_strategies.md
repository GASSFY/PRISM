# 跨模态列选择策略对比（总文档）

> 本实验是 Phase-1 **最关键对照**：在同一全局列预算下，比较多种跨模态保列协议，最终决定默认策略。  
> **下游任务绝不参与选参**；选参只用校准集 ΔCE（或该策略自身的非下游准则）。  
> 策略逐个补齐；结果统一汇总到本文。

## 共同设定

| 项 | 值 |
|----|-----|
| 模型 | LLaVA-OneVision-Qwen2-7B-OV、InternVL2-8B（**各自独立**跑完整协议） |
| 校准 | COCO ShareGPT4V，`n_samples=128`（全量，不切 hold-out） |
| 列预算 | `asd_high_precision_ratio=0.01`（`target_bit: null`） |
| 量化 | `w_bit=4`，`w_group=128`，伪量化 |
| 二阶代理 | \(K_c^m=\|W_{:,c}\|^2\cdot E[x_c^2\mid m]\)，\(m\in\{V,T\}\) |
| 归一化 | 各模态 importance **全局 max** 归一化后再进入策略公式 |
| 选参准则 | 校准集 ΔCE（越小越好）；**禁止**用 RWQA / MMMU 搜参 |
| 最终判定 | RealWorldQA `exact_match` + MMMU-val `mmmu_acc` |

## 策略一览（汇总表，随实验填充）

| # | 策略 | 可搜参数 | LLaVA RWQA | LLaVA MMMU | InternVL RWQA | InternVL MMMU | 状态 |
|---|------|----------|-----------:|-----------:|--------------:|--------------:|------|
| 1 | $	heta K^T+(1-	heta)K^V$ + 网格搜 θ | θ（每模型） | 0.6706 | 0.4878 | 0.6523 | 0.4800 | 完成 |
| 2 | 最差模态贪心 | 无 | 0.6654 | 0.4878 | 0.6523 | 0.4800 | 完成 |
| 3 | Shared → OnlyT/OnlyV | ρ=1.0（固定） | 0.6667 | 0.4844 | 0.6510 | 0.4722 | 完成 |
| 4 | 短板 + disagreement bonus | β, τ | — | — | — | — | **已剔除** |

本地产物根目录：`experiments/column_keep_strategies/`。

---

## 策略 1：线性融合 + θ 网格搜索

### 公式

\[
K=\theta\cdot\mathrm{norm}(K^T)+(1-\theta)\cdot\mathrm{norm}(K^V)
\]

- \(\theta=1\)：纯文本；\(\theta=0\)：纯视觉。

### 选参协议（每模型独立）

1. 收集（或加载缓存）视觉/文本激活能量。  
2. **8 次**校准 ΔCE 评估：  
   - 粗扫 \(\{0,0.25,0.5,0.75,1.0\}\)  
   - 在粗扫最优邻域用步长 \(0.1\) 再补 **3** 个未评估点  
3. 取 ΔCE 最小的 \(\theta^\*\)，重建 checkpoint。  
4. 仅用 \(\theta^\*\) 评下游（RWQA + MMMU）。

<!-- STRATEGY1_RESULTS_START -->
### θ 搜索轨迹

| 模型 | 粗扫最优 θ | θ\*（最终） | ΔCE\* | 8 次 trial 明细 |
|------|-----------:|----------:|------:|----------------|
| llava_ov | 1.00 | 1.00 | 0.021892 | `θ=0.00/ΔCE=0.025717/49s; θ=0.25/ΔCE=0.025724/41s; θ=0.50/ΔCE=0.024987/39s; θ=0.75/ΔCE=0.023226/38s; θ=1.00/ΔCE=0.021892/42s; θ=0.90/ΔCE=0.023772/41s; θ=0.80/ΔCE=0.024774/40s; θ=0.70/ΔCE=0.023449/40s` |
| internvl2_8b | 0.00 | 0.00 | -0.000196 | `θ=0.00/ΔCE=-0.000196/38s; θ=0.25/ΔCE=0.004250/39s; θ=0.50/ΔCE=0.004800/40s; θ=0.75/ΔCE=0.005569/39s; θ=1.00/ΔCE=0.004262/38s; θ=0.10/ΔCE=0.004505/40s; θ=0.20/ΔCE=0.004322/41s; θ=0.30/ΔCE=0.003989/41s` |

### 下游结果（仅 θ\*）

| 模型 | θ\* | RWQA exact_match | MMMU mmmu_acc | kept |
|------|----:|-----------------:|-------------:|-----:|
| llava_ov | 1.00 | 0.6706 | 0.4878 | 11325 |
| internvl2_8b | 0.00 | 0.6523 | 0.4800 | 9830 |

### 耗时（秒）

分段计时，便于后续部署/推理优化对照。

| 模型 | 能量收集 | θ 搜索（含试探伪量化+CE） | 最终 ckpt | RWQA 评测 | MMMU 评测 | 合计 |
|------|--------:|--------------------------:|----------:|----------:|----------:|-----:|
| llava_ov | 0.1 | 329.7 | 31.0 | 793.0 | 420.0 | 1689.0 |
| internvl2_8b | 0.0 | 315.8 | 30.8 | 318.0 | 272.0 | 1053.0 |
<!-- STRATEGY1_RESULTS_END -->

### 复现

```bash
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate prism
cd /root/autodl-tmp/quantization/PRISM
screen -dmS strategy1_theta bash experiments/column_keep_strategies/strategy1_theta/scripts/run_pipeline.sh
```

### 本节结论

- **llava_ov**: θ\*=1.00, ΔCE\*=0.021892; RWQA=0.6706, MMMU=0.4878。
- **internvl2_8b**: θ\*=0.00, ΔCE\*=-0.000196; RWQA=0.6523, MMMU=0.4800。
- 两模型 θ\* 都落在**单模态端点**（LLaVA 纯文本、InternVL 纯视觉），等权融合 θ=0.5 并非校准最优。

---

## 策略 2：最差模态贪心

**无任何可学习 / 可网格搜索参数。**

### 算法

1. 对各列算 \(g_c^V=\mathrm{norm}(K_c^V)\)、\(g_c^T=\mathrm{norm}(K_c^T)\)。  
2. \(G^m=\sum_c g_c^m\)，已覆盖 \(C^m\)，剩余损失 \(L^m=G^m-C^m\)。  
3. 每轮 \(m^*=\arg\max_m L^m\)，在未选列中取对 \(m^*\) 增益最大的列（另一模态作 tie-break）。  
4. 更新 \(C^V,C^T\)，直到 \(|\Omega|=B\)。

校准 ΔCE 只作参考，**不参与选列**；下游只做最终判定。

<!-- STRATEGY2_RESULTS_START -->
### 结果

| 模型 | 校准 ΔCE（参考） | RWQA exact_match | MMMU mmmu_acc | kept |
|------|-----------------:|-----------------:|-------------:|-----:|
| llava_ov | 0.025717 | 0.6654 | 0.4878 | 11325 |
| internvl2_8b | -0.000196 | 0.6523 | 0.4800 | 9830 |

### 耗时（秒）

| 模型 | 能量收集 | 贪心选列 | 伪量化 | RWQA 评测 | MMMU 评测 | 合计 |
|------|--------:|---------:|-------:|----------:|----------:|-----:|
| llava_ov | 0.0 | 12.3 | 2.5 | 794.0 | 421.0 | 1331.0 |
| internvl2_8b | 0.0 | 16.2 | 2.4 | 323.0 | 282.0 | 729.0 |
<!-- STRATEGY2_RESULTS_END -->

### 复现

```bash
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate prism
cd /root/autodl-tmp/quantization/PRISM
screen -dmS strategy2_minimax bash experiments/column_keep_strategies/strategy2_minimax/scripts/run_pipeline.sh
```

### 本节结论

（跑完后填写。）

## 策略 3：Shared → OnlyT / OnlyV

本轮固定 \(\rho=1.0\)，剩余预算对半给 OnlyV / OnlyT；**不搜融合权重**。

### 算法

1. \(\Omega^V=\mathrm{Top}_{B}(K^V)\)，\(\Omega^T=\mathrm{Top}_{B}(K^T)\)。  
2. \(\mathrm{Shared}=\Omega^V\cap\Omega^T\)，OnlyV / OnlyT 为对称差。  
3. 先取 Shared（按 \(g^V+g^T\)）；\(B'=B-|\mathrm{Shared}|\)。  
4. \(\lfloor B'/2\rfloor\) 给 OnlyV（按 \(K^V\)），其余给 OnlyT（按 \(K^T\)）。

<!-- STRATEGY3_RESULTS_START -->
### 结果

| 模型 | 校准 ΔCE（参考） | RWQA exact_match | MMMU mmmu_acc | kept |
|------|-----------------:|-----------------:|-------------:|-----:|
| llava_ov | 0.022090 | 0.6667 | 0.4844 | 11325 |
| internvl2_8b | 0.004207 | 0.6510 | 0.4722 | 9830 |

### 保列构成

| 模型 | Shared | OnlyV | OnlyT | fallback |
|------|-------:|------:|------:|---------:|
| llava_ov | 6738 | 2293 | 2294 | 0 |
| internvl2_8b | 5946 | 1942 | 1942 | 0 |

### 耗时（秒）

| 模型 | 能量收集 | 配额选列 | 伪量化 | RWQA 评测 | MMMU 评测 | 合计 |
|------|--------:|---------:|-------:|----------:|----------:|-----:|
| llava_ov | 0.0 | 10.2 | 2.5 | 797.0 | 425.0 | 1339.0 |
| internvl2_8b | 0.0 | 10.5 | 2.3 | 415.0 | 283.0 | 809.0 |
<!-- STRATEGY3_RESULTS_END -->

### 复现

```bash
screen -dmS strategy3_shared bash experiments/column_keep_strategies/strategy3_shared/scripts/run_pipeline.sh
```

### 本节结论

（跑完后填写。）

## 策略 4：短板 + disagreement bonus（已剔除，不再实验）

原公式 \(S_c=\min(K_c^V,K_c^T)+\beta\cdot|K_c^V-K_c^T|\cdot\mathbf{1}[\min\ge\tau]\)。

剔除理由：

1. 形式复杂，与「简单直观」的方法论取向冲突（奥卡姆剃刀）。
2. \(\beta\) 是无界倍率，没有像 θ 那样来自凸组合的自然上下界，网格/贝叶斯搜索都要先人为画框。
3. 两个参数异构（一个权重、一个门槛），二维搜参成本翻倍，而策略 1–3 的差异本身已在评测噪声内。

代码中的 `select_columns_disagreement_bonus` 已同步删除。

---

## 三策略对照结论（W4/g128, r=0.01）

| 结论 | 证据 |
|------|------|
| 三策略差异**全在噪声内**，此设定无法判优 | RWQA stderr ±0.0171，而最大差距仅 0.52pp（LLaVA）／0.13pp（InternVL） |
| **策略 2 退化为纯视觉 Top-B**，非独立策略 | ΔCE 与 θ=0 trial 完全一致；`\|S2∩S3\|` = 9031／7888，恰等于 S3 的 Shared+OnlyV 保留数 |
| 该设定**几乎没有量化损失余量** | InternVL ΔCE = −0.0002（量化后 CE 略低于 FP） |

策略 2 的退化根因：\(L^m=G^m-C^m\) 用未归一化总量，而 \(G^V\gg G^T\)，导致每轮恒选视觉。


