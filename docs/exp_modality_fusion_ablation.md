# 实验：跨模态列重要度融合消融（θ∈{0, 0.5, 1}）

> 阶段 1：验证拆分模态后的等权融合是否优于单模态。不做 θ 网格搜索。

## 设定

| 项 | 值 |
|----|-----|
| 模型 | LLaVA-OneVision-Qwen2-7B-OV、InternVL2-8B |
| 校准 | COCO ShareGPT4V，`n_samples=128` |
| 公式 | `K = θ·norm(K^T) + (1-θ)·norm(K^V)` |
| 归一化 | 各模态 importance 先做**全局 max 归一化**，再融合 |
| θ | `0.0` 纯视觉 / `0.5` 等权混模态 / `1.0` 纯文本 |
| 选列分数 | 纯融合 \(K\)（Ψ 已从主路径移除） |
| keep ratio | `0.01`；`w_bit=4`, `w_group=128` |
| 评测 | RealWorldQA + MMMU-val（下游真分数） |

## 结果

| 模型 | 设置 | θ | RWQA exact_match | MMMU mmmu_acc | ΔCE（校准，参考） | kept |
|------|------|---:|-----------------:|-------------:|-----------------:|-----:|
| llava_ov | vision-only (θ=0) | 0.00 | 0.6654 | 0.4878 | 0.0257 | 11325 |
| llava_ov | fusion (θ=0.5) | 0.50 | 0.6680 | 0.4867 | 0.0250 | 11325 |
| llava_ov | text-only (θ=1) | 1.00 | 0.6706 | 0.4878 | 0.0219 | 11325 |
| internvl2_8b | vision-only (θ=0) | 0.00 | 0.6523 | 0.4800 | -0.0002 | 9830 |
| internvl2_8b | fusion (θ=0.5) | 0.50 | 0.6458 | 0.4822 | 0.0048 | 9830 |
| internvl2_8b | text-only (θ=1) | 1.00 | 0.6458 | 0.4833 | 0.0043 | 9830 |

## 结论

- **θ=0.5 没有稳定优于单模态**：两模型上差距约 0.5–0.7 个点，基本落在噪声带。
- LLaVA：RWQA 随 θ 略升（文本偏一点略好）；MMMU 几乎持平。
- InternVL：RWQA 上纯视觉最好；MMMU 上纯文本略高，差距极小。
- **ΔCE 与下游不对齐**：校准最优不等于下游最优。

阶段 1 结论：简单等权混模态（θ=0.5）尚不足以作为默认策略；跨模态路径保留在代码中，后续可换融合策略或更细 θ 搜索。

## 复现

```bash
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate prism
cd /root/autodl-tmp/quantization/PRISM
screen -dmS modality_ablation bash experiments/modality_theta/scripts/run_pipeline.sh
```

本地产物：`experiments/modality_theta/`（configs / logs / metrics / results）。
