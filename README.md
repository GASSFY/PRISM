# PRISM

**PRISM**: Precision Retention by Importance Scoring for Multimodal

Phase-1：仅权重 **伪量化**（全局 \(K\) 选列 + 分组混合精度保列），用 lmms-eval 测多模态精度。  
真量化部署 / CUDA kernel / 推理加速 → **后续阶段**，本仓库暂不包含。

> 代码底座源自 ASDQ；包名 **`prism`**。  
> 算法：[docs/ALGORITHM.md](docs/ALGORITHM.md) · 实验索引：[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)

方法近邻（精度研究）：本地 `../owq`、`../SpQR-main`。

---

## 文档

| 文档 | 用途 |
|------|------|
| [docs/ALGORITHM.md](docs/ALGORITHM.md) | 算法说明 + 核心代码地图 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 实验目录（索引） |
| [docs/exp_modality_fusion_ablation.md](docs/exp_modality_fusion_ablation.md) | 跨模态 θ 融合消融 |
| [docs/exp_offline_vs_sequential.md](docs/exp_offline_vs_sequential.md) | Offline vs Sequential（已删 sequential） |
| [docs/exp_holdout_vs_full.md](docs/exp_holdout_vs_full.md) | Holdout vs Full（已删 hold-out 协议） |

本地实验产物在 `experiments/`（默认不推远程）；各子目录有 `NOTES.md` 说明删改依据。

---

## 环境

```bash
conda create -n prism python=3.10 -y && conda activate prism
cd /path/to/PRISM
pip install -r requirements.txt && pip install -e .

# 另装 LLaVA-NeXT、lmms-eval（本仓不内嵌）
```

校准：COCO JSON/JSONL + 图像目录。模型如 LLaVA-OneVision。

---

## 运行（伪量化）

```bash
bash scripts/quant.sh configs/default.yaml
bash scripts/eval.sh  configs/default.yaml
```

等价于直接调用 `main_quant.py` / `main_eval.py`。  
关键项：`target_bit` 或 `asd_high_precision_ratio`、`w_group`、可选 `split_modality` + `modality_theta`。

---

## 结构

```
PRISM/
├── main_quant.py / main_eval.py
├── configs/
├── prism/           # calibration, metrics, quantization, models
├── docs/            # ALGORITHM + EXPERIMENTS 索引 + 实验笔记
├── scripts/         # quant.sh / eval.sh（正式入口）
└── experiments/     # 本地实验产物与实验脚本（gitignore）
```
