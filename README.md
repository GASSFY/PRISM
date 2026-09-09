# PRISM

**PRISM**: Precision Retention by Importance Scoring for Multimodal

Phase-1：仅权重 **伪量化**（ASD 选列 + SpQR 风格分组），用 lmms-eval 测多模态精度。  
真量化部署 / CUDA kernel / 推理加速 → **后续阶段**，本仓库暂不包含。

> 代码底座源自 ASDQ；包名暂为 `asdq`。  
> 新会话：[docs/SESSION_START.md](docs/SESSION_START.md) → [docs/RESEARCH_HANDOFF.md](docs/RESEARCH_HANDOFF.md)

方法近邻（精度研究）：本地 `../owq`、`../SpQR-main`。

---

## 文档

| 文档 | 用途 |
|------|------|
| [docs/SESSION_START.md](docs/SESSION_START.md) | 最短入口 |
| [docs/RESEARCH_HANDOFF.md](docs/RESEARCH_HANDOFF.md) | 共识与实验路线 |
| [docs/ALGORITHM.md](docs/ALGORITHM.md) | Phase-1 算法 |

---

## 环境

```bash
conda create -n prism python=3.10 -y && conda activate prism
cd E:\LLM-learning\PRISM
pip install -r requirements.txt && pip install -e .

# 另装 LLaVA-NeXT、lmms-eval（本仓不内嵌）
```

校准：COCO JSON/JSONL + 图像目录。模型如 LLaVA-OneVision。

---

## 运行（伪量化）

```bash
python main_quant.py --config configs/default.yaml
python main_eval.py --config configs/default.yaml
```

关键项：`asd_theta1` / `asd_theta2`、`asd_high_precision_ratio`、`w_group`、`pseudo_quant: true`。

---

## 结构

```
PRISM/
├── main_quant.py / main_eval.py
├── configs/
├── asdq/          # calibration, metrics, quantization, models
├── docs/
└── scripts/       # 消融等（伪量化）
```
