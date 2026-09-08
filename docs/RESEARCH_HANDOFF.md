# PRISM 科研交接文档（RESEARCH_HANDOFF）

> **新会话必读。**  
> **PRISM** = Precision Retention by Importance Scoring for Multimodal。  
> 包名暂为 `asdq`。

---

## 0. 阶段划分（重要）

| 阶段 | 目标 | 做什么 | 不做什么 |
|------|------|--------|----------|
| **Phase-1（当前）** | 弄清伪量化后的**任务精度** | Hessian → ASD 选列 → SpQR 风格 **pseudo-quant** → lmms-eval | real-int4、CUDA kernel、推理 bench、部署压缩体积 |
| **Phase-2（以后）** | AI Infra / 真量化部署 | 再补 packing、kernel、显存/延迟 | — |

方法参照：本地 `../owq`、`../SpQR-main`（伪量化/敏感列思路）；**不是**抄它们的推理部署栈进本阶段。

---

## 1. 算法现状（Phase-1）

- **仅权重**伪量化：权重变成「量化再反量化」后的 float，前向仍是 FP matmul。  
- **保列**（输入通道）；**分组**是一行 × `w_group`。  
- \(\mathrm{ASD}=\theta_1 K+\theta_2\Psi\)，\(K=\|W_{:,c}\|^2 E[x_c^2]\)，\(\Psi\)=层内 z-score。  
- 细节见 [ALGORITHM.md](ALGORITHM.md)。

---

## 2. 新颖性提醒

- **K（绝对显著性）与 OWQ 同族**，不宜当唯一 novelty。  
- **Ψ** 需消融；无效则删。  
- 更稳叙事：全局融合 / MLLM / 后续可学保列·分模态·裁剪界的**非冗余增益**。

---

## 3. 实验路线

### Phase A — Ψ 消融

`theta2=0`（仅 K） vs 默认 `0.8/0.2`（可选再扫 θ）。其它固定。

### Phase B — 增量模块（仍伪量化）

可学保列 / 裁剪界 / 分模态；先单加再两两组合。定稿：增益明显且不冗杂。

### 校准协议

COCO 等校准集做统计与（若有）AdamW；**测试集不调参**。

---

## 4. MASQuant 可借鉴（仅权重阶段的用法）

路径：`e:\LLM-learning\MASQuant-EfficientAI`。  
可借鉴：校准上逐层重构 + 短 AdamW、分模态 mask。  
CMC / 多套 smooth 主攻 W–A，**不是 Phase-1 必做**。

---

## 5. 代码地图（Phase-1）

```text
main_quant.py
  → hessian_collector.collect_hessian_diag
  → mixed_precision + metrics/asd.py
  → quant_funcs / quantize.pseudo_quantize_model_weight
  → checkpoint.save_checkpoint   # float state_dict only

main_eval.py
  → eval_load.load_model_for_eval  # fp16 | overwrite pseudo weights
  → lmms-eval
```

已删除（勿再引用）：`real_quant.py`、`asdq/kernel/*`、`main_prompt_compare.py`、bench 推理脚本。

---

## 6. 本地对照

| 路径 | 用途 |
|------|------|
| `e:\LLM-learning\PRISM` | 本仓库 |
| `e:\LLM-learning\ASDQ` | 旧底座（只读；含已废弃部署代码） |
| `e:\LLM-learning\owq` | OWQ |
| `e:\LLM-learning\SpQR-main` | SpQR |
| `e:\LLM-learning\MASQuant-EfficientAI` | MASQuant |

---

## 7. 新会话下一步

1. 读 `SESSION_START.md` + 本文 + `ALGORITHM.md`  
2. 配 `configs/default.yaml`  
3. 冒烟：伪量化 + 小 limit eval  
4. **Phase A：Ψ 消融**
