# 新会话入口（SESSION_START）

1. 读 [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md)（阶段目标 + 实验路线）  
2. 读 [ALGORITHM.md](ALGORITHM.md)（Phase-1 伪量化算法）  
3. 包名暂为 **`asdq`**；品牌 **PRISM**  
4. **当前只做伪量化评精度**；不要加 real-int4 / CUDA / 推理 bench  
5. 实验优先：**Phase A — Ψ 消融**；校准用 COCO，测试集不调参  

```bash
python main_quant.py --config configs/default.yaml
python main_eval.py --config configs/default.yaml
```
