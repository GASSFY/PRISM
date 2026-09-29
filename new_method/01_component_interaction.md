# 实验 01：跨组件量化相互作用

结果文件还没有。远程跑完 `run.py` 之后，把 JSON 里的 `cells` 和 `interactions` 填进文末两张表，再按「怎么读结果」写结论。不要在数字出来之前写结论。

## 目的

怀疑的是这件事：把视觉编码器、投影层、语言模型各自量化时看起来不大的误差，叠在一起并不等于各误差相加。要验证的也是这件事，外加它最先出现在哪一段。

不验证「哪个组件更敏感」。组件敏感度已经有人测过（arXiv:2607.08029）。也不验证一个新的重要性分数。PRISM 的列选择不参加这个实验。

损失记成 \(\mathcal{L}(S)\)，\(S\) 是被量化的组件集合，数值越大越差，全精度的损失当作 0。两两交互是

\[
I_{A,B} = \mathcal{L}(A \cup B) - \mathcal{L}(A) - \mathcal{L}(B)
\]

三者交互把三对的贡献去掉：

\[
I_{V,P,L} = \mathcal{L}(VPL) - \mathcal{L}(VP) - \mathcal{L}(VL) - \mathcal{L}(PL) + \mathcal{L}(V) + \mathcal{L}(P) + \mathcal{L}(L)
\]

\(I>0\) 表示联合量化比相加更糟，\(I<0\) 表示彼此有补偿，\(I \approx 0\) 表示可加。公式的符号和空集项在 `interaction.py` 里，本地已经用可加例子和超加例子对过。

## 设计

模型只锁 LLaVA-OneVision（SigLIP + `mm_projector` + Qwen2）。InternVL、Qwen2.5-VL 的接口不是这块投影层，不放进这一轮。

八个格子，字母表示该组件做了伪量化：

| 格子 | 视觉编码器 | 投影层 | 语言模型 |
| --- | --- | --- | --- |
| fp | 全精度 | 全精度 | 全精度 |
| V | 量化 | 全精度 | 全精度 |
| P | 全精度 | 量化 | 全精度 |
| L | 全精度 | 全精度 | 量化 |
| VP | 量化 | 量化 | 全精度 |
| VL | 量化 | 全精度 | 量化 |
| PL | 全精度 | 量化 | 量化 |
| VPL | 量化 | 量化 | 量化 |

量化器在所有被打开的组件上是同一套 RTN：仅权重、4 bit、group 128、非对称。输入通道不能被 128 整除的线性层退回按输出通道一组，并记在 JSON 的 `quant_stats` 里。不用 AWQ、GPTQ，也不用 PRISM 的保列。那些方法自己会补偿误差，补偿和「误差是否相加」会缠在一起。

不量化的部分：patch embedding 的卷积、词嵌入、`lm_head`、归一化。投影层参数量远小于语言模型，这是这个实验要面对的事实，不是要在这一轮里改比特去迁就它。

每个格子都从全精度权重恢复，再量化，再重新跑视觉编码。校准数据不能事先做成 `inputs_embeds` 缓存，否则视觉和投影的量化不会进入后面的语言模型。探针挂在：

- `vision_out`：视觉塔输出
- `proj_out`：投影层输出
- `llm_early` / `llm_mid` / `llm_late`：语言模型第 0 层、中间层、最后一层的输出
- `logit_kl`：监督 token 上 \(\mathrm{KL}(p_{\mathrm{FP}} \Vert p_Q)\)，logits 与 labels 按因果语言模型的方式错开一位

激活误差是和全精度前向相比的均方误差。参考激活以 float16 存在 CPU 上，W4 的误差应远大于这次舍入。

隔离是否做成，由脚本里的 sanity 判断，不靠目测：

- 只量化视觉时，`vision_out` 的误差必须明显大于 0。否则钩子没挂到视觉塔，或者前面有缓存。
- 只量化投影或只量化语言模型时，`vision_out` 的误差相对「只量化视觉」应可以忽略。
- 只量化语言模型时，`proj_out` 同样应可以忽略。

sanity 失败时不计算交互项。那种 JSON 不能拿来解释耦合。

默认 32 条 COCO 图文、batch size 1。这是第一刀，用来看交互是不是大到值得继续。把它换成 128 条只改配置里的 `n_samples`。这一轮不跑 lmms-eval。榜要等 logit 上的 \(I_{V,L}\) 确实离开 0 再开，否则八组完整评测没有对象。

## 怎么跑

在 PRISM 根目录、装好 `lmms-eval` 和 LLaVA 的环境里：

```bash
python new_method/exp/component_interaction/run.py \
    --config new_method/exp/component_interaction/config.yaml
```

配置里必须填 `model_args`、`data_path`、`image_folder`。结果写到 `new_method/results/01_component_interaction/w4.json`。

本地可以不加载模型，只核对交互公式：

```bash
python new_method/exp/component_interaction/interaction.py
```

## 结果

尚未在远程运行。跑完后把 JSON 填进下面两张表。

参数量（来自 JSON 的 `param_counts`）：

| 组件 | 线性层数 | 参数量 |
| --- | --- | --- |
| V |  |  |
| P |  |  |
| L |  |  |

各格子的损失。fp 行应接近 0。空着表示还没有数。

| 格子 | vision_out MSE | proj_out MSE | llm_early MSE | llm_mid MSE | llm_late MSE | logit KL |
| --- | --- | --- | --- | --- | --- | --- |
| fp |  |  |  |  |  |  |
| V |  |  |  |  |  |  |
| P |  |  |  |  |  |  |
| L |  |  |  |  |  |  |
| VP |  |  |  |  |  |  |
| VL |  |  |  |  |  |  |
| PL |  |  |  |  |  |  |
| VPL |  |  |  |  |  |  |

交互项（来自 JSON 的 `interactions`）：

| 探针 | \(I_{V,P}\) | \(I_{V,L}\) | \(I_{P,L}\) | \(I_{V,P,L}\) |
| --- | --- | --- | --- | --- |
| vision_out |  |  |  |  |
| proj_out |  |  |  |  |
| llm_early |  |  |  |  |
| llm_mid |  |  |  |  |
| llm_late |  |  |  |  |
| logit_kl |  |  |  |  |

sanity_flags：

（跑完后粘贴。非空则下面的结论不成立，先修隔离。）

## 结论

数字写上之前，这里没有结论。有数字之后只按下面四条读，不另加一个分数：

1. 若 logit KL 上 \(I_{V,L}\)、\(I_{V,P}\)、\(I_{P,L}\) 都接近 0，联合量化和分开量化是可加的。跨组件相互作用这条线在这个模型、这个比特上停。
2. 若 \(I_{V,L}\) 在 logit KL 上明显大于 0，而 \(I_{P,L}\) 接近 0，非加性主要在视觉和语言模型之间，和 2026 年 7 月那篇小模型观察同向。接着才看它从哪一个探针开始变大。
3. 若交互在 `vision_out` 已经很大，它不是对齐接口上长出来的，因为这一路还没经过投影和语言模型。若它在 `vision_out` 接近 0、从 `proj_out` 或 `llm_early` 才变大，耦合发生在视觉进入语言的这一段。
4. 投影层自己的 \(\mathcal{L}(P)\) 可以很小，同时 \(I_{V,P}\) 或 \(I_{P,L}\) 很大。那表示参数量小的接口仍在决定两边的量化误差能不能相加。只有这一格成立，才有理由接着谈按交互分配比特。在那之前不做混合精度算法。
