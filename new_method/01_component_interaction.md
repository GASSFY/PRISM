# 实验 01：跨组件量化相互作用

结果在 `new_method/results/01_component_interaction/`：`w4.json`（LLaVA-OneVision）、`w4_qwen25vl.json`、`w4_internvl2.json`。数字和结论在文末。

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

第一轮先锁 LLaVA-OneVision（SigLIP + `mm_projector` + Qwen2）。同一套八格后来又跑了 Qwen2.5-VL 和 InternVL2-8B。这两个模型没有同名的 `mm_projector`，组件按实际算子拆开，并写进各自 JSON 的 `component_map`：

- Qwen2.5-VL：V = `visual.blocks`（merger 之前），P = `visual.merger`，L = 语言层。不能把整个 `visual` 当成 V，否则 merger 会同时算进 V 和 P。
- InternVL2：V = `vision_model`，P = `mlp1`，L = 语言层。中间的 `pixel_shuffle` 没有参数。`select_layer = -1`，视觉钩子取的是真正送进后面的最后一层隐状态。

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

来自 `new_method/results/01_component_interaction/w4.json`。LLaVA-OneVision Qwen2-7B，32 条 ShareGPT4V-COCO，监督 token 共 6100。视觉塔有 26 个线性层的输入维不能被 128 整除（SigLIP MLP 的 4304），这些层按输出通道一组，记在 JSON 的 `quant_stats` 里。

参数量（来自 JSON 的 `param_counts`）：

| 组件 | 线性层数 | 参数量 |
| --- | --- | --- |
| V | 156 | 396107296 |
| P | 2 | 16980992 |
| L | 196 | 6525417472 |

各格子的损失。fp 行是 0。

| 格子 | vision_out MSE | proj_out MSE | llm_early MSE | llm_mid MSE | llm_late MSE | logit KL |
| --- | --- | --- | --- | --- | --- | --- |
| fp | 0 | 0 | 0 | 0 | 0 | 0 |
| V | 0.7165 | 0.4863 | 0.3613 | 0.3195 | 7.520 | 0.008378 |
| P | 0 | 0.01463 | 0.01116 | 0.006371 | 0.2491 | 0.0002085 |
| L | 0 | 0 | 0.0002943 | 0.02930 | 1.606 | 0.03095 |
| VP | 0.7165 | 0.4996 | 0.3715 | 0.3230 | 7.566 | 0.008670 |
| VL | 0.7165 | 0.4863 | 0.3617 | 0.3421 | 8.309 | 0.03928 |
| PL | 0 | 0.01463 | 0.01149 | 0.03560 | 1.774 | 0.03104 |
| VPL | 0.7165 | 0.4996 | 0.3720 | 0.3457 | 8.339 | 0.03942 |

交互项（来自 JSON 的 `interactions`）：

| 探针 | \(I_{V,P}\) | \(I_{V,L}\) | \(I_{P,L}\) | \(I_{V,P,L}\) |
| --- | --- | --- | --- | --- |
| vision_out | 0 | 0 | 0 | 0 |
| proj_out | -0.001316 | 0 | 0 | 0 |
| llm_early | -0.000960 | 0.000175 | 0.000037 | 0 |
| llm_mid | -0.002889 | -0.006649 | -0.000074 | 0.000128 |
| llm_late | -0.2035 | -0.8177 | -0.0820 | 0.0661 |
| logit_kl | 0.000083 | -0.000045 | -0.000119 | -0.000036 |

sanity_flags：

`[]`

## 结论

只按下面四条读，不另加一个分数：

1. 若 logit KL 上 \(I_{V,L}\)、\(I_{V,P}\)、\(I_{P,L}\) 都接近 0，联合量化和分开量化是可加的。跨组件相互作用这条线在这个模型、这个比特上停。
2. 若 \(I_{V,L}\) 在 logit KL 上明显大于 0，而 \(I_{P,L}\) 接近 0，非加性主要在视觉和语言模型之间，和 2026 年 7 月那篇小模型观察同向。接着才看它从哪一个探针开始变大。
3. 若交互在 `vision_out` 已经很大，它不是对齐接口上长出来的，因为这一路还没经过投影和语言模型。若它在 `vision_out` 接近 0、从 `proj_out` 或 `llm_early` 才变大，耦合发生在视觉进入语言的这一段。
4. 投影层自己的 \(\mathcal{L}(P)\) 可以很小，同时 \(I_{V,P}\) 或 \(I_{P,L}\) 很大。那表示参数量小的接口仍在决定两边的量化误差能不能相加。只有这一格成立，才有理由接着谈按交互分配比特。在那之前不做混合精度算法。

这一轮落在第 1 条。

logit KL 上三个两两交互相对各自的单组件损失之和都在 1% 以内：\(I_{V,P}=8.3\times10^{-5}\)（和为 \(8.6\times10^{-3}\)），\(I_{V,L}=-4.5\times10^{-5}\)（和为 \(3.9\times10^{-2}\)），\(I_{P,L}=-1.2\times10^{-4}\)（和为 \(3.1\times10^{-2}\)）。联合量化和分开量化在 logit 上是可加的。跨组件相互作用这条线在 LLaVA-OneVision、W4 RTN、这 32 条图文上停。第 2 条不成立，因为 \(I_{V,L}\) 并没有明显大于 0。

第 3 条也不指向对齐接口。`vision_out` 上交互是 0；`proj_out` 和 `llm_early` 上相对单组件之和不到 1%。唯一离开 0 的是 `llm_late` 上的 \(I_{V,L}=-0.82\)，约为 \(\mathcal{L}(V)+\mathcal{L}(L)\) 的 9%，符号是补偿，而且没有传到 logit KL。

第 4 条不成立。投影层自己的 logit KL 很小（\(\mathcal{L}(P)=2.1\times10^{-4}\)，\(\mathcal{L}(L)=3.1\times10^{-2}\)），但 \(I_{V,P}\) 和 \(I_{P,L}\) 同样接近 0。这一轮不做按交互分配比特。

## 复核：量到的是不是交叉项

LLaVA 的八格损失与上一轮逐位相同，不是另一次随机前向。三个模型的 sanity 都是空的。每个被打开的组件，权重量化的相对 L2 都在 0.09 到 0.13，伪量化确实改了权重。

另外记了向量残差 \(\lVert e_{AB}-e_A-e_B\rVert^2\)。它小，表示联合误差真的约等于两个单独误差相加，而不是两个大误差碰巧在均方意义上正交、把一个很大的乘积项藏起来。投影层输出上，这个残差相对联合 MSE 大约是 1%：

| 模型 | \(e_{VP}\) 残差 | 联合 MSE |
| --- | --- | --- |
| LLaVA-OneVision | 0.00603 | 0.500 |
| Qwen2.5-VL | 0.00109 | 0.123 |
| InternVL2-8B | 0.000114 | 0.0109 |

所以在视觉进入语言的这一步，\(\Delta W\Delta h\) 是小项。最后一层激活上残差变大，大约是 \(\mathcal{L}(V)+\mathcal{L}(L)\) 的 15% 到 20%，但对应的 \(I_{V,L}\) 是负的，联合量化比相加稍好，不是更糟。

Qwen 的 `vision_out` MSE 是 3510，因为 merger 之前的特征 RMS 是 159。相对能量 \(\mathrm{MSE}/\mathrm{RMS}^2\) 是 0.14，和 LLaVA 的 0.21 同一量级。

## 三个模型的 logit KL

同一套 32 条 ShareGPT4V-COCO、W4 RTN。表里的比例是交互除以两个单组件损失之和。负号是补偿。

| 模型 | \(\mathcal{L}(V)\) | \(\mathcal{L}(P)\) | \(\mathcal{L}(L)\) | \(I_{V,P}\) | \(I_{V,L}\) | \(I_{P,L}\) |
| --- | --- | --- | --- | --- | --- | --- |
| LLaVA-OneVision | 0.00838 | 0.000209 | 0.0309 | +1.0% | -0.1% | -0.4% |
| Qwen2.5-VL | 0.00972 | 0.00103 | 0.0370 | -8.7% | -2.8% | -2.4% |
| InternVL2-8B | 0.0109 | 0.00235 | 0.0264 | -14% | -8.5% | -7.2% |

三个模型都没有出现「联合量化比相加更糟」。InternVL 的补偿最大，也只是让本来就小的 \(\mathcal{L}(V)+\mathcal{L}(P)\) 再小一截，没有超过语言模型自己的损失。第 2 条（\(I_{V,L}\) 明显大于 0）在三个模型上都不成立。第 4 条也不成立：投影层损失小，它和另外两侧的交互也没有大到该为它改比特。

因此，在这 32 条图文、这个比特、这个 RTN 上，不按交互做混合精度。这不覆盖榜单精度，尤其没有覆盖 OCR、图表、文档这类任务；那些还没有测。LLaVA 这一路用的是适配器里的 `pad` 单图，不是 OneVision 推理时的 anyres。
