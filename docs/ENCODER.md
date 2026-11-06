# TempoBridge 编码器

主编码器为 `model.encoder="multiview"`，提供连续波形 token 与时间序列 Transformer。
实现位于 `src/tempobridge/encoding.py`，由 `src/tempobridge/model.py` 的 `TempoBridge` 调用。
所有发布的运行配置显式选择该路径。旧 `local` 编码器保留，用于读取历史配置与 checkpoint。

## 从波形到 token

输入为单通道 `[B,1,T]`，每个 patch 有 P=`patch_size` 个采样点，N=T/P。
输入必须有限、长度整除 P，并至少含两个 patch。不使用离散分词器、词表、码本或文本模型。

`TempoBridgeTokenizer` 同时提取四路连续信息：

| 分支 | 实现顺序 | 每个 patch 的输出 |
|---|---|---|
| 原波形 | 可见点标准化 → 全序列 Conv1d → 通道 LayerNorm → patch 内均值 | D 维 |
| 相邻差分 | 相邻采样差分 → 清除端点不可见的差分 → 标准化 → 全序列 Conv1d → 通道 LayerNorm → patch 内均值 | D 维 |
| 局部均值 | 模型内部标准化前计算 patch 均值 → TempoBridgeScalarEncoder | S 维 |
| 局部标准差 | 模型内部标准化前计算 patch 样本标准差 → TempoBridgeScalarEncoder | S 维 |

两条卷积分支各自学习参数，默认输入通道 1、输出通道 D、卷积核 41、步长 1、对称 padding 20。
卷积先于 patch 池化，因而可以跨 patch 边界提取邻域信息。默认 D=256、S=32：
四路拼接为 576 维，再经线性层投影为 256 维并做 LayerNorm，输出 `[B,N,256]`。
隐藏 patch 的最终输出替换为同一个可学习 mask token。

原波形描述形状，差分强调局部变化，统计支路补充窗口标准化可能弱化的局部水平和幅度。
这增加了 token 的输入信息与计算量；差分也会放大高频噪声。固定卷积核在不同采样率下覆盖的秒数不同，
例如 41 点在 100 Hz/50 Hz 下分别约为 0.41/0.82 秒，需要在真实数据上检查是否合适。
卷积池化仍会压缩 patch 内的细节，不能保证保留每个心搏事件。

## 标量的多尺度编码

`TempoBridgeScalarEncoder` 为每个标量 v 建立多个幅度尺度视图。默认尺度为
`[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]`，每个视图有可学习的斜率、偏移和 LayerNorm：

```text
view_i = LayerNorm_i(v * slope_i + scale_i * offset_i)
weight_i = 1 / abs(log(abs(v) / scale_i + epsilon))
embedding = sum_i(normalize_sum(weight)_i * view_i)
```

默认 `scalar_epsilon=1.1`，保证权重公式可计算；均值和标准差使用两套独立的编码器。
相关计算以 float32 进行，以减轻混合精度下极端尺度的数值问题。
这是一组尺度视图的加权连续嵌入，没有把浮点数映射到离散 token ID。
它可能适应不同数值范围，但不自动消除设备差异或提供生理幅值标定。

统计量来自“进入模型的波形”，不是恢复了采集设备的原始电压。如果数据转换器已经归一化，
这些统计量反映的就是转换后的单位。不能把该分支直接解释为真实血压或灌注幅度。

## Transformer 与输出

在 N 个 patch token 后追加可学习 CLS token，形成 `[B,N+1,D]`。
每个 `TempoBridgeAttentionBlock` 使用以下结构：

```text
u = RMSNorm(tokens)
q, k, v = Linear(u)
q, k = rotary_with_XPos_scaling(q, k)
h = tokens + Linear(bidirectional_attention(q, k, v))
gate, value = Linear(RMSNorm(h))
output = h + Dropout(Linear(SiLU(gate) * value))
```

Q/K 的旋转携带 token 位置，XPos 缩放分别以正向和倒数尺度作用于 Q/K。
SwiGLU 用门控控制前馈特征。默认 6 层、8 头、D=256、前馈隐藏宽度 512；
前 2 层构成受保护的浅层，后 4 层构成上下文层。注意力是双向的，不是在线因果网络。
这些组件提供不同的表示与数值处理方式，并不保证更长序列上的泛化或下游性能。

`forward_features(ppg)` 提供 `embedding`、`tokens`、`summary`、`context` 和 `morphology`。
其中 `tokens` 不含 CLS，`context` 是这些 patch token 的均值。

| `feature_pool` | `embedding` 的构成 | 默认 D=256 时维度 |
|---|---|---|
| `combined`，默认 | CLS 与 patch token 均值拼接 | 512 |
| `cls` | CLS | 256 |
| `mean` | patch token 均值 | 256 |
| `morph_context` | 受保护浅层形态支路均值与上下文均值拼接 | 512；关闭形态支路时为 256 |

默认 `combined` 中没有拼接形态支路。关闭 `morphology_branch` 取消辅助形态重建，
不改变 `combined` 的输出维度。该辅助支路从受保护浅层的 patch token 读取信息，
经残差 MLP 与 RMSNorm 后供重建头使用。

`feature_layer=-1` 读取最终层；`feature_layer=0` 读取第一个 block 后的快照，以此类推。
该选项改变公共特征接口的 CLS、patch token 和上下文均值；形态支路仍取受保护浅层末端。
实现仍计算全部 block，没有因选择较浅快照而降低编码成本。蒸馏和重建训练默认使用完整编码器输出，
不会自动把损失转移到选定快照。校准会读取公共特征接口，因此变更该设置后必须重新校准。

## 时间尺度与固定 token 数

推荐设置 `num_tokens=0`，固定每个 patch 的采样点数：

| 配置 | 采样率 | `patch_size` | patch 时长 | 2 秒/8 秒的 patch token 数 |
|---|---|---|---|---|
| `configs/base.json` | 100 Hz | 25 | 0.25 秒 | 8 / 32 |
| `configs/pulsebound_50hz.json` | 50 Hz | 10 | 0.20 秒 | 10 / 40 |

表中不计额外 CLS。按固定物理时长分 patch，使不同窗口长度的 token 有相同时间含义。
ECG 与 PPG 仍分别按各自采样率裁剪；ECG 不直接进入学生 token。

可选 `num_tokens=32` 只是输入约束，要求 T=`32*patch_size`，不是自动插值到 32 个 token。
同时使用 2 秒和 8 秒窗口的现有训练配置不能直接改成该约束，否则某些尺度会被拒绝。
短长尺度应继续用 `num_tokens=0`；若设计固定长度单尺度实验，应一致修改数据、尺度与校准设置。

## 掩码、目标与训练边界

隐藏采样点在原波形、差分、统计提取和标准化前被清除；差分的相邻两端必须可见，末端补零。
各分支只用有效可见位置计算标准化统计。全波形卷积可以读取跨 patch 的可见邻域，
但无法通过输入数值读取隐藏目标。随后隐藏 patch 的融合 token 被 mask token 覆盖。

重建目标另行使用完整波形和可见点均值/方差标准化，并在损失端 detach。
目标标准化与 tokenizer 内部标准化是两条独立路径，不把重建目标送回 token 提取器。
改变隐藏波形只能改变监督目标，不能改变编码器的可见信息。

训练保持 PPG 掩码重建、ECG 教师、跨患者可预测性校准、尺度选择性 Huber 蒸馏与浅层梯度保护。
蒸馏前向将浅层输出 detach；PPG 自监督仍更新 tokenizer、CLS 和浅层。
教师只在训练/校准阶段提供目标；公共 `forward(ppg)` 与导出的编码器仅需 PPG。
数学细节见 [方法说明](METHOD.md)。

## 历史模型兼容与实验范围

`ModelConfig()` 默认构建 `multiview`。为了读取历史文件，`ModelConfig.from_dict` 在输入字典
缺少 `encoder` 时选择 `local`，而不是悄悄替换旧模型结构。新增/修改配置时请显式写出 `encoder`。

`local` 保留逐 patch 独立卷积、简单均值/标准差拼接、原有旋转注意力及形态/上下文输出。
新增的 `feature_pool` 和 `feature_layer` 表述主要针对 `multiview`；历史默认行为保持原样。
两条路径的 state_dict 不兼容，不能只更改配置名称就复用权重。换编码器后重新预热、训练或适配教师、
校准并导出，不复用绑定旧结构的校准产物。

本实现独立编写选定算法组件，未逐行复用上游实现，也不是完整论文训练体系的复现。
FM 文档列出的其他框架中，[PULSE](https://github.com/yenhochen/PULSE) 的扩张卷积可供后续感受野消融；
[EntroPE](https://github.com/Sachithx/EntroPE) 的动态分段依赖额外熵模型；
[LogSig-SSM](https://github.com/felix-oury/LogSigSSM) 则改变压缩表示与主干。
本轮选择保持固定时长 patch 和现有蒸馏边界，未加入这些完整系统。

是否优于 `local`、是否值得保留差分/统计支路，需在相同数据划分和训练预算下进行下游消融。
代码跑通只能说明接口和计算链条可执行，不能证明真实 ECG–PPG 数据上的效果。
