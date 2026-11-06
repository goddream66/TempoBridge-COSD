# TempoBridge 编码器

主实验配置使用 `model.encoder="selective"`，提供多感受野选择的连续波形 token 与时间序列 Transformer。
实现位于 `src/tempobridge/encoding.py`，由 `src/tempobridge/model.py` 的 `TempoBridge` 调用。
`multiview` 保留单核四路编码作为结构基线；`local` 保留历史编码器，用于读取旧配置与 checkpoint。

## 从波形到 token

输入为单通道 `[B,1,T]`，每个 patch 有 P=`patch_size` 个采样点，N=T/P。
输入必须有限、长度整除 P，并至少含两个 patch。不使用离散分词器、词表、码本或文本模型。

三种配置都接收单通道波形；`selective` 和 `multiview` 使用原波形、相邻差分、均值、标准差四类信息。
新路径没有增加频域、VQ 码本、显式峰检测或 ECG 输入通道。

## 主路径：多感受野选择

`selective` 借鉴 [Selective Kernel Networks](https://arxiv.org/abs/1903.06586) 的
Split–Fuse–Select：先产生多个感受野的分支，融合信息生成选择描述，再按通道选择分支。
本轮依据论文算法进行一维适配；没有将该论文的完整图像分类网络搬入项目，也不声称已核对其官方仓库代码。

`TempoBridgeSelectiveTokenizer` 为原波形与相邻差分分别配置独立的 `TempoBridgeSelectiveKernel`：

1. **Split**：将清理和标准化后的单通道序列映射到较小宽度 W，用不同核长的
   depthwise/pointwise 卷积分支提取时间特征，减少在每个采样点维持完整 D 通道的开销。
2. **Fuse**：从分支结果构造可见位置上的全局描述和每个 patch 的局部描述。
   覆盖率处理记录卷积支持范围内实际观察到的输入，避免把隐藏位置当作正常零值观测。
3. **Select**：小型瓶颈网络产生分支得分，沿分支轴进行 softmax。
   每个 patch、每个通道分别得到一组权重，加权合并分支后输出 W 维波形/差分表示。
4. 与独立编码的 patch 均值、标准差拼接，投影并归一化为 D 维 token；隐藏 patch 最终替换成 mask token。

默认 W=32、S=32 时，四路拼接为 `2W+2S=128` 维，再投影为 D=256 维。
每个分支的具体顺序是 `1×1 输入投影 → depthwise 时间卷积 → 覆盖率补偿 → 1×1 通道混合
→ LayerNorm → GELU → 可见位置 patch 池化`，其中输入投影由同一信号路径的多个核分支共享。

分支支持率的分母只统计记录内真实存在的位置，因此普通序列边缘的 padding 不会被算作缺失。
启用覆盖率适配时，depthwise 输出乘以 `possible_support / max(observed_support,1)`，最多放大 4 倍；
分支选择 logits 再加 `log(coverage)`。patch 内只池化中心采样可见的响应，完全隐藏的 patch
不参与全局描述，选择分配为零后由最终 mask token 覆盖。这些处理保持可见信息边界，并不填补真实隐藏波形。

默认配置：

| 字段 | 默认值 | 含义 |
|---|---|---|
| `selective_width` | 32 | 各时间分支的中间通道宽度 W |
| `selective_kernel_sizes` | `[9,21,41]` | 多个奇数卷积核，单位为采样点 |
| `selective_reduction` | 4 | 选择网络瓶颈压缩比例 |
| `selective_min_dim` | 8 | 选择网络瓶颈最小宽度 |
| `selective_fusion` | `adaptive` | 可学习选择；`mean` 为固定平均融合对照 |
| `selective_local_context` | `true` | 选择时使用 patch 局部描述 |
| `selective_coverage` | `true` | 启用覆盖率适配 |

同一个 patch 仍然只输出一个 D 维 token，多个感受野没有扩展成额外 token 序列。
全局描述有助于选择符合当前窗口背景的分支，局部描述允许不同 patch 使用不同分支组合。
覆盖率只表示观测支持程度，不是 PPG 生理质量标签，也不是经过校准的可信概率。

这些核长是采样点数：9/21/41 点在 50 Hz 下约覆盖 0.18/0.42/0.82 秒，在 100 Hz 下约为
0.09/0.21/0.41 秒。当前模块不根据采样率自动改核长；跨采样率实验须显式匹配物理感受野。
标量的“多个数值尺度”、卷积的“多个感受野”、训练输入的“多个观察时长”是三个不同概念。

自适应选择网络带来额外计算，也可能只偏向某一分支；`mean` 对照用于区分多感受野本身和
可学习选择的收益。选择网络参数在两个模式中都保留；`mean` 在有效分支间均分权重，但不运行选择网络，
因此参数结构对齐不等于 FLOPs 完全相同。覆盖率修正也不保证消除所有遮挡边缘效应。
选择器位于浅层保护边界之前，默认仅由 PPG 自监督更新，不能将其称为 ECG 蒸馏直接训练的选择器。

`forward_with_diagnostics(patches, mask)` 额外返回原波形/差分两路的 `weights_*` 和 `coverage_*`：
权重为 `[B,N,K,W]`，覆盖率为 `[B,N,K]`；K 是分支数。诊断由调用显式返回，模块不缓存计算图。

## 基线路径：单核四路编码

`model.encoder="multiview"` 的 `TempoBridgeTokenizer` 同时提取四路连续信息：

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

## 完整输入、消融与固定 token 数

主实验配置使用 `data.window_mode="full"`、`scales_seconds=[30.0]`。
数据加载器要求记录恰好覆盖 30 秒，以完整同步窗口运行学生和教师；长记录不截断，短记录不补齐。
模型设置 `num_tokens=0`，固定每个 patch 的采样点数：

| 配置 | 采样率 | `patch_size` | patch 时长 | 主实验 30 秒的 patch token 数 |
|---|---|---|---|---|
| `configs/base.json` / `configs/laptop_8gb.json` | 50 Hz | 10 | 0.20 秒 | 150 |
| `configs/pulsebound_50hz.json` | 50 Hz | 10 | 0.20 秒 | 150 |

表中不计额外 CLS。50 Hz、30 秒只对齐 PulseBound 的输入窗口；PulseBound 主预训练每个 patch
为 50 点而非这里的 10 点，而且其额外视图/register token、后缀掩码及未来预测任务未被采用。
ECG 与 PPG 分别按各自配置的采样率读取同一物理时间段，ECG 不直接进入学生 token。

`configs/ablation_2s8s.json` 单独设置 `window_mode="crop"`，从足够长的同步记录裁出 2 秒/8 秒。
50 Hz、patch=10 时对应 10/40 个 patch token。训练时两个尺度分别随机选择起点，验证时各自居中；
短窗不保证嵌套在长窗中。它是观察时长消融，不是 PulseBound 的正式训练窗口。

对结构进行比较时，使用相同 30 秒输入的 `configs/baseline_multiview_30s.json` 和
`configs/ablation_selective_mean_30s.json`，分别检查单核基线和固定平均融合。

可选 `num_tokens=32` 只是输入约束，要求 T=`32*patch_size`，不是自动插值到 32 个 token。
不能把当前主配置直接设为 32 而保持其余输入参数不变；若设计固定 token 数实验，应同时修改
数据、patch 和校准设置。2 秒/8 秒裁剪消融继续使用 `num_tokens=0`。

## 掩码、目标与训练边界

隐藏采样点在原波形、差分、统计提取和标准化前被清除；差分的相邻两端必须可见，末端补零。
各分支只用有效可见位置计算标准化统计。全波形卷积可以读取跨 patch 的可见邻域，
但无法通过输入数值读取隐藏目标。`selective` 的全局/局部描述及分支选择也遵守这一边界；
关闭 `selective_coverage` 仅关闭覆盖率适配，不会允许原始隐藏采样重新进入输入。
随后隐藏 patch 的融合 token 被 mask token 覆盖。

重建目标另行使用完整波形和可见点均值/方差标准化，并在损失端 detach。
目标标准化与 tokenizer 内部标准化是两条独立路径，不把重建目标送回 token 提取器。
改变隐藏波形只能改变监督目标，不能改变编码器的可见信息。

训练保持 PPG 掩码重建、ECG 教师、跨患者可预测性校准、尺度选择性 Huber 蒸馏与浅层梯度保护。
蒸馏前向将浅层输出 detach；PPG 自监督仍更新 tokenizer、CLS 和浅层。
教师只在训练/校准阶段提供目标；公共 `forward(ppg)` 与导出的编码器仅需 PPG。
数学细节见 [方法说明](METHOD.md)。

## 历史模型兼容与实验范围

`ModelConfig()` 及主实验配置默认构建 `selective`。为了读取历史文件，`ModelConfig.from_dict` 在输入字典
缺少 `encoder` 时选择 `local`，而不是悄悄替换旧模型结构。Python 直接构造及新增/修改配置时请显式写出 `encoder`。
新数据配置默认 `full` / `[30.0]`；旧数据字典缺少 `window_mode` 时解析为历史 `crop`，
不会把历史 2 秒/8 秒模型自动解释成完整 30 秒模型。

`local` 保留逐 patch 独立卷积、简单均值/标准差拼接、原有旋转注意力及形态/上下文输出。
`feature_pool` 和 `feature_layer` 表述适用于 `selective` 和 `multiview`；历史 `local` 默认行为保持原样。
不同路径的完整 state_dict 不兼容，不能只更改配置名称就复用权重。换编码器后重新预热、训练或适配教师、
校准并导出，不复用绑定旧结构的校准产物。

本实现独立编写选定算法组件，未逐行复用上游实现，也不是完整论文训练体系的复现。
FM 文档列出的其他框架中，[PULSE](https://github.com/yenhochen/PULSE) 的扩张卷积可供后续感受野消融；
[EntroPE](https://github.com/Sachithx/EntroPE) 的动态分段依赖额外熵模型；
[LogSig-SSM](https://github.com/felix-oury/LogSigSSM) 则改变压缩表示与主干。
本轮选择保持固定时长 patch 和现有蒸馏边界，未加入这些完整系统。

是否优于 `multiview`/`local`、自适应选择是否优于平均融合，需在相同数据划分、观察时长和训练预算下进行下游消融。
代码跑通只能说明接口和计算链条可执行，不能证明真实 ECG–PPG 数据上的效果。
