# 方法与数学约定

## 学生与梯度边界

主实验路径为 `model.encoder="selective"`。令 x 为 PPG，T 为包含多感受野选择的 tokenizer，
E0 为加入可学习 CLS 后的前若干 Transformer 层，Ec 为后层，Gm 为形态支路。

    h = E0(concat_tokens(T(x), CLS))
    hm = RMSNorm(h_patches + Gm(h_patches))
    hc = Ec(h)
    z = concat_features(hc_CLS, mean(hc_patches))

CLS 放在 patch 序列末尾，并与 patch 一起进行双向注意力。默认 `combined` 表征拼接 CLS 与
patch 均值，不包含浅层形态输出。形态支路采用残差小 MLP 与 RMSNorm，用于辅助重建。
这里的“形态/上下文”是归纳偏置，不是对潜在生理变量的可辨识分解，不保证统计独立或因果解耦。
可选 `cls`、`mean`、`morph_context` 池化；后者才拼接形态与上下文均值。
`feature_layer` 可选择某一层的表示快照，但不提前停止后续计算。

PPG 自监督分别从浅层支路和后层预测隐藏 patch，损失为 Huber(context) + α Huber(morphology)。
α 默认 0.25，不作为创新点。没有 ECG 时，此目标仍覆盖该 PPG 样本。

蒸馏另做一次完整 PPG 前向：

    t_s = stopgrad(ECG_teacher(ecg_crop_s))
    h_s = E0(concat_tokens(T(ppg_crop_s), CLS))
    p_s = head_s(mean_patches(Ec(stopgrad(h_s))))

因此，蒸馏对 tokenizer、CLS 和 E0 的导数严格为零；PPG 自监督对它们的导数仍可非零。
新增的多感受野选择器位于 tokenizer 内，也遵守这一边界。它根据 PPG 特征进行自适应融合，
不是由 ECG 蒸馏直接训练的选择器；当前没有添加绕过保护边界的适配器或额外选择器损失。
参数共享的 Ec 同时接受两种目标。关闭 protect_shallow 时移除该 stopgrad。
这是梯度路径约束，不是冻结浅层参数；AdamW 仍会根据 PPG 梯度更新浅层。
两次学生前向增加训练开销；推理只需要一次。
蒸馏与校准统一调用 `transfer_features`，固定返回最后一层 context patch 均值。
`feature_pool` 与 `feature_layer` 仅控制对外特征接口，不改变该共享定义；ECG token 不接入学生。
产物绑定完整模型配置，改变配置后仍需重新生成对应产物。

`multiview` 保留原单核四路编码，`local` 保留独立 patch 卷积、原有 Transformer 和形态/上下文均值拼接。
`ModelConfig.from_dict` 对缺少 `encoder` 的历史配置选择 `local`；新增实验显式指定编码器。
结构、输入维度与开关见 [编码器说明](ENCODER.md)。

## 多感受野选择

参考 [Selective Kernel Networks](https://arxiv.org/abs/1903.06586) 的 Split–Fuse–Select 思路，
在原波形和差分两条支路内分别进行一维多感受野编码。原论文的空间分支与全局通道选择，
在本项目中适配为低维 depthwise/pointwise 时间分支、可见点全局描述、patch 局部描述和覆盖率处理。
`selective_kernel_sizes` 是采样点数而非秒数；默认 `[9,21,41]` 在 50 Hz 与 100 Hz 的物理感受野不同。

对于第 i 个 patch、第 k 个分支、第 c 个通道，记分支表示为 U(i,k,c)：

    local_i = sum_k U(i,k,:)
    global = sum_i visible_count_i * local_i / sum_i visible_count_i
    descriptor_i = global + local_i
    logits_i = selection_mlp(descriptor_i) + log(coverage_i)
    alpha(i,k,c) = softmax_over_branches(logits_i)[k,c]
    V(i,c) = sum_k alpha(i,k,c) * U(i,k,c)

该式对应局部描述和覆盖率均启用的模式；零覆盖分支在 softmax 前排除，空 patch 最终权重为零。
卷积分支同时进行有上限的支持率补偿：记录内可用支持数除以可见支持数，最多放大 4 倍。
softmax 沿分支轴计算，
每个 patch 的每个通道有各自的分支权重，不是为整个窗口只选一个卷积核。
`selective_fusion="mean"` 在有效分支间固定平均，检验收益是否来自可学习选择本身。
`selective_local_context=false` 仅使用全局描述；`selective_coverage=false` 关闭幅值补偿与 logits 的对数覆盖率项，
但它们都不会取消隐藏样本在输入特征构造前的清除。

覆盖率描述已观察到多少输入，不等于生理 SQI 或校准后的可信概率；tokenizer 没有频域分支、
VQ 码本或显式心搏检测。独立参考信号提取的生理目标作为另一个训练基线，不改变 tokenizer。
该适配是否改善泛化由单独消融判断，不作首次或性能保证。

## 掩码的信息边界

1. 随机选择 patch，不根据教师、未来波形或事件峰位置选边界。
2. 进入 tokenizer 后先清除隐藏采样点。原波形分支仅用可见点估计标准化统计；差分只有相邻两端均可见时才保留，末端补零。
3. 原波形和差分的卷积分支在整段清理后的序列上运行，可以跨 patch 边界读取可见邻域；分支融合的全局/局部描述也仅来自可见输入。隐藏 patch 最终替换为 mask token，没有把完整目标波形传给分支或门控。
4. 均值和标准差支路在模型内部标准化之前，按 patch 计算；隐藏 patch 的统计输入被清零且其融合输出被 mask token 覆盖。
5. 重建目标独立由完整波形和可见点的均值/方差计算，仅在损失端使用并 detach。编码器不读取目标中隐藏的数值。

该边界要求：固定 mask 和可见输入时，修改隐藏波形只能改变重建目标，不能改变编码器表示。
两条卷积分支经过逐窗口标准化；统计支路额外携带进入模型时的 patch 均值和波动幅度，
因此不能把 `multiview` 或 `selective` 表述为“移除了全部 DC/增益信息”。这里也不等于保留设备标定的物理幅值：
上游数据转换、单位换算或预处理可能已改变幅值。`local` 路径仍对标准化后的 patch 提取特征。
这是 masked reconstruction 的目标隔离；模型仍在可见上下文内双向注意。
离线 resample_poly 和原始数据的上游滤波不属于原始流的严格在线因果处理。
不能把本框架称为 PulseBound 式未来前缀预测方法。

## 完整窗口与可选多尺度目标

主实验配置使用 `window_mode="full"`、`scales_seconds=[30.0]`。数据加载器要求整条同步记录
恰好为 30 秒，将完整 PPG 与 ECG 分别送入学生与教师；不会从长记录偷偷取中间 30 秒，也不会补短记录。
这是单一观察时长上的校准/蒸馏；多感受野 token 是同一次前向中的局部卷积结构，与多个裁剪时长是不同变量。

2 秒/8 秒保留为 `configs/ablation_2s8s.json` 的 `window_mode="crop"` 消融。
训练时两个尺度独立采样裁剪起点，同尺度 ECG/PPG 在物理时间上对齐；短窗不保证位于长窗内部。
验证/校准使用各尺度的居中窗口。每个尺度实际裁剪后，分别运行教师与学生。
禁止先运行长窗口教师再将其全上下文 token 池化成所谓“短尺度目标”。
两种信号按秒定位，允许不同采样率；同步偏差必须在数据准备阶段审计。
本版窗口级蒸馏不保证精确逐心搏匹配，生理时延和边界不完整心搏仍可能影响结果。

PulseBound 正式 v6 的 1500 点/50 Hz 输入同样覆盖 30 秒，但采用 1 秒 patch、连续后缀掩码
及未来心搏目标。本项目只对齐完整输入窗口及指定采样率，仍使用自己的 patch、随机掩码与蒸馏目标。
从历史 2 秒/8 秒实验切换到 30 秒后必须重新校准，不能复用绑定旧观察时长的校准权重。

## 交叉拟合

预先训练并冻结 PPG warm-up 编码器。它可以见过全部训练患者的无标签 PPG。
因此“out-of-fold”仅指可预测性探针，而非整条表征学习流水线完全排除该折患者。
验证/测试患者从未参与 warm-up、原生教师训练、probe 拟合、校准统计或权重选择。

训练患者被划分为 K 折。在每个 fit 折内：

- 对 PPG 特征和 ECG 目标按患者等权计算均值与标准差。
- 拟合 ridge 或两层 MLP，并在其余患者上预测。
- baseline 为 fit 折目标均值；没有使用 held-out 折目标均值。
- 在 fit 折目标标准化坐标中计算平方误差。

同一患者样本数多不会自动提高其贡献。对尺度 s、质量组 c，汇总患者均衡误差：

    r(s,c) = 1 - E_probe(s,c) / E_mean_baseline(s,c)
    a(s,c) = clip(r(s,c), 0, 1)

baseline 误差接近零（例如常量教师）或患者数不足时设 a=0。
默认 consensus 同时计算 ridge 与 MLP，使用两者 r 的最小值作为保守证据。
这会降低弱 probe 偶然高估的风险，也可能因 probe 欠拟合过滤有用知识；必须报告各自 r 和覆盖率。

这一量不是不可约条件方差的无偏估计，不是任务效用，不是跨模态因果关系的证明。
可学不等于值得学，真实价值必须通过独立下游与分组结果评价。

## 蒸馏损失与不退化条件

以下样本加权公式对应 PW-KD。COSD 的完整坐标、患者划分及损失契约见 [COSD.md](COSD.md)。

    w(i,s) = ECG_quality(i) * a(s, PPG_quality_bin(i)) * ECG_available(i)
    L_transfer = mean_s [ sum_i w(i,s) * mean_dim Huber(pred_i, target_i) / batch_size ]
    L = L_PPG + lambda * L_transfer

schema 3 的最终 target 均值/标准差由 basis 患者拟合；同一 cache/划分下完整目标和投影目标共用该统计。
完整目标在不足三个患者时才退回全部训练配对统计；COSD 不采用这一退回路径。OOF 评分仍使用各 fit 折独立统计。
分母固定为 batch_size，不能改成 sum(weights)，否则整体低可信度会被重新归一化抵消。
weights 与 teacher 都无梯度，selector 无法靠把自身权重降至零来最小化训练损失。
真正全零校准则允许退回纯 PPG 自监督，明确记录“无可迁移证据”，不强行设置正覆盖率。
模型没有通过约束保证数据分布中的罕见状态不被过滤，需在真实实验审计。

COSD 将标准化后的真实 ECG 输出右乘冻结的正交方向矩阵，学生直接预测 r 维目标。
候选基为原始坐标、PCA 或随机正交基；discovery 患者的方向 OOF skill 与条件 bootstrap
下界决定保留方向，独立 audit 患者只能用于报告。默认 hard 方向权重为 1，soft 权重只乘在损失上。
`normalize_direction_weights=true` 将方向权重均值归一到 1，作为独立权重总量对照。
每个尺度可显式指定 `ranks_per_scale`；随机/PCA 对照由实验计划匹配 COSD 的实际 rank。

总目标为 `ssl_weight * (L_context + alpha * L_morph) + transfer_weight * L_KD`。
默认浅层保护不变。`physio` 基线直接预测独立 ECG/ABP/RESP 标量标签，不载入神经教师。

## 可复现实验边界

校准绑定 warm-up checkpoint、教师权重与元数据、数据内容/划分、尺度、质量分组和采样率。
更换来源时拒绝继续蒸馏。校准文件是实验产物，不能在另一个训练划分直接复用。
外部教师若训练数据未知，只能说明未知，不能宣称整个预训练流程无患者重叠。

架构不保证下游增益，不宣称统计上识别了真实共享/私有生理因素。
它提供的是可检验的梯度约束、数据边界和校准规则。
