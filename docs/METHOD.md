# 方法与数学约定

## 学生与梯度边界

主路径为 `model.encoder="multiview"`。令 x 为 PPG，T 为 TempoBridgeTokenizer，
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
参数共享的 Ec 同时接受两种目标。关闭 protect_shallow 时移除该 stopgrad。
这是梯度路径约束，不是冻结浅层参数；AdamW 仍会根据 PPG 梯度更新浅层。
两次学生前向增加训练开销；推理只需要一次。
蒸馏预测头使用完整上下文层的 patch 均值，`feature_pool` 与 `feature_layer` 定义的公共特征接口
不会把 ECG token 接入学生。特征设置也会影响校准使用的 PPG 表征，因此修改后应重新校准。

旧 `local` 编码器保留独立 patch 卷积、原有 Transformer 和形态/上下文均值拼接。
`ModelConfig.from_dict` 对缺少 `encoder` 的历史配置选择 `local`；显式 `multiview` 才选择新路径。
结构、输入维度与开关见 [编码器说明](ENCODER.md)。

## 掩码的信息边界

1. 随机选择 patch，不根据教师、未来波形或事件峰位置选边界。
2. 进入 tokenizer 后先清除隐藏采样点。原波形分支仅用可见点估计标准化统计；差分只有相邻两端均可见时才保留，末端补零。
3. 两条卷积在整段清理后的序列上运行，可以跨 patch 边界读取可见邻域；隐藏 patch 最终替换为 mask token。没有把完整目标波形传给卷积分支。
4. 均值和标准差支路在模型内部标准化之前，按 patch 计算；隐藏 patch 的统计输入被清零且其融合输出被 mask token 覆盖。
5. 重建目标独立由完整波形和可见点的均值/方差计算，仅在损失端使用并 detach。编码器不读取目标中隐藏的数值。

该边界要求：固定 mask 和可见输入时，修改隐藏波形只能改变重建目标，不能改变编码器表示。
两条卷积分支经过逐窗口标准化；统计支路额外携带进入模型时的 patch 均值和波动幅度，
因此不能再把 `multiview` 表述为“移除了全部 DC/增益信息”。这里也不等于保留设备标定的物理幅值：
上游数据转换、单位换算或预处理可能已改变幅值。`local` 路径仍对标准化后的 patch 提取特征。
这是 masked reconstruction 的目标隔离；模型仍在可见上下文内双向注意。
离线 resample_poly 和原始数据的上游滤波不属于原始流的严格在线因果处理。
不能把本框架称为 PulseBound 式未来前缀预测方法。

## 多尺度目标

默认 2s/8s 只是初始实验设置。每个尺度在输入信号上裁剪后，分别运行教师与学生。
禁止先运行长窗口教师再将其全上下文 token 池化成所谓“短尺度目标”。
两种信号按秒定位，允许不同采样率；同步偏差必须在数据准备阶段审计。
本版窗口级蒸馏不保证精确逐心搏匹配，生理时延和边界不完整心搏仍可能影响结果。

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

    w(i,s) = ECG_quality(i) * a(s, PPG_quality_bin(i)) * ECG_available(i)
    L_transfer = mean_s [ sum_i w(i,s) * mean_dim Huber(pred_i, target_i) / batch_size ]
    L = L_PPG + lambda * L_transfer

target 在训练患者全部可靠配对上估计的固定均值/标准差中标准化。
这些全训练统计用于最终训练；OOF评分仍使用各 fit 折独立统计。
分母固定为 batch_size，不能改成 sum(weights)，否则整体低可信度会被重新归一化抵消。
weights 与 teacher 都无梯度，selector 无法靠把自身权重降至零来最小化训练损失。
真正全零校准则允许退回纯 PPG 自监督，明确记录“无可迁移证据”，不强行设置正覆盖率。
模型没有通过约束保证数据分布中的罕见状态不被过滤，需在真实实验审计。

## 可复现实验边界

校准绑定 warm-up checkpoint、教师权重与元数据、数据内容/划分、尺度、质量分组和采样率。
更换来源时拒绝继续蒸馏。校准文件是实验产物，不能在另一个训练划分直接复用。
外部教师若训练数据未知，只能说明未知，不能宣称整个预训练流程无患者重叠。

架构不保证下游增益，不宣称统计上识别了真实共享/私有生理因素。
它提供的是可检验的梯度约束、数据边界和校准规则。
