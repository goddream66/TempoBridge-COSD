# TempoBridge-COSD：完整研究方案与实验开发计划

**研究定位：** ECG 辅助的 PPG Foundation Model\
**数据：** MIMIC-III Waveform Database Matched Subset + VitalDB\
**预处理协议：** 50 Hz、30 s、1500 samples/window\
**部署约束：** PPG-only inference\
**研究阶段：** Backbone 已有原型；论文 Idea 尚待实验验证\
**方案范围：** A（可预测性加权的完整教师蒸馏）+
B（可观测教师子空间蒸馏）+ 全部关键消融

------------------------------------------------------------------------

## 0. 执行摘要

本方案不把 ECG 与 PPG 简单拼接成双模态推理模型，而采用
privileged-modality training：

-   训练时：对具有有效同步 ECG 的窗口，ECG 作为冻结教师提供辅助监督；PPG
    Student 同时接受 PPG 自监督目标。
-   训练时：对没有有效 ECG 的窗口，继续执行 PPG-only 自监督，不伪造
    ECG、不把零占位波形当作有效 ECG。
-   推理时：只保留 PPG Encoder，ECG
    Teacher、校准器和蒸馏头均不进入部署路径。

本研究比较两条方法线：

### A. Predictability-Weighted Full-Teacher Distillation（PW-KD）

保留完整 ECG Teacher embedding 作为目标；使用患者级 cross-fitting
估计不同观察尺度/PPG 质量条件下，PPG 表征预测 ECG Teacher
的能力；将校准结果作为蒸馏权重。

A 是当前 TempoBridge 方法的严格、可复现化版本，也是 B 必须击败的强基线。

### B. Cross-Fitted Observable-Subspace Distillation（COSD）

不要求 PPG Student 拟合完整 ECG Teacher
embedding。先在独立患者折上估计教师表征各方向是否能被 PPG
表征稳定预测，再构造教师侧可观测子空间投影；Student 只拟合该子空间中的
ECG Teacher target，同时继续学习 PPG 自监督目标。

核心假设：

> 对 PPG-only Foundation Encoder 而言，完整 ECG Teacher
> 表征中存在一部分在当前 PPG
> 观测条件下难以可靠预测的信息。无差别拟合全部教师维度可能造成无效监督或负迁移；在患者级交叉拟合中识别可预测教师子空间，并只迁移该子空间，可能在保持
> PPG 原生信息的同时改善跨患者、跨数据源迁移。

这是一项**待验证假设**，不是已证实的生理因果分解，也不是已证明的 ICML
创新。可预测教师信息可能来自患者身份、设备、数据源、记录环境或真正共享的生理动态；研究必须通过分组分析、负对照和下游任务审计排除这些替代解释。

### 最重要的方法学修正

不能把探针预测结果 `g(PPG)` 直接当作 B 的唯一蒸馏目标，再声称 ECG
知识被迁移。因为 `g(PPG)` 是由既有 PPG 表征确定性生成的，Student
可能只是复制自己的 warm-up 表征。

本方案将：

1.  使用 cross-fitting 在训练患者上估计哪些 **ECG Teacher 输出方向**
    可由 PPG 表征预测；
2.  根据 OOF 预测技能与稳定性选择教师侧投影矩阵 `P_obs`；
3.  对真实 ECG Teacher target 做投影：`E_obs = E @ P_obs`；
4.  Student 预测 `E_obs`；
5.  在验证集/测试集上独立评估投影方向的可预测性、稳定性和任务效用。

因此，B 的目标仍然来自真实 ECG Teacher，而不是单纯复制 PPG probe
的输出。

------------------------------------------------------------------------

# 1. 研究问题、目标与边界

## 1.1 研究问题

ECG 与 PPG
观测的是同一心血管系统的不同侧面，但二者并不包含完全相同的信息：

-   ECG 主要反映心脏电活动；
-   PPG 反映外周血容量随时间的光学变化；
-   ECG 到外周脉搏之间存在生理传播时延；
-   PPG 还受到灌注、接触压力、运动、传感器和滤波流程影响。

因此，不能默认 ECG Teacher 的每个 latent dimension 都适合被 PPG Student
拟合。

本研究回答三个递进问题：

**RQ1：** 在患者独立评估下，PPG Encoder 能在多大程度上预测 ECG Teacher
表征？

**RQ2：** 对完整 ECG Teacher embedding
做预测性加权蒸馏（A），是否优于无条件全维度蒸馏？

**RQ3：** 只蒸馏在患者级 OOF 评估中稳定可预测的 ECG Teacher
子空间（B），是否能比 A 更好地迁移到 PPG-only
下游任务，尤其是跨患者、跨来源和不同信号质量条件？

## 1.2 研究目标

1.  建立稳定的 50 Hz、30 s、1500 点 ECG/PPG 配对与非配对训练协议。
2.  保留现有 TempoBridge PPG Encoder，避免过早将未验证的结构创新固化进
    Backbone。
3.  构建 PPG-only SSL、Vanilla KD、PW-KD（A）和 COSD（B）公平比较体系。
4.  证明 OOF
    可预测性和教师子空间筛选是否带来真实下游收益，而非只降低蒸馏损失。
5.  对患者身份、来源、质量、时间对齐和数据泄漏进行系统审计。

## 1.3 不在当前方案中预设的内容

当前不将以下内容作为必需 Backbone 创新：

-   State--Trait Factorization；
-   Future-beat Forecasting；
-   PulseBound 式前缀信息边界；
-   新的 delay-aware cross-attention；
-   新的生理图结构；
-   新的动态 patch segmentation；
-   更换 Transformer 为 SSM/Mamba；
-   多个新的质量门控分支。

这些方向可以在 COSD
失败或发现新的经验现象后再评估。当前方案聚焦于可观测教师目标的选择与迁移机制。

------------------------------------------------------------------------

# 2. 已有 TempoBridge 代码基线与必须修正的问题

本节根据上传仓库
`src/tempobridge/`、`docs/METHOD.md`、`docs/ENCODER.md`、`docs/RESEARCH.md`、`docs/TEACHER.md`
和配置文件整理。

## 2.1 固定输入协议

主实验沿用 `configs/pulsebound_50hz.json` 的输入协议：

  参数                                          设置
  ---------------------------- ---------------------
  PPG sampling rate                            50 Hz
  ECG sampling rate                            50 Hz
  Window duration                               30 s
  Samples per modality                          1500
  Patch size                              10 samples
  Patch duration                               0.2 s
  PPG temporal tokens                            150
  Encoder dimension                              256
  Transformer depth                                6
  Attention heads                                  8
  Protected shallow depth                          2
  Context depth                                    4
  Selective kernels              9 / 21 / 41 samples
  Selective width                                 32
  Scalar embedding dimension                      32

注意：这与 PulseBound 的采样率、窗口时长和采样点数相同，但**不等于复现
PulseBound Backbone**。PulseBound 使用 50-sample（1 s）patch、30 个时间
patch、多个派生视图、registers 和 future-beat objective；TempoBridge
使用 10-sample（0.2 s）patch、150 个时间 token、Selective Kernel
tokenizer、双向 Transformer 和 masked reconstruction / ECG
distillation。

## 2.2 当前 Encoder 结构

当前 `TempoBridge` 主要路径：

1.  `TempoBridgeSelectiveTokenizer`；
2.  原始 PPG waveform 多感受野卷积分支；
3.  一阶差分分支；
4.  patch mean / std 的 `TempoBridgeScalarEncoder`；
5.  coverage-aware adaptive fusion；
6.  CLS/summary token；
7.  前 2 层 shallow blocks；
8.  后 4 层 context blocks；
9.  waveform reconstruction head；
10. morphology reconstruction head。

保留这些结构作为统一 Backbone。Selective
Kernel、RMSNorm、RoPE/XPos、SwiGLU 等不作为本文独立创新。

## 2.3 必须先修复：Calibration 与实际 KD 输入表示不一致

从当前源码可见：

`calibration.py` 中：

``` python
features[s].append(model(x).cpu().numpy())
```

`model.forward()` 调用 `forward_features()` 并返回 `embedding`。在默认
`feature_pool="combined"` 下，embedding 是 summary/CLS 与 context patch
mean 的拼接，维度为 `2 * dim = 512`。

但 `training.py` 的 `transfer_loss()` 中：

``` python
tokens = model.encode(
    ppg[selected],
    detach_shallow=protect_shallow
)["context"]
pred = heads(tokens, scale_index)
```

`TransferHeads` 对 context tokens 做 `mean(1)`，输入维度为 `dim = 256`。

所以目前：

-   calibration probe 的 PPG 输入特征：512 维；
-   训练时 KD head 的 PPG 输入特征：256 维；
-   Teacher target：当前 native Teacher 的 `context` mean，256 维。

这会造成校准估计的可预测性不完全对应实际 KD 路径。它不是简单的 target
dimension 报错，因为 calibration probe 可以把 512 维映射到 256
维；真正的问题是**校准时评估的学生表示与蒸馏时使用的学生表示不同**。

### P0 修复

在 `model.py` 中建立显式接口：

``` python
def transfer_features(self, ppg, detach_shallow=False):
    out = self.encode(ppg, detach_shallow=detach_shallow)
    return out["context"].mean(dim=1)
```

校准阶段和蒸馏阶段必须调用相同的特征定义。建议先用 256 维 context mean
作为主实验的共同表示。另将 `summary`、`combined`、`morph_context`
作为单独表示消融，不要混在同一校准产物中。

每次改变以下任何内容，都必须重新 warm-up / calibration
或至少重新生成与其严格绑定的校准 artifact：

-   `feature_pool`；
-   `feature_layer`；
-   Encoder 结构；
-   输入采样率；
-   窗口时长；
-   patch size；
-   数据划分；
-   Teacher checkpoint；
-   数据 manifest。

## 2.4 当前 calibration 的真实能力边界

当前 `calibration.py` 已经实现：

-   subject-balanced sample weights；
-   train patient K-fold；
-   Ridge 与 MLP probes；
-   consensus 使用两个 probe 中较保守的预测技能；
-   按 PPG quality bin 汇总；
-   target mean/std；
-   对训练患者每折 OOF probe prediction error 评分。

但其 OOF 仅指**固定 PPG warm-up representation 上的 probe OOF**。PPG
warm-up encoder 可以见过所有训练患者的无标签 PPG；因此不能称为整条
representation learning pipeline 的严格 patient-level OOF。

当前 calibration 输出的是按尺度/PPG
质量组的标量权重，并非教师维度级投影矩阵。

## 2.5 当前训练机制

当前 `training.py`：

-   PPG masked reconstruction 始终执行；
-   ECG Teacher frozen；
-   有有效 ECG 的样本可进入蒸馏；
-   无 ECG 的 PPG 仍执行 PPG SSL；
-   `protect_shallow=True` 时蒸馏前向对 shallow output detach；
-   PPG SSL 仍更新浅层；
-   KD head 对 context patch mean 进行预测；
-   蒸馏损失采用固定 batch denominator，避免低权重被重新归一化抵消；
-   inference export 只保留 PPG Encoder。

这些设计保留为共同训练框架。

------------------------------------------------------------------------

# 3. 总体方法：A/B 两条方法线

## 3.1 统一符号

对第 `i` 个同步窗口：

-   `x_i^p`：PPG waveform；
-   `x_i^e`：ECG waveform；
-   `m_i`：ECG availability mask；
-   `q_i^p`：PPG quality score；
-   `q_i^e`：ECG quality score；
-   `f_theta`：PPG Student Encoder；
-   `g_psi`：冻结 ECG Teacher；
-   `z_i^p = f_theta(x_i^p)`：PPG Student transfer feature；
-   `z_i^e = g_psi(x_i^e)`：ECG Teacher feature；
-   `s_i`：source dataset；
-   `r_i`：真实患者/受试者 ID；
-   `c_i`：观察尺度（主实验为 30 s）。

主实验建议先使用一个固定 30 s 尺度；多尺度 2 s/8 s
仅作为独立扩展消融，不与主实验混淆。

## 3.2 统一 PPG 自监督目标

对 PPG 随机 mask patch，保持当前 waveform 与 morphology reconstruction：

\[ `\mathcal `{=tex}L\_{SSL} = `\mathcal `{=tex}L\_{context} +
`\alpha`{=tex}*{morph}`\mathcal `{=tex}L*{morph} \]

其中：

\[ `\mathcal `{=tex}L\_{context} = `\operatorname{SmoothL1}`{=tex}
(`\hat `{=tex}X\_{masked}\^{context},X\_{masked}) \]

\[ `\mathcal `{=tex}L\_{morph} = `\operatorname{SmoothL1}`{=tex}
(`\hat `{=tex}X\_{masked}\^{morph},X\_{masked}) \]

默认 `alpha_morph = 0.25`，以当前配置为起点。

目标只在 masked patch 上计算。ECG 缺失样本仍参与该目标。

## 3.3 Teacher 约定

主实验使用一个固定 ECG Teacher checkpoint：

-   `modality="ecg"`；
-   `global_step > 0`；
-   `requires_grad=False`；
-   `eval()`；
-   Teacher checkpoint 与元数据有 SHA256 fingerprint；
-   Teacher 的训练患者不得与下游 validation/test 患者重叠；
-   Teacher 预训练数据来源未知时，必须明确记为 unknown，不得声称无重叠。

优先使用与 PPG Student 可比较的 256 维 context
representation。若使用外部 ECG
Teacher，必须记录其输入采样率、预处理、特征层、训练数据和许可。

------------------------------------------------------------------------

# 4. 方法 A：Predictability-Weighted Full-Teacher Distillation（PW-KD）

A 是当前 TempoBridge calibrated transfer 的严格对照，也是 B
的核心比较对象。

## 4.1 A 的目标

A 不筛选教师维度。它保留完整 Teacher feature：

\[ z_i\^e `\in `{=tex}`\mathbb `{=tex}R\^D \]

Student transfer head 预测：

\[ `\hat `{=tex}z_i\^e = h\_`\phi`{=tex}(z_i\^p) \]

完整教师目标损失：

\[ `\ell`{=tex}\_i\^A = `\operatorname{Huber}`{=tex}
(`\hat `{=tex}z_i^e,`\operatorname{sg}`{=tex}(z_i^e)) \]

## 4.2 患者级 cross-fitting

将训练患者划分为 `K=5` 折：

\[ `\mathcal `{=tex}S=`\bigcup`{=tex}\_{k=1}\^{K}`\mathcal `{=tex}S_k \]

对第 `k` 折：

-   fit set：所有非 `k` 折训练患者；
-   score set：第 `k` 折患者；
-   仅在 fit set 拟合标准化参数和 probe；
-   在 score set 上得到 OOF 预测；
-   不得使用 held-out fold target 均值作为 baseline。

Probe family：

1.  Ridge；
2.  两层 MLP；
3.  Consensus：取 Ridge 与 MLP 中较保守的结果。

患者等权：每个患者的窗口总权重相同，避免长记录患者主导 probe。

## 4.3 A 的 OOF predictability score

对质量组 `q`、尺度 `c`：

\[ R\_{probe}(c,q) = `\operatorname{PatientMacroMSE}`{=tex}
(z^e,`\hat `{=tex}z^e) \]

\[ R\_{null}(c,q) = `\operatorname{PatientMacroMSE}`{=tex}
(z^e,`\bar `{=tex}z^e\_{fit}) \]

定义：

\[ O_A(c,q) = `\operatorname{clip}`{=tex} `\left`{=tex}(
1-`\frac{R_{probe}(c,q)}{R_{null}(c,q)+\epsilon}`{=tex}, 0,1
`\right`{=tex}) \]

若患者数不足、null baseline 近零或 probe 失败，则该组权重设为
0，并报告原因。

A 的蒸馏权重：

\[ w_i\^A = m_i `\cdot `{=tex}q_i\^e `\cdot `{=tex}O_A(c_i,q_i\^p) \]

其中 `m_i` 表示 ECG
有效。质量值必须是有定义、有来源的质量分数；若没有经过校准的
SQI，不得将默认 1.0 解释为真实高质量。

蒸馏损失：

\[ `\mathcal `{=tex}L_A = `\frac{1}{B}`{=tex} `\sum`{=tex}\_{i=1}\^{B}
w_i\^A `\ell`{=tex}\_i\^A \]

使用固定 batch denominator，不除以权重和。

总损失：

\[ `\boxed{
\mathcal L_{PW-KD}
=
\mathcal L_{SSL}
+
\lambda_A\mathcal L_A
}`{=tex} \]

## 4.4 A 的实验目的

A 用于回答：

-   当前 cross-patient predictability calibration 是否优于 vanilla KD？
-   质量分组是否能减少不可靠蒸馏？
-   观察尺度是否改变可预测性？
-   当前浅层 gradient protection 是否有帮助？
-   A 的下游收益是否真正来自 ECG 配对关系？

------------------------------------------------------------------------

# 5. 方法 B：Cross-Fitted Observable-Subspace Distillation（COSD）

## 5.1 B 的关键区别

A：保留完整 ECG Teacher feature，只调整蒸馏强度。

B：先确定 ECG Teacher feature 中哪些方向能被 PPG Student
稳定预测，再对真实 ECG Teacher target 做教师侧投影，只蒸馏保留方向。

B 的核心不是 `g(PPG)` 本身，而是从 OOF
证据确定教师空间中的可迁移子空间。

## 5.2 教师方向的候选空间

将 ECG Teacher 输出：

\[ Z^e`\in`{=tex}`\mathbb `{=tex}R^{N`\times `{=tex}D} \]

在训练患者上做 patient-balanced centering/scaling。PCA/SVD
只在训练患者上拟合：

\[ Z^e_c=U`\Sigma `{=tex}V^`\top`{=tex} \]

候选教师方向：

\[ v_1,`\ldots`{=tex},v_D \]

对应每个方向的 teacher score：

\[ t\_{ij}=z_i\^e v_j \]

建议第一版先在标准化后的 teacher coordinates 上做方向级分析；PCA basis
作为独立消融。这样可以区分"可预测方向选择"和"PCA 压缩"本身的效果。

## 5.3 Cross-fitted direction predictability

对每个方向 `j`，使用 PPG transfer feature 预测教师方向分数：

\[ `\hat `{=tex}t\_{ij}\^{(-k)} = g\_{j,-k}(z_i\^p) \]

其中 probe 只在非 `k` 折患者上训练，在 `k` 折患者上评估。

方向级 OOF skill：

\[ O_j = 1- `\frac{
\operatorname{PatientMacroMSE}(t_j,\hat t_j^{OOF})
}{
\operatorname{PatientMacroMSE}(t_j,\bar t_{j,fit})
+\epsilon
}`{=tex} \]

方向稳定性：

\[ S_j = 1-`\operatorname{Dispersion}`{=tex} `\left`{=tex}(
O_j^{(1)},`\ldots`{=tex},O_j^{(K)},O_j\^{seed} `\right`{=tex}) \]

实际实现中应将稳定性映射到 `[0,1]`，并预先规定 dispersion
计算方式。建议用 bootstrap patient resampling 或 repeated patient
folds，不能把窗口当成独立重复实验。

方向选择集合：

\[ `\mathcal `{=tex}J\_{obs} = {j: O_j\>`\tau`{=tex}\_O,;
S_j\>`\tau`{=tex}\_S} \]

阈值不能看 test performance 后再调。应在训练/validation
上预注册候选阈值或直接用固定 top-k rank 规则，并报告敏感性曲线。

## 5.4 教师侧投影矩阵

如果使用标准化原始 teacher coordinate：

\[ P\_{obs} = `\operatorname{diag}`{=tex}(a_1,`\ldots`{=tex},a_D) \]

其中 `a_j` 为二值保留 mask，或经过预先定义的平滑权重。

如果使用 PCA basis `V`：

\[ P\_{obs}=V`\operatorname{diag}`{=tex}(a)V\^`\top`{=tex} \]

对真实 ECG Teacher feature 投影：

\[ `\boxed{
z_i^{e,obs}
=
P_{obs}(z_i^e-\mu_e)
}`{=tex} \]

这里的目标仍由真实 ECG Teacher 输出产生。投影矩阵只由训练患者 OOF
可预测性与稳定性确定。

**关键防泄漏要求：** `P_obs`、teacher normalization、direction
selection、rank/threshold 都只能在训练患者中确定；validation
只用于模型选择，test 只用于最终评估。

## 5.5 B 的蒸馏目标

Student prediction head：

\[ `\hat `{=tex}z_i^{e,obs}=h\_{`\phi`{=tex},B}(z_i^p) \]

B 损失：

\[ `\ell`{=tex}\_i\^B = `\operatorname{Huber}`{=tex} `\left`{=tex}(
`\hat `{=tex}z_i\^{e,obs}, `\operatorname{sg}`{=tex}(z_i\^{e,obs})
`\right`{=tex}) \]

B 权重可从 A 的质量可靠性框架开始：

\[ w_i\^B = m_i q_i\^e q_i\^p \]

方向可预测性已经进入 `P_obs`，因此第一版不要同时把相同 `O_j`
再乘入每个方向的损失权重，以免重复计算可预测性。方向权重版本作为单独消融。

\[ `\mathcal `{=tex}L_B = `\frac{1}{B}`{=tex}`\sum`{=tex}\_i
w_i\^B`\ell`{=tex}\_i\^B \]

\[ `\boxed{
\mathcal L_{COSD}
=
\mathcal L_{SSL}
+
\lambda_B\mathcal L_B
}`{=tex} \]

可选的方向加权形式：

\[ `\mathcal `{=tex}L\_{B-weighted} = `\frac{1}{B}`{=tex} `\sum`{=tex}*i
w_i\^B `\sum`{=tex}*j a_j `\rho`{=tex}(O_j,S_j)
`\operatorname{Huber}`{=tex}(`\hat `{=tex}t*{ij},t*{ij}) \]

此形式应作为 B 的扩展消融，不能和 hard top-k projection
同时引入过多未验证自由度。

## 5.6 重要的解释边界

`z_e - P_obs z_e` 是"当前探针/数据/PPG
表征下未被选入的教师残差"，不是已证明的 ECG 私有生理信息。

可预测性可能来自：

-   心血管动态共享信息；
-   患者身份/基础状态；
-   设备或采集站点；
-   数据集来源；
-   窗口时段和病房环境；
-   ECG Teacher 自身的低层统计。

因此论文中使用：

-   predictable teacher subspace；
-   PPG-predictable teacher directions；
-   cross-fitted observable target；

避免直接使用 causal shared physiology 或 true shared latent state
等过强术语，除非有独立证据支持。

------------------------------------------------------------------------

# 6. 统一训练阶段

## Stage 0：数据和教师审计

1.  建立 MIMIC-III 和 VitalDB 统一窗口 manifest。
2.  统一 50 Hz、30 s、1500 samples。
3.  对 ECG/PPG 时间戳和同步关系做审计。
4.  对 ECG missingness、flatline、clipping、baseline drift、noise
    进行质量审计。
5.  建立 subject/patient/case/record/window 层级 ID。
6.  按真实患者或可验证的 group ID 做 train/val/test split。
7.  固定 manifest hash。

## Stage 1：ECG Teacher pretraining / selection

训练或选择 ECG Teacher。Teacher 必须先达到真实数据上的可用水平。

建议：

-   使用相同数据来源时，Teacher train/val/test 患者划分与 PPG Student
    一致；
-   Teacher 不得看到下游 test patients；
-   固定 Teacher checkpoint；
-   保存训练来源、患者列表、预处理和特征层；
-   比较不同 Teacher checkpoint 的 OOF transfer quality。

Teacher 的预训练目标可以先采用同类 ECG SSL reconstruction
baseline。不要在 COSD 主实验中同时更换 Teacher 架构、Teacher objective
和 Student distillation method。

## Stage 2：PPG-only warm-up

使用现有 TempoBridge：

\[ `\mathcal `{=tex}L=`\mathcal `{=tex}L\_{SSL} \]

所有训练 PPG 窗口都可参加，包括没有 ECG 的样本。

保存：

-   warm-up checkpoint；
-   model config；
-   data manifest hash；
-   train subject list；
-   preprocessing fingerprint；
-   seed；
-   validation reconstruction curve。

## Stage 3：A/B Calibration

对固定 PPG warm-up encoder 和固定 ECG Teacher：

-   统一使用 `transfer_features()`；
-   仅使用 train split 的有效配对样本；
-   patient-balanced sample；
-   K-fold cross-fitting；
-   计算 A 的全教师 OOF skill；
-   计算 B 的方向级 OOF skill/stability；
-   保存投影矩阵和完整审计日志。

Calibration artifact 必须绑定：

-   Student checkpoint SHA256；
-   Teacher checkpoint SHA256；
-   manifest SHA256；
-   train subject IDs；
-   source IDs；
-   input protocol；
-   feature extraction config；
-   quality definition；
-   folds/seeds；
-   projection basis；
-   thresholds/rank；
-   code version。

## Stage 4：A/B Distillation

两条训练线分别从同一个 PPG warm-up checkpoint 初始化。

**A：**

\[
`\mathcal `{=tex}L=`\mathcal `{=tex}L\_{SSL}+`\lambda`{=tex}\_A`\mathcal `{=tex}L_A
\]

**B：**

\[
`\mathcal `{=tex}L=`\mathcal `{=tex}L\_{SSL}+`\lambda`{=tex}\_B`\mathcal `{=tex}L_B
\]

要求：

-   相同初始化；
-   相同样本顺序或可复现 sampler；
-   相同 batch size；
-   相同 optimizer；
-   相同训练 step；
-   相同训练总窗口预算；
-   相同 Teacher；
-   相同 PPG SSL；
-   相同 checkpoint selection；
-   相同下游 head 和 evaluation split。

如果 A/B
由于权重筛选产生不同有效蒸馏样本量，必须同时报告有效配对样本数、总权重和实际
transfer gradient norm，并增加 matched-coverage / matched-total-weight
对照。

## Stage 5：PPG-only export

最终只导出 PPG Encoder：

-   不导出 ECG Teacher；
-   不导出 calibration probe；
-   不导出 projection head；
-   不导出 transfer head；
-   不要求 ECG input；
-   不要求 ECG availability metadata。

验证导出的模型与训练时 PPG-only Encoder 在同一输入上的 embedding 一致。

------------------------------------------------------------------------

# 7. A/B 核心实验矩阵

## 7.1 必须比较的主方法

  ----------------------------------------------------------------------------------------
  ID             Method             ECG teacher target   Calibration        PPG SSL
  -------------- ------------------ -------------------- ------------------ --------------
  M0             Random-init PPG    None                 None               Fine-tune
                                                                            baseline

  M1             PPG-only SSL       None                 None               Yes

  M2             Vanilla KD         Full teacher         None               Yes

  M3             Quality-weighted   Full teacher         Signal quality     Yes
                 KD                                      only               

  M4             PW-KD (A)          Full teacher         OOF global         Yes
                                                         predictability     

  M5             COSD (B)           Projected teacher    OOF direction      Yes
                                                         predictability +   
                                                         stability          

  M6             B-soft             Projected/weighted   Continuous         Yes
                                    teacher              direction weights  

  M7             Shuffled-pair KD   Mispaired teacher    Same as A/B        Yes
  ----------------------------------------------------------------------------------------

M0 只用于下游任务微调基线；M1--M7 的预训练预算应严格匹配。

## 7.2 主比较假设

-   H1：M4 优于 M2，说明 patient-level predictability calibration
    比无条件 full-teacher KD 更有效。
-   H2：M5 优于 M4，说明教师子空间筛选比只调整全局蒸馏强度更有效。
-   H3：M5 不劣于 M1，且在部分 ECG 相关任务上优于 M1，说明 COSD
    没有以牺牲 PPG 原生信息换取教师对齐。
-   H4：M7 明显弱于正确配对的 M4/M5，说明提升依赖 ECG--PPG
    的真实对应关系，而不是额外训练量或数据来源标签。
-   H5：M5
    在跨来源测试、患者宏平均和低质量分层下仍保持收益，才有理由主张方法具有较强泛化价值。

所有假设均允许被否定。

------------------------------------------------------------------------

# 8. 全部关键消融实验

## A. Backbone 与 PPG SSL 消融

  -----------------------------------------------------------------------
  ID                      Encoder                 Purpose
  ----------------------- ----------------------- -----------------------
  A1                      local                   历史基础 Encoder

  A2                      multiview               固定多视图基线

  A3                      selective-mean          固定平均融合

  A4                      selective-adaptive      当前主 Backbone

  A5                      selective without       coverage contribution
                          coverage                

  A6                      selective without       morphology auxiliary
                          morphology branch       contribution

  A7                      patch size 10 vs 25 vs  temporal token
                          50                      granularity
  -----------------------------------------------------------------------

主 COSD 论文不应把所有 tokenizer
变体都作为新贡献；这一组用于确认结果不是单纯来自更强的 PPG Backbone。

## B. A 方法校准消融

  ID   Probe                   Quality             Scale
  ---- ----------------------- ------------------- ----------------------
  B1   No probe / Vanilla KD   None                30 s
  B2   Ridge                   None                30 s
  B3   MLP                     None                30 s
  B4   Ridge + MLP consensus   None                30 s
  B5   Consensus               PPG quality bins    30 s
  B6   Consensus               PPG + ECG quality   30 s
  B7   Consensus               Quality             2/8/30 s exploratory

对每种 probe 报告 OOF skill、patient macro
error、质量分层覆盖率、患者数、窗口数和稳定性。

## C. B 子空间消融

### C1. 子空间选择方法

-   Original teacher coordinates；
-   PCA teacher basis；
-   Random projection matched rank；
-   Random direction selection；
-   OOF skill top-k；
-   OOF skill + stability top-k；
-   OOF skill + stability + quality strata。

### C2. Projection rank

候选 rank：

\[ r`\in`{=tex}{16,32,64,128,192,256} \]

同时报告：

-   retained dimension ratio；
-   OOF teacher prediction skill；
-   projection stability；
-   downstream linear probe；
-   full fine-tuning；
-   task-specific gains/losses；
-   effective KD coverage。

若只有完整 256 维效果最好，则 B
的核心子空间筛选假设不成立或当前方向评分不足。

### C3. Hard vs Soft selection

Hard：

\[ a_j`\in`{=tex}{0,1} \]

Soft：

\[ a_j=`\operatorname{clip}`{=tex}(`\rho`{=tex}(O_j,S_j),0,1) \]

需要匹配投影能量或总权重，避免 soft 版本只是保留了更多教师信号。

## D. Reliability × Observability 消融

  ID   PPG quality   ECG quality   Observability
  ---- ------------- ------------- ---------------
  D1   No            No            No
  D2   Yes           No            No
  D3   No            Yes           No
  D4   Yes           Yes           No
  D5   No            No            Yes
  D6   Yes           Yes           Yes

报告低质量窗口覆盖率、低质量任务性能、不同来源的有效蒸馏量。

不要只报告总体平均。

## E. Paired / Unpaired 数据消融

  ID   Paired PPG+ECG         Unpaired PPG
  ---- ---------------------- --------------
  E1   Yes                    No
  E2   No                     Yes
  E3   Yes                    Yes
  E4   Shuffled ECG pairing   Yes
  E5   ECG temporal shift     Yes

E3 为主方案。

E4 至少包括：

-   同患者不同时间窗口错配；
-   跨患者错配；
-   跨来源错配。

E5
使用多个预先规定的时间偏移量，并保留同一时间窗边界。注意人为错位是负对照，不是学习真实生理时延的方法。

## F. Gradient protection 消融

  ID   protect_shallow      PPG SSL gradient    KD gradient
  ---- -------------------- ------------------- -------------------
  F1   True                 Shallow + Context   Context only
  F2   False                Shallow + Context   Shallow + Context
  F3   True + reduced SSL   Reduced             Context only

报告不同层的 gradient norm、参数更新量、下游结果和训练稳定性。

浅层保护是优化/梯度路径约束，不应包装成新的生理信息分解。

## G. Teacher 与目标质量消融

-   Native ECG Teacher；
-   独立预训练的强 ECG Teacher；
-   不同 teacher layers；
-   不同 teacher pooling；
-   ECG quality threshold；
-   Teacher embedding normalization；
-   Teacher random/untrained negative
    control（仅作方法诊断，不作正式模型比较）。

教师层选择必须只在训练/validation 进行，不能根据 test 选择。

## H. 表征审计与混杂因素消融

-   Patient ID probe；
-   Source ID probe；
-   Case/record ID probe；
-   ECG quality prediction；
-   PPG quality prediction；
-   HR/RMSSD probe；
-   SBP/DBP/MAP probe；
-   Rhythm probe；
-   Within-record temporal prediction；
-   Cross-source probe。

如果 COSD 保留的子空间主要编码 patient/source/device
信息，而非对下游生理任务有增量价值，则不能声称其学习到了共享生理知识。

------------------------------------------------------------------------

# 9. 数据划分、采样与泄漏控制

## 9.1 数据源

### MIMIC-III

明确使用实际可用的波形来源/匹配子集，而不是笼统地把整个临床 MIMIC-III
当成完整同步 ECG--PPG 波形集。

记录：

-   subject ID；
-   recording ID；
-   waveform record；
-   PPG availability；
-   ECG availability；
-   sampling rate；
-   time alignment；
-   signal quality；
-   split；
-   preprocessing fingerprint。

### VitalDB

必须区分：

-   patient/subject ID；
-   case ID；
-   recording ID；
-   window ID。

不能仅因 `case_id` 不同就认定是不同患者。需要核实 VitalDB case
与患者身份的关系；若公开数据无法建立真实 patient mapping，论文必须将
group-disjoint claim 限定为 case-disjoint，而不是 patient-disjoint。

## 9.2 分割

主实验采用 source-specific group split：

-   Train：90%
-   Validation：5%
-   Test：5%

比例可沿用当前数据协议，但划分必须先于切窗，且以可靠的 group ID 进行。

如果 subject mapping 不可用，应明确报告 split unit 为
case/record，并降低泛化结论强度。

## 9.3 采样

当前 `SourceBlockSampler` 实现来源级 source mixture，但不是严格
patient-balanced sampler。

主实验建议：

1.  先按来源抽样；
2.  在来源内按患者/group 均衡抽样；
3.  再从患者中抽记录；
4.  最后抽 30 s window。

对比 source mixture：

-   80:20；
-   60:40；
-   50:50。

每种设置匹配总采样窗口数、训练 step 和优化预算。

## 9.4 ECG 缺失

-   `ecg_valid=False`：不计算 Teacher KD；
-   PPG SSL 仍正常执行；
-   不得把 zero placeholder 当作有效 ECG；
-   缺失 ECG 的窗口不能参与 ECG calibration；
-   报告每个来源配对/非配对窗口和患者比例。

## 9.5 同步与时延

50 Hz 统一网格不等于 ECG 与 PPG 的生理事件已经同步到同一相位。

必须区分：

-   acquisition timestamp alignment；
-   resampling grid alignment；
-   physiological pulse arrival delay；
-   detector error；
-   device clock offset。

不应为了让 R-peak 与 PPG peak
对齐而直接移动波形，这可能删除真实生理传播时延。

主方案暂时不增加新 delay-alignment
网络；先审计现有同步与窗口配对，避免将同步错误误认为跨模态不可预测性。

------------------------------------------------------------------------

# 10. 下游任务与评价协议

## 10.1 任务优先级

优先使用 MIMIC-III/VitalDB 中有独立可信参考信号或标签的任务。

  -----------------------------------------------------------------------
  Task                    Reference               Primary metric
  ----------------------- ----------------------- -----------------------
  Heart rate              ECG R-peak              MAE (bpm)

  RMSSD / HRV             ECG R-R intervals       MAE / correlation

  Rhythm classification   ECG-derived rhythm      AUROC / AUPRC / F1
                          labels                  

  SBP                     ABP                     MAE (mmHg)

  DBP                     ABP                     MAE (mmHg)

  MAP                     ABP                     MAE (mmHg)

  Respiratory rate        RESP reference          MAE

  PPG signal quality      Independently defined   Macro-AUROC
                          SQI/annotations         
  -----------------------------------------------------------------------

任务是否可用取决于数据集中真实标签/参考信号的可用率，不能假定每个窗口都具备
ABP、RESP 或 ECG 标签。

对于 HR/HRV，明确 ECG R-R 与 PPG pulse interval 的区别，不能混用作为同一
gold standard。

对于 BP，优先用有效 ABP waveform 或明确记录的 ABP-derived
reference；不能用模型自身预测或 PPG fallback 作为独立 ground truth。

## 10.2 评估设置

每个任务分别报告：

1.  Frozen linear probe（主表）；
2.  Full fine-tuning（次表）；
3.  Label efficiency：1%、5%、10%、25%、100%；
4.  Within-source patient/group-disjoint；
5.  Cross-source zero-shot；
6.  Cross-source adaptation（可选）。

## 10.3 跨来源评估

必须区分：

**Joint-source held-out groups：** MIMIC 和 VitalDB
都参与预训练，但测试患者/group 未参与训练。

**True source-held-out zero-shot：** 训练时完全不使用目标来源。示例：

-   MIMIC-only pretraining → VitalDB test；
-   VitalDB-only pretraining → MIMIC test。

不能把 joint-source pretraining 后的 VitalDB test 称为 unseen-source
zero-shot。

## 10.4 指标

分类：

-   AUROC；
-   AUPRC；
-   Macro-F1；
-   Sensitivity / Specificity。

回归：

-   MAE；
-   RMSE；
-   Pearson/Spearman；
-   patient-macro MAE；
-   calibration/error by subgroup。

表示：

-   Linear probe；
-   Few-shot；
-   Full fine-tuning；
-   OOF teacher predictability；
-   projection rank；
-   retained teacher energy；
-   patient/source decodability。

鲁棒性：

-   PPG quality strata；
-   ECG quality strata；
-   ECG missing；
-   injected noise；
-   time shift；
-   source/device/case strata。

效率：

-   parameter count；
-   FLOPs；
-   PPG-only inference latency；
-   training GPU-hours；
-   Teacher overhead；
-   calibration wall-clock/memory。

------------------------------------------------------------------------

# 11. 统计分析与结果报告

## 11.1 随机种子

建议至少 3 个独立 pretraining seeds；资源允许时使用 5 个。

不要把同一 checkpoint 的多个下游 head seed 当成多个独立预训练重复。

报告：

-   每个 pretraining seed 的结果；
-   mean ± sample SD；
-   patient-level bootstrap CI；
-   每个来源单独结果；
-   patient-macro 结果。

## 11.2 配对统计

A/B 应在完全相同测试患者、窗口和标签支持上比较。

-   Regression：patient-cluster bootstrap paired difference；
-   Classification：patient-cluster bootstrap AUROC/AUPRC difference；
-   多任务：逐任务报告，不用不透明的异单位算术平均作为唯一结论；
-   多重比较：预先指定 primary tasks，其余作为 secondary/exploratory。

## 11.3 不允许的论证方式

不能仅凭：

-   KD loss 更低；
-   probe R² 更高；
-   OOF error 更低；
-   训练收敛更快；
-   一个任务提升；
-   一个 seed 提升；
-   质量分组权重看起来合理；

就宣称 COSD 学到了更好的生理共享表示。

必须将 OOF 可预测性、表示审计、下游效用和跨来源泛化联系起来。

------------------------------------------------------------------------

# 12. Go / No-Go 判据

在正式写论文前，设置以下判据。

## 12.1 Go：继续发展 COSD

需要同时看到：

1.  OOF direction scores 在不同 patient folds/seeds 下有可重复性；
2.  B 的投影方向不是由 source/patient/device shortcut 主导；
3.  B 在至少一组预先指定的核心下游任务上稳定优于 A；
4.  B 不明显损害 PPG-only SSL 代表的原生 PPG 任务；
5.  B 的优势在 patient-level bootstrap 和多个 seeds 下保持；
6.  正确 ECG--PPG pairing 明显优于 shuffled pairing；
7.  跨来源结果不完全依赖某一个数据源；
8.  相同预算下 B 的收益不是由有效样本数或蒸馏权重总量差异造成。

## 12.2 No-Go：不再把 B 作为论文主创新

出现以下任一情况，应认真考虑放弃或重新定义 B：

-   B 与 A 没有稳定差异；
-   B 只降低 KD loss，却没有下游收益；
-   B 的可预测方向主要编码患者/来源/设备；
-   B 对某些关键 PPG 任务造成系统性退化；
-   rank/threshold 对随机种子高度敏感；
-   shuffled ECG pairing 与正确 pairing 结果接近；
-   B 的优势完全来自更少的蒸馏监督；
-   简单 Ridge baseline 已达到同样效果，且没有更强的科学解释。

此时可以保留 A
作为研究结论，或转向其他从真实实验中发现的问题。不要为了保住 Idea
而反复增加复杂模块。

------------------------------------------------------------------------

# 13. 逐文件工程改造清单

## 13.1 `src/tempobridge/model.py`

**保留：** 现有 tokenizer、shallow/context blocks、重建头和 PPG-only
forward。

**新增：**

``` python
def transfer_features(self, ppg, detach_shallow=False):
    result = self.encode(ppg, detach_shallow=detach_shallow)
    return result["context"].mean(dim=1)
```

建议增加单元测试：

-   输出 shape `[B, 256]`；
-   calibration 与 training 调用得到相同特征；
-   `detach_shallow=True` 时 shallow/tokenizer 无 KD 梯度；
-   PPG SSL 仍可更新 shallow/tokenizer。

## 13.2 `src/tempobridge/calibration.py`

将现有标量 calibration 扩展为两个独立 artifact：

**A artifact：**

``` python
{
    "method": "pw_kd",
    "feature_spec": "context_mean",
    "fold_assignments": ...,
    "probe_metrics": ...,
    "quality_bin_weights": ...,
    "target_mean": ...,
    "target_scale": ...,
    "provenance": ...
}
```

**B artifact：**

``` python
{
    "method": "cosd",
    "feature_spec": "context_mean",
    "teacher_mean": ...,
    "teacher_scale": ...,
    "basis": ...,
    "selected_directions": ...,
    "direction_oof_skill": ...,
    "direction_stability": ...,
    "rank": ...,
    "thresholds": ...,
    "fold_assignments": ...,
    "provenance": ...
}
```

Artifact 需要版本号和 hash 校验。不得静默复用旧的 calibration 文件。

## 13.3 `src/tempobridge/teacher.py`

**保留：** frozen teacher、native/TorchScript wrapper、sampling rate
检查、训练患者元数据。

**新增：**

-   明确 teacher feature spec；
-   输出固定 256 维或配置指定维度；
-   支持完整 batch 的 target 提取；
-   保存 teacher layer/pooling provenance；
-   对 teacher 输出做 finite/shape/range 检查。

## 13.4 `src/tempobridge/training.py`

建议将 transfer objective 抽象为策略：

``` python
if transfer_method == "vanilla":
    target = teacher(ecg)

elif transfer_method == "pw_kd":
    target = teacher(ecg)
    weight = pw_calibrator.lookup(...)

elif transfer_method == "cosd":
    target = project_teacher(teacher(ecg), cosd_artifact)
    weight = quality_weight(...)
```

共同保持：

``` python
total_loss = ppg_ssl_loss + transfer_weight * transfer_loss
```

注意：

-   `teacher` 永远无梯度；
-   projection matrix 永远无梯度；
-   calibration weights 永远无梯度；
-   PPG SSL 对 PPG Encoder 的梯度边界不因 COSD 改变；
-   transfer head 不进入 PPG-only export；
-   batch 内没有有效 ECG 时只执行 SSL；
-   固定 batch denominator；
-   记录有效 paired count、weight sum、KD gradient norm。

## 13.5 `src/tempobridge/config.py`

新增配置建议：

``` python
transfer_method: str = "pw_kd"  # none / vanilla / pw_kd / cosd
projection_basis: str = "standardized"  # standardized / pca
projection_rank: int = 64
projection_selection: str = "topk_oof_skill"
projection_stability: bool = True
projection_threshold: float = 0.0
calibration_folds: int = 5
calibration_probe: str = "consensus"
```

参数必须经过严格校验。不要在默认 config 中静默启用
COSD；旧实验配置应保持兼容。

## 13.6 `src/tempobridge/sampling.py`

现有 `SourceBlockSampler` 解决 source mixture，不等于 patient-balanced
sampling。

新增 patient/group-aware sampler，记录每 epoch：

-   每来源窗口数；
-   每来源 group 数；
-   每 group 窗口数分布；
-   paired/unpaired 数；
-   有效 ECG 数；
-   重复/遗漏统计。

## 13.7 `src/tempobridge/preprocess.py`

增加 ECG 质量与同步审计字段，但不把质量评分伪装成临床可信概率。

建议保存：

``` text
ppg_valid
ecg_valid
ppg_sqi
ecg_sqi
ppg_missing_ratio
ecg_missing_ratio
ppg_clipping_ratio
ecg_clipping_ratio
ppg_flatline_score
ecg_flatline_score
timestamp_alignment_status
source_id
subject_id
case_id
record_id
window_id
```

## 13.8 `src/tempobridge/inference.py`

保持 PPG-only inference：

``` python
embedding = exported_encoder(ppg)
```

不加载 ECG Teacher、calibration artifact 或 projection matrix。

导出测试需要验证：

-   同一 PPG 输入在 training encoder 与 exported encoder 得到相同
    embedding；
-   不提供 ECG 也能运行；
-   输出维度固定；
-   checkpoint 不包含 Teacher 权重。

------------------------------------------------------------------------

# 14. 建议的实验执行顺序

  -----------------------------------------------------------------------------------
  阶段              实验                         产出              继续条件
  ----------------- ---------------------------- ----------------- ------------------
  P0                数据                         manifest + audit  group split 可信
                    ID、配对、采样率、同步审计                     

  P1                PPG-only SSL                 baseline          真实数据训练稳定
                                                 checkpoint        

  P2                Vanilla ECG KD               baseline          paired KD 有效
                                                 checkpoint        

  P3                A：PW-KD                     A checkpoint +    A 优于 vanilla
                                                 OOF report        或提供清晰诊断

  P4                B：COSD                      B checkpoint +    方向可预测性稳定
                                                 projection report 

  P5                A/B 全部关键消融             ablation matrix   B 有独立增益证据

  P6                下游和跨来源评估             final benchmark   patient-level
                                                                   稳定提升

  P7                表征审计、负对照、统计检验   evidence package  排除主要替代解释

  P8                论文撰写                     Method/Results    Go 判据满足
  -----------------------------------------------------------------------------------

不要在 P0 尚未完成时启动大规模正式预训练。

------------------------------------------------------------------------

# 15. 预期论文叙事

如果 B 的实验支持假设，论文可以组织为：

## Title（暂定）

**Learning What PPG Can Observe: Cross-Fitted Observable-Subspace
Distillation for ECG-Guided Physiological Foundation Models**

## Contributions（待结果支持后再确定）

1.  **Problem formulation：** 将 ECG-guided PPG pretraining
    中的全教师拟合问题转化为患者级 OOF 可预测教师子空间选择问题。
2.  **Method：** 提出 cross-fitted teacher-direction scoring 与
    observable-subspace target distillation，在不把 ECG
    输入部署模型的情况下迁移筛选后的教师目标。
3.  **Evaluation：** 在 MIMIC-III 和 VitalDB 上比较 PPG-only
    SSL、vanilla KD、quality-weighted KD、PW-KD 和
    COSD，并通过患者/来源分组审计、错配负对照及跨来源实验验证方法边界。

如果 B 不优于
A，则不要把上述三条写成已成立贡献。可以把研究结论调整为：全教师可预测性加权是否已足够、子空间筛选何时有益/无益。

------------------------------------------------------------------------

# 16. 与参考工作的关系和新颖性边界

以下归纳基于用户提供的 PulseBound PDF、PPG.zip 中论文、FM.docx
论文清单及当前仓库研究文档。部分论文为
under-review/preprint，方法细节应在投稿前通过论文全文、公开代码和最新版本再次核对。

## 16.1 PulseBound

已明确内容：

-   50 Hz、30 s、1500 samples；
-   1 s patch；
-   prefix-only normalization；
-   suffix replacement before derived views；
-   aligned masking；
-   future-beat state forecasting；
-   可选 ECG-derived PAT 只作训练监督；
-   PPG-only inference。

TempoBridge 当前只采用相同采样率/窗口长度，不复用其完整
architecture/objective。本文不把 future-beat forecasting 或
stored-suffix invariance 作为自己的创新。

## 16.2 PhysioDyT / BEATFORMER

这些工作已经覆盖 ECG--PPG 正向延迟、动态对齐、beat-level
对应关系等方向。因此本研究不将 delay estimation 或 cross-modal alignment
单独作为贡献。

## 16.3 QRAFT / RAMP / Learning What to Trust

这些工作提示可靠性条件化建模已有直接先例。本研究应区分 signal quality
reliability 与 teacher-direction
predictability，不把普通质量加权作为主要创新。

## 16.4 Long-Horizon Physiological FM

其多模态与 PPG-only streams 的联合训练思想与本项目设置相关。因此
paired + unpaired PPG training 本身不是创新。本文的重点应是 OOF
估计教师可预测子空间，以及该目标选择对 PPG-only transfer 的作用。

## 16.5 People over Segments

提示生理预训练数据规模必须同时关注独立个体数量和每个个体的窗口数量。因此本项目应报告
group 数量、每组窗口数和 patient-macro 指标，不能只报告总窗口量。

## 16.6 Decodability-Grounded Audit

提示 representation claims
需要通过可解码性、协变量调整和信息擦除等审计。COSD
必须证明保留方向不只是 patient/source/device shortcut。

## 16.7 FM.docx 中的时间序列 Foundation Models

Mantis、PULSE、FlowState、EntroPE、LogSig-SSM、OATS、Kairos 等可作为
Backbone、长时序、采样率适配或数据增强对照来源；不应将
RMSNorm、RoPE、SwiGLU、动态 patch 或普通多尺度模块直接包装为本文主贡献。

## 16.8 需要继续核对的近邻工作

当前仓库 `docs/RESEARCH.md` 已列出 ATLAS、RPM-Distill、Teaching PPG How,
Not Who、CardioState-JEPA
等近邻方向。它们与可预测教师目标、跨模态蒸馏、身份捷径和跨模态时延有直接关联。

投稿前必须逐项核对：

-   是否已有教师可观测子空间投影；
-   是否已有患者级 cross-fitted teacher target selection；
-   是否已有按 student predictability 选择 teacher directions；
-   是否已有完整 teacher residual 被显式排除；
-   是否已有同样的 paired/unpaired PPG training 组合。

如果已有方法在数学和训练目标上实质相同，必须调整研究问题或放弃该主张，不能仅靠更换名称宣称新颖。

------------------------------------------------------------------------

# 17. 关键失败模式与应对

  ----------------------------------------------------------------------------------------------
  失败模式                诊断                      应对
  ----------------------- ------------------------- --------------------------------------------
  OOF skill 低            PPG 无法稳定预测 Teacher  不强行做 COSD；检查 Teacher/数据同步

  OOF skill               可预测不等于有用          加入任务效用分析和 shortcut probes
  高但下游无收益                                    

  B 只在一个来源有效      source confounding        source-held-out、source-balanced、分源投影

  B 主要预测 patient ID   identity shortcut         within-record analysis、patient-adjusted
                                                    probe

  B 丢失 PPG 原生任务     过度迁移                  保留 SSL、调低 transfer weight

  B 比 A                  coverage confound         matched coverage / matched total weight
  低但训练样本也更少                                

  随机配对也有效          可能是额外训练/来源捷径   强化 pairing controls

  ECG quality score       质量估计错误              人工审阅子集或独立 SQI validation
  不可信                                            

  Teacher checkpoint      潜在患者重叠              透明报告未知，不声称严格无泄漏
  来源不清                                          

  子空间 rank 不稳定      方向选择过拟合            多 seed、bootstrap、固定 rank sweep
  ----------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 18. 最终执行建议

当前最合理的开发策略是：

1.  **先修正 transfer feature 对齐问题。** Calibration 与训练 KD
    必须使用完全相同的 256 维 context mean（或明确选择另一共同接口）。
2.  **先做 M0/M1/M2。** 确定 PPG-only SSL、vanilla KD 的真实数据基线。
3.  **实现 A。** 这是当前 calibration 的自然、严格基线。
4.  **实现 B。** 使用 OOF direction scores 构造教师侧
    projection，再对真实 Teacher targets 做投影。
5.  **先在固定数据子集上完成 rank/threshold/negative controls。**
    不要直接启动最大规模训练。
6.  **A/B 同预算比较。** 以患者宏平均、跨来源和下游效用为主，而不是只看
    KD loss。
7.  **满足 Go 判据后再写 ICML 故事。**

## 最终研究判断

COSD 是当前最值得验证的候选方法，但它的风险也很明确：

-   "可预测性校准"已有相关研究；
-   "学生可观测教师目标"有直接近邻；
-   "ECG 辅助 PPG"已有成熟工作；
-   "配对与非配对联合训练"也不是新颖性来源。

因此，本文真正需要证明的不是"我们也做了 ECG-guided PPG"，而是：

> **患者级交叉拟合所识别的教师输出子空间，是否能比完整教师目标更有效地指导
> PPG-only
> 表征学习，并且这种优势是否能在独立患者、独立来源、质量分层和负对照中成立？**

这就是本方案的核心可证伪命题。

------------------------------------------------------------------------

# 附录 A：首轮配置建议

``` json
{
  "model": {
    "encoder": "selective",
    "dim": 256,
    "depth": 6,
    "protected_depth": 2,
    "heads": 8,
    "patch_size": 10,
    "feature_pool": "mean",
    "feature_layer": -1,
    "selective_kernel_sizes": [9, 21, 41],
    "selective_width": 32,
    "selective_fusion": "adaptive",
    "selective_local_context": true,
    "selective_coverage": true
  },
  "data": {
    "ppg_fs": 50,
    "ecg_fs": 50,
    "scales_seconds": [30.0],
    "window_mode": "full",
    "source_weights": {
      "mimic": 0.8,
      "vitaldb": 0.2
    }
  },
  "train": {
    "transfer_method": "pw_kd",
    "protect_shallow": true,
    "mask_ratio": 0.5
  },
  "calibration": {
    "folds": 5,
    "probe": "consensus",
    "max_windows_per_subject": 16
  },
  "cosd": {
    "basis": "standardized",
    "rank": 64,
    "selection": "topk_oof_skill",
    "stability": true
  }
}
```

注：这是研究配置草案，不保证与现有 `ExperimentConfig`
直接兼容。`transfer_method` 和 `cosd`
字段需要在代码中实现后才能加载。`feature_pool="mean"` 是为了使
calibration 与训练 transfer feature 统一为 256 维 context
mean；如选择其他表示，必须在两个路径同步实现。

------------------------------------------------------------------------

# 附录 B：首轮实验清单

-   [ ] 确认 MIMIC-III 与 VitalDB 的 split unit 和真实 group ID。
-   [ ] 生成固定 manifest 和 SHA256。
-   [ ] 统计 50 Hz / 30 s / 1500 点窗口数量。
-   [ ] 统计 ECG paired/unpaired 比例。
-   [ ] 审计 ECG/PPG 时间戳与同步误差。
-   [ ] 完成 ECG Teacher checkpoint 与 provenance。
-   [ ] 完成 PPG-only SSL baseline。
-   [ ] 完成 Vanilla ECG KD baseline。
-   [ ] 修复 transfer feature mismatch。
-   [ ] 实现 A 的 patient-level OOF calibration。
-   [ ] 实现 B 的 direction-wise OOF scoring。
-   [ ] 实现 teacher-side projection artifact。
-   [ ] 实现 COSD projected-target training。
-   [ ] 完成 rank sweep。
-   [ ] 完成 shuffled-pair / shifted-ECG controls。
-   [ ] 完成 quality × observability ablation。
-   [ ] 完成 patient/source ID probes。
-   [ ] 完成 patient-macro linear probe。
-   [ ] 完成 cross-source zero-shot。
-   [ ] 完成 bootstrap 与多 seed 报告。
-   [ ] 验证 PPG-only export 不依赖 ECG。

------------------------------------------------------------------------

## 参考材料与来源边界

本方案依据以下用户提供材料形成：

1.  `2027 - PulseBound Future-Beat State Forecasting Under an Explicit Information Boundary.pdf`。
2.  `PPG.zip` 中的 20 篇论文，包括 PulseBound、Learning What to
    Trust、PhysioDyT、OPLR、Structured Trajectory
    Supervision、DPIKAN、Individual-level Public-Private
    Decoupling、MSMA-PPG、wHuBERT Layer Selection、Unlocking the
    Specialization、PhysPPG、Pulse2View、People over
    Segments、PhysioSkeptic、BeatFlow、Long-Horizon
    Physiologic、Physiology-Aware
    Hierarchical、RAMP、BEATFORMER、Decodability-Grounded Audit。
3.  `FM.docx` 中列出的 18 个时间序列 Foundation Model 工作及其仓库地址。
4.  上传的 `.github.zip` 中 `src/tempobridge/` 和 `docs/`
    的代码/文档，尤其是
    `METHOD.md`、`ENCODER.md`、`RESEARCH.md`、`TEACHER.md`、`DATA.md` 和
    `configs/pulsebound_50hz.json`。

其中论文处于 under-review 或 preprint
状态的内容，应在正式投稿前核对最终版本与公开实现。本文将"来源中明确描述的工作"与"本方案提出的设计/假设"分开表述，不将尚未完成的实验写成结果。
