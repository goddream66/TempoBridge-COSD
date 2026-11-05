# 研究定位与实验设计

## 可检验的主张

1. **尺度与条件相关的可预测性**：ECG 表征并非在每种 PPG 观察时长和质量条件下都同样可学。
2. **负迁移的范围控制**：不加区分的蒸馏可能损害 PPG 形态任务；保留自身监督并限制梯度有机会缓解。
3. **跨患者校准优于当前训练误差筛选**：需要与强基线比较，而非只展示更低蒸馏误差。

这些主张都可能被实验否定。若统一蒸馏已同样好，或者收益只来自少训练几个样本，
不应继续宣称选择机制有价值。现阶段没有“首次”“理论保证提升”“顶会可录用”的结论。

## 与 PulseBound 的关系

审阅对象：用户提供的 PulseBound Future-Beat State Forecasting Under an Explicit Information Boundary
投稿 PDF 与 PulseBound_ICLR 补充包，正式训练路径为 v6。

该版本无 ECG 教师网络。其主线是严格存储窗口前缀边界、未来四心搏代理状态预测、
PPG token/波形重建、多任务 scaffold 与有效性 mask。ECG 提供 HR、RMSSD、节律及可选 PAT 目标。
不能把这些内容重新命名成本文创新。本项目不实现该未来预测任务，也不复用其损失组合或 register 架构。

## 必须纳入相关工作的边界

| 工作 | 已有内容 | 对本项目的约束 |
|---|---|---|
| Mantis / MantisV2 | 可复用时序 Transformer 组件 | 不将 RMSNorm/RoPE/SwiGLU 当创新 |
| CardioState-JEPA | 共享心脏表示、时延对齐、跨模态潜在预测 | 不能把加入对齐或 JEPA 独立当新贡献 |
| Teaching PPG How, Not Who | ECG→PPG 蒸馏的身份捷径与 fixed-effects 修正 | 不能把记录内中心化声称首次 |
| RPM-Distill | 生理跨模态蒸馏、门控、元学习权重 | 不能把按学生收益调权声称首次 |
| ATLAS | 学生可观测结构与可达到教师目标 | 可预测性筛选存在直接概念先例，须进一步核对全文 |

本项目采用跨患者探针评分和时间尺度条件化，不采用 ATLAS 摘要中的图谱投影。
差异不等于新颖性已经成立。需要完整 literature review 及直接实验对比。

原始来源：

- https://github.com/vfeofanov/mantis
- https://arxiv.org/html/2608.12944
- https://arxiv.org/html/2610.10662
- https://arxiv.org/html/2606.28089v1
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7346910
- https://aclanthology.org/2022.acl-long.485/

## 实验矩阵

所有方法匹配数据、backbone、有效 batch、优化步数、目标标准化与 checkpoint 选择。

1. PPG 自监督；固定 ECG 生理 scaffold（后续比较实现）；全量 embedding 蒸馏。
2. ECG 质量加权；单尺度/多尺度；校准加权；打乱权重及严格匹配覆盖/总权重对照。
3. 去形态支路；去梯度保护；去 PPG 自监督（研究对照，当前训练器故意始终保留 SSL）。
4. ridge / MLP / consensus，质量分组敏感性，不同校准样本数与 teacher。
5. 同一记录内打乱 ECG 配对、跨记录打乱、时间偏移、独立 ECG/PPG 噪声。

建议按患者 bootstrap 区间，多随机种子；至少同时验证节律与形态/血流相关任务，
并进行跨站点/设备测试。冻结线性探针与微调分开报告。
核心指标包括下游性能、目标可预测性、有效监督权重、罕见状态覆盖率和训练/推理成本。
不得用低 KD loss 替代下游获益。不要只按平均值掩盖低质量或少见状态的退化。

## 已知风险

- 小 probe 不会学不代表信息不存在；consensus 可能过于保守。
- 质量标签若关联设备或人群，权重可能放大偏差。
- 记录级质量分组无法充分表达逐心搏可迁移性。
- 身份/设备背景可能使跨模态目标容易预测但对任务无用。
- 浅层梯度保护降低负迁移的同时也可能限制正向适配。
- 长窗口平均特征可能抹平罕见局部事件。
- 自监督教师可能主要学到低层统计，需要更强教师与独立任务验证。

研发顺序应是先验证问题存在，再验证最小方法，最后扩展规模；不要为了增加“创新点”不断叠加模块。
