# COSD 研究方案工程核对

范围：按根目录原始研究方案完成可运行的研究代码与实验管理。用户明确要求只在当前机器
检查代码，不进行真实数据/GPU 训练。因此“已实现”指接口与 CPU 合成验证，不指研究假设成立。
原始方案文件保留原文；工程统计细节以 COSD.md 和下表为准。

| 原方案内容 | 对应实现/入口 | 工程边界 |
|---|---|---|
| §2 校准/KD 表征一致 | model.py 的 transfer_features；feature_cache.py | 固定最终 context mean；导出 pooling 独立；三种 encoder 覆盖 |
| §3–4 PW-KD | calibration.py、transfer.py；pw_kd_30s.json | 患者折、fit-fold 标准化/null、ridge/MLP/consensus；OOF 仅指探针 |
| §5 COSD | subspace.py、calibration.py | basis/discovery/audit 患者互斥；候选基、条件 bootstrap、真实教师投影、零 rank 退回 SSL |
| §6 阶段训练 | cache-features/calibrate-cache/pretrain/distill/export | 冻结 teacher/projection；共享 warm-up；epoch 恢复；PPG-only 导出 |
| §7 M0–M7 | experiments.py、experiment_matrix.py | 随机初始化微调 M0；core M1–M6；pairing 族 M7；相同窗口/步数并不代表相同计算量 |
| §8 A Backbone | backbone 族、encoding.py、model.py | local/multiview/mean/adaptive/coverage/morphology/patch；切换结构重新 warm-up |
| §8 B 校准 | calibration 族 | probe、质量组、2/8/30 秒探索；每种协议重建 cache |
| §8 C 子空间 | subspace 族、random_rank/pca_rank、soft_weight_matched | 实际逐尺度 rank 匹配；缺少非恒定方向时拒绝伪装匹配；稳定性是条件 OOF bootstrap |
| §8 D 可靠性 | quality 族；training/transfer 的分来源与质量日志 | PPG/ECG 权重独立；质量未知不解释为临床高质量 |
| §8 E 数据与配对 | controls.py、manifest_tools.py；data/pairing 族 | 真正 waveform-valid paired/unpaired；错误配对与正确对照匹配有效覆盖率/可靠性 |
| §8 F 梯度约束 | gradient 族；transfer.py/training.py | 各层 SSL/KD 梯度、实际参数更新；浅层仍能接受 SSL 更新 |
| §8 G 教师目标 | teacher.py、teacher_controls.py；teacher 族 | 原生层/池化/L2、校准质量阈值、显式 random_control；外部强教师需用户提供权重和来源证明 |
| §8 H 混杂审计 | audit-representation | 身份闭集诊断、source OOF、质量/独立生理标签、保留/残差、记录内变化；不能证明因果共享生理因素 |
| §9 数据/采样 | preprocess.py、data.py、h5data.py、sampling.py | NPZ/HDF5/WFDB/VitalDB；保留来源/组/记录/时间；同时间网格不等于时延校正；未知身份不冒充患者 |
| §10 独立参考 | reference-targets、physio_targets.py、physio_30s.json | ECG/ABP/RESP 原采样率参考 NPZ 或外部 CSV；内置算法仅代理标签；诊断节律/SQI 需外部标注 |
| §10 评价协议 | evaluate-probe/finetune/compare-predictions | 标签比例、val-only 选择、group-disjoint、真正 source-held-out、患者宏平均和分组指标 |
| §10 鲁棒性 | features --noise-std；分层报告；配对时间偏移 | 白噪声为单独诊断，非真实运动伪迹模拟；缺失/质量/来源/病例/设备在有元数据时报告 |
| §10 效率 | scripts/benchmark.py、profiling.py；state.json | 参数、推理/教师耗时、阶段 wall-clock/RSS、部分 profiler FLOPs；未声称完整理论 FLOPs/真实 GPU-hours |
| §11 统计 | experiment_reporting.py、evaluation.py | 默认三 seed；seed 方差与患者 bootstrap 分开；相同测试支持的配对比较；不自动挑选最好 seed |
| §12 Go/No-Go | summary.json + 研究方案原判据 | 必须依赖真实数据结果；没有自动把合成测试通过当成 Go |
| §13 文件与代码管理 | src 布局、configs/tests/docs/scripts、CHANGELOG、CI、Git 分阶段提交 | import 名 tempobridge 保持兼容；运行产物/环境不纳入 Git；输入与输出路径由参数指定 |
| §14–18 执行顺序与论文主张 | DEVELOPMENT_PLAN.md、COSD.md、EXPERIMENTS.md | 不把工程实现当成新颖性或性能证据；原文近邻工作与失败条件保留 |

可选 cross-source adaptation 没有作为强制自动实验；当前优先实现可严格核查的 source-held-out。
原始 WFDB/VitalDB 转换器主要负责模型输入，独立参考原生窗口通过专门 JSONL/NPZ 接口接收，
不会从已降采样 PPG 或学生预测伪造独立生理真值。
研究还需要真实 ECG 教师质量验证、参考标签审核、真实三 seed 下游实验及近邻工作持续核对。
这些是未执行的实证工作，不能从本次 CPU 跑通结果推断。

验证记录见 VALIDATION.md；可复用命令见 EXPERIMENTS.md。目录为 TempoBridge-COSD，
Python 分发和 import 名继续使用 tempobridge，避免破坏既有调用方。
