# 实验计划、对照与表征审计

所有路径由命令行传入。以下命令是运行方式示例；开发验证只使用 CPU 合成信号。
训练配置中的 `device` 决定执行设备。不要把默认 `auto` 的正式配置当成 CPU smoke 配置。

## 先生成计划

```powershell
python -m tempobridge plan-experiments --config configs/cosd_30s.json --manifest data/manifest.json --output runs/study --families all --seeds 17 29 43
python -m tempobridge run-experiments --plan runs/study/plan.json
```

这两条命令不训练。第二条只显示选择的方法和预算。明确执行时：

```powershell
python -m tempobridge run-experiments --plan runs/study/plan.json --execute --only ssl vanilla quality pw_kd cosd cosd_pw random_rank pca_rank
```

`plan.json` 是预声明配置，`state.json` 记录阶段、内容哈希及运行资源；`summary.json` 汇总已经完成的方法。
`complete=false` 表示计划中仍有未执行方法，不等于运行失败。仅选随机/PCA 对照时会先执行其 COSD rank 依赖。
更改代码、计划、标签、波形内容或教师后，需建立新计划目录；不能继续复用旧状态。
同一计划一次只运行一个执行进程。失败的非训练产物会移至 `artifacts/interrupted/`，不删除原产物。
预训练阶段支持 epoch checkpoint 恢复；探针与完整微调阶段中断后重新执行该阶段。

## 实验族与比较条件

| family | 内容 |
|---|---|
| core | SSL、vanilla、quality、PW-KD、COSD、COSD+PW 权重、实际 rank 匹配的随机/PCA、soft 及方向权重总量对照 |
| backbone | local、multiview、selective-mean、无 coverage、无形态支路、patch 10/25/50 |
| calibration | ridge/MLP/consensus × 全局/质量组，2/8/30 秒裁剪 |
| subspace | 原始/PCA/随机基、随机方向、无稳定性、质量组稳定性、rank 与阈值 |
| quality | PPG 质量、ECG 质量、方向可观测性的六种组合 |
| pairing | 同记录、跨患者、跨来源错配，−30/+30/+60 秒；各有匹配覆盖率和质量的正确配对对照 |
| data | paired-only、unpaired-only SSL、全数据、每个来源单独预训练及其他来源测试 |
| gradient | 浅层保护、无保护、降低 SSL；各层梯度范数与实际参数更新量 |
| teacher | 原生教师层/池化/L2 输出、校准 ECG 质量阈值、显式随机教师诊断 |
| sampling | 两来源时的 80/20、60/40、50/50；顺序为计划中的来源名称排序 |

各族自动包含 core 参照。未提供独立参考标签时 physio 标为 unavailable；单来源时跨来源错配不可用。
有标签但缺少 `train.physio_targets` 指定目标，也不会自动加入 physio。
严禁把 unavailable 或 rank-zero 当成缺失结果后悄悄排除；报告原因与有效覆盖率。
部分数据集可能没有足够的 discovery 患者、时间偏移候选或下游类别，此时工具明确失败，需要修订预声明计划。

同一学生结构/数据协议/子集/种子共享 PPG warm-up，教师与 feature cache 按输入契约复用。
计划固定患者采样、每 epoch 窗口数、batch/accumulation、epoch 数和 `last` checkpoint。
来源配比使用可整除的共同窗口预算；小数据不足一个预算单位时，显式设置合法的 `samples_per_epoch`。
不同 backbone 的计算量仍可能不同；相同步数不能解释为相同 FLOPs。
`cosd_pw` 与 PW-KD 共用标量权重定义，摘要会核对实际覆盖率和总权重；rank-zero 时可能无法匹配。
随机/PCA 若没有足够非恒定方向实现参照 rank，会拒绝声称匹配成功。

## 接入独立标签与下游任务

```powershell
python -m tempobridge plan-experiments --config configs/cosd_30s.json --manifest data/manifest.json --output runs/labeled --families core data --labels data/targets.csv --targets hr_bpm:regression rhythm:classification --label-fractions 0.01 0.05 0.1 0.25 1 --seeds 17 29 43
```

默认运行冻结探针，以及预训练/随机初始化的完整微调；`--skip-finetune` 仅保留探针。
标签比例按整组患者抽取，验证集独立选择探针超参数或微调 epoch；测试标签不参与选择。
真正 source-held-out 使用保持原始划分的来源视图，预训练、教师、下游拟合与验证都不使用目标来源。
联合来源预训练后的分来源测试仍属于已见来源的 group-disjoint 评估。
报告包括总体、patient macro、来源、质量组、ECG 可用性、病例/记录分组；缺失元数据明确标记，病例明细超过 256 组时注明截断。
质量分层是输入诊断，并不意味着质量分数已获得临床校准；可用性元数据也不等于实际 ECG 波形通过清洗。

摘要同时输出各 seed 的值、均值和样本标准差，以及同一 seed、同一测试支持上的配对患者 bootstrap。
它不会把多个 seed 混成伪独立患者，也不会自动判断 Go。样本标准差在只有一个 seed 时为 null。

## 教师与表征审计

```powershell
python -m tempobridge teacher-variant --checkpoint runs/ecg/last.pt --output runs/ecg_layer2.pt --layer 2 --pool context_mean
python -m tempobridge teacher-variant --checkpoint runs/ecg/last.pt --output runs/random_teacher.pt --random-seed 17
python -m tempobridge audit-representation --cache runs/cache --calibration runs/cosd.json --output runs/audit.json --labels data/targets.csv --targets hr_bpm rmssd_ms sbp_mmhg dbp_mmhg map_mmhg
```

随机教师只能通过 `--teacher-kind random_control` 使用，保留零训练步、无训练患者和诊断标记。
外部强教师通过 `plan-experiments --teacher ... --teacher-kind torchscript` 接入独立计划，需带训练来源和输入协议元数据。
工具不能验证其“强”或独立性；未知训练来源会阻止 source-held-out 声明。外部 adapter 自行定义内部层和池化。
原生 `summary` 对 local encoder 等同 context mean；对含 summary token 的 encoder 才是不同读出。

审计比较 PPG、完整教师、保留子空间和残差：source OOF 解码、质量预测、独立生理标签预测、身份诊断、记录内变化。
身份诊断在同一组身份下按时间前后分半，不是新患者识别任务。记录内去均值使用审计记录的统计，
是回顾性动态审计，不是未来预测或线上算法。audit 患者不反向影响方向选择；所有 OOF 都只指冻结表征上的探针。

## 扰动、效率和 CPU 检查

```powershell
python -m tempobridge features --checkpoint runs/encoder.pt --manifest data/manifest.json --split test --output runs/test_noise.npz --device cpu --noise-std 0.2 --noise-seed 17
python scripts/benchmark.py --config configs/cosd_30s.json --output runs/benchmark.json --teacher runs/ecg/last.pt
python scripts/smoke_experiments.py --output runs/cpu_matrix_check
```

噪声是相对每个窗口标准差的白高斯扰动，按 sample ID 固定；真实 PPG 运动伪迹不等于该模型。
用干净 train/val 特征与扰动 test 特征调用 `evaluate-probe`，避免通过扰动测试选择超参数。
benchmark 的设备来自配置；只有 smoke 固定 CPU。报告 batch 推理延迟、吞吐、参数量、可选教师前向耗时、
GPU allocated memory，以及 profiler 能计数的部分 FLOPs。该 FLOPs 数不包含所有算子，不能作为完整理论 FLOPs。
每个计划阶段记录 wall-clock 与采样的整个进程 RSS；不是隔离模块的精确峰值，也不是实测大规模 GPU-hours。
