# 验证记录

日期：2026-11-05，数据接入补充验证于 2026-11-06。全部在 **CPU** 上完成；遵照用户要求，没有运行 GPU 训练或 GPU benchmark。
这里只证明工程流程和指定不变量能够运行，不证明真实数据上的表征质量、临床有效性或论文录用可能性。

## 环境

- Windows，Python 3.13.9。
- PyTorch 2.15.0+cpu；NumPy 2.5.4；SciPy 1.18.1。
- 隔离的 `.venv-cpu`，没有继承系统 site-packages。
- 可选精确依赖快照：`requirements/validated-cpu.txt`；其他 Python/平台使用 pyproject.toml 重新解析兼容版本。
- 曾遇到系统 Anaconda MKL/OpenMP 重复加载；通过隔离环境并安装独立依赖解决，没有设置忽略 OpenMP 冲突的环境变量。

## 已执行检查

| 检查 | 实际结果 |
|---|---|
| 初始完整 pytest suite | 38 passed，38.35 秒 |
| 新增蒸馏续训测试 | 1 passed，2.83 秒；模型与预测头逐参数完全一致 |
| Ruff | All checks passed |
| pip check | No broken requirements found |
| Python wheel 构建 | `tempobridge-0.1.0-py3-none-any.whl` 成功 |
| 完整 CLI smoke | 数据审计、两种预训练、校准、蒸馏、导出、提取特征全部成功 |
| 正式尺寸网络 CPU 检查 | 一次前向/反向成功，梯度有限，没有优化器训练 |
| 仓库外迁移加载 | 从临时目录加载 wheel 和相对路径 encoder 成功 |

3 条 warning 来自 PyTorch 2.15 对可选 TorchScript 教师兼容接口的弃用提示。
该接口仍可运行；原生教师与主训练路径不使用 TorchScript。
CI workflow 已提供，但没有声称在 GitHub 托管 runner 上执行过。

## 科学与工程不变量

测试实际覆盖：

- 修改隐藏 patch 的波形数值，不改变编码器输出。
- 修改裁剪外内容，不改变短窗口输入对应的表示。
- 蒸馏梯度在保护开启时不进入 stem、浅层、形态支路或教师；上下文层与预测头梯度非零。
- 关闭保护的消融可以更新浅层。
- 全零/无效 ECG 权重跳过教师，但 PPG 自监督仍可更新模型。
- 将权重缩小一半会将蒸馏损失缩小一半，不被分母重新归一化抵消。
- 线性可共享的合成目标获得高 OOF 权重，独立随机教师噪声被拒绝。
- 常量教师、样本不足的质量组不会得到正权重。
- fold 患者不重叠，长记录不通过样本数量获得更高患者权重。
- 患者跨 split、采样率缺失、波形时长不匹配、坏输入和错误配置被拒绝。
- 教师保持 eval 且无梯度；随机/PPG checkpoint 不能冒充 ECG 教师。
- 原生教师训练患者与当前 held-out 患者重叠时被拒绝。
- CPU 下带 dropout 的预训练续训、带预测头的蒸馏续训与连续运行一致。
- PPG-only 导出保留推理结果，移除训练头权重；输入文件里的坏 ECG 不影响特征提取。

## 完整 CLI smoke

命令（在仓库根目录，使用已安装环境）：

```text
python scripts/smoke.py --output runs/cpu_smoke
```

使用 18 位合成人员 × 6 窗口：训练 14 位/84 窗口，验证和测试分别 2 位/12 窗口。
一个训练窗口缺 ECG，PPG 仍保留。三阶段各只运行 2 个小 epoch，用于跑通检查。
输出测试特征形状 `[12,64]`。

校准得到的两个尺度、三个质量组权重：

```text
scale 1: 0.128522, 0.178584, 0.170670
scale 2: 0.097055, 0.169478, 0.268918
```

蒸馏平均有效权重约 0.159152，实际执行了非零蒸馏，未仅走全零退回分支。
这些数字来自简单周期性合成信号，没有临床解释。
原始运行产物保存在 `runs/cpu_smoke/`，由 .gitignore 排除。

## 正式网络尺寸

- 参数：5,272,850（包括两个预训练重建头）。
- 测试输入 `[2,1,800]`，输出 `[2,512]`。
- 6 层、256 维、8 头、patch 25。
- CPU 上完成一次遮挡重建前向与反向，所有已产生梯度均有限。
- 未在用户 GPU 上进行训练或测显存，不能据此报告 GPU 训练速度或峰值显存。

证据：`runs/base_cpu_check.json`。

## 可迁移性检查

```text
python -m build --wheel --no-isolation
python scripts/check_portability.py --wheel dist/tempobridge-0.1.0-py3-none-any.whl --encoder runs/cpu_smoke/encoder.pt
```

将 wheel 和 encoder 放到仓库外的新临时目录，显式确认导入来自该目录，
用相对路径加载 encoder 并得到有限 CPU 特征 `[2,64]`。
核心源码、配置、测试和工具中没有当前机器的固定盘符或用户名路径。
这证明了本机目录迁移；尚未在 Linux 机器实测运行。

## 尚未验证

真实 ECG–PPG 数据预训练、独立下游任务提升、跨设备/站点泛化、罕见状态公平性、
多随机种子、强外部 ECG 教师、CUDA AMP、DDP 或千万级数据吞吐。
这些属于后续实验范围，不应从 smoke 测试推断出来。

## 2026-11-06 数据接入补充

完成原始 WFDB、本地 VitalDB `.vital`、已有 HDF5 三种接入方式；测试数据全部在本地生成，
没有运行用户真实临床数据转换或模型预训练。

| 检查 | 实际结果 |
|---|---|
| 完整 pytest suite | 63 passed，9.90 秒 |
| Ruff / git diff --check | 通过 |
| pip check | No broken requirements found |
| 原始 WFDB 格式转换 | 95 秒记录保存三个 30 秒窗口；尾部丢弃；磁盘故障后恢复无重复窗口 |
| 实际 `.vital` 格式转换 | 100 Hz PPG / 200 Hz ECG 共用时间轴；保留 ECG 开始处 1 秒缺口，保存两个 30 秒窗口 |
| HDF5 完整闭环 | CPU 预热、ECG 教师、分组校准、蒸馏、导出及 PPG 特征提取成功 |
| HDF5 epoch 恢复 | 含 dropout 的逐参数结果与不中断运行完全一致 |
| Windows 多进程读取 | 两个 spawn worker 成功读取分片；句柄不被 pickle 传递 |
| 划分/完整性 | 未知 ID 仅 train 且不进入校准；case 回退、组间隔离、同尺寸文件篡改、相对路径迁移通过 |
| 来源采样 | 80:20 精确 epoch 配额、无放回、随机种子复现及超容量拒绝通过 |

另外直接载入用户提供的 PulseBound `v6/group_split.py`，独立交叉核验：

- 14 类标识值归一化和 group hash 一致。
- 0 到 1,000 个组的 90/5/5 整数分配全部一致。
- 10 种组数的实际组分配一致。
- 原代码生成的两来源、四分片清单成功导入；包括跨 chunk 重复组、case 回退和未知 ID 行。

本地对照记录位于 `runs/data_compatibility/reference_report.json`；用于生成原清单的临时合成分片已自动释放。
新数据依赖实际版本：h5py 3.16.0、WFDB 4.3.1、VitalDB 1.7.2。
测试中的警告来自既有 TorchScript 接口以及 WFDB 对新版 pandas 的废弃接口调用，未造成失败。
这不证明历史 VitalDB 多模态转换逐样本相同，也不验证真实大数据的吞吐量或临床效果。
