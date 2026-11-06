# 数据格式

原始 MIMIC WFDB、VitalDB `.vital` 及已处理 PulseBound HDF5 的转换、清洗、90/5/5 划分与命令见
[PulseBound 数据接入](PULSEBOUND_DATA.md)。下文保留原 CSV + NPZ 接口，两种 manifest 均能进入同一训练流程。

CSV 必需列：

```csv
sample_id,subject_id,recording_id,split,path,ppg_quality,ecg_quality
s001_w001,s001,r001,train,records/s001_w001.npz,0.8,0.95
s002_w001,s002,r002,val,records/s002_w001.npz,0.9,1.0
s003_w001,s003,r003,test,records/s003_w001.npz,0.7,0.9
```

- sample_id 全局唯一；同一 subject_id 只允许一个 split。
- recording_id 对应单次记录且只能属于一个 subject。
- split 为 train/val/test；训练需要 train。推理 manifest 可以只有 test。
- path 相对 CSV 所在目录，也接受绝对路径。
- quality 在 [0,1]，越大越可靠。两列缺省时为 1，意味着**没有质量筛选信息**，不能宣称模型判断了质量。
- 质量标签应由无测试标签的、固定且可复现的 SQI/质控流程产生。默认代码不伪造已校准概率。
- quality 是记录/窗口级标签。如果局部质量变化显著，应先分成更细的同步记录条目；本版未实现逐时刻 SQI。

NPZ 内容：

```python
import numpy as np
np.savez("record.npz",
         ppg=ppg.astype(np.float32), ppg_fs=np.array(100),
         ecg=ecg.astype(np.float32), ecg_fs=np.array(250))
```

波形为单导联 [T] 或单例通道 [1,T]/[T,1]。采样率必须明确、为正整数。
PPG 必需；缺失 ECG 时完全省略 ecg/ecg_fs，不能填 NaN 或伪造零信号并标为有效。
已有 ECG 时必须有限、同起点、同时长（允许一个采样点量级的舍入差）。
数据读取采用 allow_pickle=False，不接受对象数组。

可从不同采样率用 resample_poly 抗混叠重采样到配置采样率。
模型输入长度必须为 patch_size 的整数倍。主配置采用 `window_mode="full"`、
`scales_seconds=[30.0]`，PPG/ECG 均为 50 Hz：重采样后每条同步波形必须恰好 1500 点，
完整送入模型，过长报错而不自动裁剪，过短报错而不补齐。
可选 `window_mode="crop"` 才执行裁剪：记录至少覆盖最大尺度，训练随机裁剪、验证居中裁剪，
同尺度 ECG/PPG 使用同一起点。2 秒/8 秒消融见 `configs/ablation_2s8s.json`。
ECG 很短但 PPG 很长时，先在预处理阶段明确切成同步部分和 PPG-only 部分。
本仓库不会自动估计设备时钟偏移或临床意义的 PAT。

建议先按患者/站点建立划分，再从记录切窗。相邻重叠窗口必须保持在同一患者划分。
程序可以发现 ID、路径冲突，无法凭波形保证两个不同 ID 实际不是同一个人；这需要数据来源审计。

### 规模与性能

样本按需读取，不一次加载全部波形。相同记录每个 epoch 会重新读取和重采样；真实大规模训练建议预先转为目标采样率。
不要将几小时记录作为每个 item；主实验应离线生成恰好 30 秒的同步窗口，保留原 recording_id。
校准阶段每个已知训练组最多抽取 `calibration.max_windows_per_subject` 个窗口（默认 16），
再筛选可用 ECG，并在内存中收集表征而非波形。内存随组数 × 每组上限增长；
MLP 探针使用受 `probe_batch_size` 限制的训练批。尚未实现完全磁盘流式校准。
