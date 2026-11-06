# MIMIC、PulseBound HDF5 与 VitalDB 接入

三种入口共用同一套 HDF5 索引、分组划分、训练/校准/推理读取器。原来的 CSV + NPZ 仍可用。
转换和索引都是 CPU 操作；以下命令不会启动模型训练，也不会自动下载临床数据。
命令从仓库根目录执行，`python` 指安装本项目的环境；Windows 本地可替换为 `.\.venv-cpu\Scripts\python.exe`。
所有示例路径均可替换，没有固定盘符。读取用户本地已获得的数据，不需要账号写入代码。

## 1. 安装数据依赖

```powershell
python -m pip install -e ".[dev,data]"
```

`data` 包含 h5py、WFDB 和 VitalDB。先按 README 安装 CPU PyTorch 即可进行代码验证。

## 2. 已有 PulseBound HDF5：直接建立索引

复制 `configs/data_sources.example.json` 为自己的配置，例如：

```json
{
  "sources": {
    "mimic": {"root": "../data/processed/mimic", "pattern": "*.h5", "fs": 50},
    "vitaldb": {"root": "../data/processed/vitaldb", "pattern": "*.h5", "fs": 50}
  }
}
```

`root` 相对这个 sources JSON 所在目录，也接受绝对路径。`recursive: true` 允许递归寻找分片。
只使用一个来源时删除另一个来源；原始 MIMIC 与已处理 MIMIC 不应重复加入同一批数据。

```powershell
python -m tempobridge index-h5 --sources configs/data_sources.example.json --output data/pulsebound_manifest.json
python -m tempobridge audit-data --config configs/pulsebound_50hz.json --manifest data/pulsebound_manifest.json --max-windows 32
```

已有波形不重复滤波、不重复归一化，也不转成海量 NPZ 文件。
审计默认校验全部分片 SHA256，但每个 split 只抽查最多 32 个波形；输出同时列明总数和检查数。
`--max-windows 0` 才会逐条检查全部波形。全量 SHA256 会顺序读取所有分片字节，首次大数据审计可能较慢。

如果已持有 PulseBound 原始 `group_split_manifest.json`，使用：

```powershell
python -m tempobridge index-h5 --sources configs/data_sources.example.json --output data/pulsebound_manifest.json --pulsebound-split path/to/group_split_manifest.json
```

输出路径必须尚不存在。此模式逐项核对来源名称、文件集合、内容 SHA256、行分组及组的 split；
不同文件/子集不会被冒充为原实验划分。文件 mtime 变化不影响内容相同的数据迁移。
没有原清单时使用相同确定性协议重新划分；**不意味着已复现论文未提供的私有数据清单**。

## 3. 原始 MIMIC WFDB：离线转换

```powershell
python -m tempobridge preprocess --kind mimic --input-root data/raw/mimic --output data/processed/mimic
```

支持本地 `.hea/.dat` 记录；优先读取根目录 `RECORDS` 指定的记录或患者目录，无该文件时递归寻找 header。
跳过 `_layout`、数字通道 `*n` 和没有单段通道表的父 MultiRecord，以避免重复读入叶段。
自动识别 PLETH/PPG 和常用单导联 ECG 名称；存在多个候选时按通道顺序选第一个。
如需固定导联，显式传入 `--ppg-channel PLETH --ecg-channel II`。找不到指定通道时记录错误而不擅自替换。

默认处理：

1. 同步取非重叠 30 秒 ECG–PPG 窗口，丢弃尾部不足 30 秒部分，每条记录最多 200 个窗口。
2. **插值前**检查 NaN/Inf 比例；PPG 缺失超过 20% 或为常量时丢弃窗口。
3. 允许的缺失点线性插值、边缘最近值补齐；PPG 用二阶 Butterworth 0.5–8 Hz 零相位带通。
4. Fourier 重采样至 50 Hz，再分别 min-max 归一化到 `[-1,1]`；ECG 不额外带通。
5. 写入每秒 SQI 和幅度/偏度特征；每片最多 50,000 窗口，增量写盘。
6. ECG 无效时保留 PPG，写入零占位 ECG 并设置 `has_ecg=false`，训练端不会把占位波形当作监督。

窗口、目标采样率、缺失阈值、分片大小和记录上限均为参数，可用 `preprocess --help` 查看。
`--max-windows-per-record 0` 表示取消上限；改变它会改变语料，与原默认设置不能混称。
先用 `--max-records 2` 和独立输出目录检查真实数据的通道映射，再进行正式转换。

`conversion_report.json` 记录保存数、各类拒绝计数、记录错误、前 100 条问题及与参考代码的差异。
计数仅涵盖实际尝试的完整窗口，不包含尾段和被 200 窗口上限截断的部分。
`conversion_contract.json` 保存输入文件大小/mtime、ID 映射与参数，用于防止不同输入混用断点。

```powershell
python -m tempobridge preprocess --kind mimic --input-root data/raw/mimic --output data/processed/mimic --resume
```

中断后从最后一个完整分片的记录/窗口游标恢复；未发布的 `.h5.partial` 会重写。
不覆盖已发布分片，磁盘写入失败会立即报错。恢复必须保持原输入清单、参数和 ID 映射一致。
原始文件身份检查使用大小/mtime，不是逐字节内容证明；最终 HDF5 索引使用 SHA256。

## 4. 本地 VitalDB `.vital`

```powershell
python -m tempobridge preprocess --kind vitaldb --input-root data/raw/vitaldb --output data/processed/vitaldb
```

按共同时间轴读取 ECG 和 PPG，保留原始时间缺口，随后执行相同的同步切窗/清洗/重采样。
不会分别删除 NaN 后拼接两种信号；也不会自动把 ECG–PPG 的生理延迟对齐到零。
必要时使用 `--ppg-channel SNUADC/PLETH --ecg-channel SNUADC/ECG_II` 明确设备轨道。
当前接口接收本地 `.vital`，不自动下载网上数据、不假定独立轨道 CSV 已经对时。
VitalDB 每次载入一个病例的选中轨道到内存，超长病例转换仍需相应主机 RAM。

VitalDB 默认从纯数字文件名取得 `case_id`（例如 `00042.vital` → 42）。
MIMIC 默认从路径 `p000012` 一类片段取得 `subject_id`。可用 `--id-map mapping.csv` 覆盖：

```csv
record_path,subject_id,case_id
p00/p000012/record.hea,12,
00042.vital,987,42
```

`record_path` 相对输入根目录且包含扩展名；缺失值留空，ID 使用非负整数。
**case/record 分组不等同于患者分组**。同一患者可能有多个病例时，应提供统一的 subject_id 映射。
自动识别不到的 ID 保持未知，不会把每个窗口伪装成一名新患者。

## 5. 划分与训练读取约定

- 按 `subject_id → patient_id → case_id → record_id` 逐行选择首个有效标识。
- 使用 PulseBound 默认 salt `pulsebound-iclr2027-source-split-v1`、来源作用域的 SHA256 分组和排序协议。
- 每个来源独立按组数分配 90%/5%/5%，使用相同的 Hamilton 整数分配及平局规则。
- 同组在所有分片中固定属于一个 split；窗口比例不保证恰好 90/5/5。源数据更改后必须重新索引，分组分配也可能改变。
- 未知 ID 窗口仅进入 train，排除于交叉校准；无法据此证明未知患者与测试集绝无重叠。
- 数值 ID `12` 与字符串 ID `"12"` 按原协议区分；同一真实患者的 ID 类型/别名需在源数据统一。

读取必需 `signals[N,L]` 和明确采样率（HDF5 `target_fs`/`fs`，或来源 JSON 的 `fs`）。
ECG 使用可选 `ecg_wave[N,L]` 与 `has_ecg[N]`。缺字段、标记无效、非有限或常量 ECG 均不提供教师监督。
已有 HDF5 的非有限/常量 PPG 会报错，需修复源分片后重新索引，不会静默跳行改变实验样本集。
PPG 质量优先使用 `ppg_quality`，其次 `metrics` 中 `sqi`，再其次 `patch_sqi` 均值；缺省 1 表示无筛选信息。
ECG 质量优先使用 `ecg_quality`，缺省 1；新转换器使用有限样本覆盖率，不声称它是经过校准的生理可信度。

训练内存保存分组行区间和少量打开的分片句柄，不建立每窗口 Python 对象列表。
`data.max_open_shards` 限制每个进程的句柄缓存；Windows 多 worker 启动时不传递打开的 HDF5 句柄。
`data.integrity="metadata"` 可跳过全量内容哈希，但仅检查文件大小，**无法发现同尺寸内容修改**；复现实验默认保留 `sha256`。

`configs/pulsebound_50hz.json` 配置两个来源 80:20、无放回的每 epoch 精确配额。
epoch 容量由较稀缺来源决定，并截断至 `LCM(batch_size × accumulation, 比例分母)` 的整数倍；不会偷偷过采样。
混合比例是每 epoch 的比例，不保证每个小 batch 都是 80:20；块打乱保留一定磁盘局部性。
不设置 `source_weights`（`{}`）时每个 epoch 使用全部训练窗口，按原始规模决定来源占比。
单来源数据请将权重改为 `{}`；需要限制 epoch 样本数时同时显式设置来源权重和 `samples_per_epoch`。

新配置使用 50 Hz、patch=10（0.2 秒），模型仍从已存储的 30 秒窗口同步裁出 2 秒/8 秒用于原有方法。
索引建立后，原训练命令中的 `--manifest data/manifest.csv` 替换为 `--manifest data/pulsebound_manifest.json`，
配置替换为 `--config configs/pulsebound_50hz.json` 即可。该配置默认 CPU/fp32；本次不启动真实数据训练。

校准默认每个已知训练组最多抽取 16 窗口，再筛选可用 ECG，按组交叉拟合。
`calibration.max_windows_per_subject` 控制上限，`probe_batch_size` 限制 MLP 探针每步样本数；选择数与可靠配对数写入校准文件。
表征仍在 RAM 中收集，内存随已知组数 × 每组上限增长，尚不是完全磁盘流式校准。
原生教师 checkpoint 自动记录上述来源作用域的组哈希。外部教师元数据中的 `training_subjects`
也必须使用同一 ID 命名体系才能进行交集核验；原始数字 ID 与哈希 ID 不会自动相互匹配。

## 6. 路径迁移及与 PulseBound 的边界

生成的索引尽可能使用相对根路径。整体搬迁数据和索引不会改变语料摘要。
可在实验配置中设置 `data.source_roots={"mimic":"新的根目录","vitaldb":"新的根目录"}`；
相对覆盖路径以 manifest 所在目录为基准。改根路径不会修改源 HDF5，也不会改变分组哈希。
训练续跑仍要求实验配置一致；整体搬迁且保持相对目录布局时无需修改配置。

兼容的是已核对的分组协议、波形字段及名义清洗流程。保留了以下有意差异：

1. 原始代码在插值后检查 NaN 比例；这里在插值前统计 NaN/Inf，防止长时间缺失被插值掩盖。
2. 原始部分分片代码要求所有行都有 ECG 才写 `ecg_wave`；这里逐行写可用性，避免一行缺失导致整片 ECG 丢失。
3. 当前目标不使用 PulseBound 的未来事件、呼吸、血压等 scaffold 标签，因此不伪造这些字段；生成文件不是完整 scaffold 预训练包。
4. 所提供补充材料没有完整的历史多模态 VitalDB 原始转换器；本项目的 `.vital` 同步转换器是新增实现，不能宣称逐样本复刻历史处理。
5. 零相位滤波、Fourier 重采样和整窗 min-max 都属于离线窗口处理；短 crop 的遮挡边界不等同于原始连续信号的严格因果边界。

已有 HDF5 直接读取最适合保持已有波形；若要严格比较同一语料，应共同使用冻结的原始分片和经核验的同一划分清单。
