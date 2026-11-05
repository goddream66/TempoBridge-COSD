# ECG 教师接口

原生模式直接使用本仓库 ECG 自监督预训练产生的 checkpoint。程序检查：

- modality 必须为 ecg，global_step > 0；随机初始化不能冒充预训练教师。
- 教师 sampling rate 与当前配置一致。
- 已记录的教师训练患者不得与当前 val/test 患者重叠。
- requires_grad=False，eval 始终生效，forward 在 no_grad 中执行。

原生教师的目标是 context tokens 的均值，不需要标签或指定下游任务。
其强度需真实数据验证，训练几个合成 batch 不会产生有效的 ECG foundation model。

## 接入已有 ECG 编码器

把可信来源的 ECG 模型包装为 TorchScript 模块，接口：

    forward(ecg: Tensor[B,1,T]) -> Tensor[B,D]

该模块负责该教师所需的输入归一化，且必须正确支持配置中的每个真实裁剪长度。
不要把模型分类 logits 无说明地当成 backbone feature，也不要把冻结的随机 projection 当预训练知识。
若原模型必须固定长度，应在 wrapper 内采用明确有效区域/attention mask 的适配并验证 padding 不引入目标信息。

保存 `teacher.ts`，并在其旁边创建 **`teacher.ts.json`**：

```json
{
  "input_fs": 250,
  "target_dim": 768,
  "training_subjects": null,
  "provenance": {
    "repository": "真实来源 URL",
    "revision": "固定 commit 或版本",
    "checkpoint": "权重来源与许可",
    "pretraining_data": "已知数据来源；未知必须写明未知",
    "normalization": "wrapper 内所做的处理"
  }
}
```

如果知道训练患者 ID，training_subjects 填字符串列表；null 表示来源无法据此验证，
**不代表没有患者重叠**。教师文件及 sidecar 元数据共同参与 artifact 哈希。
TorchScript 是可执行模型，请只加载可信的本地文件。

校准和蒸馏命令均增加 `--teacher-kind torchscript`。PPG 与 ECG 可设置不同采样率，
但每个尺度在两种采样率下的采样点数都需满足当前 patch 配置验证。
外部教师应报告真实部署大小和训练成本；学生推理不保留其权重。
