# Amadeus Base Model Training Guide

> **目标**: 在 RTX 4060 Ti 8GB 上训练 FullDuplexDiT 基础模型  
> **数据集**: HDTF (High-definition Talking Face Dataset), 300 个预处理片段  
> **预计时间**: ~2.3 小时 (200 epoch)  
> **最后验证**: 2026-06-07, 3-epoch smoke test 通过 (loss 0.032 → 0.008)

---

## 目录

1. [环境准备](#1-环境准备)
2. [训练数据](#2-训练数据)
3. [启动训练](#3-启动训练)
4. [监控训练](#4-监控训练)
5. [训练后验证](#5-训练后验证)
6. [故障排查](#6-故障排查)
7. [下一步](#7-下一步)

---

## 1. 环境准备

### 1.1 激活 conda 环境

```bash
conda activate amadeus
```

确认环境:

```bash
python --version                # 应显示 Python 3.11.5
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 应显示: 2.6.0+cu124 True
```

### 1.2 设置 HuggingFace 镜像

```bash
# 中国大陆用户必须设置,否则模型下载会超时
set HF_ENDPOINT=https://hf-mirror.com
```

### 1.3 验证模型权重已缓存

```bash
# 应该已经下载过,如果缺失会自动下载
python -c "
from transformers import HubertModel, BertModel, BertTokenizer
HubertModel.from_pretrained('facebook/hubert-base-ls960')
BertModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
print('All models cached OK')
"
```

---

## 2. 训练数据

### 2.1 数据位置

预处理后的训练数据在:

```
data/preprocessed/hdtf_subset/
├── *.npz          # 300 个文件,每文件 81 帧 × 45 Live2D params
├── *_audio.wav    # 对应音频(HDTF 视频无音频轨道,为静默)
└── *_meta.json    # 元数据
```

每个 `.npz` 包含:

| 字段 | 形状 | 说明 |
|------|------|------|
| `live2d_params` | (81, 45) | 45 个 Live2D 参数值,范围 [0, 1] |
| `blendshapes` | (81, 52) | 原始 ARKit blendshapes |
| `head_angles` | (81, 3) | 头部角度 [pitch, yaw, roll] |
| `bad_frames` | (N,) | 人脸检测失败的帧索引 |
| `fps` | float | 提取帧率 (25) |
| `identity_id` | int | 角色 ID |

### 2.2 预处理更多数据(可选)

如果需要处理更多 HDTF 片段:

```bash
# 处理单个视频
python -m src.motion.preprocess.pipeline --input video.mp4 --output data/preprocessed/

# 批量处理目录(会自动跳过已处理的)
python -m src.motion.preprocess.pipeline data/raw/hdtf/clips/clips/ --output_dir data/preprocessed/hdtf_full/
```

> **注意**: HDTF clips.zip 共 16,914 个片段,全量处理约需 5 小时。建议先用 300-500 片段训练,确认模型收敛后再扩。

---

## 3. 启动训练

### 3.1 标准训练 (推荐)

```bash
conda activate amadeus
set HF_ENDPOINT=https://hf-mirror.com

python scripts\train_base.py ^
    --data_dir data\preprocessed\hdtf_subset ^
    --output_dir models\motion\base_model ^
    --num_epochs 200 ^
    --warmup_steps 200 ^
    --ema_decay 0.999 ^
    --early_stopping_patience 50
```

### 3.2 低层 CLI (更细粒度控制)

```bash
python -m src.motion.training.train ^
    --data_dir data\preprocessed\hdtf_subset ^
    --output_dir models\motion\base_model ^
    --num_params 45 ^
    --hidden_dim 320 ^
    --num_layers 4 ^
    --epochs 200 ^
    --batch_size 1 ^
    --grad_accum 4 ^
    --lr 1e-4 ^
    --weight_decay 0.01 ^
    --warmup_steps 200 ^
    --ema_decay 0.999 ^
    --early_stopping_patience 50 ^
    --val_split 0.1
```

### 3.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_epochs` | 200 | 总训练轮数 |
| `--batch_size` | 1 | 单 GPU 8GB 显存建议保持 1 |
| `--grad_accum` | 4 | 梯度累积步数,等效 batch size = 4 |
| `--lr` | 1e-4 | 学习率 |
| `--weight_decay` | 0.01 | L2 正则化 |
| `--warmup_steps` | 200 | 线性 warmup 步数 |
| `--ema_decay` | 0.999 | EMA 衰减率(0=禁用) |
| `--early_stopping_patience` | 50 | 验证 loss 连续 N 轮不下降则停止(0=禁用) |
| `--val_split` | 0.1 | 验证集比例 |

### 3.4 恢复训练

```bash
python -m src.motion.training.train ^
    --data_dir data\preprocessed\hdtf_subset ^
    --output_dir models\motion\base_model ^
    --resume models\motion\base_model\full_duplex_dit_epoch_0050.pt ^
    --epochs 300
```

> 完整快照包含: model + optimizer + scheduler + AMP scaler + EMA + epoch。恢复后 optimizer 动量和 LR 调度器状态也完全恢复。

### 3.5 LoRA 微调

```bash
python -m src.motion.training.train ^
    --data_dir data\preprocessed\character_data ^
    --output_dir models\lora\my_character ^
    --use_lora ^
    --lora_rank 8 ^
    --lora_alpha 16 ^
    --epochs 50
```

---

## 4. 监控训练

### 4.1 实时输出

训练过程中每 epoch 打印:

```
Epoch 1/200 | Loss: 0.032003 | LR: 7.84e-05 | Val: 0.317241
Epoch 2/200 | Loss: 0.009432 | LR: 2.68e-05 | Val: 0.149171
Epoch 3/200 | Loss: 0.008121 | LR: 0.00e+00 | Val: 0.094625
```

- **Loss**: 训练集上的 x-prediction MSE
- **Val**: 验证集上的 x-prediction MSE (EMA 权重,如果启用)
- **LR**: 当前学习率(warmup → cosine 衰减)

### 4.2 查看日志

```bash
# 训练日志(每 epoch 一行)
type models\motion\base_model\train.log

# 标准输出(含所有 INFO)
type models\motion\base_model\training_output.log
```

### 4.3 生成 Loss 曲线

训练结束后 `scripts\train_base.py` 会自动生成 `loss_curve.png`。也可手动:

```bash
python -m src.motion.training.visualize loss ^
    --log models\motion\base_model\train.log ^
    --out models\motion\base_model\loss_curve.png
```

### 4.4 生成样本对比图

需要从验证集中选一个 `.npz` 文件,并用模型生成预测:

```bash
# 1. 找一个验证集样本
dir data\preprocessed\hdtf_subset\*.npz

# 2. 用模型生成预测(需要写一个小推理脚本,或等 visualize 工具扩展)
# 目前 visualize 支持: 比较 ground truth vs predicted motion
python -m src.motion.training.visualize motion ^
    --gt_npz data\preprocessed\hdtf_subset\RD_Radio10_000_0_80.npz ^
    --pred_npy models\motion\base_model\sample_pred.npy ^
    --out models\motion\base_model\comparison.png
```

### 4.5 生成动画 GIF

```bash
python -m src.motion.training.visualize gif ^
    --input data\preprocessed\hdtf_subset\RD_Radio10_000_0_80.npz ^
    --out models\motion\base_model\sample.gif ^
    --fps 30
```

---

## 5. 训练后验证

### 5.1 检查点文件

训练完成后检查:

```bash
dir models\motion\base_model\
```

应包含:

| 文件 | 说明 |
|------|------|
| `full_duplex_dit.pt` | 最终模型(完整快照,含 EMA 权重) |
| `full_duplex_dit_epoch_0010.pt` | 每 10 epoch 的中间检查点 |
| `train.log` | 训练日志 |
| `loss_curve.png` | Loss 曲线图 |
| `lora_config.pt` | (仅 LoRA 模式) |

### 5.2 验证 Loss 曲线

`loss_curve.png` 应显示:
- 训练 loss 从 ~0.03 下降到 ~0.001 以下
- 验证 loss 同步下降,与训练 loss 差距不大(无严重过拟合)
- 如果启用早停,会在验证 loss 不再下降时停止

### 5.3 推理测试

```bash
# 用训练好的模型推理(需启动 Amadeus 或写推理脚本)
python -m src.main
```

---

## 6. 故障排查

### 6.1 CUDA 内存不足

```
RuntimeError: CUDA out of memory.
```

**解决**:
- 确保 `batch_size=1`
- 确保 `grad_accum=4`(显存不随 accum 增加)
- 关闭 AMP: `--no_amp`(AMP 在 8GB 卡上可能不稳定)
- 减少 `hidden_dim` 到 256

### 6.2 模型下载超时

```
Connection to huggingface.co timed out.
```

**解决**:
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

### 6.3 Loss 不下降

如果 loss 卡在 ~0.08 以上不下降:

- 检查 `--lr` 是否太低(建议 1e-4)
- 检查 `--warmup_steps` 是否太长(建议 200)
- 检查 `--ema_decay` 是否开启(0.999 需要 100+ epoch 才见效)
- 检查数据预处理是否成功(运行 `python -m src.motion.preprocess.pipeline --input video.mp4 --output data/test/`)

### 6.4 验证 loss 远高于训练 loss

```
Epoch 50/200 | Loss: 0.002 | Val: 0.500
```

- 验证集太小(默认 10% 的 300 = 30 个样本,统计噪声大)
- 增加 `--val_split` 到 0.2 或使用更多数据

---

## 7. 下一步

训练完成后:

| 任务 | 说明 |
|------|------|
| **角色 LoRA 微调** | 用角色视频数据微调 LoRA,产出 ~MB 的适配器 |
| **推理测试** | 启动 Amadeus,用 `set_character_id()` 加载模型 |
| **更多训练数据** | 处理更多 HDTF 片段(16,914 个全量) |
| **MEAD 数据集** | 申请 MEAD 后,用 `--dataset_type mead` 训练 |
| **音频驱动训练** | 使用含真实音频的视频(当前 HDTF 音频为静默) |

---

*本文档最后更新: 2026-06-07*  
*对应分支: `fix/training-pipeline-issues`*  
*问题历史: `docs/TRAINING_PIPELINE_REVIEW.md`*  
*架构图: `docs/ARCHITECTURE_DIAGRAMS.md`*
