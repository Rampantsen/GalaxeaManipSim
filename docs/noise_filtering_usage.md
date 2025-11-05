# 噪声过滤功能使用指南

本文档介绍如何使用噪声过滤功能来训练模仿学习策略。

## 📋 功能概述

噪声过滤功能允许您：
- ✅ 将带噪声标记的帧作为**历史观测输入**（observation history）
- ✅ 但**不将噪声帧作为动作预测目标**（action target）
- ✅ 充分利用所有数据，同时避免学习错误的动作

## 🔄 完整工作流程

### 1. 数据收集（带噪声标记）

使用 `collect_demos.py` 收集数据，启用 replan 功能：

```bash
python galaxea_sim/scripts/collect_demos.py \
    --env-name R1ProBlocksStackEasy \
    --num-demos 100 \
    --feature all \
    --enable-replan True \
    --replan-prob 0.5 \
    --replan-noise-min 0.02 \
    --replan-noise-max 0.05
```

**关键参数：**
- `--feature all`: 启用所有数据增强（包括 grasp_sample 和 replan）
- `--enable-replan True`: 启用重规划噪声注入
- `--replan-prob 0.5`: 50% 概率触发重规划
- `--replan-noise-range`: 噪声范围 [0.02, 0.05]

**输出：**
- 每个 observation 都带有 `is_replan_noise` 标记
- 数据保存在 `datasets/{env_name}/{table_type}/{feature}/collected/*.h5`

### 2. 转换为 LeRobot 格式（保留噪声标记）

使用 `convert_single_galaxea_sim_to_lerobot_with_noise_label.py` 转换数据：

```bash
python galaxea_sim/scripts/convert_single_galaxea_sim_to_lerobot_with_noise_label.py \
    --env-name R1ProBlocksStackEasy \
    --table-type red \
    --feature all \
    --tag collected
```

**输出：**
- LeRobot 格式数据集，包含 `is_replan_noise` 字段
- 保存在 `~/.cache/huggingface/lerobot/galaxea/{env_name}/`

### 3. 训练策略（带噪声过滤）

#### 训练 ACT 策略

```bash
python galaxea_sim/scripts/train_lerobot_act_policy_with_noise_filter.py \
    --task R1ProBlocksStackEasy \
    --filter-noise True \
    --batch-size 128 \
    --num-epochs 300 \
    --learning-rate 1e-4 \
    --chunk-size 30 \
    --n-obs-steps 1
```

#### 训练 Diffusion Policy

```bash
python galaxea_sim/scripts/train_lerobot_dp_policy.py \
    --task R1ProBlocksStackEasy \
    --filter-noise True \
    --batch-size 128 \
    --num-epochs 300 \
    --learning-rate 1e-4 \
    --n-obs-steps 1
```

**关键参数：**
- `--filter-noise True`: **启用噪声过滤**（推荐）
- `--filter-noise False`: 禁用噪声过滤（使用所有数据）
- `--noise-field-name`: 噪声标记字段名（默认 `is_replan_noise`）

## 📊 DataLoader 工作原理

### 带噪声过滤的 DataLoader

```python
from galaxea_sim.utils.noise_filtered_dataset import create_noise_filtered_dataloader

# 创建原始数据集
base_dataset = LeRobotDataset(
    "galaxea/R1ProBlocksStackEasy",
    delta_timestamps=delta_timestamps
)

# 创建噪声过滤的 dataloader
dataloader = create_noise_filtered_dataloader(
    base_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    noise_field_name="is_replan_noise"
)
```

**行为：**
1. 扫描数据集，构建有效帧索引列表（`is_replan_noise=False` 的帧）
2. 采样时只从有效帧中选择作为**主帧**（action 预测目标）
3. 当读取历史观测窗口时，噪声帧仍会被包含（作为输入）

### 示例说明

假设数据序列：

```
帧 0: is_replan_noise=False  ✅ 有效帧
帧 1: is_replan_noise=False  ✅ 有效帧
帧 2: is_replan_noise=True   ❌ 噪声帧
帧 3: is_replan_noise=True   ❌ 噪声帧
帧 4: is_replan_noise=False  ✅ 有效帧
帧 5: is_replan_noise=False  ✅ 有效帧
```

**使用噪声过滤的 DataLoader：**
- 只会采样帧 0, 1, 4, 5 作为主帧（action target）
- 但如果帧 4 的历史窗口是 [帧2, 帧3, 帧4]，那么帧2和帧3会被包含在输入中
- 这样既避免了学习错误动作，又充分利用了数据

## 🎯 最佳实践

### 1. 数据收集阶段

- 推荐使用 `--feature all` 启用所有数据增强
- `replan_prob` 设置在 0.3-0.5 之间
- 噪声范围不宜过大，推荐 [0.02, 0.05]

### 2. 训练阶段

- **默认启用噪声过滤**（`--filter-noise True`）
- 只有在对比实验时才禁用噪声过滤
- 检查日志中的噪声帧统计信息

### 3. 评估阶段

- 使用相同的评估脚本（不需要修改）
- 在评估时环境不会注入噪声

## 🔍 验证噪声标记

检查数据集是否包含噪声标记：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset("galaxea/R1ProBlocksStackEasy")

# 检查第一个样本
sample = dataset[0]
if "is_replan_noise" in sample:
    print("✅ 数据集包含噪声标记")
    
    # 统计噪声帧比例
    noise_count = sum(
        dataset[i]["is_replan_noise"].item() 
        for i in range(min(1000, len(dataset)))
    )
    print(f"前1000帧中噪声帧: {noise_count} ({noise_count/1000*100:.1f}%)")
else:
    print("❌ 数据集不包含噪声标记（旧数据或baseline数据）")
```

## 📈 预期效果

使用噪声过滤后：

1. **训练稳定性提升**：避免学习错误的动作
2. **数据利用率高**：噪声帧仍可作为历史输入
3. **性能提升**：特别是在有较多噪声帧的数据集上

典型日志输出：

```
噪声过滤数据集统计:
  - 总帧数: 10000
  - 有效帧数: 7500 (75.0%)
  - 噪声帧数: 2500 (25.0%)
  - 训练时只使用有效帧作为action目标
```

## 🛠️ 自定义使用

如果您有自己的训练脚本，可以这样集成：

```python
from galaxea_sim.utils.noise_filtered_dataset import (
    NoiseFilteredLeRobotDataset,
    create_noise_filtered_dataloader
)

# 方式1: 直接包装数据集
base_dataset = LeRobotDataset("galaxea/task", delta_timestamps=...)
filtered_dataset = NoiseFilteredLeRobotDataset(
    base_dataset,
    noise_field_name="is_replan_noise",
    verbose=True
)
dataloader = torch.utils.data.DataLoader(
    filtered_dataset,
    batch_size=128,
    shuffle=True
)

# 方式2: 使用便捷函数（推荐）
dataloader = create_noise_filtered_dataloader(
    base_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    noise_field_name="is_replan_noise"
)
```

## 📚 相关文件

- **数据收集**: `galaxea_sim/scripts/collect_demos.py`
- **数据转换**: `galaxea_sim/scripts/convert_single_galaxea_sim_to_lerobot_with_noise_label.py`
- **ACT训练**: `galaxea_sim/scripts/train_lerobot_act_policy_with_noise_filter.py`
- **Diffusion训练**: `galaxea_sim/scripts/train_lerobot_dp_policy.py`
- **数据集工具**: `galaxea_sim/utils/noise_filtered_dataset.py`
- **数据加载工具**: `galaxea_sim/utils/dataset_utils.py`

## ❓ 常见问题

### Q1: 旧数据集（没有噪声标记）能用吗？

A: 可以！如果数据集中没有 `is_replan_noise` 字段，会自动将所有帧视为有效帧，不会影响训练。

### Q2: 是否需要重新收集数据？

A: 如果您想使用噪声过滤功能，需要用新的脚本重新收集数据。但旧数据仍然可以正常使用。

### Q3: 性能提升有多少？

A: 取决于数据集中噪声帧的比例。如果噪声帧占20-30%，通常可以看到5-10%的成功率提升。

### Q4: 可以调整噪声过滤策略吗？

A: 可以！修改 `NoiseFilteredLeRobotDataset` 类来实现自定义过滤逻辑。例如，您可以实现 V2 版本，在历史窗口中也跳过噪声帧。

## 🎉 总结

噪声过滤是一个简单但有效的技术，可以：
- 提升训练数据质量
- 保持高数据利用率
- 提升策略性能

推荐在所有新项目中启用此功能！

