# 噪声过滤训练指南

## 概述

本项目支持在数据收集时添加 replan 噪声来增加数据多样性。但在训练时，我们希望：
- ✅ **噪声帧可以作为历史observation输入**（提供上下文）
- ❌ **噪声帧不应作为action预测目标**（避免学习错误动作）

## 工作流程

### 1. 数据收集（已完成）

使用 `collect_demos.py` 收集数据时，会自动记录 `is_replan_noise` 标记：

```bash
python galaxea_sim/scripts/collect_demos.py \
    --env-name R1ProBlocksStackEasy-v0 \
    --num-demos 100 \
    --feature all \
    --enable-replan \
    --replan-prob 0.5
```

数据中会包含：
- `observation`: 图像和状态
- `action`: 动作指令
- `is_replan_noise`: 布尔值，标记该帧是否为噪声帧

### 2. 数据转换

使用新的转换脚本，将噪声标记也保存到 lerobot 格式：

```bash
# 方法1: 保留噪声标记的转换脚本（推荐）
python galaxea_sim/scripts/convert_single_galaxea_sim_to_lerobot_with_noise_label.py \
    --env-name R1ProBlocksStackEasy-v0 \
    --table-type white \
    --feature all \
    --robot r1_pro

# 方法2: 原始转换脚本（不保留标记，需要重新转换）
python galaxea_sim/scripts/convert_single_galaxea_sim_to_lerobot.py \
    --env-name R1ProBlocksStackEasy-v0 \
    --table-type white \
    --feature all \
    --robot r1_pro
```

**重要**: 使用方法1的脚本会将 `is_replan_noise` 字段也保存到数据集中。

### 3. 训练（核心改动）

#### 方式A: 使用新的训练脚本（推荐）

```bash
# 使用噪声过滤训练ACT
python galaxea_sim/scripts/train_lerobot_act_policy_with_noise_filter.py \
    --task R1ProBlocksStackEasy-v0 \
    --batch-size 128 \
    --num-epochs 300 \
    --filter-noise  # 启用噪声过滤
```

#### 方式B: 修改现有训练脚本

在任何使用 `LeRobotDataset` 的训练脚本中，只需要改动 dataloader 部分：

**修改前:**
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(f"galaxea/{dataset_name}", delta_timestamps=delta_timestamps)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)
```

**修改后:**
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from galaxea_sim.utils.noise_filtered_dataset import create_noise_filtered_dataloader

# 创建原始数据集
base_dataset = LeRobotDataset(f"galaxea/{dataset_name}", delta_timestamps=delta_timestamps)

# 使用噪声过滤的dataloader
dataloader = create_noise_filtered_dataloader(
    base_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)
```

## 技术细节

### NoiseFilteredLeRobotDataset 工作原理

```python
class NoiseFilteredLeRobotDataset(Dataset):
    """
    1. 在初始化时遍历整个数据集，构建有效帧索引列表
    2. __len__() 返回有效帧数量（而非总帧数）
    3. __getitem__(idx) 将索引映射到有效帧
    4. 历史观测窗口由base_dataset处理，可能包含噪声帧（这是期望的）
    """
```

### 数据流示例

假设一个轨迹有10帧，其中第3、7帧是噪声：

```
原始索引:  0  1  2  3  4  5  6  7  8  9
噪声标记:  ✓  ✓  ✓  ❌  ✓  ✓  ✓  ❌  ✓  ✓
         (✓=正常, ❌=噪声)

有效索引: [0, 1, 2, 4, 5, 6, 8, 9]  # 8个有效帧
```

训练时：
- 采样 idx=4 → 实际获取原始索引6的数据
- 如果 n_obs_steps=2，会获取帧5和帧6的observation
- 帧6的action用作预测目标（不是噪声）
- 帧5和帧7都可能在历史窗口中（即使帧7是噪声）

### 统计信息输出

```
噪声过滤数据集统计:
  - 总帧数: 15000
  - 有效帧数: 14250 (95.0%)
  - 噪声帧数: 750 (5.0%)
  - 训练时只使用有效帧作为action目标
```

## API 参考

### create_noise_filtered_dataloader

```python
def create_noise_filtered_dataloader(
    base_dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    noise_field_name: str = "is_replan_noise",
    **kwargs
) -> DataLoader
```

**参数:**
- `base_dataset`: 原始的 LeRobotDataset
- `batch_size`: batch 大小
- `shuffle`: 是否打乱数据
- `num_workers`: 数据加载的工作进程数
- `pin_memory`: 是否使用 pinned memory
- `noise_field_name`: 噪声标记的字段名（默认 "is_replan_noise"）
- `**kwargs`: 其他传递给 DataLoader 的参数

### NoiseFilteredLeRobotDataset

```python
class NoiseFilteredLeRobotDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        noise_field_name: str = "is_replan_noise",
        verbose: bool = True,
    )
```

## 适用算法

这个方案适用于所有基于 lerobot 的算法：

- ✅ **ACT (Action Chunking with Transformers)**
- ✅ **Diffusion Policy**
- ✅ **其他使用 LeRobotDataset 的算法**

## 常见问题

### Q1: 我需要重新收集数据吗？

**不需要！** 如果你已经用 `enable_replan=True` 收集了数据，数据中已经包含了 `is_replan_noise` 标记。只需要：
1. （可选）重新转换数据，使用 `convert_single_galaxea_sim_to_lerobot_with_noise_label.py`
2. 使用新的训练脚本或修改现有训练脚本的 dataloader

### Q2: 如果我的数据没有噪声标记怎么办？

如果数据集中没有 `is_replan_noise` 字段，`NoiseFilteredLeRobotDataset` 会将所有帧视为有效帧，相当于不做过滤。

### Q3: 会影响训练速度吗？

- 初始化时需要遍历一次数据集来构建索引（一次性开销）
- 训练时几乎没有额外开销
- 实际上由于过滤掉了噪声帧，每个 epoch 的步数会略微减少

### Q4: 历史窗口中的噪声帧会影响训练吗？

不会有负面影响。历史窗口中的噪声帧可以：
- 提供更丰富的上下文信息
- 帮助模型学习从偏差状态恢复
- 增强模型的鲁棒性

关键是：**噪声帧的 action 不会用作训练目标**。

### Q5: 我应该用 V1 还是 V2？

- **V1 (NoiseFilteredLeRobotDataset)**: 推荐。噪声帧可以出现在历史窗口中。
- **V2 (NoiseFilteredLeRobotDatasetV2)**: 更激进的过滤，历史窗口中也跳过噪声帧。

通常情况下使用 V1 即可。

## 完整示例

```python
from pathlib import Path
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from galaxea_sim.utils.noise_filtered_dataset import create_noise_filtered_dataloader

# 1. 加载数据集元数据
dataset_metadata = LeRobotDatasetMetadata("galaxea/R1ProBlocksStackEasy-v0")

# 2. 配置 delta_timestamps
delta_timestamps = {
    "observation.images.rgb_head": [0.0],
    "observation.state": [0.0],
    "action": [0.0],
}

# 3. 创建原始数据集
base_dataset = LeRobotDataset(
    "galaxea/R1ProBlocksStackEasy-v0",
    delta_timestamps=delta_timestamps
)

# 4. 创建噪声过滤的 dataloader
dataloader = create_noise_filtered_dataloader(
    base_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
)

# 5. 正常训练
for batch in dataloader:
    # batch 中不包含噪声帧作为 action target
    # 但噪声帧可能在 observation history 中
    observations = batch["observation.state"]
    actions = batch["action"]
    # ... 训练代码 ...
```

## 总结

通过使用 `NoiseFilteredLeRobotDataset`，你可以：

✅ 保留原始数据不变  
✅ 充分利用噪声帧作为上下文  
✅ 避免学习噪声动作  
✅ 提高模型训练质量  
✅ 增强模型鲁棒性  

只需要在训练脚本中添加几行代码即可实现！

