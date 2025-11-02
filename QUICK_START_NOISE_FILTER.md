# 快速开始：噪声过滤训练

## 核心思想

您的数据中已经包含 `is_replan_noise` 标记。训练时：
- ✅ 噪声帧可以作为**输入**（observation history）
- ❌ 噪声帧不作为**输出目标**（action target）

这样可以避免模型学习到噪声动作，同时充分利用所有数据。

## 使用方法（3步）

### 步骤1: 转换数据（带噪声标记）

```bash
python galaxea_sim/scripts/convert_single_galaxea_sim_to_lerobot_with_noise_label.py \
    --env-name R1ProBlocksStackEasy-v0 \
    --table-type white \
    --feature all \
    --robot r1_pro
```

### 步骤2: 训练（自动过滤噪声）

```bash
# 使用新的训练脚本
python galaxea_sim/scripts/train_lerobot_act_policy_with_noise_filter.py \
    --task R1ProBlocksStackEasy-v0 \
    --batch-size 128 \
    --num-epochs 300 \
    --filter-noise
```

### 步骤3: 验证

训练时会看到类似输出：

```
噪声过滤数据集统计:
  - 总帧数: 15000
  - 有效帧数: 14250 (95.0%)
  - 噪声帧数: 750 (5.0%)
  - 训练时只使用有效帧作为action目标
```

## 修改现有训练脚本

如果你想修改现有的训练脚本，只需改动 dataloader 部分：

**修改前：**
```python
dataset = LeRobotDataset(f"galaxea/{dataset_name}", delta_timestamps=delta_timestamps)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
```

**修改后：**
```python
from galaxea_sim.utils.noise_filtered_dataset import create_noise_filtered_dataloader

base_dataset = LeRobotDataset(f"galaxea/{dataset_name}", delta_timestamps=delta_timestamps)
dataloader = create_noise_filtered_dataloader(base_dataset, batch_size=128, shuffle=True)
```

就这么简单！

## 适用算法

- ✅ ACT
- ✅ Diffusion Policy  
- ✅ 其他任何使用 LeRobotDataset 的算法

## 详细文档

查看完整文档：`docs/NOISE_FILTERING_GUIDE.md`

