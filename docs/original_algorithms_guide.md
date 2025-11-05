# 使用原版 ACT 和 Diffusion Policy 算法指南

本文档说明如何使用原版论文实现而不是 LeRobot 版本。

## 1. 原版 ACT 算法

### 1.1 安装原版 ACT

```bash
cd ~/workspace/galaxea
git clone https://github.com/tonyzhaozh/act.git
cd act
pip install -e .
```

### 1.2 数据格式

原版 ACT 使用 HDF5 格式，与我们的 `collect_demos.py` 收集的格式兼容！

**数据结构示例：**
```python
episode_0.hdf5
├── qpos: (T, 14)              # 机器人关节位置
├── qvel: (T, 14)              # 机器人关节速度  
├── action: (T, 14)            # 动作指令
├── images
│   ├── top: (T, H, W, 3)      # 相机图像
│   ├── left_wrist: (T, H, W, 3)
│   └── right_wrist: (T, H, W, 3)
```

### 1.3 适配您的数据

创建数据转换脚本 `convert_to_original_act.py`:

```python
"""
将 GalaxeaManipSim 数据转换为原版 ACT 格式
"""
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_single_episode(src_path, dst_path):
    """转换单个 episode"""
    with h5py.File(src_path, 'r') as src_f:
        # 读取数据
        qpos = src_f['upper_body_observations']['qpos'][:]
        action = src_f['upper_body_action_dict']['joint_position_cmd'][:]
        
        # 图像数据
        img_head = src_f['upper_body_observations']['rgb_head'][:]
        img_left = src_f['upper_body_observations']['rgb_left_hand'][:]
        img_right = src_f['upper_body_observations']['rgb_right_hand'][:]
        
        # 创建新的 HDF5 文件
        with h5py.File(dst_path, 'w') as dst_f:
            dst_f.create_dataset('qpos', data=qpos)
            dst_f.create_dataset('action', data=action)
            
            # 图像数据
            dst_f.create_dataset('images/top', data=img_head)
            dst_f.create_dataset('images/left_wrist', data=img_left)
            dst_f.create_dataset('images/right_wrist', data=img_right)
            
            # 元数据
            dst_f.attrs['compress'] = True

def convert_dataset(src_dir, dst_dir):
    """转换整个数据集"""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    demo_files = sorted(src_dir.glob("demo_*.h5"))
    
    for i, demo_file in enumerate(tqdm(demo_files)):
        dst_file = dst_dir / f"episode_{i}.hdf5"
        convert_single_episode(demo_file, dst_file)
    
    print(f"✅ 转换完成: {len(demo_files)} episodes")

if __name__ == "__main__":
    convert_dataset(
        src_dir="datasets/R1ProBlocksStackEasy/red/all/collected",
        dst_dir="datasets_original_act/R1ProBlocksStackEasy"
    )
```

### 1.4 训练原版 ACT

```bash
cd ~/workspace/galaxea/act

python train.py \
    --dataset_dir ../GalaxeaManipSim/datasets_original_act/R1ProBlocksStackEasy \
    --ckpt_dir ./checkpoints/R1ProBlocksStackEasy \
    --policy_class ACT \
    --task_name R1ProBlocksStackEasy \
    --batch_size 8 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0 \
    --chunk_size 100 \
    --kl_weight 10 \
    --hidden_dim 512 \
    --dim_feedforward 3200
```

## 2. 原版 Diffusion Policy

### 2.1 安装原版 Diffusion Policy

```bash
cd ~/workspace/galaxea
git clone https://github.com/real-stanford/diffusion_policy.git
cd diffusion_policy
pip install -e .
```

### 2.2 数据格式

Diffusion Policy 使用 Zarr 格式：

```python
dataset.zarr
├── data
│   ├── img (N, T, H, W, 3)        # 图像序列
│   ├── state (N, T, state_dim)     # 状态序列
│   └── action (N, T, action_dim)   # 动作序列
└── meta
    └── episode_ends (num_episodes,)
```

### 2.3 适配您的数据

创建转换脚本 `convert_to_diffusion_policy.py`:

```python
"""
将 GalaxeaManipSim 数据转换为 Diffusion Policy 格式
"""
import zarr
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

def convert_to_zarr(src_dir, dst_path):
    """转换为 Zarr 格式"""
    src_dir = Path(src_dir)
    demo_files = sorted(src_dir.glob("demo_*.h5"))
    
    all_images = []
    all_states = []
    all_actions = []
    episode_ends = []
    
    total_frames = 0
    
    for demo_file in tqdm(demo_files):
        with h5py.File(demo_file, 'r') as f:
            # 读取数据
            state = f['upper_body_observations']['qpos'][:]
            action = f['upper_body_action_dict']['joint_position_cmd'][:]
            img = f['upper_body_observations']['rgb_head'][:]
            
            all_states.append(state)
            all_actions.append(action)
            all_images.append(img)
            
            total_frames += len(state)
            episode_ends.append(total_frames)
    
    # 创建 Zarr 数据集
    root = zarr.open(dst_path, mode='w')
    
    # 保存数据
    root.create_dataset('data/img', 
                       data=np.concatenate(all_images, axis=0),
                       chunks=(1, *all_images[0].shape[1:]),
                       dtype=np.uint8)
    
    root.create_dataset('data/state',
                       data=np.concatenate(all_states, axis=0),
                       dtype=np.float32)
    
    root.create_dataset('data/action',
                       data=np.concatenate(all_actions, axis=0),
                       dtype=np.float32)
    
    root.create_dataset('meta/episode_ends',
                       data=np.array(episode_ends),
                       dtype=np.int64)
    
    print(f"✅ 转换完成: {len(demo_files)} episodes, {total_frames} frames")

if __name__ == "__main__":
    convert_to_zarr(
        src_dir="datasets/R1ProBlocksStackEasy/red/all/collected",
        dst_path="datasets_diffusion_policy/R1ProBlocksStackEasy.zarr"
    )
```

### 2.4 训练原版 Diffusion Policy

```bash
cd ~/workspace/galaxea/diffusion_policy

python train.py \
    --config-name=train_diffusion_unet_image_workspace \
    task.dataset_path=../GalaxeaManipSim/datasets_diffusion_policy/R1ProBlocksStackEasy.zarr \
    task.name=R1ProBlocksStackEasy \
    training.batch_size=64 \
    training.num_epochs=500 \
    training.lr=1e-4
```

## 3. 性能对比

### LeRobot vs 原版实现

| 特性 | LeRobot | 原版 ACT | 原版 Diffusion Policy |
|------|---------|----------|----------------------|
| **易用性** | ✅ 统一接口 | ⚠️ 需要适配 | ⚠️ 需要适配 |
| **数据格式** | 统一格式 | HDF5 | Zarr |
| **维护** | 活跃 | 稳定 | 活跃 |
| **性能** | 优化过 | 论文基准 | 论文基准 |
| **自定义** | ⚠️ 较难 | ✅ 容易 | ✅ 容易 |

## 4. 推荐方案

### 场景 1：快速实验和部署
**推荐**: LeRobot 版本
- ✅ 开箱即用
- ✅ 统一的数据和训练流程
- ✅ 已经集成噪声过滤

### 场景 2：研究和论文对比
**推荐**: 原版实现
- ✅ 与论文结果直接可比
- ✅ 完全控制超参数
- ✅ 便于修改和扩展

### 场景 3：生产环境
**推荐**: LeRobot 版本
- ✅ 更好的工程化
- ✅ 持续维护
- ✅ 社区支持

## 5. 混合方案

您也可以同时使用两者：

```bash
# 使用 LeRobot 快速验证
python train_lerobot_act_policy_with_noise_filter.py --task R1ProBlocksStackEasy

# 使用原版进行详细对比
python convert_to_original_act.py
cd ~/workspace/galaxea/act && python train.py ...
```

## 6. 注意事项

### 数据兼容性
- ✅ 您的 HDF5 数据格式与原版 ACT 很接近
- ⚠️ 需要转换才能用于 Diffusion Policy
- ✅ 噪声标记可以在转换时保留

### 超参数
- 原版实现的默认超参数可能需要调整
- 建议从论文中的设置开始
- LeRobot 的超参数已经过优化

### 计算资源
- 原版实现可能需要更多显存
- LeRobot 有更好的内存管理
- 建议在小数据集上先测试

## 7. 快速开始脚本

创建 `setup_original_algorithms.sh`:

```bash
#!/bin/bash

# 克隆原版仓库
cd ~/workspace/galaxea
git clone https://github.com/tonyzhaozh/act.git
git clone https://github.com/real-stanford/diffusion_policy.git

# 安装依赖
cd act && pip install -e . && cd ..
cd diffusion_policy && pip install -e . && cd ..

echo "✅ 原版算法仓库已准备好"
echo "接下来："
echo "1. 运行数据转换脚本"
echo "2. 查看各仓库的 README 了解详细用法"
```

运行：
```bash
bash setup_original_algorithms.sh
```

