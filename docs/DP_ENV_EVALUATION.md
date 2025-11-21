# Diffusion Policy 环境评估完整指南

## 🎯 实现说明

我们创建了一个Gymnasium兼容的环境评估器，不依赖旧版gym的工具类。

## 📦 文件位置

```
policy/dp/diffusion_policy/env_runner/galaxea_image_runner.py
```

按照原版Diffusion Policy的结构组织，与其他runner放在一起：
- `pusht_image_runner.py` - PushT任务评估器
- `robomimic_image_runner.py` - Robomimic任务评估器
- `galaxea_image_runner.py` - Galaxea任务评估器 ✨

## ✨ 关键特性

### 1. **Gymnasium兼容**
- ✅ 使用 `gymnasium.make()` 而不是旧版 `gym.make()`
- ✅ 兼容Galaxea环境的新API
- ✅ 不依赖AsyncVectorEnv等旧版工具

### 2. **三相机支持**
- 自动提取 `rgb_head`, `rgb_left_hand`, `rgb_right_hand`
- 自动处理图像normalize和维度转换
- 拼接状态为16维qpos

### 3. **WandB集成**
返回标准格式的评估指标：
- `test/mean_score`: 平均奖励
- `test/success_rate`: 成功率
- `test/max_reward_mean`: 平均最大奖励
- `test/max_reward_std`: 奖励标准差
- `test/avg_length`: 平均episode长度

## ⚙️ 配置

### 在训练脚本中

`scripts/train_dp_with_noise.sh` 会自动生成：

```yaml
env_runner:
  _target_: diffusion_policy.env_runner.galaxea_image_runner.GalaxeaImageRunner
  output_dir: null
  env_name: R1ProBlocksStackEasy-v0
  n_test: 5  # 每次评估5个episode
  n_test_vis: 0  # 暂不支持视频
  test_start_seed: 100000
  max_steps: 300
  n_obs_steps: ${n_obs_steps}  # 2
  n_action_steps: ${n_action_steps}  # 8
  fps: 15
  past_action: False
  tqdm_interval_sec: 1.0
```

### 评估频率

在 `train_galaxea_diffusion_unet_image_workspace.yaml` 中：

```yaml
training:
  rollout_every: 50  # 每50个epoch评估一次
```

## 🚀 使用方法

### 启用评估（默认）

```bash
./scripts/train_dp_with_noise.sh R1ProBlocksStackEasy all true
```

训练流程：
1. Epoch 0-49: 正常训练
2. Epoch 50: 训练 + 运行5个测试episode
3. 记录成功率和奖励到WandB
4. Epoch 51-99: 继续训练
5. Epoch 100: 再次评估
6. ...

### 调整评估参数

```bash
# 更频繁评估
cd policy/dp && python train.py \
  --config-name=train_galaxea_diffusion_unet_image_workspace \
  task=galaxea_R1ProBlocksStackEasy_all \
  training.rollout_every=10 \
  task.env_runner.n_test=3

# 更多测试episodes
cd policy/dp && python train.py \
  ... \
  task.env_runner.n_test=20 \
  task.env_runner.max_steps=500
```

### 禁用评估

修改 `scripts/train_dp_with_noise.sh`：
```yaml
env_runner: null
```

然后需要修改checkpoint监控指标为 `train_loss`。

## 📊 评估指标说明

### test/mean_score
- 所有测试episode的平均累计奖励
- Checkpoint默认监控这个指标（越高越好）

### test/success_rate
- 成功完成任务的episode比例
- 范围：0.0 - 1.0
- 最重要的性能指标

### test/avg_length
- 平均episode长度
- 可以看出策略是否能快速完成任务

## 🔧 实现细节

### 观测处理流程

```python
# 1. Galaxea原始观测
obs_dict = {
    'upper_body_observations': {
        'rgb_head': (720, 1280, 3),
        'rgb_left_hand': (240, 320, 3),
        'rgb_right_hand': (240, 320, 3),
        'left_arm_joint_position': (7,),
        ...
    }
}

# 2. 提取并处理
img_head = obs['rgb_head']  # HWC
img_head = img_head.permute(2, 0, 1) / 255.0  # CHW, [0,1]
state = np.concatenate([left_7, left_gripper_1, right_7, right_gripper_1])  # 16

# 3. 堆叠历史（n_obs_steps=2）
obs_seq = {
    'img_head': torch.stack([obs[t] for t in history], dim=0),  # (2, 3, H, W)
    'state': torch.stack([obs[t] for t in history], dim=0),  # (2, 16)
}

# 4. 添加batch维度
obs_seq = {k: v.unsqueeze(0) for k, v in obs_seq.items()}  # (1, 2, ...)

# 5. 传给策略
action_dict = policy.predict_action(obs_seq)
action = action_dict['action'][0].cpu().numpy()  # 第一步动作
```

### 与原版的区别

| 特性 | 原版 (pusht/robomimic) | 我们的实现 |
|------|----------------------|-----------|
| 环境库 | 旧版 `gym` | 新版 `gymnasium` |
| 并行环境 | AsyncVectorEnv | 单个环境串行 |
| 视频录制 | VideoRecordingWrapper | 暂不支持 |
| 观测历史 | MultiStepWrapper | 手动管理deque |
| 速度 | 快（并行） | 稍慢（串行） |

## ⚠️ 限制和未来改进

### 当前限制

1. **串行评估**：episodes按顺序运行，不是并行
2. **无视频录制**：避免依赖VideoRecordingWrapper
3. **速度较慢**：5个episodes约需10-15秒

### 未来改进

1. **添加视频支持**：使用mediapy或其他库录制
2. **并行评估**：使用gymnasium的vector env
3. **缓存环境**：避免每次创建新环境

## 💡 最佳实践

### 调试阶段
```yaml
env_runner:
  n_test: 2  # 少量episode
  max_steps: 50  # 短episode
training:
  rollout_every: 5  # 频繁评估
```

### 正式训练
```yaml
env_runner:
  n_test: 10  # 足够统计
  max_steps: 300  # 完整episode
training:
  rollout_every: 50  # 标准频率
```

### 快速训练（不评估）
```yaml
env_runner: null
training:
  rollout_every: 10000
checkpoint:
  topk:
    monitor_key: train_loss  # 改用训练loss
    mode: min
```

## 🎯 监控训练

在WandB中关注：
- `test/success_rate` - 主要指标
- `test/mean_score` - 辅助指标
- `train_loss` - 训练是否正常

当success_rate达到60-80%时，策略已经很好了！

## 📁 相关文件

- **评估器**: `policy/dp/diffusion_policy/env_runner/galaxea_image_runner.py`
- **数据集**: `galaxea_sim/utils/dp_noise_filtered_dataset.py`
- **配置**: `policy/dp/diffusion_policy/config/train_galaxea_diffusion_unet_image_workspace.yaml`
- **脚本**: `scripts/train_dp_with_noise.sh`
