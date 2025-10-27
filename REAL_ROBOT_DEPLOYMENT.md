# 真机部署指南 - ACT策略

## 概述

这份指南介绍如何将在仿真环境中训练的ACT策略部署到真实的Galaxea机器人上。

## 系统架构

```
真机传感器 → ROS话题 → ACT策略推理 → ROS话题 → 真机执行器

相机图像 ─┐
关节状态 ─┤→ 观测 → 策略 → 动作 ─┤→ 关节命令
夹爪状态 ─┘                        └→ 夹爪命令
```

## 前置要求

### 1. 硬件要求
- Galaxea R1/R1Pro/R1Lite机器人
- 相机（头部相机、腕部相机等）
- 运行Ubuntu 20.04的控制电脑
- GPU（推荐用于策略推理）

### 2. 软件要求
```bash
# ROS Noetic
sudo apt install ros-noetic-desktop-full

# Python依赖
pip install torch lerobot opencv-python
pip install rospkg cv_bridge

# Galaxea SDK
# 参考: https://userguide-galaxea.github.io/Product_User_Guide/Guide/A1/Software_Guide/
```

## 部署步骤

### 步骤1: 准备训练好的模型

```bash
# 确认模型路径
ls /path/to/your/model/checkpoints/last/pretrained_model/
# 应该包含: config.json, model.safetensors 等
```

### 步骤2: 配置相机话题映射

编辑真机部署脚本，设置正确的相机话题：

```python
camera_topics = {
    # 根据你的实际相机话题修改
    'head_camera': '/camera/head/color/image_raw',
    'left_wrist_camera': '/camera/left_wrist/color/image_raw',  
    'right_wrist_camera': '/camera/right_wrist/color/image_raw',
}
```

**查找相机话题**：
```bash
# 启动机器人后，查看所有图像话题
rostopic list | grep image

# 查看某个话题的信息
rostopic info /camera/head/color/image_raw

# 预览图像
rosrun image_view image_view image:=/camera/head/color/image_raw
```

### 步骤3: 配置关节映射

确认关节顺序与训练数据一致：

```python
# 在部署脚本中
def get_observation(self):
    joint_positions = np.array(self.current_joint_state.position)
    
    # ⚠️ 关键：确保关节顺序与训练数据一致
    # 检查训练数据中的关节顺序
    # 然后在这里进行正确的映射
    
    # 示例（需要根据实际情况调整）:
    left_arm_indices = [0, 1, 2, 3, 4, 5, 6]  # 左臂7个关节
    right_arm_indices = [7, 8, 9, 10, 11, 12, 13]  # 右臂7个关节
    
    obs['left_arm_joint_position'] = joint_positions[left_arm_indices]
    obs['right_arm_joint_position'] = joint_positions[right_arm_indices]
```

**检查关节顺序**：
```bash
# 查看关节状态话题
rostopic echo /joint_states_host -n 1

# 输出会显示：
# name: ['joint1', 'joint2', ...]
# position: [0.1, 0.2, ...]
```

### 步骤4: 启动机器人

```bash
# 终端1: 启动机器人驱动
cd A1_SDK/install
source setup.bash
roslaunch mobiman single_arm_node.launch  # 或双臂启动脚本

# 终端2: 启动相机节点（如果需要）
roslaunch realsense2_camera rs_camera.launch

# 终端3: 检查话题
rostopic list
```

### 步骤5: 运行部署脚本

```bash
python -m galaxea_sim.scripts.deploy_real_robot_act \
  --pretrained-policy-path /path/to/model/checkpoints/last/pretrained_model \
  --dataset-repo-id galaxea/R1ProBlocksStackEasy/traj_augmented \
  --device cuda \
  --temporal-ensemble \
  --control-freq 15 \
  --num-episodes 10
```

## 关键配置说明

### 1. 控制频率匹配

```python
control_freq = 15  # 必须与训练数据的fps一致！
```

训练数据的fps可以在数据集metadata中查看：
```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
metadata = LeRobotDatasetMetadata("your-dataset-id")
print(metadata.fps)  # 应该是15
```

### 2. 观测格式匹配

确保真机观测的keys和shapes与训练数据完全一致：

```python
# 训练数据的观测格式
{
    'head_camera': (480, 640, 3),
    'left_wrist_camera': (480, 640, 3),
    'right_wrist_camera': (480, 640, 3),
    'left_arm_joint_position': (7,),
    'right_arm_joint_position': (7,),
    'left_arm_gripper_position': (1,),
    'right_arm_gripper_position': (1,),
    ...
}
```

### 3. 动作格式匹配

策略输出的动作格式：
```python
action = [
    left_joint1, left_joint2, ..., left_joint7,  # 左臂关节
    left_gripper,                                  # 左夹爪
    right_joint1, right_joint2, ..., right_joint7, # 右臂关节
    right_gripper,                                 # 右夹爪
]
```

## 调试技巧

### 1. 验证观测格式

```python
# 在部署前，先打印观测
obs = deployer.get_observation()
for key, value in obs.items():
    print(f"{key}: {value.shape if hasattr(value, 'shape') else type(value)}")

# 对比训练数据
# 确保keys和shapes完全一致
```

### 2. 可视化相机图像

```python
import cv2

for camera_name, image in deployer.camera_images.items():
    cv2.imshow(camera_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
```

### 3. 慢速执行

```python
# 降低控制频率，便于观察
control_freq = 5  # Hz（训练时是15Hz）

# 或者在每步之间添加延迟
time.sleep(0.5)
```

### 4. 记录执行过程

```python
# 记录观测和动作，用于离线分析
trajectory = []
for step in range(max_steps):
    obs = get_observation()
    action = policy.select_action(obs)
    
    trajectory.append({
        'observation': obs,
        'action': action,
        'timestamp': time.time()
    })
    
    execute_action(action)

# 保存
import pickle
with open('real_robot_trajectory.pkl', 'wb') as f:
    pickle.dump(trajectory, f)
```

## 安全注意事项

⚠️ **在真机上运行前，请务必**：

1. **设置急停按钮**: 随时准备紧急停止
2. **清空工作空间**: 移除可能碰撞的物体
3. **慢速测试**: 第一次运行时降低速度
4. **监督执行**: 人员在场监督，观察异常行为
5. **限制工作空间**: 设置软件限位，避免超出安全范围

### 添加安全检查

```python
def is_safe_action(self, action):
    """检查动作是否安全"""
    # 检查关节限位
    joint_limits_lower = np.array([-3.14, -2.0, ...])
    joint_limits_upper = np.array([3.14, 2.0, ...])
    
    if np.any(action < joint_limits_lower) or np.any(action > joint_limits_upper):
        rospy.logerr("⚠️ 动作超出关节限位！")
        return False
    
    # 检查速度
    if self.last_action is not None:
        velocity = np.abs(action - self.last_action) / self.dt
        max_velocity = np.array([2.0, 2.0, ...])  # rad/s
        
        if np.any(velocity > max_velocity):
            rospy.logerr("⚠️ 动作速度过快！")
            return False
    
    return True

def execute_action(self, action):
    """执行动作（带安全检查）"""
    if not self.is_safe_action(action):
        rospy.logerr("动作未通过安全检查，停止执行")
        rospy.signal_shutdown("Safety check failed")
        return
    
    # 发送命令
    self.joint_command_pub.publish(...)
    self.last_action = action
```

## 常见问题

### Q1: 图像尺寸不匹配怎么办？

```python
# 在get_observation中调整图像大小
for camera_name, image in self.camera_images.items():
    # 调整到训练数据的尺寸
    image = cv2.resize(image, (640, 480))
    obs[camera_name] = image
```

### Q2: 真机执行速度不稳定？

- 检查控制频率是否稳定
- 使用`rospy.Rate`控制循环频率
- 监控ROS话题的发布频率：`rostopic hz /joint_states_host`

### Q3: 策略输出的动作不合理？

可能原因：
1. 观测格式不匹配（检查keys和shapes）
2. 数据归一化不一致（检查是否需要反归一化）
3. 坐标系不一致（检查world frame vs base frame）

### Q4: 如何使用不同的控制器？

根据[Galaxea文档](https://userguide-galaxea.github.io/Product_User_Guide/Guide/A1/Software_Guide/#joint-position-movement-interface)，有多种控制接口：

**关节位置控制**（推荐）:
```python
# 发布到 /arm_joint_target_position
joint_cmd = JointState()
joint_cmd.position = action  # 直接是关节位置
```

**末端位姿控制**:
```python
# 发布到 /a1_ee_target
ee_cmd = PoseStamped()
ee_cmd.pose.position.x = x
ee_cmd.pose.position.y = y
ee_cmd.pose.position.z = z
ee_cmd.pose.orientation = quaternion
```

## 完整部署流程示例

```bash
# 1. 启动机器人
roslaunch mobiman single_arm_node.launch

# 2. 启动相机（另一个终端）
roslaunch realsense2_camera rs_camera.launch

# 3. 检查话题（另一个终端）
rostopic list | grep -E "joint_states|camera"

# 4. 测试订阅（另一个终端）
rostopic echo /joint_states_host -n 1
rostopic echo /camera/head/color/image_raw -n 1

# 5. 运行部署脚本（另一个终端）
python -m galaxea_sim.scripts.deploy_real_robot_act \
  --pretrained-policy-path ./outputs/ACT/.../checkpoints/last/pretrained_model \
  --dataset-repo-id galaxea/R1ProBlocksStackEasy/traj_augmented \
  --device cuda \
  --temporal-ensemble \
  --num-episodes 5
```

## 相机配置参考

根据你的机器人配置，相机话题可能是：

```python
# RealSense相机
camera_topics = {
    'head_camera': '/camera/color/image_raw',
    'left_wrist_camera': '/left_camera/color/image_raw',
    'right_wrist_camera': '/right_camera/color/image_raw',
}

# 或者自定义相机
camera_topics = {
    'head_camera': '/head_cam/image',
    'left_wrist_camera': '/left_wrist_cam/image',
    'right_wrist_camera': '/right_wrist_cam/image',
}
```

## sim2real差异处理

### 1. 视觉差异
- **图像质量**: 真机相机可能与仿真渲染不同
- **光照条件**: 真实环境光照变化
- **建议**: 在多种光照条件下收集数据

### 2. 动力学差异
- **摩擦力**: 真实关节有摩擦
- **惯性**: 真实机器人有惯性和延迟
- **建议**: 使用轨迹增强训练（已实现的执行噪声方案）

### 3. 控制延迟
- **通信延迟**: ROS话题传输有延迟
- **执行延迟**: 电机响应需要时间
- **建议**: 调整控制频率，添加预测补偿

## 优化建议

### 1. 使用轨迹增强训练的模型

```bash
# 使用执行噪声训练的模型更鲁棒
bash collect_demo.sh \
  --env-name R1ProBlocksStackEasy-traj_aug \
  --feature traj_augmented \
  --num-demos 1500
```

这样训练的策略能够：
- ✅ 应对执行误差
- ✅ 自动纠偏
- ✅ 更好的sim2real效果

### 2. Domain Randomization

在训练时添加：
- 视觉随机化（光照、颜色、纹理）
- 物理随机化（摩擦系数、质量）
- 位置随机化（物体初始位置）

### 3. 在线微调

收集真机数据，进行在线微调：
```bash
# 1. 在真机上用遥操作收集少量demo（50-100个）
# 2. 转换为LeRobot格式
# 3. 与仿真数据混合训练
# 4. 部署微调后的模型
```

## 故障排除

### 问题1: 策略输出异常动作

**症状**: 机器人突然大幅度移动或震荡

**排查**:
```python
# 添加调试输出
print(f"观测keys: {list(obs.keys())}")
print(f"观测shapes: {[(k, v.shape) for k, v in obs.items()]}")
print(f"动作范围: min={action.min()}, max={action.max()}")

# 检查数据归一化
# 如果训练时归一化了，推理时也要归一化
```

### 问题2: ROS话题接收不到数据

**排查**:
```bash
# 检查话题是否发布
rostopic list

# 检查话题频率
rostopic hz /joint_states_host
rostopic hz /camera/head/color/image_raw

# 检查话题内容
rostopic echo /joint_states_host -n 1
```

### 问题3: 控制频率不稳定

**解决**:
```python
# 使用rospy.Rate严格控制频率
rate = rospy.Rate(control_freq)

for step in range(max_steps):
    # ... 推理和执行 ...
    
    rate.sleep()  # 严格控制循环频率
```

## 性能监控

```python
import time

# 监控推理时间
start = time.time()
action = policy.select_action(obs)
inference_time = time.time() - start

print(f"推理时间: {inference_time*1000:.2f}ms")
print(f"控制周期: {1/control_freq*1000:.2f}ms")

if inference_time > 1/control_freq:
    print("⚠️ 警告：推理时间超过控制周期！")
```

## 相关文档

- [Galaxea A1 软件指南](https://userguide-galaxea.github.io/Product_User_Guide/Guide/A1/Software_Guide/)
- `deploy_real_robot_act.py` - 真机部署脚本
- `TRAJ_AUGMENTED_README.md` - 轨迹增强训练方案

## 快速检查清单

部署前检查：
- [ ] 模型已训练完成
- [ ] ROS话题正常发布
- [ ] 相机图像清晰
- [ ] 关节映射正确
- [ ] 控制频率匹配
- [ ] 安全措施就位
- [ ] 工作空间清空

开始执行：
- [ ] 观测格式验证通过
- [ ] 动作范围合理
- [ ] 第一步慢速测试成功
- [ ] 完整episode测试成功

---

**重要提示**: 真机部署时请务必小心，从慢速测试开始，逐步提高速度和复杂度。

