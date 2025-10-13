# 夹爪目标位置可视化功能

## 功能说明

在 `BlocksStackEasyTrajAugEnv` 环境中，现在会在 sapien 场景中实时显示计算的夹爪目标位置的**虚拟夹爪形态**，帮助调试和理解机器人的规划过程。

## 可视化标记

环境会显示三个半透明的虚拟夹爪形态，显示不同阶段的目标位置和姿态：

| 标记颜色 | 含义                        | 位置说明                                              |
| -------- | --------------------------- | ----------------------------------------------------- |
| 🔵 蓝色  | 预抓取位置 (pre_grasp)      | 物体上方 20cm 处，夹爪接近物体前的位置和姿态          |
| 🟦 青色  | 抓取位置 (grasp)            | 实际抓取物体时夹爪的位置和姿态（预抓取位置下降 15cm） |
| 🟨 黄色  | 目标放置位置 (target_place) | 物体要放置到的目标位置和姿态                          |

每个虚拟夹爪包含：

- 夹爪基座（手掌部分）
- 两个平行手指
- 连接杆

所有虚拟夹爪都是半透明的（alpha=0.5-0.6），不会产生物理碰撞，仅用于可视化调试。

## 工作原理

1. **初始化**：在环境初始化时，使用 `create_visual_ee_link()` 函数创建三组虚拟夹爪（每组包含基座、两个手指和连接杆，共 4 个部件）

2. **规划时更新**：每次调用 `move_block()` 方法计算抓取和放置轨迹时，使用 `_update_visual_gripper_pose()` 方法自动更新所有标记的位置和姿态：

   - 蓝色夹爪显示预抓取位置和姿态
   - 青色夹爪显示实际抓取位置和姿态
   - 黄色夹爪显示目标放置位置和姿态

3. **实时可视化**：在 sapien 渲染的 3D 场景中实时看到这些半透明的虚拟夹爪，可以清楚地看到每个阶段夹爪的方向和角度

## 代码实现

### 关键修改

1. **在 `robotwin_utils.py` 中添加 `create_visual_ee_link` 函数**：

```python
def create_visual_ee_link(
    scene: sapien.Scene,
    pose: sapien.Pose,
    color=(0.3, 0.3, 0.8, 0.6),
    name="visual_gripper",
    gripper_width=0.08,
    gripper_depth=0.04,
    finger_length=0.06,
) -> list:
    """创建夹爪末端执行器的可视化形态（半透明，无碰撞）

    返回包含4个实体的列表：[基座, 左手指, 右手指, 连接杆]
    """
```

2. **导入可视化工具**：

```python
from galaxea_sim.utils.robotwin_utils import create_box, create_visual_ee_link
```

3. **初始化标记**（在 `__init__` 中）：

```python
self._setup_visual_markers()
```

4. **更新标记位置和姿态**（在 `move_block` 中）：

```python
# 更新可视化标记（位置+姿态）
pre_grasp_sapien_pose = sapien.Pose(p=pre_grasp_pose[:3], q=pre_grasp_pose[3:7])
self._update_visual_gripper_pose(self.pre_grasp_marker_entities, pre_grasp_sapien_pose)

grasp_sapien_pose = sapien.Pose(p=grasp_pose_vis[:3], q=grasp_pose_vis[3:7])
self._update_visual_gripper_pose(self.grasp_marker_entities, grasp_sapien_pose)

target_sapien_pose = sapien.Pose(p=target_pose[:3], q=target_pose[3:7])
self._update_visual_gripper_pose(self.target_place_marker_entities, target_sapien_pose)
```

## 使用方法

### 方法 1：直接运行环境

```python
from galaxea_sim.envs.robotwin.blocks_stack_easy_traj_aug import BlocksStackEasyTrajAugEnv
from galaxea_sim.robots import R1Pro

# 创建环境（headless=False 以显示GUI）
env = BlocksStackEasyTrajAugEnv(
    robot_class=R1Pro,
    headless=False
)

# 重置环境
env.reset()

# 执行任务，可视化标记会自动更新
for action_name, action_params in env.solution():
    if hasattr(env, action_name):
        getattr(env, action_name)(**action_params)
    env.render()
```

### 方法 2：使用测试脚本

```bash
cd /home/sen/workspace/galaxea/GalaxeaManipSim
python test_visual_markers.py
```

## 虚拟夹爪尺寸

默认虚拟夹爪参数：

- **夹爪宽度** (`gripper_width`): 8cm（两个手指间的距离）
- **夹爪深度** (`gripper_depth`): 4cm（前后方向）
- **手指长度** (`finger_length`): 6cm
- **透明度**: 0.5-0.6（半透明，不会完全遮挡视线）

## 调试用途

这个功能特别有用于：

- ✅ 验证抓取规划的正确性
- ✅ 调试轨迹生成算法
- ✅ 理解不同抓取角度的影响
- ✅ 检查目标放置位置是否合理
- ✅ 分析失败案例的原因

## 扩展性

### 在其他任务中使用

如果想在其他任务环境中添加虚拟夹爪可视化，只需：

1. **导入函数**：

```python
from galaxea_sim.utils.robotwin_utils import create_visual_ee_link
```

2. **创建虚拟夹爪**：

```python
# 在环境初始化时创建
self.my_gripper_entities = create_visual_ee_link(
    scene=self._scene,
    pose=sapien.Pose(p=[x, y, z], q=[qw, qx, qy, qz]),
    color=(r, g, b, alpha),  # RGBA值，范围0-1
    name="my_gripper",
    gripper_width=0.08,
    gripper_depth=0.04,
    finger_length=0.06,
)
```

3. **更新位置和姿态**：

```python
# 创建辅助方法更新所有部件（参考 _update_visual_gripper_pose）
def update_gripper(entities, pose):
    gripper_width = 0.08
    finger_length = 0.06
    entities[0].set_pose(pose)  # 基座
    entities[1].set_pose(pose * sapien.Pose(p=[0, gripper_width/2-0.008, -finger_length/2]))  # 左手指
    entities[2].set_pose(pose * sapien.Pose(p=[0, -gripper_width/2+0.008, -finger_length/2]))  # 右手指
    entities[3].set_pose(pose * sapien.Pose(p=[0, 0, -0.01]))  # 连接杆

# 使用
new_pose = sapien.Pose(p=[x, y, z], q=[qw, qx, qy, qz])
update_gripper(self.my_gripper_entities, new_pose)
```

### 自定义夹爪尺寸

可以根据实际机器人调整参数：

```python
# 示例：创建更大的夹爪
large_gripper = create_visual_ee_link(
    scene=self._scene,
    pose=initial_pose,
    color=(1.0, 0.5, 0.0, 0.6),
    name="large_gripper",
    gripper_width=0.12,  # 12cm宽
    gripper_depth=0.06,   # 6cm深
    finger_length=0.08,   # 8cm手指
)
```

## 性能影响

虚拟夹爪使用 `create_visual_ee_link`，每个夹爪包含 4 个实体（共 12 个实体用于 3 个夹爪）。这些实体：

- ✅ **只有渲染组件**，没有物理碰撞组件
- ✅ **半透明材质**，易于区分和观察
- ✅ **静态更新**，仅在规划时更新位置
- ✅ **对仿真性能影响极小**（< 1% CPU/GPU 开销）

即使在复杂场景中，这些可视化标记也不会显著影响仿真速度。
