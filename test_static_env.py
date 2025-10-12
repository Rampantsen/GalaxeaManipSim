#!/usr/bin/env python3
"""
示例：启动环境但机器人不执行任何动作
只是可视化环境和机器人的初始状态
可以自定义设置方块的位姿
"""

import numpy as np
import sapien
from galaxea_sim.robots.r1_pro import R1ProRobot
from galaxea_sim.envs.robotwin.blocks_stack_easy_traj_aug import (
    BlocksStackEasyTrajAugEnv,
)

# R1Pro 机器人的初始关节位置
R1PRO_INIT_QPOS = [
    0.7,
    -1.4,
    -0.9,
    0.0,
    -0.4,
    -0.4,
    1.3,
    -1.3,
    -0.7,
    0.7,
    -1.57,
    -1.57,
    1.3,
    -1.3,
    -0.4,
    -0.4,
    -0.8,
    0.8,
    0,
    0,
    0,
    0,
]

# 创建环境
env = BlocksStackEasyTrajAugEnv(
    robot_class=R1ProRobot,
    robot_kwargs=dict(init_qpos=R1PRO_INIT_QPOS),
    headless=False,  # 显示可视化窗口
    ray_tracing=False,  # 不使用光线追踪（更快）
)

print("=" * 60)
print("环境已创建！")
print("机器人将保持在初始位置不动。")
print("=" * 60)

# 方式1：重置环境到随机状态
# obs, info = env.reset()

# 方式2：重置环境并设置自定义的方块位姿
# 物体位姿格式: [x, y, z, qx, qy, qz, qw]
# 坐标系: 桌面中心约在 [0.7, 0, 0.9]

quat1 = [0.856813, 7.98373e-06, -2.22692e-05, -0.515628]

pose1_sapien = sapien.Pose(p=[0.678886, 0.112749, 0.92])  # 位置
pose1_sapien.set_q(quat1)  # 绕Z轴旋转180度
block1_pose = np.concatenate([pose1_sapien.p, pose1_sapien.q])

# 设置第二个方块的位姿
pose2_sapien = sapien.Pose(p=[0, 0, 0.92])  # 位置
pose2_sapien.set_rpy([0, 0, 0])  # 无旋转
block2_pose = np.concatenate([pose2_sapien.p, pose2_sapien.q])

# 重置环境并应用自定义位姿
reset_info = {
    "block1_pose": block1_pose,
    "block2_pose": block2_pose,
}

obs, info = env.reset()
# 重置后手动设置物体位姿
env.unwrapped.reset_world(reset_info)

print(f"方块1位姿: 位置={block1_pose[:3]}, 四元数={block1_pose[3:]}")
print(f"方块2位姿: 位置={block2_pose[:3]}, 四元数={block2_pose[3:]}")
print("按 Ctrl+C 退出...")
print("=" * 60)

# 主循环：只渲染，不执行任何动作
try:
    while True:
        env.render()  # 只渲染可视化，不执行动作
except KeyboardInterrupt:
    print("\n退出程序")

# 关闭环境
env.close()
