#!/usr/bin/env python3
"""
最简单的示例：使用 gymnasium 注册的环境
启动环境但机器人不执行任何动作
"""

import gymnasium as gym
import galaxea_sim.envs.robotwin  # 导入以注册环境

# 使用 gymnasium 创建环境（更简单）
# 可选的环境：
# - R1ProBlocksStackEasy-v0: 方块堆叠任务
# - R1ProBlocksStackEasy-traj_aug: 带轨迹增强的方块堆叠
# - R1ProDualBottlesPickEasy-v0: 双瓶抓取任务
# 等等...

env = gym.make(
    "R1ProBlocksStackEasy-traj_aug",
    headless=False,  # 显示可视化窗口
)

print("=" * 60)
print("环境已创建！机器人将保持静止。")
print("=" * 60)
print(f"环境名称: {env.spec.id}")
print(f"机器人: {env.unwrapped.robot.name}")
print("按 Ctrl+C 退出...")
print("=" * 60)

# 重置环境
obs, info = env.reset()

# 主循环：只渲染，不执行任何动作
try:
    step_count = 0
    while True:
        env.render()
        step_count += 1

        # 每500帧打印一次信息
        if step_count % 500 == 0:
            print(f"已运行 {step_count} 帧...")

except KeyboardInterrupt:
    print("\n退出程序")
finally:
    env.close()
