#!/usr/bin/env python3
"""
测试replan噪声标记功能

验证：
1. 数据收集时正确添加标记
2. 数据加载时可以根据标记过滤
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
import tempfile
from loguru import logger

import galaxea_sim.envs
from galaxea_sim.planners.bimanual import BimanualPlanner
from galaxea_sim.utils.data_utils import save_dict_list_to_hdf5
from galaxea_sim.utils.dataset_utils import load_demo_with_filter


def test_label_replan_noise():
    """测试标记replan噪声的功能"""
    
    logger.info("=" * 60)
    logger.info("测试：标记replan噪声数据")
    logger.info("=" * 60)
    
    # 创建环境，100%触发replan
    env = gym.make(
        "R1ProBlocksStackEasy-v0",
        control_freq=15,
        headless=True,
        obs_mode="state",
        ray_tracing=False,
        enable_replan=True,
        replan_prob=1.0,  # 100%触发
        replan_noise_range=(0.03, 0.06),
    )
    
    # 创建planner
    planner = BimanualPlanner(
        urdf_path=f"{env.unwrapped.robot.name}/robot.urdf",
        srdf_path=None,
        left_arm_move_group=env.unwrapped.left_ee_link_name,
        right_arm_move_group=env.unwrapped.right_ee_link_name,
        active_joint_names=env.unwrapped.active_joint_names,
        control_freq=env.unwrapped.control_freq,
    )
    
    if hasattr(env.unwrapped, 'set_planner'):
        env.unwrapped.set_planner(planner)
    
    # 收集一个轨迹
    env.reset()
    traj = []
    
    logger.info("\n收集轨迹数据...")
    for substep_idx, substep in enumerate(env.unwrapped.solution()):
        method, kwargs = substep
        is_replan_noise = kwargs.get("_is_replan_noise", False)
        
        # 移除内部标记
        kwargs_for_planner = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        
        actions = planner.solve(
            (method, kwargs_for_planner),
            env.unwrapped.robot.get_qpos(),
            env.unwrapped.last_gripper_cmd,
            verbose=False,
        )
        
        if actions is not None:
            for action in actions:
                obs, _, _, _, info = env.step(action)
                
                # 添加标记
                obs["is_replan_noise"] = is_replan_noise
                traj.append(obs)
    
    logger.info(f"收集完成，共 {len(traj)} 步")
    
    # 统计标记
    num_noise = sum(1 for step in traj if step.get("is_replan_noise", False))
    num_valid = len(traj) - num_noise
    
    logger.info(f"\n轨迹统计:")
    logger.info(f"  - 总步数: {len(traj)}")
    logger.info(f"  - 噪声步数: {num_noise} ({num_noise/len(traj)*100:.1f}%)")
    logger.info(f"  - 有效步数: {num_valid} ({num_valid/len(traj)*100:.1f}%)")
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        logger.info(f"\n保存到临时文件: {tmp_path}")
        save_dict_list_to_hdf5(traj, tmp_path)
    
    # 测试加载：不过滤
    logger.info("\n" + "=" * 60)
    logger.info("测试1：加载数据（不过滤噪声）")
    logger.info("=" * 60)
    traj_unfiltered, stats_unfiltered = load_demo_with_filter(
        tmp_path, 
        filter_replan_noise=False
    )
    logger.info(f"加载结果:")
    logger.info(f"  - 返回步数: {len(traj_unfiltered)}")
    logger.info(f"  - 统计: {stats_unfiltered}")
    
    # 测试加载：过滤噪声
    logger.info("\n" + "=" * 60)
    logger.info("测试2：加载数据（过滤噪声）")
    logger.info("=" * 60)
    traj_filtered, stats_filtered = load_demo_with_filter(
        tmp_path,
        filter_replan_noise=True
    )
    logger.info(f"加载结果:")
    logger.info(f"  - 返回步数: {len(traj_filtered)}")
    logger.info(f"  - 统计: {stats_filtered}")
    
    # 验证
    logger.info("\n" + "=" * 60)
    logger.info("验证结果")
    logger.info("=" * 60)
    
    assert len(traj_unfiltered) == len(traj), "未过滤时应该返回所有步骤"
    assert len(traj_filtered) == num_valid, "过滤后应该只返回有效步骤"
    assert stats_unfiltered["num_noise_steps"] == num_noise, "噪声步数统计不正确"
    assert stats_filtered["num_returned_steps"] == num_valid, "过滤后返回步数不正确"
    
    # 验证过滤后的数据都是有效的
    for step in traj_filtered:
        assert not step.get("is_replan_noise", False), "过滤后不应该有噪声步骤"
    
    logger.info("✅ 所有测试通过！")
    logger.info(f"\n总结:")
    logger.info(f"  - 原始数据: {len(traj)} 步")
    logger.info(f"  - 包含噪声: {num_noise} 步 ({num_noise/len(traj)*100:.1f}%)")
    logger.info(f"  - 过滤后: {len(traj_filtered)} 步 ({len(traj_filtered)/len(traj)*100:.1f}%)")
    logger.info(f"  - 数据已打标签，训练时可选择是否过滤")
    
    # 清理临时文件
    tmp_path.unlink()
    logger.info(f"\n已删除临时文件: {tmp_path}")


if __name__ == "__main__":
    test_label_replan_noise()

