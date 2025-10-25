import datetime
import uuid

from typing import Literal, Optional

import gymnasium as gym
import tyro
from loguru import logger
from pathlib import Path
import random
import numpy as np

from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
from galaxea_sim.planners.bimanual import BimanualPlanner
from galaxea_sim.utils.data_utils import save_dict_list_to_hdf5, save_dict_list_to_json


def main(
    env_name: str,
    num_demos: int = 100,
    dataset_dir: str = "datasets",
    control_freq: int = 15,
    headless: bool = True,
    obs_mode: Literal["state", "image"] = "state",
    table_type: Literal["red", "white"] = "red",
    feature: Literal[
        "no-retry",
        "no-grasp_sample",
        "grasp_sample_only",
        "retry_only",
        "baseline",
        "traj_augmented",  
        "all",
        "test",
    ] = "all",
    tag: Literal["collected"] = "collected",
    ray_tracing: bool = True,
    seed: Optional[int] = None,  # 添加seed参数
):
    # 设置全局随机种子以确保可复现性
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"设置随机种子: {seed}")

    env = gym.make(
        env_name,
        control_freq=control_freq,
        headless=headless,
        obs_mode=obs_mode,
        ray_tracing=ray_tracing,
    )
    
    # 设置环境的随机种子
    if seed is not None:
        env.seed(seed)
        logger.info(f"环境随机种子已设置: {seed}")
    assert isinstance(env.unwrapped, BimanualManipulationEnv)
    planner = BimanualPlanner(
        urdf_path=f"{env.unwrapped.robot.name}/robot.urdf",
        srdf_path=None,
        left_arm_move_group=env.unwrapped.left_ee_link_name,
        right_arm_move_group=env.unwrapped.right_ee_link_name,
        active_joint_names=env.unwrapped.active_joint_names,
        control_freq=env.unwrapped.control_freq,
        env=env,
    )
    
    # 设置planner到环境中（用于grasp_sample的IK测试）
    if hasattr(env.unwrapped, 'set_planner'):
        env.unwrapped.set_planner(planner)

    save_dir = Path(dataset_dir) / env_name / table_type / feature / tag
    num_collected = 0
    num_tries = 0
    meta_info_list = []
    # 记录收集参数
    meta_info_list.append(
        dict(
            env_name=env_name,
            feature=feature,
            seed=seed,
            num_demos=num_demos,
        )
    )

    # 检查是否需要启用累积误差跟踪
    # 当feature包含traj_augmented时，启用误差跟踪（用于触发重新规划）
    enable_error_tracking = "traj_augmented" in feature or feature in ["all"] or feature in ["no-retry"] or feature in ["test"]
    if enable_error_tracking:
        planner.enable_traj_augmented_mode(True)
        logger.info("已启用轨迹增强模式（执行噪声）：记录正确action，但发送带噪声的action给机器人")

    while num_collected < num_demos:

        num_steps = 0
        traj = []
        info = {}
        _, rest_info = env.reset()
        
        # 每个demo开始时重置累积误差
        if enable_error_tracking:
            planner.reset_accumulated_error()
        
        if not headless:
            env.render()
        for substep in env.unwrapped.solution():
            method, kwargs = substep
            
            result = planner.solve(
                substep,
                env.unwrapped.robot.get_qpos(),
                env.unwrapped.last_gripper_cmd,
                verbose=False,
            )
            if result is not None:
                # 检查返回值是否是tuple（有执行噪声）
                if isinstance(result, tuple):
                    executed_actions, planned_actions = result
                    
                    # 记录哪个手臂在活动（从substep的kwargs中获取）
                    from mplib.pymp import Pose
                    left_pose_in_substep = kwargs.get('left_pose')
                    right_pose_in_substep = kwargs.get('right_pose')
                    
                    # 记录原始目标位姿（用于重新规划）
                    final_planned_action = planned_actions[-1]
                    current_qpos_temp = env.unwrapped.robot.get_qpos()
                    final_qpos = current_qpos_temp.copy()
                    final_qpos[env.unwrapped.left_arm_joint_indices] = final_planned_action[:planner.left_arm_action_dim]
                    final_qpos[env.unwrapped.right_arm_joint_indices] = final_planned_action[planner.left_arm_action_dim+1:planner.left_arm_action_dim+1+planner.right_arm_action_dim]
                    
                    # 使用FK计算目标末端位姿
                    planner._fk.compute_forward_kinematics(final_qpos)
                    original_target_left = Pose(p=planner._fk.get_link_pose(0).p, q=planner._fk.get_link_pose(0).q)
                    original_target_right = Pose(p=planner._fk.get_link_pose(1).p, q=planner._fk.get_link_pose(1).q)
                    
                    # 使用带噪声的action执行，但记录正确的action
                    i = 0
                    while i < len(executed_actions):
                        executed_action = executed_actions[i]
                        planned_action = planned_actions[i]
                        num_steps += 1
                        
                        # 执行带噪声的action
                        obs, _, _, _, info = env.step(executed_action)
                        
                        # 如果启用了误差跟踪，计算末端执行器位置误差
                        if enable_error_tracking:
                            # 获取执行后的实际qpos
                            actual_qpos = env.unwrapped.robot.get_qpos()
                            
                            # 构建目标qpos（将planned_action填入实际qpos）
                            target_qpos = actual_qpos.copy()
                            target_qpos[env.unwrapped.left_arm_joint_indices] = planned_action[:planner.left_arm_action_dim]
                            target_qpos[env.unwrapped.right_arm_joint_indices] = planned_action[planner.left_arm_action_dim+1:planner.left_arm_action_dim+1+planner.right_arm_action_dim]
                            
                            # 使用FK计算末端执行器位置误差
                            ee_pos_error, _ = planner.compute_pose_error(target_qpos, actual_qpos)
                            
                            # 只记录最大的单步误差，而不是累加所有误差
                            planner.accumulated_position_error = max(
                                planner.accumulated_position_error, 
                                ee_pos_error
                            )
                            
                            # 每隔一定步数检查是否需要重新规划
                            if (i + 1) % 10 == 0 and planner.check_replan_needed():
                                # 从当前位置重新规划到原目标位姿
                                logger.info(f"触发重新规划! 当前误差: {planner.accumulated_position_error:.4f}m")
                                
                                current_qpos = env.unwrapped.robot.get_qpos()
                                
                                # 只给原始substep中活动的手臂传入目标位姿，静止的手臂传入None
                                replan_target_left = original_target_left if left_pose_in_substep is not None else None
                                replan_target_right = original_target_right if right_pose_in_substep is not None else None
                                
                                print(f"🔄 重新规划: left_pose={left_pose_in_substep is not None}, right_pose={right_pose_in_substep is not None}")
                                print(f"   replan_target_left={replan_target_left is not None}, replan_target_right={replan_target_right is not None}")
                                
                                replan_result = planner.replan_from_current(
                                    target_pose_left=replan_target_left,
                                    target_pose_right=replan_target_right,
                                    current_qpos=planner.sim2mplib_mapping(current_qpos),
                                    verbose=False
                                )
                                
                                if replan_result is not None:
                                    # 重新规划成功，生成新的轨迹替换剩余部分
                                    init_pos = np.zeros(planner.action_dim)
                                    init_pos[:planner.left_arm_action_dim] = current_qpos[env.unwrapped.left_arm_joint_indices]
                                    init_pos[planner.left_arm_action_dim] = env.unwrapped.last_gripper_cmd[0]
                                    init_pos[-planner.right_arm_action_dim-1:-1] = current_qpos[env.unwrapped.right_arm_joint_indices]
                                    init_pos[-1] = env.unwrapped.last_gripper_cmd[1]
                                    
                                    new_trajectory = planner.get_move_trajectory(init_pos, *replan_result)
                                    
                                    # 根据原始substep确定活动手臂（只给活动的手臂添加噪声）
                                    if left_pose_in_substep is not None and right_pose_in_substep is not None:
                                        active_arm_for_replan = "both"
                                    elif left_pose_in_substep is not None:
                                        active_arm_for_replan = "left"
                                    elif right_pose_in_substep is not None:
                                        active_arm_for_replan = "right"
                                    else:
                                        active_arm_for_replan = "both"  # fallback
                                    
                                    # 应用执行噪声到新轨迹（只给活动的手臂添加）
                                    new_executed, new_planned = planner.add_execution_noise(
                                        new_trajectory, active_arm=active_arm_for_replan,
                                        noise_probability=0.4, position_noise_std=0.015
                                    )
                                    
                                    # 替换剩余的轨迹
                                    executed_actions = new_executed
                                    planned_actions = new_planned
                                    i = 0  # 重置索引
                                    planner.reset_accumulated_error()
                                    logger.info("✅ 重新规划完成，继续执行新轨迹")
                                    continue
                                else:
                                    logger.warning("⚠️ 重新规划失败，继续执行原轨迹")
                                    planner.reset_accumulated_error()
                        
                        # 在obs中记录的是正确的action（用于训练）
                        obs['upper_body_action_dict']['left_arm_joint_position_cmd'] = planned_action[:planner.left_arm_action_dim]
                        obs['upper_body_action_dict']['left_arm_gripper_position_cmd'] = np.array([planned_action[planner.left_arm_action_dim]])
                        obs['upper_body_action_dict']['right_arm_joint_position_cmd'] = planned_action[planner.left_arm_action_dim+1:planner.left_arm_action_dim+1+planner.right_arm_action_dim]
                        obs['upper_body_action_dict']['right_arm_gripper_position_cmd'] = np.array([planned_action[planner.left_arm_action_dim+1+planner.right_arm_action_dim]])
                        traj.append(obs)
                        if not headless:
                            env.render()
                        
                        i += 1  # 下一步
                else:
                    # 正常模式
                    actions = result
                    for action in actions:
                        num_steps += 1
                        obs, _, _, _, info = env.step(action)
                        traj.append(obs)
                        if not headless:
                            env.render()
        num_tries += 1
        if info["success"]:
            save_dict_list_to_hdf5(traj, save_dir / f"demo_{num_collected}.h5")
            num_collected += 1
            meta_info = dict(
                reset_info=rest_info,
                success=info["success"],
                total_steps=num_steps,
                num_collected=num_collected,
                num_tries=num_tries,
            )
            meta_info_list.append(meta_info)
            save_dict_list_to_json(meta_info_list, save_dir / "meta_info.json")
            logger.info(
                f"Collected {num_collected} demos in {num_tries} tries. Success rate: {int(num_collected/num_tries*100)}%"
            )


if __name__ == "__main__":
    tyro.cli(main)
