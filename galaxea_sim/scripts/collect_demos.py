import datetime
import uuid
import numpy as np
import random
from typing import Literal, Optional

import gymnasium as gym
import tyro
from loguru import logger
from pathlib import Path

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
        "baseline",
        "grasp_sample_only",
        "replan",  # 新增replan模式
        "all",
        "test",
    ] = "all",
    tag: Literal["collected"] = "collected",
    ray_tracing: bool = False,
    seed: Optional[int] = None,  # 添加seed参数
    enable_grasp_sample: bool = False,  # 是否启用grasp_sample
    enable_replan: bool = False,  # 是否启用replan
    replan_prob: float = 0.5,  # replan触发概率
    replan_noise_min: float = 0.02,  # 噪声范围最小值
    replan_noise_max: float = 0.05,  # 噪声范围最大值
):
    # 根据feature自动设置enable_grasp_sample和enable_replan
    if feature == "grasp_sample_only":
        enable_grasp_sample = True
    elif feature == "replan":
        enable_replan = True
    elif feature == "all":
        enable_grasp_sample = True
        enable_replan = True
    
    
    env = gym.make(
        env_name,
        control_freq=control_freq,
        headless=headless,
        obs_mode=obs_mode,
        ray_tracing=ray_tracing,
        enable_grasp_sample=enable_grasp_sample,
        enable_replan=enable_replan,
        replan_prob=replan_prob,
        replan_noise_range=(replan_noise_min, replan_noise_max),
        # table_type=table_type,
    )
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"设置随机种子: {seed}")

    assert isinstance(env.unwrapped, BimanualManipulationEnv)
    planner = BimanualPlanner(
        urdf_path=f"{env.unwrapped.robot.name}/robot.urdf",
        srdf_path=None,
        left_arm_move_group=env.unwrapped.left_ee_link_name,
        right_arm_move_group=env.unwrapped.right_ee_link_name,
        active_joint_names=env.unwrapped.active_joint_names,
        control_freq=env.unwrapped.control_freq,
    )
    
    # 设置planner到环境中，用于grasp_sample的IK测试
    if hasattr(env.unwrapped, 'set_planner'):
        env.unwrapped.set_planner(planner)

    save_dir = Path(dataset_dir) / env_name / table_type / feature /tag
    num_collected = 0
    num_tries = 0
    meta_info_list = []
    fail_meta_info_list = []
    meta_info_list.append(
        dict(
            env_name=env_name,
            feature=feature,
            seed=seed,
            num_demos=num_demos,
            enable_grasp_sample=enable_grasp_sample,
            enable_replan=enable_replan,
            replan_prob=replan_prob if enable_replan else None,
            replan_noise_range=(replan_noise_min, replan_noise_max) if enable_replan else None,
            has_replan_noise_label=enable_replan,  # 数据是否包含replan噪声标记
        )
    )
    while num_collected < num_demos:
        num_steps = 0
        traj = []
        info = {}
        _, rest_info = env.reset()
        if not headless:
            env.render()
        for substep in env.unwrapped.solution():
            # 提取substep的metadata
            method, kwargs = substep
            is_replan_noise = kwargs.get("_is_replan_noise", False)
            
            # 移除内部标记，避免传递给planner
            kwargs_for_planner = {k: v for k, v in kwargs.items() if not k.startswith("_")}
            
            actions = planner.solve(
                (method, kwargs_for_planner),
                env.unwrapped.robot.get_qpos(),
                env.unwrapped.last_gripper_cmd,
                verbose=False,
            )
            if actions is not None:
                for i, action in enumerate(actions):
                    num_steps += 1
                    obs, _, _, _, info = env.step(action)

                    # 添加replan噪声标记（保存所有数据，在读取时决定是否使用）
                    obs["is_replan_noise"] = is_replan_noise
                    
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
        else:
            fail_meta_info = dict(
                reset_info=rest_info,
            )
            fail_meta_info_list.append(fail_meta_info)
            save_dict_list_to_json(fail_meta_info_list, save_dir / "fail_meta_info.json")

if __name__ == "__main__":
    tyro.cli(main)
