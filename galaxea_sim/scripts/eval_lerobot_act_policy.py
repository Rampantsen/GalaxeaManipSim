from pathlib import Path
import gymnasium as gym
import imageio
import numpy as np
import pickle
import torch
import tyro
import cv2
import random
from tqdm import tqdm

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from galaxea_sim.utils.data_utils import save_dict_list_to_json

import numpy as np
import time


def evaluate(
    task: str,
    pretrained_policy_path: str = "/home/sen/workspace/galaxea/GalaxeaManipSim/outputs/ACT/R1ProBlocksStackEasy-traj_aug/all-20251010214500/checkpoints/last/pretrained_model",
    dataset_repo_id: str = "galaxea/R1ProBlocksStackEasy/normal",
    target_controller_type: str = "bimanual_joint_position",
    device: str = "cuda",
    headless: bool = True,
    num_evaluations: int = 100,
    temporal_ensemble: bool = True,  # ACT 特有的时序集成
    save_video: bool = False,
    ray_tracing: bool = False,  # 添加ray_tracing参数
    seed: int = 10,  # 添加seed参数
):
    """在模拟环
    境中多次评估预训练的 ACT 策略。"""
    # 设置全局随机种子以确保可复现性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_directory = (
        Path(pretrained_policy_path).parent.parent
        / "evaluations"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    dataset_stats = dataset_metadata.stats

    policy = ACTPolicy.from_pretrained(
        pretrained_policy_path,
        dataset_stats=dataset_stats,
    )
    policy.eval()
    policy.to(device)

    # 启用时序集成（ACT 特有功能）
    # 注意：使用 temporal ensemble 时，n_action_steps 必须为 1
    # temporal_ensembler 只在初始化时创建，所以需要在评估前手动创建
    if temporal_ensemble:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        policy.config.temporal_ensemble_coeff = 0.1
        policy.config.n_action_steps = 30
        policy.temporal_ensembler = ACTTemporalEnsembler(
            temporal_ensemble_coeff=0.1, chunk_size=policy.config.chunk_size
        )

    infos = []
    infos.append(
        dict(
            task=task,
            target_controller_type=target_controller_type,
            temporal_ensemble=temporal_ensemble,
            ray_tracing=ray_tracing,
            seed=seed,
            num_evaluations=num_evaluations,
            pretrained_policy_path=pretrained_policy_path,
        )
    )
    env = gym.make(
        task,
        control_freq=15,  # 必须与训练数据的 fps 匹配！
        headless=headless,
        max_episode_steps=400,
        controller_type=target_controller_type,
        ray_tracing=ray_tracing,  # 添加ray_tracing参数
    )

    success_count_running = 0
    for eval_idx in tqdm(range(num_evaluations), desc="评估进度", unit="episode"):
        policy.reset()
        numpy_observation, reset_info = env.reset()

        # 保存初始方块位姿信息
        initial_block_poses = {}
        if "block1_pose" in reset_info:
            initial_block_poses["block1_initial_pose"] = reset_info[
                "block1_pose"
            ].tolist()
        if "block2_pose" in reset_info:
            initial_block_poses["block2_initial_pose"] = reset_info[
                "block2_pose"
            ].tolist()

        if save_video:
            env.render()

        rewards = []
        frames = []
        if save_video:
            frames.append(env.render())

        step = 0
        done = False

        while not done:
            obs = numpy_observation["upper_body_observations"]

            # 根据控制器类型准备状态
            if target_controller_type == "bimanual_joint_position":
                state = (
                    torch.cat(
                        [
                            torch.from_numpy(obs["left_arm_joint_position"]),
                            torch.from_numpy(obs["left_arm_gripper_position"]),
                            torch.from_numpy(obs["right_arm_joint_position"]),
                            torch.from_numpy(obs["right_arm_gripper_position"]),
                        ],
                        dim=-1,
                    )
                    .to(torch.float32)
                    .to(device, non_blocking=True)
                    .unsqueeze(0)
                )
            else:
                state = (
                    torch.cat(
                        [
                            torch.from_numpy(obs["left_arm_ee_pose"]),
                            torch.from_numpy(obs["left_arm_gripper_position"]),
                            torch.from_numpy(obs["right_arm_ee_pose"]),
                            torch.from_numpy(obs["right_arm_gripper_position"]),
                        ],
                        dim=-1,
                    )
                    .to(torch.float32)
                    .to(device, non_blocking=True)
                    .unsqueeze(0)
                )

            # 调整图像大小并归一化
            resized_image = {
                k: cv2.resize(obs[k], (224, 224))
                for k in ["rgb_head", "rgb_left_hand", "rgb_right_hand"]
            }
            images = {
                k: torch.from_numpy(resized_image[k]).to(torch.float32).permute(2, 0, 1)
                / 255.0
                for k in ["rgb_head", "rgb_left_hand", "rgb_right_hand"]
            }
            images = {
                k: v.to(device, non_blocking=True).unsqueeze(0)
                for k, v in images.items()
            }

            observation = {
                "observation.state": state,
                **{f"observation.images.{k}": v for k, v in images.items()},
            }

            # 使用 ACT 策略选择动作
            with torch.inference_mode():
                action = policy.select_action(observation)

            numpy_observation, reward, terminated, truncated, info = env.step(
                action.squeeze(0).cpu().numpy()
            )
            rewards.append(reward)
            if save_video:
                frames.append(env.render())
            done = terminated or truncated or done
            step += 1

        # 将seed、eval_idx、方块初始位姿和其他统计信息添加到info中
        info["eval_idx"] = eval_idx
        info["total_steps"] = step
        # 添加初始方块位姿信息
        info.update(initial_block_poses)

        infos.append(info)

        # 更新实时成功率
        if info.get("success", False):
            success_count_running += 1
        current_success_rate = success_count_running / (eval_idx + 1) * 100
        tqdm.write(
            f"Episode {eval_idx + 1}: {'✓ 成功' if terminated else '✗ 失败'} | 步数: {step} | 当前成功率: {current_success_rate:.1f}%"
        )

        save_dict_list_to_json(infos, output_directory / "info.json")

        if save_video:
            fps = env.unwrapped.control_freq
            video_path = output_directory / f"rollout_{eval_idx + 1}.mp4"
            imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
            print(f"视频已保存至 {video_path}")

    # 计算并打印统计信息
    success_count = sum(1 for info in infos if info.get("success", False))
    success_rate = success_count / num_evaluations
    infos.insert(
        1,  # 插入到索引1的位置（配置信息之后，episode信息之前）
        dict(
            success_count=success_count,
            success_rate=success_rate,
        ),
    )
    save_dict_list_to_json(infos, output_directory / "info.json")
    print(f"\n{'='*50}")
    print(f"评估完成!")
    print(f"{'='*50}")
    print(f"总评估次数: {num_evaluations}")
    print(f"成功次数: {success_count}")
    print(f"成功率: {success_rate * 100:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    tyro.cli(evaluate)
