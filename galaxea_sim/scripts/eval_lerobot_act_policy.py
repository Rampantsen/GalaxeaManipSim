from pathlib import Path
import gymnasium as gym
import imageio
import numpy as np
import pickle
import torch
import tyro
import cv2

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from galaxea_sim.utils.data_utils import save_dict_list_to_json

import numpy as np
import time


def evaluate(
    task: str,
    pretrained_policy_path: str = "/home/sen/workspace/galaxea/GalaxeaManipSim/outputs/ACT/R1ProBlocksStackEasy-traj_aug/all-20251010214500/checkpoints/last/pretrained_model",
    dataset_repo_id: str = "galaxea/R1ProBlocksStackEasy-traj_aug/all",
    target_controller_type: str = "bimanual_relaxed_ik",
    device: str = "cuda",
    headless: bool = True,
    num_evaluations: int = 100,
    temporal_ensemble: bool = True,  # ACT 特有的时序集成
    save_video: bool = True,
):
    """在模拟环境中多次评估预训练的 ACT 策略。"""
    output_directory = (
        Path(pretrained_policy_path).parent.parent
        / "evaluations"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    # 加载数据集元数据和策略
    # 检查是否存在旧格式的 dataset_metadata.pkl
    # metadata_pkl_path = Path(pretrained_policy_path) / "dataset_metadata.pkl"
    # if metadata_pkl_path.exists():
    #     with open(metadata_pkl_path, "rb") as f:
    #         dataset_metadata: LeRobotDatasetMetadata = pickle.load(f)
    #     dataset_stats = dataset_metadata.stats
    # else:
    #     # 如果没有 pkl 文件，从数据集repo加载元数据
    #     print(f"未找到 dataset_metadata.pkl，从数据集 {dataset_repo_id} 加载元数据...")
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
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(
            temporal_ensemble_coeff=0.1, chunk_size=policy.config.chunk_size
        )

    infos = []
    env = gym.make(
        task,
        control_freq=15,  # 必须与训练数据的 fps 匹配！
        headless=headless,
        max_episode_steps=600,
        controller_type=target_controller_type,
    )

    for eval_idx in range(num_evaluations):
        print(f"开始评估 {eval_idx + 1}/{num_evaluations}")

        policy.reset()
        numpy_observation, info = env.reset(seed=42)
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

        print("成功!" if terminated else "失败!")
        print(f"总步数: {step}, 总奖励: {sum(rewards):.2f}")
        infos.append(info)
        save_dict_list_to_json(infos, output_directory / "info.json")

        if save_video:
            fps = env.unwrapped.control_freq
            video_path = output_directory / f"rollout_{eval_idx + 1}.mp4"
            imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
            print(f"视频已保存至 {video_path}")

    # 计算并打印统计信息
    success_count = sum(1 for info in infos if info.get("success", False))
    success_rate = success_count / num_evaluations
    print(f"\n评估完成:")
    print(f"总评估次数: {num_evaluations}")
    print(f"成功次数: {success_count}")
    print(f"成功率: {success_rate * 100:.2f}%")


if __name__ == "__main__":
    tyro.cli(evaluate)
