"""
使用噪声过滤的 Diffusion Policy 训练脚本

相比普通训练脚本的改动：
1. 使用 NoiseFilteredLeRobotDataset 包装原始数据集
2. 噪声帧可以作为历史observation输入，但不作为action预测目标
3. 这样可以充分利用数据，同时避免学习噪声动作
"""

from pathlib import Path
import datetime

import torch
import tyro
import tqdm
import pickle
from typing import Literal, Optional
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType

from loguru import logger

# 导入噪声过滤数据集工具
from galaxea_sim.utils.noise_filtered_dataset import create_noise_filtered_dataloader


def main(
    task: str,
    output_dir: str = "outputs/DiffusionPolicy",
    batch_size: int = 128,
    num_epochs: int = 300,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
    # 噪声过滤相关参数
    filter_noise: bool = True,  # 是否过滤噪声帧
    noise_field_name: str = "is_replan_noise",  # 噪声标记字段名
    # Diffusion Policy配置参数
    n_obs_steps: int = 8,  # 历史观测帧数：使用过去8帧作为输入
    drop_n_last_frames: int = 8,
    # 数据集特征类型（可选，如果提供则添加到数据集名称后面）
    feature: Optional[str] = None,
):
    # 创建输出目录
    exp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if feature:
        output_directory = Path(output_dir) / task / feature / exp_id
    else:
        output_directory = Path(output_dir) / task / exp_id
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建数据集路径
    # repo_id 必须是两级格式（namespace/repo_name）
    # 使用 root 参数指定本地子目录
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
    
    dataset_repo_id = f"galaxea/{task}"  # 两级格式
    if feature:
        # 数据在子目录中
        dataset_root = HF_LEROBOT_HOME / "galaxea" / task / feature
    else:
        dataset_root = HF_LEROBOT_HOME / "galaxea" / task
    
    logger.info(f"加载数据集: {dataset_repo_id}")
    logger.info(f"数据集目录: {dataset_root}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # 配置Diffusion Policy策略
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        crop_shape=(224, 224),
        crop_is_random=False,
        use_separate_rgb_encoder_per_camera=True,
        optimizer_lr=learning_rate,
        n_obs_steps=n_obs_steps,
        drop_n_last_frames=drop_n_last_frames,
    )
    
    # 创建策略
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # 配置delta_timestamps
    delta_timestamps = {
        "observation.images.rgb_head": [0.0],
        "observation.images.rgb_left_hand": [0.0],
        "observation.images.rgb_right_hand": [0.0],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    
    # 创建原始数据集（指定root从本地加载）
    base_dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps
    )
    
    logger.info(f"原始数据集大小: {len(base_dataset)} 帧")
    
    # 根据filter_noise参数决定是否使用噪声过滤
    if filter_noise:
        logger.info("✅ 使用噪声过滤数据集（推荐）")
        logger.info("   - 噪声帧可以作为历史observation输入")
        logger.info("   - 但噪声帧不会作为action预测目标")
        
        dataloader = create_noise_filtered_dataloader(
            base_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type != "cpu",
            noise_field_name=noise_field_name,
        )
    else:
        logger.info("⚠️  未使用噪声过滤（可能会学习到噪声动作）")
        
        dataloader = torch.utils.data.DataLoader(
            base_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=device.type != "cpu",
            drop_last=True,
        )
    
    # 计算训练步数
    training_steps = num_epochs * len(dataloader)
    log_freq = 50
    save_freq = 1000
    
    logger.info(f"训练配置:")
    logger.info(f"  - 任务: {task}")
    logger.info(f"  - 数据集: {dataset_repo_id}")
    logger.info(f"  - 特征类型: {feature if feature else 'default'}")
    logger.info(f"  - 总epoch数: {num_epochs}")
    logger.info(f"  - 每epoch步数: {len(dataloader)}")
    logger.info(f"  - 总训练步数: {training_steps}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"Diffusion Policy配置:")
    logger.info(f"  - Observation steps: {n_obs_steps}")
    logger.info(f"  - Drop last frames: {drop_n_last_frames}")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)

    # 训练循环
    step = 0
    done = False
    
    logger.info("开始训练...")
    
    while not done:
        for batch in dataloader:
            batch.pop("task", None)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if step % log_freq == 0:
                logger.info(
                    f"Step: {step}/{training_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                    f"Progress: {step / training_steps * 100:.1f}%"
                )
            
            if step % save_freq == 0 and step > 0:
                checkpoint_dir = output_directory / f"checkpoint-{step}"
                policy.save_pretrained(checkpoint_dir)
                with open(checkpoint_dir / "dataset_metadata.pkl", "wb") as f:
                    pickle.dump(dataset_metadata, f)
                logger.info(f"💾 保存checkpoint: {checkpoint_dir}")
            
            step += 1
            if step >= training_steps:
                done = True
                break
    
    # 保存最终模型
    final_checkpoint = output_directory / "checkpoint-final"
    policy.save_pretrained(final_checkpoint)
    with open(final_checkpoint / "dataset_metadata.pkl", "wb") as f:
        pickle.dump(dataset_metadata, f)
    
    logger.info(f"✅ 训练完成！最终模型保存至: {final_checkpoint}")


if __name__ == "__main__":
    tyro.cli(main)