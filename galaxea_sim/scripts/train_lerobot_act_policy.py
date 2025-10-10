from pathlib import Path
import datetime

import torch
import tyro
import tqdm
import pickle

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

from loguru import logger


def main(task: str, feature: str):
    # 创建目录来存储训练检查点
    exp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = Path(f"outputs/train/{task}/{feature}/act/{exp_id}")
    output_directory.mkdir(parents=True, exist_ok=True)

    # 选择设备
    device = torch.device("cuda")

    # 从数据集元数据中获取输入输出特征
    dataset_name = f"{task}/{feature}"
    dataset_metadata = LeRobotDatasetMetadata(f"galaxea/{dataset_name}")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    # 使用 ACTConfig 初始化策略配置
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        # 训练参数
        optimizer_lr=1e-5,
        optimizer_weight_decay=1e-4,
        # 观察和动作参数
        n_obs_steps=1,
        chunk_size=30,  # ACT 预测的动作序列长度
        n_action_steps=30,  # 每次推理使用的动作步数
        # Transformer 架构参数
        dim_model=512,  # Transformer 主隐藏层维度
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=1,  # 原始实现虽然是7层，但因bug只使用第一层
        n_heads=8,
        pre_norm=False,
        feedforward_activation="relu",
        # VAE 参数
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=4,
        # 训练参数
        dropout=0.1,
        kl_weight=10.0,
        # 视觉编码器参数
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
    )

    # 使用配置和数据集统计信息实例化策略
    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # 为每个策略配置 delta_timestamps
    # ACT 期望的输入形状（见 modeling_act.py 第 396-404 行）：
    # - observation.state: (B, state_dim) - 无时间维度
    # - observation.images: (B, n_cameras, C, H, W) - 无时间维度
    # - action: (B, chunk_size, action_dim) - 有时间维度
    # 所以只为 action 设置 delta_timestamps，其他特征保持原始形状
    delta_timestamps = {
        # 图像和状态不设置 delta_timestamps，保持 (B, ...) 的形状
        "action": [
            i / dataset_metadata.fps for i in cfg.action_delta_indices
        ],  # chunk_size 个动作
    }

    # 使用 delta_timestamps 配置实例化数据集
    dataset = LeRobotDataset(
        f"galaxea/{dataset_name}", delta_timestamps=delta_timestamps
    )
    # num_epochs = 1000
    batch_size = 16  # ACT 通常使用较小的 batch size
    # training_steps = num_epochs * len(dataset) // batch_size
    training_steps = 100000
    log_freq = 50
    save_freq = 5000

    print(f"训练 {training_steps} 步。")
    print(f"每 {log_freq} 步记录一次。")

    # 创建优化器和数据加载器
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer_lr)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # 运行训练循环
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch.pop("task", None)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                # 记录总损失和各个组件的损失
                kld_loss = loss_dict.get("kld_loss", 0.0)
                l1_loss = loss_dict.get("l1_loss", 0.0)
                logger.info(
                    f"step: {step} loss: {loss.item():.3f} kld_loss: {kld_loss:.3f} l1_loss: {l1_loss:.3f} "
                    f"lr: {optimizer.param_groups[0]['lr']:.6f} progress: {step / training_steps * 100:.2f}%"
                )
            if step % save_freq == 0:
                # 保存策略检查点
                output_directory.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(output_directory / f"checkpoint-{step}")
                with open(
                    output_directory / f"checkpoint-{step}/dataset_metadata.pkl", "wb"
                ) as f:
                    pickle.dump(dataset_metadata, f)
            step += 1
            if step >= training_steps:
                done = True
                break

    # 保存最终策略检查点
    policy.save_pretrained(output_directory / "checkpoint-final")
    with open(output_directory / f"checkpoint-final/dataset_metadata.pkl", "wb") as f:
        pickle.dump(dataset_metadata, f)


if __name__ == "__main__":
    tyro.cli(main)
