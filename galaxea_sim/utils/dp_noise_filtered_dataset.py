"""
Diffusion Policy的噪声过滤数据集 (修复版)
直接继承原版的SequenceSampler，保证采样逻辑正确
"""

import copy
import numpy as np
import torch
import zarr
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

# 添加policy/dp到path
import sys
workspace_root = Path(__file__).parent.parent.parent
dp_path = workspace_root / "policy" / "dp"
if str(dp_path) not in sys.path:
    sys.path.insert(0, str(dp_path))

try:
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.common.replay_buffer import ReplayBuffer
    from diffusion_policy.common.sampler import (
        SequenceSampler, get_val_mask, downsample_mask)
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.dataset.base_dataset import BaseImageDataset
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
except ImportError as e:
    logger.error(f"导入Diffusion Policy模块失败: {e}")
    logger.error(f"请确保已安装Diffusion Policy: pip install -e policy/dp")
    raise


class NoiseAwareSequenceSampler(SequenceSampler):
    """
    支持噪声过滤的序列采样器
    继承自原版SequenceSampler，在索引构建完成后过滤噪声序列
    """
    
    def __init__(
        self,
        replay_buffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        episode_mask: Optional[np.ndarray] = None,
        filter_noise: bool = False,
        noise_labels: Optional[np.ndarray] = None
    ):
        # 先调用父类初始化（构建所有索引）
        super().__init__(
            replay_buffer=replay_buffer,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=episode_mask
        )
        
        # 如果需要过滤噪声，额外过滤
        if filter_noise and noise_labels is not None:
            self._filter_noise_indices(noise_labels)
    
    def _filter_noise_indices(self, noise_labels: np.ndarray):
        """过滤包含噪声的序列"""
        original_count = len(self.indices)
        filtered_indices = []
        
        for i in range(len(self.indices)):
            # 原版indices格式：[buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            buffer_start, buffer_end, _, _ = self.indices[i]
            
            # 检查序列中是否包含噪声帧
            has_noise = np.any(noise_labels[buffer_start:buffer_end])
            if not has_noise:
                filtered_indices.append(self.indices[i])
        
        self.indices = np.array(filtered_indices, dtype=self.indices.dtype)
        filtered_count = original_count - len(self.indices)
        
        if filtered_count > 0:
            logger.info(f"  - 过滤掉 {filtered_count} 个噪声序列 ({filtered_count/original_count*100:.1f}%)")


class GalaxeaImageDataset(BaseImageDataset):
    """
    Galaxea的Diffusion Policy图像数据集
    支持噪声过滤功能和三相机输入
    """
    
    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        filter_noise: bool = False,
    ):
        """
        Args:
            zarr_path: Zarr数据集路径
            horizon: 序列长度（预测的时间步数）
            pad_before: 序列前的padding
            pad_after: 序列后的padding
            seed: 随机种子
            val_ratio: 验证集比例
            max_train_episodes: 最大训练episodes数
            filter_noise: 是否过滤噪声帧作为action目标
        """
        super().__init__()
        
        self.zarr_path = Path(zarr_path)
        self.filter_noise = filter_noise
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        # 加载数据
        logger.info(f"加载数据集: {zarr_path}")
        
        # 检查数据集中可用的键
        zarr_root = zarr.open(str(self.zarr_path), 'r')
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
        
        # 加载噪声标签（如果存在）
        self.noise_labels = None
        if 'data/is_replan_noise' in zarr_root:
            self.noise_labels = zarr_root['data/is_replan_noise'][:]
            noise_count = np.sum(self.noise_labels)
            total_count = len(self.noise_labels)
            logger.info(f"发现噪声标签: {noise_count}/{total_count} 帧为噪声 ({noise_count/total_count*100:.1f}%)")
            logger.info(f"噪声过滤: {'开启' if filter_noise else '关闭'}")
        
        # 划分训练集和验证集
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )
        
        # 创建采样器（使用噪声感知版本）
        self.sampler = NoiseAwareSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            filter_noise=filter_noise,
            noise_labels=self.noise_labels
        )
        
        self.train_mask = train_mask
        
        logger.info(f"数据集初始化完成:")
        logger.info(f"  - Episodes: {self.replay_buffer.n_episodes}")
        logger.info(f"  - 序列数: {len(self.sampler)}")
        logger.info(f"  - Horizon: {horizon}")
        # 多相机：显示每个相机的形状
        logger.info(f"  - 头部相机形状: {self.replay_buffer['img_head'].shape[1:]}")
        logger.info(f"  - 左手相机形状: {self.replay_buffer['img_left'].shape[1:]}")
        logger.info(f"  - 右手相机形状: {self.replay_buffer['img_right'].shape[1:]}")
        logger.info(f"  - 状态维度: {self.replay_buffer['state'].shape[-1]}")
        logger.info(f"  - 动作维度: {self.replay_buffer['action'].shape[-1]}")
    
    def get_validation_dataset(self):
        """获取验证集"""
        val_set = copy.copy(self)
        val_set.sampler = NoiseAwareSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            filter_noise=self.filter_noise,
            noise_labels=self.noise_labels
        )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        """
        获取数据归一化器
        参考原版实现：直接传递zarr数组或先读取后过滤
        """
        if self.filter_noise and self.noise_labels is not None:
            # 需要过滤噪声：先读取数据到内存再过滤
            # 注意：这里只读取state和action（~2.5MB），不读取图像（10GB）
            logger.info("读取state和action数据用于归一化计算...")
            all_actions = self.replay_buffer['action'][:]
            all_states = self.replay_buffer['state'][:]
            
            valid_mask = ~self.noise_labels
            data = {
                'action': all_actions[valid_mask],
                'state': all_states[valid_mask]
            }
            logger.info(f"使用 {np.sum(valid_mask)} 个非噪声帧计算归一化统计")
        else:
            # 不过滤噪声：直接传递zarr数组引用（参考原版）
            data = {
                'action': self.replay_buffer['action'],  # zarr数组引用
                'state': self.replay_buffer['state']     # zarr数组引用
            }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # 为每个相机设置normalizer
        normalizer['img_head'] = get_image_range_normalizer()
        normalizer['img_left'] = get_image_range_normalizer()
        normalizer['img_right'] = get_image_range_normalizer()
        
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        """获取所有动作（用于某些算法）"""
        # 先读取所有动作到内存
        all_actions = self.replay_buffer['action'][:]
        
        # 如果过滤噪声，只返回非噪声帧的动作
        if self.filter_noise and self.noise_labels is not None:
            valid_mask = ~self.noise_labels
            all_actions = all_actions[valid_mask]
        
        return torch.from_numpy(all_actions)
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个训练样本
        
        Returns:
            包含以下键的字典:
            - obs: 包含图像和state的观测字典
            - action: 动作序列
        """
        # 使用父类的sample_sequence（保证返回固定长度）
        sample = self.sampler.sample_sequence(idx)
        
        # 处理三个相机的图像
        img_head = np.moveaxis(sample['img_head'], -1, 1) / 255.0
        img_left = np.moveaxis(sample['img_left'], -1, 1) / 255.0
        img_right = np.moveaxis(sample['img_right'], -1, 1) / 255.0
        
        # 每个相机作为obs中的单独键
        data = {
            'obs': {
                'img_head': img_head.astype(np.float32),  # T, 3, H, W
                'img_left': img_left.astype(np.float32),  # T, 3, H, W
                'img_right': img_right.astype(np.float32),  # T, 3, H, W
                'state': sample['state'].astype(np.float32),  # T, state_dim
            },
            'action': sample['action'].astype(np.float32),  # T, action_dim
        }
        
        # 转换为torch tensors
        torch_data = dict_apply(data, torch.from_numpy)
        
        return torch_data

