"""
Diffusion Policy 带噪声过滤的数据集

功能：
1. 保留噪声标签的数据
2. 根据filter_noise参数决定训练行为：
   - filter_noise=True: 噪声帧只作为obs历史，不作为action目标
   - filter_noise=False: 正常训练，使用所有数据

使用方法：
    from galaxea_sim.utils.dp_noise_filtered_dataset import GalaxeaImageDataset
    
    dataset = GalaxeaImageDataset(
        zarr_path='datasets_diffusion_policy/R1ProBlocksStackEasy_with_noise.zarr',
        filter_noise=True,  # 是否过滤噪声action
        horizon=16,
    )
"""

from typing import Dict, Optional, List
import torch
import numpy as np
import copy
import zarr
from pathlib import Path
from loguru import logger

# 导入Diffusion Policy的基础类
try:
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.common.replay_buffer import ReplayBuffer
    from diffusion_policy.common.sampler import (
        SequenceSampler, get_val_mask, downsample_mask)
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.dataset.base_dataset import BaseImageDataset
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
except ImportError:
    logger.warning("Diffusion Policy未安装，请先安装: pip install -e policy/dp")
    BaseImageDataset = object


class NoiseAwareSequenceSampler:
    """
    噪声感知的序列采样器
    
    在filter_noise=True时：
    - 只从非噪声帧中采样主帧（action target）
    - 但历史窗口可以包含噪声帧（作为observation）
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
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.episode_mask = episode_mask
        self.filter_noise = filter_noise
        self.noise_labels = noise_labels
        
        # 构建索引
        self._build_indices()
    
    def _build_indices(self):
        """构建可采样的索引"""
        episode_ends = self.replay_buffer.episode_ends
        n_episodes = len(episode_ends)
        
        # 如果有episode mask，只使用指定的episodes
        if self.episode_mask is not None:
            episode_idxs = np.where(self.episode_mask)[0]
        else:
            episode_idxs = np.arange(n_episodes)
        
        # 收集所有有效的索引
        indices = []
        for ep_idx in episode_idxs:
            start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
            end_idx = episode_ends[ep_idx]
            
            # 确保序列不会跨越episode边界
            for idx in range(start_idx + self.pad_before, 
                           end_idx - self.sequence_length - self.pad_after + 1):
                # 如果需要过滤噪声，检查主帧是否为噪声
                if self.filter_noise and self.noise_labels is not None:
                    # 主帧是序列的最后一帧（action target）
                    main_frame_idx = idx + self.sequence_length - 1
                    if self.noise_labels[main_frame_idx]:
                        continue  # 跳过噪声帧作为action target
                
                indices.append(idx)
        
        self.indices = np.array(indices, dtype=np.int64)
        logger.info(f"序列采样器: 总共 {len(self.indices)} 个有效序列")
        
        if self.filter_noise and self.noise_labels is not None:
            # 统计信息
            total_possible = sum(episode_ends[i] - (0 if i == 0 else episode_ends[i-1]) 
                               - self.sequence_length - self.pad_before - self.pad_after + 1
                               for i in episode_idxs if (episode_ends[i] - (0 if i == 0 else episode_ends[i-1])) 
                               >= self.sequence_length + self.pad_before + self.pad_after)
            filtered_count = total_possible - len(self.indices)
            logger.info(f"  - 过滤掉 {filtered_count} 个噪声序列 ({filtered_count/total_possible*100:.1f}%)")
    
    def __len__(self):
        return len(self.indices)
    
    def sample_sequence(self, idx: int) -> Dict[str, np.ndarray]:
        """采样一个序列"""
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for {len(self.indices)} sequences")
        
        start_idx = self.indices[idx] - self.pad_before
        end_idx = self.indices[idx] + self.sequence_length + self.pad_after
        
        # 获取数据
        result = {}
        for key in self.replay_buffer.keys():
            result[key] = self.replay_buffer[key][start_idx:end_idx]
        
        # 如果有噪声标签，也包含进去（用于调试）
        if self.noise_labels is not None:
            result['is_noise'] = self.noise_labels[start_idx:end_idx]
        
        return result


class GalaxeaImageDataset(BaseImageDataset):
    """
    Galaxea的Diffusion Policy图像数据集
    支持噪声过滤功能和多相机输入
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
        available_keys = list(zarr_root['data'].keys())
        
        # 确保是多相机数据集
        if 'img_head' not in available_keys:
            raise ValueError(f"数据集必须包含多相机数据 (img_head, img_left, img_right)，当前键: {available_keys}")
        
        # 加载三个相机的数据
        logger.info("加载多相机数据集")
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img_head', 'img_left', 'img_right', 'state', 'action'])
        
        # 加载噪声标签（如果存在）
        self.noise_labels = None
        if 'data/is_replan_noise' in zarr_root:
            self.noise_labels = zarr_root['data/is_replan_noise'][:]
            noise_count = np.sum(self.noise_labels)
            total_count = len(self.noise_labels)
            logger.info(f"发现噪声标签: {noise_count}/{total_count} 帧为噪声 ({noise_count/total_count*100:.1f}%)")
            logger.info(f"噪声过滤: {'开启' if filter_noise else '关闭'}")
        else:
            logger.info("数据集中没有噪声标签")
            if filter_noise:
                logger.warning("filter_noise=True 但数据集中没有噪声标签，将使用所有数据")
        
        # 划分训练/验证集
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)
        
        # 创建采样器
        self.sampler = NoiseAwareSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            filter_noise=filter_noise and self.noise_labels is not None,
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
            filter_noise=self.filter_noise and self.noise_labels is not None,
            noise_labels=self.noise_labels
        )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        """获取数据归一化器"""
        data = {
            'action': self.replay_buffer['action'],
            'state': self.replay_buffer['state']
        }
        
        # 如果过滤噪声，只使用非噪声帧计算统计量
        if self.filter_noise and self.noise_labels is not None:
            valid_mask = ~self.noise_labels
            data = {
                'action': self.replay_buffer['action'][valid_mask],
                'state': self.replay_buffer['state'][valid_mask]
            }
            logger.info(f"使用 {np.sum(valid_mask)} 个非噪声帧计算归一化统计")
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # 为每个相机设置normalizer
        normalizer['img_head'] = get_image_range_normalizer()
        normalizer['img_left'] = get_image_range_normalizer()
        normalizer['img_right'] = get_image_range_normalizer()
        
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        """获取所有动作（用于某些算法）"""
        actions = self.replay_buffer['action']
        
        # 如果过滤噪声，只返回非噪声帧的动作
        if self.filter_noise and self.noise_labels is not None:
            valid_mask = ~self.noise_labels
            actions = actions[valid_mask]
        
        return torch.from_numpy(actions)
    
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
