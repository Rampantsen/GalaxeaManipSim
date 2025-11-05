"""
原版ACT和Diffusion Policy的噪声过滤数据集适配器

功能：
1. 从HDF5文件加载数据
2. 噪声帧可以作为历史observation输入
3. 但噪声帧不会作为action预测目标
4. 与原版ACT/DP格式兼容
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from torch.utils.data import Dataset
from loguru import logger


class NoiseFilteredH5Dataset(Dataset):
    """
    支持噪声过滤的HDF5数据集
    
    适用于原版ACT和Diffusion Policy算法
    
    参数：
        dataset_dir: 包含.h5文件的目录
        n_obs_steps: 历史观测步数
        n_action_steps: 未来action步数
        filter_noise: 是否过滤噪声帧
        noise_field_name: 噪声标记字段名
    """
    
    def __init__(
        self,
        dataset_dir: str,
        n_obs_steps: int = 1,
        n_action_steps: int = 1,
        filter_noise: bool = True,
        noise_field_name: str = "is_replan_noise",
        max_episodes: Optional[int] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.filter_noise = filter_noise
        self.noise_field_name = noise_field_name
        
        # 加载所有episode
        self.episodes = self._load_episodes(max_episodes)
        
        # 构建有效样本索引
        self.valid_indices = self._build_valid_indices()
        
        logger.info(f"数据集加载完成:")
        logger.info(f"  - Episode数量: {len(self.episodes)}")
        logger.info(f"  - 总样本数: {len(self.valid_indices)}")
        logger.info(f"  - 观测步数: {n_obs_steps}")
        logger.info(f"  - Action步数: {n_action_steps}")
        logger.info(f"  - 噪声过滤: {filter_noise}")
    
    def _load_episodes(self, max_episodes: Optional[int]) -> List[Dict]:
        """加载所有episode数据"""
        episodes = []
        h5_files = sorted(self.dataset_dir.glob("demo_*.h5"))
        
        if max_episodes:
            h5_files = h5_files[:max_episodes]
        
        logger.info(f"加载 {len(h5_files)} 个episode...")
        
        for h5_path in h5_files:
            with h5py.File(h5_path, 'r') as f:
                # 读取数据
                episode_data = {
                    'qpos': f['upper_body_observations']['qpos'][:],
                    'action': f['upper_body_action_dict']['joint_position_cmd'][:],
                    'gripper_cmd': np.concatenate([
                        f['upper_body_action_dict']['left_arm_gripper_position_cmd'][:],
                        f['upper_body_action_dict']['right_arm_gripper_position_cmd'][:]
                    ], axis=-1),
                }
                
                # 如果有图像
                if 'rgb_head' in f['upper_body_observations']:
                    episode_data['images'] = {
                        'rgb_head': f['upper_body_observations']['rgb_head'][:],
                        'rgb_left_hand': f['upper_body_observations']['rgb_left_hand'][:],
                        'rgb_right_hand': f['upper_body_observations']['rgb_right_hand'][:],
                    }
                
                # 读取噪声标记
                if self.noise_field_name in f:
                    episode_data['is_noise'] = f[self.noise_field_name][:]
                else:
                    # 没有标记，默认全部为有效帧
                    episode_data['is_noise'] = np.zeros(len(episode_data['qpos']), dtype=bool)
                
                episodes.append(episode_data)
        
        return episodes
    
    def _build_valid_indices(self) -> List[Tuple[int, int]]:
        """
        构建有效样本索引
        
        返回: [(episode_idx, frame_idx), ...]
        只有非噪声帧才能作为主帧（预测target）
        """
        valid_indices = []
        total_frames = 0
        noise_frames = 0
        
        for ep_idx, episode in enumerate(self.episodes):
            ep_len = len(episode['qpos'])
            total_frames += ep_len
            
            for frame_idx in range(ep_len):
                is_noise = episode['is_noise'][frame_idx]
                
                if is_noise:
                    noise_frames += 1
                
                # 检查是否可以作为有效样本
                # 1. 主帧不能是噪声（如果启用过滤）
                # 2. 需要有足够的历史帧（observation）
                # 3. 需要有足够的未来帧（action）
                can_be_sample = True
                
                if self.filter_noise and is_noise:
                    can_be_sample = False
                
                if frame_idx < self.n_obs_steps - 1:
                    can_be_sample = False
                
                if frame_idx + self.n_action_steps > ep_len:
                    can_be_sample = False
                
                if can_be_sample:
                    valid_indices.append((ep_idx, frame_idx))
        
        if self.filter_noise:
            logger.info(f"噪声过滤统计:")
            logger.info(f"  - 总帧数: {total_frames}")
            logger.info(f"  - 噪声帧数: {noise_frames} ({noise_frames/total_frames*100:.1f}%)")
            logger.info(f"  - 有效样本数: {len(valid_indices)}")
        
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        获取一个样本
        
        返回:
            {
                'qpos': (n_obs_steps, qpos_dim),  # 历史观测（可能包含噪声帧）
                'action': (n_action_steps, action_dim),  # 未来动作（不包含噪声）
                'image': (n_obs_steps, H, W, C) if exists,
            }
        """
        ep_idx, frame_idx = self.valid_indices[idx]
        episode = self.episodes[ep_idx]
        
        # 获取历史观测（包括当前帧）
        obs_start = frame_idx - (self.n_obs_steps - 1)
        obs_end = frame_idx + 1
        obs_qpos = episode['qpos'][obs_start:obs_end]  # (n_obs_steps, qpos_dim)
        
        # 获取未来动作
        action_start = frame_idx
        action_end = frame_idx + self.n_action_steps
        action = episode['action'][action_start:action_end]  # (n_action_steps, action_dim)
        gripper = episode['gripper_cmd'][action_start:action_end]
        
        # 拼接action和gripper
        full_action = np.concatenate([action, gripper], axis=-1)
        
        sample = {
            'qpos': obs_qpos.astype(np.float32),
            'action': full_action.astype(np.float32),
        }
        
        # 如果有图像
        if 'images' in episode:
            sample['image'] = np.stack([
                episode['images']['rgb_head'][obs_start:obs_end],
                episode['images']['rgb_left_hand'][obs_start:obs_end],
                episode['images']['rgb_right_hand'][obs_start:obs_end],
            ], axis=1)  # (n_obs_steps, 3_cameras, H, W, C)
        
        return sample


def create_original_policy_dataloader(
    dataset_dir: str,
    n_obs_steps: int = 1,
    n_action_steps: int = 1,
    batch_size: int = 64,
    num_workers: int = 4,
    filter_noise: bool = True,
    **kwargs
):
    """
    创建原版算法兼容的数据加载器
    
    使用示例:
        dataloader = create_original_policy_dataloader(
            dataset_dir="datasets/R1ProBlocksStackEasy/red/all/collected",
            n_obs_steps=2,      # Diffusion Policy推荐
            n_action_steps=8,   # Diffusion Policy推荐
            batch_size=64,
            filter_noise=True
        )
    """
    import torch
    
    dataset = NoiseFilteredH5Dataset(
        dataset_dir=dataset_dir,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        filter_noise=filter_noise,
        **kwargs
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试
    dataloader = create_original_policy_dataloader(
        dataset_dir="datasets/R1ProBlocksStackEasy/red/all/collected",
        n_obs_steps=2,
        n_action_steps=8,
        batch_size=4,
        filter_noise=True,
        max_episodes=5
    )
    
    print("\n测试数据加载:")
    for batch in dataloader:
        print(f"qpos shape: {batch['qpos'].shape}")
        print(f"action shape: {batch['action'].shape}")
        if 'image' in batch:
            print(f"image shape: {batch['image'].shape}")
        break

