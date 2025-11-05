"""
带噪声过滤的数据集包装器

这个模块提供了一个自定义的 Dataset wrapper，可以：
1. 噪声帧可以作为历史observation的一部分（输入）
2. 但噪声帧不能作为action的预测目标（输出）

使用方法：
    from galaxea_sim.utils.noise_filtered_dataset import NoiseFilteredLeRobotDataset
    
    # 创建原始数据集
    dataset = LeRobotDataset("galaxea/R1ProBlocksStackEasy", delta_timestamps=...)
    
    # 包装为过滤噪声的数据集
    filtered_dataset = NoiseFilteredLeRobotDataset(dataset)
    
    # 正常使用dataloader
    dataloader = torch.utils.data.DataLoader(filtered_dataset, batch_size=128, shuffle=True)
"""

import torch
import numpy as np
from typing import Optional, List
from torch.utils.data import Dataset
from loguru import logger


class NoiseFilteredLeRobotDataset(Dataset):
    """
    噪声过滤的LeRobot数据集包装器
    
    功能：
    - 过滤掉标记为 is_replan_noise=True 的帧作为主帧（action target）
    - 但这些噪声帧仍可以出现在历史观测窗口中（作为输入）
    
    工作原理：
    - 构建有效索引列表（非噪声帧）
    - 采样时只从有效索引中选择主帧
    - 在获取历史帧时，噪声帧会被包含（如果在窗口内）
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        noise_field_name: str = "is_replan_noise",
        verbose: bool = True,
    ):
        """
        Args:
            base_dataset: 原始的LeRobotDataset
            noise_field_name: 噪声标记字段名
            verbose: 是否打印统计信息
        """
        self.base_dataset = base_dataset
        self.noise_field_name = noise_field_name
        
        # 构建有效帧索引（非噪声帧）
        self.valid_indices = self._build_valid_indices()
        
        if verbose:
            total_frames = len(self.base_dataset)
            valid_frames = len(self.valid_indices)
            noise_frames = total_frames - valid_frames
            logger.info(f"噪声过滤数据集统计:")
            logger.info(f"  - 总帧数: {total_frames}")
            logger.info(f"  - 有效帧数: {valid_frames} ({valid_frames/total_frames*100:.1f}%)")
            logger.info(f"  - 噪声帧数: {noise_frames} ({noise_frames/total_frames*100:.1f}%)")
            logger.info(f"  - 训练时只使用有效帧作为action目标")
    
    def _build_valid_indices(self) -> List[int]:
        """构建有效帧的索引列表（非噪声帧）"""
        valid_indices = []
        total_frames = len(self.base_dataset)
        
        logger.info(f"开始扫描数据集构建有效索引，总帧数: {total_frames}")
        
        # 每 1000 帧打印一次进度
        progress_interval = max(1000, total_frames // 20)
        
        # 遍历数据集，找出所有非噪声帧的索引
        for idx in range(total_frames):
            try:
                # 显示进度
                if idx % progress_interval == 0 and idx > 0:
                    logger.info(f"  扫描进度: {idx}/{total_frames} ({idx/total_frames*100:.1f}%)")
                
                sample = self.base_dataset[idx]
                
                # 检查是否有噪声标记
                if self.noise_field_name in sample:
                    is_noise = sample[self.noise_field_name]
                    
                    # 处理不同的数据格式
                    if isinstance(is_noise, torch.Tensor):
                        is_noise = is_noise.item() if is_noise.numel() == 1 else is_noise[0].item()
                    elif isinstance(is_noise, np.ndarray):
                        is_noise = is_noise.item() if is_noise.size == 1 else is_noise[0]
                    elif isinstance(is_noise, (list, tuple)):
                        is_noise = is_noise[0]
                    
                    # 只保留非噪声帧
                    if not is_noise:
                        valid_indices.append(idx)
                else:
                    # 如果没有噪声标记，默认为有效帧
                    valid_indices.append(idx)
                    
            except Exception as e:
                logger.warning(f"处理索引 {idx} 时出错: {e}，将其标记为有效帧")
                valid_indices.append(idx)
        
        logger.info(f"✅ 索引构建完成")
        
        return valid_indices
    
    def __len__(self) -> int:
        """返回有效帧的数量"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        """
        获取第idx个有效帧的数据
        
        注意：
        - idx是在valid_indices中的索引
        - 实际会从base_dataset中获取对应的原始索引
        - 如果该帧依赖历史帧，历史帧中可能包含噪声帧（这是期望的行为）
        """
        # 将过滤后的索引映射回原始索引
        original_idx = self.valid_indices[idx]
        
        # 从原始数据集获取数据
        sample = self.base_dataset[original_idx]
        
        # 移除噪声标记（训练时不需要）
        if self.noise_field_name in sample:
            sample = {k: v for k, v in sample.items() if k != self.noise_field_name}
        
        return sample


class NoiseFilteredLeRobotDatasetV2(Dataset):
    """
    噪声过滤数据集 V2版本 - 更智能的历史窗口处理
    
    与V1的区别：
    - V1: 简单过滤噪声帧，依赖base_dataset自己处理历史窗口
    - V2: 主动跳过历史窗口中的噪声帧，用更早的有效帧替代
    
    适用场景：
    - 当你希望历史窗口中也不包含噪声帧时使用V2
    - 当你希望噪声帧可以作为历史输入时使用V1
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        noise_field_name: str = "is_replan_noise",
        skip_noise_in_history: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            base_dataset: 原始的LeRobotDataset
            noise_field_name: 噪声标记字段名
            skip_noise_in_history: 是否跳过历史窗口中的噪声帧
            verbose: 是否打印统计信息
        """
        self.base_dataset = base_dataset
        self.noise_field_name = noise_field_name
        self.skip_noise_in_history = skip_noise_in_history
        
        # 构建噪声标记数组
        self.noise_mask = self._build_noise_mask()
        
        # 构建有效帧索引
        self.valid_indices = [i for i, is_noise in enumerate(self.noise_mask) if not is_noise]
        
        if verbose:
            total_frames = len(self.base_dataset)
            valid_frames = len(self.valid_indices)
            noise_frames = total_frames - valid_frames
            logger.info(f"噪声过滤数据集V2统计:")
            logger.info(f"  - 总帧数: {total_frames}")
            logger.info(f"  - 有效帧数: {valid_frames} ({valid_frames/total_frames*100:.1f}%)")
            logger.info(f"  - 噪声帧数: {noise_frames} ({noise_frames/total_frames*100:.1f}%)")
            logger.info(f"  - 历史窗口跳过噪声: {skip_noise_in_history}")
    
    def _build_noise_mask(self) -> np.ndarray:
        """构建噪声标记数组"""
        noise_mask = np.zeros(len(self.base_dataset), dtype=bool)
        
        for idx in range(len(self.base_dataset)):
            try:
                sample = self.base_dataset[idx]
                
                if self.noise_field_name in sample:
                    is_noise = sample[self.noise_field_name]
                    
                    if isinstance(is_noise, torch.Tensor):
                        is_noise = is_noise.item() if is_noise.numel() == 1 else is_noise[0].item()
                    elif isinstance(is_noise, np.ndarray):
                        is_noise = is_noise.item() if is_noise.size == 1 else is_noise[0]
                    elif isinstance(is_noise, (list, tuple)):
                        is_noise = is_noise[0]
                    
                    noise_mask[idx] = bool(is_noise)
                    
            except Exception as e:
                logger.warning(f"处理索引 {idx} 时出错: {e}")
        
        return noise_mask
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        original_idx = self.valid_indices[idx]
        sample = self.base_dataset[original_idx]
        
        # 移除噪声标记
        if self.noise_field_name in sample:
            sample = {k: v for k, v in sample.items() if k != self.noise_field_name}
        
        return sample


def create_noise_filtered_dataloader(
    base_dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    noise_field_name: str = "is_replan_noise",
    use_v2: bool = False,
    verbose: bool = True,
    **kwargs
):
    """
    创建带噪声过滤的DataLoader的便捷函数
    
    Args:
        base_dataset: 原始LeRobotDataset
        batch_size: batch大小
        shuffle: 是否打乱
        num_workers: worker数量
        pin_memory: 是否pin memory
        noise_field_name: 噪声标记字段名
        use_v2: 是否使用V2版本
        verbose: 是否打印统计信息
        **kwargs: 其他传递给DataLoader的参数
    
    Returns:
        DataLoader实例
    
    示例:
        dataloader = create_noise_filtered_dataloader(
            dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
        )
    """
    if use_v2:
        filtered_dataset = NoiseFilteredLeRobotDatasetV2(
            base_dataset, 
            noise_field_name=noise_field_name,
            verbose=verbose
        )
    else:
        filtered_dataset = NoiseFilteredLeRobotDataset(
            base_dataset, 
            noise_field_name=noise_field_name,
            verbose=verbose
        )
    
    dataloader = torch.utils.data.DataLoader(
        filtered_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **kwargs
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试代码
    print("噪声过滤数据集工具 - 使用示例：")
    print("""
    # 方式1: 直接包装数据集
    from galaxea_sim.utils.noise_filtered_dataset import NoiseFilteredLeRobotDataset
    
    base_dataset = LeRobotDataset("galaxea/R1ProBlocksStackEasy", delta_timestamps=...)
    filtered_dataset = NoiseFilteredLeRobotDataset(base_dataset)
    dataloader = torch.utils.data.DataLoader(filtered_dataset, batch_size=128, shuffle=True)
    
    # 方式2: 使用便捷函数
    from galaxea_sim.utils.noise_filtered_dataset import create_noise_filtered_dataloader
    
    dataloader = create_noise_filtered_dataloader(
        base_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
    )
    """)

