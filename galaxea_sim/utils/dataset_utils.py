"""
数据集加载工具

提供带replan噪声标记过滤的数据加载功能
"""

import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger


def load_demo_with_filter(
    demo_path: Path,
    filter_replan_noise: bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    加载一个demo，可选择是否过滤replan噪声数据
    
    Args:
        demo_path: demo文件路径 (*.h5)
        filter_replan_noise: 是否过滤掉标记为噪声的步骤
    
    Returns:
        trajectory: 轨迹数据列表
        stats: 统计信息
    """
    def recursive_read(group, index=None):
        """递归读取HDF5的group或dataset"""
        result = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                # 如果提供了index，读取该timestep的数据
                if index is not None:
                    data = item[index]
                else:
                    data = item[:]
                # 处理标量值
                if isinstance(data, np.ndarray) and data.shape == ():
                    result[key] = data.item()
                else:
                    result[key] = data
            elif isinstance(item, h5py.Group):
                # 递归读取group
                result[key] = recursive_read(item, index)
        return result
    
    with h5py.File(demo_path, 'r') as f:
        # 首先读取is_replan_noise标记（如果存在）
        is_replan_noise_array = None
        if 'is_replan_noise' in f:
            is_replan_noise_array = f['is_replan_noise'][:]
        
        # 获取episode长度（从任意一个dataset推断）
        episode_length = 0
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                episode_length = f[key].shape[0]
                break
            elif isinstance(f[key], h5py.Group):
                # 在group中找第一个dataset
                for subkey in f[key].keys():
                    if isinstance(f[key][subkey], h5py.Dataset):
                        episode_length = f[key][subkey].shape[0]
                        break
                if episode_length > 0:
                    break
        
        trajectory = []
        num_total = episode_length
        num_noise = 0
        num_valid = 0
        
        # 遍历所有timesteps
        for i in range(episode_length):
            # 判断当前step是否是噪声
            is_noise = False
            if is_replan_noise_array is not None:
                is_noise = bool(is_replan_noise_array[i])
            
            if is_noise:
                num_noise += 1
            else:
                num_valid += 1
            
            # 根据filter设置决定是否读取和添加
            if not filter_replan_noise or not is_noise:
                obs = recursive_read(f, index=i)
                obs['is_replan_noise'] = is_noise
                trajectory.append(obs)
    
    stats = {
        "num_total_steps": num_total,
        "num_noise_steps": num_noise,
        "num_valid_steps": num_valid,
        "filtered": filter_replan_noise,
        "num_returned_steps": len(trajectory),
    }
    
    return trajectory, stats


def load_dataset(
    dataset_dir: Path,
    filter_replan_noise: bool = True,
    max_demos: Optional[int] = None,
) -> Tuple[List[List[Dict]], Dict]:
    """
    加载整个数据集
    
    Args:
        dataset_dir: 数据集目录
        filter_replan_noise: 是否过滤replan噪声
        max_demos: 最多加载多少个demo (None表示全部)
    
    Returns:
        trajectories: 所有轨迹的列表
        dataset_stats: 数据集统计信息
    """
    demo_files = sorted(dataset_dir.glob("demo_*.h5"))
    if max_demos is not None:
        demo_files = demo_files[:max_demos]
    
    trajectories = []
    total_steps = 0
    total_noise_steps = 0
    total_valid_steps = 0
    
    for demo_file in demo_files:
        traj, stats = load_demo_with_filter(demo_file, filter_replan_noise)
        trajectories.append(traj)
        total_steps += stats["num_total_steps"]
        total_noise_steps += stats["num_noise_steps"]
        total_valid_steps += stats["num_valid_steps"]
    
    dataset_stats = {
        "num_demos": len(trajectories),
        "total_steps": total_steps,
        "total_noise_steps": total_noise_steps,
        "total_valid_steps": total_valid_steps,
        "filter_enabled": filter_replan_noise,
        "returned_total_steps": sum(len(traj) for traj in trajectories),
    }
    
    if total_noise_steps > 0:
        logger.info(f"数据集统计:")
        logger.info(f"  - 总demos: {dataset_stats['num_demos']}")
        logger.info(f"  - 总步数: {total_steps}")
        logger.info(f"  - 噪声步数: {total_noise_steps} ({total_noise_steps/total_steps*100:.1f}%)")
        logger.info(f"  - 有效步数: {total_valid_steps} ({total_valid_steps/total_steps*100:.1f}%)")
        if filter_replan_noise:
            logger.info(f"  - 过滤后返回: {dataset_stats['returned_total_steps']} 步")
        else:
            logger.info(f"  - 未过滤，返回: {dataset_stats['returned_total_steps']} 步")
    
    return trajectories, dataset_stats


def load_for_training(
    dataset_dir: Path,
    filter_replan_noise: bool = True,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载数据用于训练（返回numpy格式）
    
    Args:
        dataset_dir: 数据集目录
        filter_replan_noise: 是否过滤噪声步骤
        **kwargs: 传递给load_dataset的额外参数
    
    Returns:
        observations: (N, obs_dim) 所有observation
        actions: (N, action_dim) 所有action
    
    示例:
        obs, actions = load_for_training(
            Path("datasets/R1ProBlocksStackEasy/red/all/collected"),
            filter_replan_noise=True  # 训练ACT/Diffusion时设为True
        )
    """
    trajectories, stats = load_dataset(dataset_dir, filter_replan_noise, **kwargs)
    
    # 提取observations和actions
    # 注意：这里需要根据实际的observation/action结构调整
    all_obs = []
    all_actions = []
    
    for traj in trajectories:
        for step in traj:
            # 根据你的数据结构提取observation和action
            # 这里是示例，需要根据实际情况调整
            upper_obs = step.get("upper_body_observations", {})
            upper_action = step.get("upper_body_action_dict", {})
            
            # 例如：提取关节位置作为observation
            if isinstance(upper_obs, dict):
                obs = upper_obs.get("qpos", [])
                all_obs.append(obs)
            
            # 例如：提取关节指令作为action
            if isinstance(upper_action, dict):
                action = upper_action.get("joint_position_cmd", [])
                all_actions.append(action)
    
    observations = np.array(all_obs)
    actions = np.array(all_actions)
    
    logger.info(f"训练数据准备完成:")
    logger.info(f"  - observations shape: {observations.shape}")
    logger.info(f"  - actions shape: {actions.shape}")
    
    return observations, actions


# 使用示例
if __name__ == "__main__":
    from pathlib import Path
    
    # 示例1：加载并过滤噪声
    dataset_dir = Path("datasets/R1ProBlocksStackEasy/red/all/collected")
    
    print("\n=== 示例1：过滤噪声数据（用于训练） ===")
    trajectories, stats = load_dataset(dataset_dir, filter_replan_noise=True, max_demos=5)
    print(f"返回 {len(trajectories)} 个demos")
    print(f"第一个demo有 {len(trajectories[0])} 步")
    
    print("\n=== 示例2：不过滤噪声（用于分析） ===")
    trajectories, stats = load_dataset(dataset_dir, filter_replan_noise=False, max_demos=5)
    print(f"返回 {len(trajectories)} 个demos")
    print(f"第一个demo有 {len(trajectories[0])} 步")
    
    print("\n=== 示例3：检查第一个demo的标记 ===")
    demo_path = dataset_dir / "demo_0.h5"
    traj, stats = load_demo_with_filter(demo_path, filter_replan_noise=False)
    print(f"总步数: {stats['num_total_steps']}")
    print(f"噪声步数: {stats['num_noise_steps']}")
    print(f"有效步数: {stats['num_valid_steps']}")
    
    # 显示前10步的标记
    print("\n前10步的is_replan_noise标记:")
    for i, step in enumerate(traj[:10]):
        is_noise = step.get("is_replan_noise", False)
        print(f"  Step {i}: {'❌ 噪声' if is_noise else '✅ 正确'}")

