"""
将 GalaxeaManipSim 数据转换为原版 ACT 格式

原版 ACT: https://github.com/tonyzhaozh/act

使用方法:
    python -m galaxea_sim.scripts.convert_to_original_act \
        --src-dir datasets/R1ProBlocksStackEasy/red/all/collected \
        --dst-dir datasets_original_act/R1ProBlocksStackEasy \
        --filter-noise True
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tyro
from loguru import logger


def convert_single_episode(src_path: Path, dst_path: Path, filter_noise: bool = False):
    """
    转换单个 episode
    
    Args:
        src_path: 源 HDF5 文件路径
        dst_path: 目标 HDF5 文件路径
        filter_noise: 是否过滤噪声帧
    
    Returns:
        num_frames: 帧数
        num_noise_filtered: 过滤的噪声帧数
    """
    with h5py.File(src_path, 'r') as src_f:
        # 读取数据
        qpos = src_f['upper_body_observations']['qpos'][:]
        
        # Action
        left_joint_cmd = src_f['upper_body_action_dict']['left_arm_joint_position_cmd'][:]
        left_gripper_cmd = src_f['upper_body_action_dict']['left_arm_gripper_position_cmd'][:]
        right_joint_cmd = src_f['upper_body_action_dict']['right_arm_joint_position_cmd'][:]
        right_gripper_cmd = src_f['upper_body_action_dict']['right_arm_gripper_position_cmd'][:]
        
        action = np.concatenate([
            left_joint_cmd, 
            left_gripper_cmd,
            right_joint_cmd, 
            right_gripper_cmd
        ], axis=-1)
        
        # 图像数据
        img_head = src_f['upper_body_observations']['rgb_head'][:]
        img_left = src_f['upper_body_observations']['rgb_left_hand'][:]
        img_right = src_f['upper_body_observations']['rgb_right_hand'][:]
        
        # 检查噪声标记
        is_replan_noise = None
        if 'is_replan_noise' in src_f:
            is_replan_noise = src_f['is_replan_noise'][:]
        
        # 过滤噪声帧
        num_noise_filtered = 0
        if filter_noise and is_replan_noise is not None:
            valid_indices = np.where(~is_replan_noise)[0]
            
            qpos = qpos[valid_indices]
            action = action[valid_indices]
            img_head = img_head[valid_indices]
            img_left = img_left[valid_indices]
            img_right = img_right[valid_indices]
            
            num_noise_filtered = len(is_replan_noise) - len(valid_indices)
        
        # 创建新的 HDF5 文件（原版 ACT 格式）
        with h5py.File(dst_path, 'w') as dst_f:
            # 机器人状态和动作
            dst_f.create_dataset('qpos', data=qpos, compression='gzip')
            dst_f.create_dataset('qvel', data=np.zeros_like(qpos), compression='gzip')  # ACT 需要但我们没有
            dst_f.create_dataset('action', data=action, compression='gzip')
            
            # 图像数据（原版 ACT 格式）
            images_group = dst_f.create_group('observations/images')
            images_group.create_dataset('top', data=img_head, compression='gzip')
            images_group.create_dataset('left_wrist', data=img_left, compression='gzip')
            images_group.create_dataset('right_wrist', data=img_right, compression='gzip')
            
            # 元数据
            dst_f.attrs['sim'] = True
            dst_f.attrs['compress'] = True
    
    return len(qpos), num_noise_filtered


def convert_dataset(
    src_dir: str,
    dst_dir: str,
    filter_noise: bool = False,
):
    """
    转换整个数据集
    
    Args:
        src_dir: 源数据集目录
        dst_dir: 目标数据集目录
        filter_noise: 是否过滤噪声帧
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    if not src_dir.exists():
        logger.error(f"源目录不存在: {src_dir}")
        return
    
    demo_files = sorted(src_dir.glob("demo_*.h5"))
    
    if len(demo_files) == 0:
        logger.error(f"未找到 demo 文件: {src_dir}")
        return
    
    logger.info(f"找到 {len(demo_files)} 个 demo 文件")
    logger.info(f"过滤噪声: {'是' if filter_noise else '否'}")
    
    total_frames = 0
    total_noise_filtered = 0
    
    for i, demo_file in enumerate(tqdm(demo_files, desc="转换中")):
        dst_file = dst_dir / f"episode_{i}.hdf5"
        num_frames, num_noise = convert_single_episode(demo_file, dst_file, filter_noise)
        total_frames += num_frames
        total_noise_filtered += num_noise
    
    logger.info(f"✅ 转换完成!")
    logger.info(f"  - Episodes: {len(demo_files)}")
    logger.info(f"  - 总帧数: {total_frames}")
    if filter_noise:
        logger.info(f"  - 过滤的噪声帧: {total_noise_filtered}")
    logger.info(f"  - 输出目录: {dst_dir}")
    
    # 创建 README
    readme_path = dst_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"# 原版 ACT 数据集\n\n")
        f.write(f"- 源目录: {src_dir}\n")
        f.write(f"- Episodes: {len(demo_files)}\n")
        f.write(f"- 总帧数: {total_frames}\n")
        f.write(f"- 噪声过滤: {'是' if filter_noise else '否'}\n")
        if filter_noise:
            f.write(f"- 过滤的噪声帧: {total_noise_filtered}\n")
        f.write(f"\n使用原版 ACT 训练:\n")
        f.write(f"```bash\n")
        f.write(f"cd ~/workspace/galaxea/act\n")
        f.write(f"python train.py --dataset_dir {dst_dir.absolute()} ...\n")
        f.write(f"```\n")


def main(
    src_dir: str = "datasets/R1ProBlocksStackEasy/red/all/collected",
    dst_dir: str = "datasets_original_act/R1ProBlocksStackEasy",
    filter_noise: bool = False,
):
    """
    转换数据集为原版 ACT 格式
    
    Args:
        src_dir: 源数据集目录
        dst_dir: 目标数据集目录
        filter_noise: 是否过滤标记为噪声的帧
    """
    convert_dataset(src_dir, dst_dir, filter_noise)


if __name__ == "__main__":
    tyro.cli(main)

