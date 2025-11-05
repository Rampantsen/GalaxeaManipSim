"""
将 GalaxeaManipSim 数据转换为 Diffusion Policy 格式

原版 Diffusion Policy: https://github.com/real-stanford/diffusion_policy

使用方法:
    python -m galaxea_sim.scripts.convert_to_diffusion_policy \
        --src-dir datasets/R1ProBlocksStackEasy/red/all/collected \
        --dst-path datasets_diffusion_policy/R1ProBlocksStackEasy.zarr \
        --filter-noise True
"""

import zarr
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import tyro
from loguru import logger


def convert_to_zarr(
    src_dir: str,
    dst_path: str,
    filter_noise: bool = False,
    image_key: str = "rgb_head",
):
    """
    转换为 Zarr 格式（Diffusion Policy 使用）
    
    Args:
        src_dir: 源数据集目录
        dst_path: 目标 Zarr 文件路径
        filter_noise: 是否过滤噪声帧
        image_key: 使用哪个相机的图像
    """
    src_dir = Path(src_dir)
    dst_path = Path(dst_path)
    
    if not src_dir.exists():
        logger.error(f"源目录不存在: {src_dir}")
        return
    
    demo_files = sorted(src_dir.glob("demo_*.h5"))
    
    if len(demo_files) == 0:
        logger.error(f"未找到 demo 文件: {src_dir}")
        return
    
    logger.info(f"找到 {len(demo_files)} 个 demo 文件")
    logger.info(f"过滤噪声: {'是' if filter_noise else '否'}")
    logger.info(f"使用图像: {image_key}")
    
    all_images = []
    all_states = []
    all_actions = []
    episode_ends = []
    
    total_frames = 0
    total_noise_filtered = 0
    
    for demo_file in tqdm(demo_files, desc="加载数据"):
        with h5py.File(demo_file, 'r') as f:
            # 读取状态
            qpos = f['upper_body_observations']['qpos'][:]
            
            # 读取动作
            left_joint_cmd = f['upper_body_action_dict']['left_arm_joint_position_cmd'][:]
            left_gripper_cmd = f['upper_body_action_dict']['left_arm_gripper_position_cmd'][:]
            right_joint_cmd = f['upper_body_action_dict']['right_arm_joint_position_cmd'][:]
            right_gripper_cmd = f['upper_body_action_dict']['right_arm_gripper_position_cmd'][:]
            
            action = np.concatenate([
                left_joint_cmd,
                left_gripper_cmd,
                right_joint_cmd,
                right_gripper_cmd
            ], axis=-1)
            
            # 读取图像
            img = f['upper_body_observations'][image_key][:]
            
            # 检查噪声标记
            is_replan_noise = None
            if 'is_replan_noise' in f:
                is_replan_noise = f['is_replan_noise'][:]
            
            # 过滤噪声帧
            if filter_noise and is_replan_noise is not None:
                valid_indices = np.where(~is_replan_noise)[0]
                
                qpos = qpos[valid_indices]
                action = action[valid_indices]
                img = img[valid_indices]
                
                total_noise_filtered += len(is_replan_noise) - len(valid_indices)
            
            all_states.append(qpos)
            all_actions.append(action)
            all_images.append(img)
            
            total_frames += len(qpos)
            episode_ends.append(total_frames)
    
    logger.info("开始创建 Zarr 数据集...")
    
    # 创建 Zarr 数据集
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(dst_path), mode='w')
    
    # 拼接所有数据
    all_images = np.concatenate(all_images, axis=0)
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    episode_ends = np.array(episode_ends, dtype=np.int64)
    
    # 保存数据
    logger.info(f"保存图像数据: {all_images.shape}")
    root.create_dataset(
        'data/img',
        data=all_images,
        chunks=(1, *all_images.shape[1:]),
        dtype=np.uint8,
        compressor=zarr.Blosc(cname='lz4', clevel=5)
    )
    
    logger.info(f"保存状态数据: {all_states.shape}")
    root.create_dataset(
        'data/state',
        data=all_states,
        dtype=np.float32,
        compressor=zarr.Blosc(cname='lz4', clevel=5)
    )
    
    logger.info(f"保存动作数据: {all_actions.shape}")
    root.create_dataset(
        'data/action',
        data=all_actions,
        dtype=np.float32,
        compressor=zarr.Blosc(cname='lz4', clevel=5)
    )
    
    logger.info(f"保存元数据")
    root.create_dataset(
        'meta/episode_ends',
        data=episode_ends,
        dtype=np.int64
    )
    
    logger.info(f"✅ 转换完成!")
    logger.info(f"  - Episodes: {len(demo_files)}")
    logger.info(f"  - 总帧数: {total_frames}")
    if filter_noise:
        logger.info(f"  - 过滤的噪声帧: {total_noise_filtered}")
    logger.info(f"  - 图像形状: {all_images.shape}")
    logger.info(f"  - 状态维度: {all_states.shape[-1]}")
    logger.info(f"  - 动作维度: {all_actions.shape[-1]}")
    logger.info(f"  - 输出路径: {dst_path}")
    
    # 创建 README
    readme_path = dst_path.parent / f"{dst_path.stem}_README.md"
    with open(readme_path, 'w') as f:
        f.write(f"# Diffusion Policy 数据集\n\n")
        f.write(f"- 源目录: {src_dir}\n")
        f.write(f"- Episodes: {len(demo_files)}\n")
        f.write(f"- 总帧数: {total_frames}\n")
        f.write(f"- 噪声过滤: {'是' if filter_noise else '否'}\n")
        if filter_noise:
            f.write(f"- 过滤的噪声帧: {total_noise_filtered}\n")
        f.write(f"- 图像: {image_key}\n")
        f.write(f"- 图像形状: {all_images.shape}\n")
        f.write(f"- 状态维度: {all_states.shape[-1]}\n")
        f.write(f"- 动作维度: {all_actions.shape[-1]}\n")
        f.write(f"\n使用原版 Diffusion Policy 训练:\n")
        f.write(f"```bash\n")
        f.write(f"cd ~/workspace/galaxea/diffusion_policy\n")
        f.write(f"python train.py task.dataset_path={dst_path.absolute()} ...\n")
        f.write(f"```\n")


def main(
    src_dir: str = "datasets/R1ProBlocksStackEasy/red/all/collected",
    dst_path: str = "datasets_diffusion_policy/R1ProBlocksStackEasy.zarr",
    filter_noise: bool = False,
    image_key: str = "rgb_head",
):
    """
    转换数据集为 Diffusion Policy 格式
    
    Args:
        src_dir: 源数据集目录
        dst_path: 目标 Zarr 文件路径
        filter_noise: 是否过滤标记为噪声的帧
        image_key: 使用哪个相机 (rgb_head, rgb_left_hand, rgb_right_hand)
    """
    convert_to_zarr(src_dir, dst_path, filter_noise, image_key)


if __name__ == "__main__":
    tyro.cli(main)

