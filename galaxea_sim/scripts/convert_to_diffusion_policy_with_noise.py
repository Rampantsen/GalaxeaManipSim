"""
将 GalaxeaManipSim 数据转换为 Diffusion Policy 格式（保留噪声标签）

原版 Diffusion Policy: https://github.com/real-stanford/diffusion_policy

使用方法:
    python -m galaxea_sim.scripts.convert_to_diffusion_policy_with_noise \
        --src-dir datasets/R1ProBlocksStackEasy/red/all/collected \
        --dst-path datasets_diffusion_policy/R1ProBlocksStackEasy_with_noise.zarr \
        --use_multi_camera
"""

import zarr
import numpy as np
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm
import tyro
from loguru import logger

# 配置loguru输出到控制台
import sys
logger.remove()  # 移除默认处理器
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def convert_to_zarr_with_noise(
    src_dir: str,
    dst_path: str,
    use_multi_camera: bool = True,
    target_width: int = 224,
    target_height: int = 224,
):
    """
    转换为 Zarr 格式（Diffusion Policy 使用），保留噪声标签
    
    Args:
        src_dir: 源数据集目录
        dst_path: 目标 Zarr 文件路径
        use_multi_camera: 是否使用多相机（True使用所有三个相机，False只使用头部相机）
        target_width: 目标图像宽度（所有相机统一）
        target_height: 目标图像高度（所有相机统一）
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
    logger.info(f"使用多相机: {'是' if use_multi_camera else '否'}")
    logger.info(f"⚠️ 保留噪声标签用于训练时过滤")
    
    # 第一遍扫描：获取数据维度和总帧数
    logger.info("第一遍扫描：获取数据维度...")
    total_frames = 0
    episode_ends = []
    
    # 统一的目标图像尺寸
    img_shape = (target_height, target_width, 3)
    logger.info(f"目标图像尺寸: {img_shape}")
    
    state_dim = None
    action_dim = None
    
    for demo_file in tqdm(demo_files, desc="扫描维度"):
        with h5py.File(demo_file, 'r') as f:
            n_frames = len(f['upper_body_observations']['rgb_head'])
            total_frames += n_frames
            episode_ends.append(total_frames)
            
            if state_dim is None:
                # 显示原始图像尺寸信息
                img_head_shape = f['upper_body_observations']['rgb_head'][0].shape
                img_hand_shape = f['upper_body_observations']['rgb_left_hand'][0].shape
                logger.info(f"原始图像尺寸 - 头部: {img_head_shape}, 手部: {img_hand_shape}")
                
                # 计算状态和动作维度
                left_arm_joint = f['upper_body_observations']['left_arm_joint_position'][0]
                left_gripper = f['upper_body_observations']['left_arm_gripper_position'][0]
                right_arm_joint = f['upper_body_observations']['right_arm_joint_position'][0]
                right_gripper = f['upper_body_observations']['right_arm_gripper_position'][0]
                state_dim = len(left_arm_joint) + len(left_gripper) + len(right_arm_joint) + len(right_gripper)
                
                left_joint_cmd = f['upper_body_action_dict']['left_arm_joint_position_cmd'][0]
                left_gripper_cmd = f['upper_body_action_dict']['left_arm_gripper_position_cmd'][0]
                right_joint_cmd = f['upper_body_action_dict']['right_arm_joint_position_cmd'][0]
                right_gripper_cmd = f['upper_body_action_dict']['right_arm_gripper_position_cmd'][0]
                action_dim = len(left_joint_cmd) + len(left_gripper_cmd) + len(right_joint_cmd) + len(right_gripper_cmd)
    
    logger.info(f"数据集统计: 总帧数={total_frames}, 图像形状={img_shape}, 状态维度={state_dim}, 动作维度={action_dim}")
    
    # 创建 Zarr 数据集并预分配空间
    logger.info("创建 Zarr 数据集...")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(dst_path), mode='w')
    
    # 预分配数组
    if use_multi_camera:
        img_head_arr = root.create_dataset(
            'data/img_head',
            shape=(total_frames, *img_shape),
            chunks=(1, *img_shape),
            dtype=np.uint8,
            compressor=zarr.Blosc(cname='lz4', clevel=5)
        )
        img_left_arr = root.create_dataset(
            'data/img_left',
            shape=(total_frames, *img_shape),
            chunks=(1, *img_shape),
            dtype=np.uint8,
            compressor=zarr.Blosc(cname='lz4', clevel=5)
        )
        img_right_arr = root.create_dataset(
            'data/img_right',
            shape=(total_frames, *img_shape),
            chunks=(1, *img_shape),
            dtype=np.uint8,
            compressor=zarr.Blosc(cname='lz4', clevel=5)
        )
    else:
        img_arr = root.create_dataset(
            'data/img',
            shape=(total_frames, *img_shape),
            chunks=(1, *img_shape),
            dtype=np.uint8,
            compressor=zarr.Blosc(cname='lz4', clevel=5)
        )
    
    state_arr = root.create_dataset(
        'data/state',
        shape=(total_frames, state_dim),
        chunks=(100, state_dim),
        dtype=np.float32,
        compressor=zarr.Blosc(cname='lz4', clevel=5)
    )
    
    action_arr = root.create_dataset(
        'data/action',
        shape=(total_frames, action_dim),
        chunks=(100, action_dim),
        dtype=np.float32,
        compressor=zarr.Blosc(cname='lz4', clevel=5)
    )
    
    noise_arr = root.create_dataset(
        'data/is_replan_noise',
        shape=(total_frames,),
        chunks=(1000,),
        dtype=bool,
        compressor=zarr.Blosc(cname='lz4', clevel=5)
    )
    
    # 第二遍扫描：增量写入数据
    logger.info("第二遍扫描：写入数据...")
    current_idx = 0
    total_noise_frames = 0
    
    for demo_file in tqdm(demo_files, desc="写入数据"):
        with h5py.File(demo_file, 'r') as f:
            # 读取状态（组合各个关节位置）
            left_arm_joint = f['upper_body_observations']['left_arm_joint_position'][:]
            left_gripper = f['upper_body_observations']['left_arm_gripper_position'][:]
            right_arm_joint = f['upper_body_observations']['right_arm_joint_position'][:]
            right_gripper = f['upper_body_observations']['right_arm_gripper_position'][:]
            
            # 组合成完整的状态向量
            qpos = np.concatenate([
                left_arm_joint,
                left_gripper,
                right_arm_joint,
                right_gripper
            ], axis=-1)
            
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
            
            # 读取噪声标签
            if 'is_replan_noise' in f:
                is_replan_noise = f['is_replan_noise'][:]
            else:
                # 如果没有噪声标签，默认全部为False
                is_replan_noise = np.zeros(len(qpos), dtype=bool)
            
            # 统计噪声帧数量
            total_noise_frames += np.sum(is_replan_noise)
            
            # 获取当前demo的帧数
            n_frames = len(qpos)
            end_idx = current_idx + n_frames
            
            # 直接写入状态和动作数据
            state_arr[current_idx:end_idx] = qpos
            action_arr[current_idx:end_idx] = action
            noise_arr[current_idx:end_idx] = is_replan_noise
            
            # 批量处理图像（避免一次性加载过多图像到内存）
            batch_size = 10  # 每次处理10帧
            for i in range(0, n_frames, batch_size):
                batch_end = min(i + batch_size, n_frames)
                batch_start_idx = current_idx + i
                batch_end_idx = current_idx + batch_end
                
                if use_multi_camera:
                    # 读取批量图像
                    img_head_batch = f['upper_body_observations']['rgb_head'][i:batch_end]
                    img_left_batch = f['upper_body_observations']['rgb_left_hand'][i:batch_end]
                    img_right_batch = f['upper_body_observations']['rgb_right_hand'][i:batch_end]
                    
                    # Resize到统一尺寸
                    img_head_batch = np.array([cv2.resize(img, (target_width, target_height)) 
                                              for img in img_head_batch])
                    img_left_batch = np.array([cv2.resize(img, (target_width, target_height)) 
                                              for img in img_left_batch])
                    img_right_batch = np.array([cv2.resize(img, (target_width, target_height)) 
                                              for img in img_right_batch])
                    
                    # 写入Zarr数组
                    img_head_arr[batch_start_idx:batch_end_idx] = img_head_batch
                    img_left_arr[batch_start_idx:batch_end_idx] = img_left_batch
                    img_right_arr[batch_start_idx:batch_end_idx] = img_right_batch
                else:
                    # 读取并resize单个相机图像
                    img_batch = f['upper_body_observations']['rgb_head'][i:batch_end]
                    img_batch = np.array([cv2.resize(img, (target_width, target_height)) 
                                         for img in img_batch])
                    img_arr[batch_start_idx:batch_end_idx] = img_batch
            
            current_idx = end_idx
    
    # 保存元数据
    logger.info(f"保存元数据")
    root.create_dataset(
        'meta/episode_ends',
        data=np.array(episode_ends, dtype=np.int64),
        dtype=np.int64
    )
    
    # 计算统计信息
    noise_ratio = total_noise_frames / total_frames * 100 if total_frames > 0 else 0
    
    logger.info(f"✅ 转换完成!")
    logger.info(f"  - Episodes: {len(demo_files)}")
    logger.info(f"  - 总帧数: {total_frames}")
    logger.info(f"  - 噪声帧数: {total_noise_frames} ({noise_ratio:.1f}%)")
    if use_multi_camera:
        logger.info(f"  - 头部相机形状: {total_frames} × {img_shape}")
        logger.info(f"  - 左手相机形状: {total_frames} × {img_shape}")
        logger.info(f"  - 右手相机形状: {total_frames} × {img_shape}")
    else:
        logger.info(f"  - 图像形状: {total_frames} × {img_shape}")
    logger.info(f"  - 状态维度: {state_dim}")
    logger.info(f"  - 动作维度: {action_dim}")
    logger.info(f"  - 输出路径: {dst_path}")
    
    # 创建 README
    readme_path = dst_path.parent / f"{dst_path.stem}_README.md"
    with open(readme_path, 'w') as f:
        f.write(f"# Diffusion Policy 数据集（带噪声标签）\n\n")
        f.write(f"- 源目录: {src_dir}\n")
        f.write(f"- Episodes: {len(demo_files)}\n")
        f.write(f"- 总帧数: {total_frames}\n")
        f.write(f"- 噪声帧数: {total_noise_frames} ({noise_ratio:.1f}%)\n")
        if use_multi_camera:
            f.write(f"- 相机配置: 多相机（头部+左手+右手）\n")
            f.write(f"- 图像数据形状: {total_frames} × {img_shape}\n")
        else:
            f.write(f"- 相机配置: 单相机（仅头部）\n")
            f.write(f"- 图像数据形状: {total_frames} × {img_shape}\n")
        f.write(f"- 状态维度: {state_dim}\n")
        f.write(f"- 动作维度: {action_dim}\n")
        f.write(f"\n## 噪声标签说明\n")
        f.write(f"数据集包含噪声标签 `data/is_replan_noise`，可在训练时选择：\n")
        f.write(f"- `filter_noise=True`: 噪声帧只作为observation历史，不作为action目标\n")
        f.write(f"- `filter_noise=False`: 正常训练，使用所有数据\n")
        f.write(f"\n## 使用方法\n")
        f.write(f"```python\n")
        f.write(f"from galaxea_sim.utils.dp_noise_filtered_dataset import GalaxeaImageDataset\n\n")
        f.write(f"# 过滤噪声的action（推荐）\n")
        f.write(f"dataset = GalaxeaImageDataset(\n")
        f.write(f"    zarr_path='{dst_path.absolute()}',\n")
        f.write(f"    filter_noise=True,  # 过滤噪声帧的action\n")
        f.write(f"    horizon=16,\n")
        f.write(f")\n\n")
        f.write(f"# 或不过滤（使用所有数据）\n")
        f.write(f"dataset = GalaxeaImageDataset(\n")
        f.write(f"    zarr_path='{dst_path.absolute()}',\n")
        f.write(f"    filter_noise=False,  # 使用所有数据\n")
        f.write(f"    horizon=16,\n")
        f.write(f")\n")
        f.write(f"```\n")


def main(
    src_dir: str = "datasets/R1ProBlocksStackEasy/all/collected",
    dst_path: str = "datasets_diffusion_policy/R1ProBlocksStackEasy_with_noise.zarr",
    use_multi_camera: bool = True,
    target_width: int = 224,
    target_height: int = 224,
):
    """
    转换数据集为 Diffusion Policy 格式（保留噪声标签）
    
    Args:
        src_dir: 源数据集目录
        dst_path: 目标 Zarr 文件路径
        use_multi_camera: 是否使用多相机（True使用所有三个相机，False只使用头部相机）
        target_width: 目标图像宽度（推荐224或480）
        target_height: 目标图像高度（推荐224或640）
    """
    convert_to_zarr_with_noise(src_dir, dst_path, use_multi_camera, target_width, target_height)


if __name__ == "__main__":
    tyro.cli(main)

