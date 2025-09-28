import h5py
import numpy as np
from pathlib import Path
import cv2
import os


def extract_rgb_images(h5_file, save_dir=None):
    """
    从 h5 文件中提取RGB图像并保存为图片文件。

    Args:
        h5_file: str, 输入的 h5 文件路径
        save_dir: str, 输出图片的保存目录 (可选)

    Returns:
        images_dict: dict, 包含不同相机的图像数据
    """
    # 创建保存目录
    if save_dir is None:
        save_dir = f"{Path(h5_file).stem}_images"

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    images_dict = {}

    with h5py.File(h5_file, "r") as f:
        # 探索h5文件的结构
        print("H5文件结构:")

        def print_structure(name, obj):
            print(f"{name}: {type(obj).__name__}")
            if isinstance(obj, h5py.Dataset):
                print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")

        f.visititems(print_structure)

        # 查找包含图像数据的组
        if "upper_body_observations" in f:
            obs_group = f["upper_body_observations"]
            print("\n在 upper_body_observations 中查找RGB数据...")

            for key in obs_group.keys():
                if key.startswith("rgb_"):
                    print(f"\n处理相机: {key}")
                    rgb_data = obs_group[key]
                    print(f"  RGB数据形状: {rgb_data.shape}")

                    # 创建相机保存目录
                    camera_save_dir = save_path / key
                    camera_save_dir.mkdir(exist_ok=True)

                    # 保存每一帧图像
                    for frame_idx in range(rgb_data.shape[0]):
                        # 获取单帧图像
                        frame = rgb_data[frame_idx]

                        # 确保图像数据在正确的范围内 (0-255)
                        if frame.dtype != np.uint8:
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)

                        # 保存图像
                        image_filename = f"frame_{frame_idx:06d}.png"
                        image_path = camera_save_dir / image_filename
                        cv2.imwrite(str(image_path), frame)

                    images_dict[key] = rgb_data
                    print(f"  ✅ 保存了 {rgb_data.shape[0]} 帧图像到 {camera_save_dir}")

            # 同时处理深度数据
            print("\n在 upper_body_observations 中查找深度数据...")
            for key in obs_group.keys():
                if key.startswith("depth_"):
                    print(f"\n处理深度相机: {key}")
                    depth_data = obs_group[key]
                    print(f"  深度数据形状: {depth_data.shape}")

                    # 创建深度保存目录
                    depth_save_dir = save_path / key
                    depth_save_dir.mkdir(exist_ok=True)

                    # 保存每一帧深度图
                    for frame_idx in range(depth_data.shape[0]):
                        # 获取单帧深度图
                        depth_frame = depth_data[frame_idx]

                        # 深度数据通常是uint16，需要转换为可视化格式
                        # 方法1: 直接保存原始深度数据（推荐用于后续处理）
                        depth_filename = f"depth_{frame_idx:06d}.png"
                        depth_path = depth_save_dir / depth_filename
                        cv2.imwrite(str(depth_path), depth_frame)

                        # 方法2: 保存可视化深度图（用于查看）
                        if depth_frame.max() > 0:
                            # 归一化到0-255范围
                            depth_normalized = (
                                (depth_frame - depth_frame.min())
                                / (depth_frame.max() - depth_frame.min())
                                * 255
                            ).astype(np.uint8)
                            depth_viz_filename = f"depth_viz_{frame_idx:06d}.png"
                            depth_viz_path = depth_save_dir / depth_viz_filename
                            cv2.imwrite(str(depth_viz_path), depth_normalized)

                    print(
                        f"  ✅ 保存了 {depth_data.shape[0]} 帧深度图到 {depth_save_dir}"
                    )

        elif "observation" in f:
            obs_group = f["observation"]
            if "images" in obs_group:
                images_group = obs_group["images"]

                for camera_name in images_group.keys():
                    print(f"\n处理相机: {camera_name}")
                    camera_data = images_group[camera_name]

                    # 检查是否是RGB数据
                    if "rgb" in camera_data:
                        rgb_data = camera_data["rgb"]
                        print(f"  RGB数据形状: {rgb_data.shape}")

                        # 创建相机保存目录
                        camera_save_dir = save_path / camera_name
                        camera_save_dir.mkdir(exist_ok=True)

                        # 保存每一帧图像
                        for frame_idx in range(rgb_data.shape[0]):
                            # 获取单帧图像
                            frame = rgb_data[frame_idx]

                            # 确保图像数据在正确的范围内 (0-255)
                            if frame.dtype != np.uint8:
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                else:
                                    frame = frame.astype(np.uint8)

                            # 保存图像
                            image_filename = f"frame_{frame_idx:06d}.png"
                            image_path = camera_save_dir / image_filename
                            cv2.imwrite(str(image_path), frame)

                        images_dict[camera_name] = rgb_data
                        print(
                            f"  ✅ 保存了 {rgb_data.shape[0]} 帧图像到 {camera_save_dir}"
                        )

        # 如果没找到标准结构，尝试直接查找图像数据
        else:
            print("未找到标准observation/images结构，尝试直接查找图像数据...")
            for key in f.keys():
                if (
                    "image" in key.lower()
                    or "rgb" in key.lower()
                    or "camera" in key.lower()
                ):
                    print(f"找到可能的图像数据: {key}")
                    data = f[key]
                    if isinstance(data, h5py.Dataset) and len(data.shape) >= 3:
                        print(f"  数据形状: {data.shape}")
                        # 这里可以添加保存逻辑

    return images_dict


if __name__ == "__main__":
    h5_file = "./datasets/R1ProBlocksStackEasy/all/replayed/demo_0.h5"

    # 检查文件是否存在
    if not os.path.exists(h5_file):
        print(f"错误: 文件 {h5_file} 不存在")
        print("请检查文件路径是否正确")
        exit(1)

    print(f"读取h5文件: {h5_file}")
    images = extract_rgb_images(h5_file)

    if images:
        print(f"\n✅ 成功提取并保存了 {len(images)} 个相机的RGB图像")
    else:
        print("\n❌ 未找到RGB图像数据")
