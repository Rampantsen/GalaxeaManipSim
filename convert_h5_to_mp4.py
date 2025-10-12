import h5py
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm


def h5_to_mp4(h5_file, output_dir, fps=30):
    """
    将单个h5文件中的RGB图像转换为mp4视频。

    Args:
        h5_file: str或Path, 输入的h5文件路径
        output_dir: str或Path, 输出mp4文件的保存目录
        fps: int, 视频帧率 (默认30)

    Returns:
        bool: 转换是否成功
    """
    h5_file = Path(h5_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = False

    try:
        with h5py.File(h5_file, "r") as f:
            # 查找包含图像数据的组
            if "upper_body_observations" in f:
                obs_group = f["upper_body_observations"]

                # 处理所有RGB相机
                for key in obs_group.keys():
                    if key.startswith("rgb_"):
                        rgb_data = obs_group[key][:]  # 读取所有帧

                        # 获取视频参数
                        num_frames, height, width, channels = rgb_data.shape

                        # 确保图像数据在正确的范围内 (0-255)
                        if rgb_data.dtype != np.uint8:
                            if rgb_data.max() <= 1.0:
                                rgb_data = (rgb_data * 255).astype(np.uint8)
                            else:
                                rgb_data = rgb_data.astype(np.uint8)

                        # 创建输出文件名
                        output_file = output_dir / f"{h5_file.stem}_{key}.mp4"

                        # 创建视频写入器 - 使用H.264编码器
                        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264编码
                        video_writer = cv2.VideoWriter(
                            str(output_file), fourcc, fps, (width, height)
                        )

                        if not video_writer.isOpened():
                            print(f"  ⚠️  无法创建视频写入器，尝试使用备用编码器")
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(
                                str(output_file), fourcc, fps, (width, height)
                            )

                        # 写入所有帧
                        for frame_idx in range(num_frames):
                            frame = rgb_data[frame_idx]
                            # OpenCV使用BGR格式，如果是RGB需要转换
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            video_writer.write(frame_bgr)

                        video_writer.release()
                        print(f"  ✅ 创建视频: {output_file} ({num_frames}帧)")
                        success = True

            elif "observation" in f:
                obs_group = f["observation"]
                if "images" in obs_group:
                    images_group = obs_group["images"]

                    for camera_name in images_group.keys():
                        camera_data = images_group[camera_name]

                        if "rgb" in camera_data:
                            rgb_data = camera_data["rgb"][:]

                            # 获取视频参数
                            num_frames, height, width, channels = rgb_data.shape

                            # 确保图像数据在正确的范围内 (0-255)
                            if rgb_data.dtype != np.uint8:
                                if rgb_data.max() <= 1.0:
                                    rgb_data = (rgb_data * 255).astype(np.uint8)
                                else:
                                    rgb_data = rgb_data.astype(np.uint8)

                            # 创建输出文件名
                            output_file = (
                                output_dir / f"{h5_file.stem}_{camera_name}.mp4"
                            )

                            # 创建视频写入器 - 使用H.264编码器
                            fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264编码
                            video_writer = cv2.VideoWriter(
                                str(output_file), fourcc, fps, (width, height)
                            )

                            if not video_writer.isOpened():
                                print(f"  ⚠️  无法创建视频写入器，尝试使用备用编码器")
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                video_writer = cv2.VideoWriter(
                                    str(output_file), fourcc, fps, (width, height)
                                )

                            # 写入所有帧
                            for frame_idx in range(num_frames):
                                frame = rgb_data[frame_idx]
                                # OpenCV使用BGR格式，如果是RGB需要转换
                                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                video_writer.write(frame_bgr)

                            video_writer.release()
                            print(f"  ✅ 创建视频: {output_file} ({num_frames}帧)")
                            success = True
            else:
                print(f"  ⚠️  未找到标准的observation结构: {h5_file.name}")

    except Exception as e:
        print(f"  ❌ 处理失败 {h5_file.name}: {str(e)}")
        return False

    return success


def batch_convert_h5_to_mp4(input_dir, output_dir, fps=30):
    """
    批量将目录中的所有h5文件转换为mp4视频。

    Args:
        input_dir: str, 包含h5文件的输入目录
        output_dir: str, 输出mp4视频的保存目录
        fps: int, 视频帧率 (默认30)

    Returns:
        tuple: (成功数量, 失败数量)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # 查找所有h5文件
    h5_files = sorted(input_dir.glob("*.h5"))

    if not h5_files:
        print(f"⚠️  在 {input_dir} 中未找到h5文件")
        return 0, 0

    print(f"找到 {len(h5_files)} 个h5文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"视频帧率: {fps} fps\n")

    success_count = 0
    fail_count = 0

    # 使用进度条处理所有文件
    for h5_file in tqdm(h5_files, desc="转换h5到mp4"):
        print(f"\n处理: {h5_file.name}")
        if h5_to_mp4(h5_file, output_dir, fps):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n" + "=" * 60)
    print(f"转换完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    return success_count, fail_count


if __name__ == "__main__":
    # 设置输入和输出路径
    input_dir = "/home/sen/workspace/galaxea/GalaxeaManipSim/datasets/R1ProBlocksStackEasy-traj_aug/grasp_sample_only/replayed"
    output_dir = "/home/sen/workspace/galaxea/GalaxeaManipSim/datasets/R1ProBlocksStackEasy-traj_aug/grasp_sample_only/video"

    # 执行批量转换
    batch_convert_h5_to_mp4(input_dir, output_dir, fps=30)
