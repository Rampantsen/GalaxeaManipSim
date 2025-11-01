import numpy as np
import sapien
import transforms3d as t3d

def rand_pose(
    xlim: np.ndarray | list,
    ylim: np.ndarray | list,
    zlim: np.ndarray | list,
    ylim_prop: bool = False,
    rotate_rand: bool = False,
    rotate_lim: list = [0, 0, 0],
    qpos: list = [1, 0, 0, 0],
) -> sapien.Pose:
    if len(xlim) < 2 or xlim[1] < xlim[0]:
        xlim = np.array([xlim[0], xlim[0]])
    if len(ylim) < 2 or ylim[1] < ylim[0]:
        ylim = np.array([ylim[0], ylim[0]])
    if len(zlim) < 2 or zlim[1] < zlim[0]:
        zlim = np.array([zlim[0], zlim[0]])

    x = np.random.uniform(xlim[0], xlim[1])
    y = np.random.uniform(ylim[0], ylim[1])

    while ylim_prop and abs(x) < 0.15 and y > 0:
        y = np.random.uniform(ylim[0], 0)

    z = np.random.uniform(zlim[0], zlim[1])

    rotate = qpos
    if rotate_rand:
            z_angle = np.random.uniform(-rotate_lim[2], rotate_lim[2])
            rotate_quat = t3d.euler.euler2quat(0, 0, z_angle)
            rotate = rotate_quat

    return sapien.Pose([x, y, z], rotate)


def add_pose_noise(
    pose: list,
    position_noise_range: tuple = (0.02, 0.05),
    orientation_noise_range: float = 0.1,
    noise_axes: list = [True, True, True],  # [x, y, z]
) -> list:
    """
    为pose添加随机噪声
    
    Args:
        pose: [x, y, z, qw, qx, qy, qz] 格式的姿态
        position_noise_range: 位置噪声范围 (min, max)，单位米
        orientation_noise_range: 旋转噪声范围，单位弧度
        noise_axes: 哪些轴添加噪声 [x, y, z]
    
    Returns:
        添加噪声后的pose
    """
    noisy_pose = pose.copy()
    
    # 添加位置噪声
    for i, add_noise in enumerate(noise_axes):
        if add_noise:
            noise_magnitude = np.random.uniform(position_noise_range[0], position_noise_range[1])
            noise_direction = np.random.choice([-1, 1])
            noisy_pose[i] += noise_direction * noise_magnitude
    
    # 可选：添加旋转噪声
    if orientation_noise_range > 0:
        # 转换为rotation matrix
        original_quat = pose[3:]
        rot_mat = t3d.quaternions.quat2mat(original_quat)
        
        # 添加小的旋转扰动
        noise_angle = np.random.uniform(-orientation_noise_range, orientation_noise_range)
        noise_axis = np.random.randn(3)
        noise_axis = noise_axis / np.linalg.norm(noise_axis)
        
        noise_rot = t3d.axangles.axangle2mat(noise_axis, noise_angle)
        new_rot = noise_rot @ rot_mat
        
        # 转回quaternion
        new_quat = t3d.quaternions.mat2quat(new_rot)
        noisy_pose[3:] = new_quat
    
    return noisy_pose
