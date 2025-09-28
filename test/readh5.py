import h5py
import numpy as np
from pathlib import Path

def extract_actions(h5_file, save_path=None, use_eef=False):
    """
    从 h5 文件中提取动作 (action)，保存为 numpy 数组。
    
    Args:
        h5_file: str, 输入的 h5 文件路径
        save_path: str, 输出的 npy 文件路径 (可选)
        use_eef: bool, 是否使用末端位姿作为 action（默认 False，使用关节空间）
    
    Returns:
        actions: np.ndarray, shape (T, D)
                 T = 轨迹长度, D = action 维度
    """
    with h5py.File(h5_file, "r") as f:
        # 基础：关节角度 + 夹爪
        left_arm = f["upper_body_action_dict"]["left_arm_joint_position_cmd"][:]   # (T, dL)
        right_arm = f["upper_body_action_dict"]["right_arm_joint_position_cmd"][:] # (T, dR)
        left_gripper = f["upper_body_action_dict"]["left_arm_gripper_position_cmd"][:]  # (T, 1)
        right_gripper = f["upper_body_action_dict"]["right_arm_gripper_position_cmd"][:]  # (T, 1)

        if use_eef:
            # 使用末端位姿替代关节角
            left_arm = f["upper_body_action_dict"]["left_arm_ee_pose_cmd"][:]      # (T, 7) pos+quat
            right_arm = f["upper_body_action_dict"]["right_arm_ee_pose_cmd"][:]    # (T, 7)

        # 拼接完整 action
        actions = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=-1)

    if save_path:
        np.save(save_path, actions)
        print(f"✅ Saved actions to {save_path}, shape={actions.shape}")
    return actions


if __name__ == "__main__":
    h5_file = "./datasets/R1ProBlocksStackEasy/collected/demo_0.h5"
    save_path = "demo_0_actions.npy"

    actions = extract_actions(h5_file, save_path, use_eef=False)
    print("第一步动作:", actions[0])
    print("总步数:", actions.shape[0])
