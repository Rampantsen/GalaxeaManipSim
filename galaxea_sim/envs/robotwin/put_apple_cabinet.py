from copy import deepcopy
import math
import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import create_obj, rand_create_urdf_obj
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class PutAppleCabinetEnv(RoboTwinBaseEnv):
    def __init__(
        self,
        *args,
        enable_replan=False,
        enable_grasp_sample=False,
        enable_visual=False,
        table_type="red", # "redwood" or "whitewood"
        replan_noise_range=(0.02, 0.05),  # replan位置噪声范围
        replan_prob=0.5,  # replan触发概率
        **kwargs,
    ):
        # 将 table_type 传递给父类
        super().__init__(*args, table_type=table_type, **kwargs)
        self.current_arm = None  # 跟踪当前使用的机械臂
        self.enable_replan= enable_replan  # 控制是否启用retry逻辑
        self.enable_grasp_sample = enable_grasp_sample  # 控制是否启用抓取角度采样
        self.enable_visual = enable_visual  # 控制是否启用可视化
        self.planner = None
        self.replan_noise_range = replan_noise_range
        self.replan_prob = replan_prob 
    def _setup_apple(self):
        apple_pose = rand_pose(
            xlim=[-0.2, -0.1],
            ylim=[-0.32, -0.2],
            zlim=[0.04],
            rotate_rand=False,
        )
        self.apple, self.apple_data = create_obj(
            self._scene,
            pose=apple_pose,
            modelname="035_apple",
            convex=True,
            tabeltop_center_in_world=self.tabletop_center_in_world,
        )   
        
    def _setup_cabinet(self):
        self.cabinet, _ = rand_create_urdf_obj(
            self._scene,
            modelname="036_cabine",
            xlim=[0.2, 0.2],
            ylim=[0.1, 0.1],
            zlim=[0.22],
            rotate_rand=False,
            qpos=[1, 0, 0, 0],
            scale=0.27,
            tabletop_center_in_world=self.tabletop_center_in_world,
        )
        self.cabinet_active_joints = self.cabinet.get_active_joints()
        for joint in self.cabinet_active_joints:
            joint.set_drive_property(stiffness=10, damping=5, force_limit=1000, mode="force")
        self.cabinet_all_joints = self.cabinet.get_joints()

    def reset_world(self, reset_info=None):
        # chek if red bottle on the scene, then remove it
        if hasattr(self, "apple"):
            self._scene.remove_actor(self.apple)
        if hasattr(self, "cabinet"):
            self.cabinet.set_qpos([0, 0, 0])
        else:
            self._setup_cabinet()
        self._setup_apple()
        if reset_info is not None:
            self.apple.set_pose(sapien.Pose(p=reset_info["init_apple_pose"][:3], q=reset_info["init_apple_pose"][3:]))

    def rot_down_grip_pose(self, pose: sapien.Pose,grasp_angle: float = math.pi / 4):
        pose_mat = pose.to_transformation_matrix()
        (lower_trans_quat := sapien.Pose()).set_rpy(rpy=(np.array([0, grasp_angle, 0])))
        lower_trans_mat = lower_trans_quat.to_transformation_matrix()

        new_pos = np.dot(pose_mat, lower_trans_mat)
        new_pose = sapien.Pose(matrix=new_pos)
        return new_pose

    def tf_to_grasp(self, pose: list, grasp_angle: float = math.pi / 4,is_apple: bool = True):
        # 根据 grasp_angle 动态选择 offset
        if is_apple:
            grasp_apple_offsets = {
                math.pi / 6: [-0.055, 0, 0.025], 
                math.pi / 5: [-0.052, 0, 0.023],   
                math.pi / 4: [-0.05, 0, 0.02],  
            }
            grasp_offset = grasp_apple_offsets.get(grasp_angle, [-0.05, 0, 0.02])
        else:
            grasp_cabinet_offsets = {
                math.pi / 6: [-0.055, 0, 0.025], 
                math.pi / 5: [-0.052, 0, 0.023],   
                math.pi / 4: [-0.05, 0, 0.02],  
            }
            grasp_offset = grasp_cabinet_offsets.get(grasp_angle, [-0.05, 0, 0.02])
        origin_pose = sapien.Pose(p=pose[:3], q=pose[3:])
        pose_mat = origin_pose.to_transformation_matrix()
        tf_mat = np.array(
            [[1, 0, 0, grasp_offset[0]], [0, 1, 0, grasp_offset[1]], [0, 0, 1, grasp_offset[2]], [0, 0, 0, 1]]
        )
        new_pos = np.dot(pose_mat, tf_mat)
        new_pose = sapien.Pose(matrix=new_pos)
        return list(new_pose.p) + list(new_pose.q)        
    def solution(self):
        pose0 = [0.58273, 0.225689, 0.91621,0.598498, -0.324811, -0.64746, -0.342189]
        yield ("move_to_pose", {"left_pose": deepcopy(pose0)})

        # pre_pose0 = list(self.cabinet.get_pose().p + [-0.25, 0.07, 0]) + [0.707, 0.707, 0, 0]
        # pose0 = list(self.cabinet.get_pose().p + [-0.25, 0.07, -0.09]) + [0.707, 0.707, 0, 0]
        # pose1 = list(self.apple.get_pose().p + [0, 0, 0.17]) + [0.707, 0, 0.707, 0]
        # yield ("move_to_pose", {"left_pose": deepcopy(pre_pose0), "right_pose": deepcopy(pose1)})
        # yield ("move_to_pose", {"left_pose": deepcopy(pose0)})
        # yield ("open_gripper", {"action_mode": "both"})
        # pose0[0] += 0.075
        # pose1[2] -= 0.15
        # yield ("move_to_pose", {"left_pose": deepcopy(pose0), "right_pose": deepcopy(pose1)})
        # yield ("close_gripper", {"action_mode": "both"})
        # pose0[0] -= 0.15
        # pose1[2] += 0.15
        # yield ("move_to_pose", {"left_pose": deepcopy(pose0), "right_pose": deepcopy(pose1)})
        
        # pose2 = list(self.cabinet.get_pose().p + [-0.18, -0.12, 0.02]) + [0.707, 0, 0, 0.707]
        # yield ("move_to_pose", {"right_pose": deepcopy(pose2)})
        # yield ("open_gripper", {"action_mode": "right", "steps": 30})
        # pose0[0] += 0.15
        # pose2[1] -= 0.15
        # yield ("move_to_pose", {"left_pose": deepcopy(pose0), "right_pose": deepcopy(pose2)})
        # yield ("open_gripper", {"action_mode": "left"})
        # pose0[0] -= 0.05
        # yield ("move_to_pose", {"left_pose": deepcopy(pose0)})
        # yield ("move_to_pose", {"left_pose": self.robot.left_init_ee_pose, "right_pose": self.robot.right_init_ee_pose})
        
    def _get_info(self):
        # 获取苹果和柜子的位置
        apple_pos = self.apple.get_pose().p
        cabinet_pos = self.cabinet.get_pose().p
        
        # 定义柜子内部的有效范围（基于柜子位置和实际尺寸）
        # 柜子位置大约在 (0.2, 0.1, 0.22)，scale=0.27
        # 考虑柜子内部空间，设置合理的边界
        cabinet_x_min = cabinet_pos[0] - 0.15  # 柜子深度范围
        cabinet_x_max = cabinet_pos[0] + 0.05  # 柜子前面（门的位置）
        cabinet_y_min = cabinet_pos[1] - 0.10  # 柜子宽度范围
        cabinet_y_max = cabinet_pos[1] + 0.10
        cabinet_z_min = cabinet_pos[2] - 0.05  # 柜子底部
        cabinet_z_max = cabinet_pos[2] + 0.15  # 柜子顶部
        
        # 检查苹果是否在柜子的3D边界内
        x_in_range = cabinet_x_min < apple_pos[0] < cabinet_x_max
        y_in_range = cabinet_y_min < apple_pos[1] < cabinet_y_max
        z_in_range = cabinet_z_min < apple_pos[2] < cabinet_z_max
        
        success = x_in_range and y_in_range and z_in_range
        
        return dict(
            success=success,
            # 添加调试信息
            apple_pos=apple_pos.tolist(),
            cabinet_pos=cabinet_pos.tolist(),
            x_in_range=x_in_range,
            y_in_range=y_in_range,
            z_in_range=z_in_range
        )
    
    def _get_reset_info(self):
        return dict(
            init_apple_pose=np.concatenate([self.apple.get_pose().p, self.apple.get_pose().q]),
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "put the apple in the cabinet" 
    
    def get_object_dict(self):
        return dict(
            apple=np.concatenate([self.apple.get_pose().p, self.apple.get_pose().q]),
            cabinet_qpos=np.array(self.cabinet.get_qpos()),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0