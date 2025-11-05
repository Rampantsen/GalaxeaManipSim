from copy import deepcopy, copy

import sapien
import numpy as np
import transforms3d as t3d

from sapien import Pose

from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
from galaxea_sim.utils.robotwin_utils import create_table, create_box, rand_create_glb
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class DualBottlesPickEasyEnv(RoboTwinBaseEnv):
    def __init__(
        self,
        *args,
        enable_replan=False,
        enable_grasp_sample=False,
        enable_visual=False,
        table_type="white", # "redwood" or "whitewood"
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
    
    def set_planner(self, planner):
        """设置planner引用，用于grasp_sample的IK测试"""
        self.planner = planner
    @property
    def table_height(self):
        return 0.755
    
    @property
    def tabletop_center_x(self):
        return 0.7
    
    def _setup_red_bottle(self):
        self.red_bottle_xlim = [-0.2, 0.]
        self.red_bottle_ylim = [-0.25, -0.05]
        self.red_bottle_zlim = [0.125]
        self.red_bottle_qpos = [0.707, 0.707, 0, 0]
        self.red_bottle, _ = rand_create_glb(
            scene=self._scene,
            modelname="001_bottles",
            xlim=self.red_bottle_xlim,
            ylim=self.red_bottle_ylim,
            zlim=self.red_bottle_zlim,
            rotate_rand=False,
            qpos=self.red_bottle_qpos,
            scale=(0.132, 0.132, 0.132),
            model_id=13,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            convex=True
        )
        # 手动添加绕z轴的随机旋转
        current_pose = self.red_bottle.get_pose()
        z_angle = np.random.uniform(-np.pi, np.pi)
        z_rot_quat = t3d.euler.euler2quat(0, 0, z_angle)
        new_quat = t3d.quaternions.qmult(z_rot_quat, [current_pose.q[0], current_pose.q[1], current_pose.q[2], current_pose.q[3]])
        self.red_bottle.set_pose(Pose(current_pose.p, new_quat))     
        
    def _setup_green_bottle(self):
        self.green_bottle_xlim = [-0.2, 0.]
        self.green_bottle_ylim = [0.05, 0.25]
        self.green_bottle_zlim = [0.125]
        self.green_bottle_qpos = [0.707, 0.707, 0, 0]
        self.green_bottle, _ = rand_create_glb(
            scene=self._scene,
            modelname="001_bottles",
            xlim=self.green_bottle_xlim,
            ylim=self.green_bottle_ylim,
            zlim=self.green_bottle_zlim,
            rotate_rand=False,
            qpos=self.green_bottle_qpos,
            scale=(0.132, 0.132, 0.132),
            model_id=16,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            convex=True
        )
        # 手动添加绕z轴的随机旋转
        current_pose = self.green_bottle.get_pose()
        z_angle = np.random.uniform(-np.pi, np.pi)
        z_rot_quat = t3d.euler.euler2quat(0, 0, z_angle)
        new_quat = t3d.quaternions.qmult(z_rot_quat, [current_pose.q[0], current_pose.q[1], current_pose.q[2], current_pose.q[3]])
        self.green_bottle.set_pose(Pose(current_pose.p, new_quat))
        
        self.green_bottle_initial_pose_on_table = self.green_bottle.get_pose()
        self.red_bottle_initial_pose_on_table = self.red_bottle.get_pose()
        
    def reset_world(self, reset_info=None):
        if hasattr(self, "red_bottle"):
            self._scene.remove_actor(self.red_bottle)
        if hasattr(self, "green_bottle"):
            self._scene.remove_actor(self.green_bottle)
        self._setup_red_bottle()
        self._setup_green_bottle()
        if reset_info is not None:
            self.red_bottle.set_pose(sapien.Pose(reset_info["red_bottle_initial_pos_on_table"], self.red_bottle_qpos))
            self.green_bottle.set_pose(sapien.Pose(reset_info["green_bottle_initial_pos_on_table"], self.green_bottle_qpos))
            self.green_bottle_initial_pose_on_table = self.green_bottle.get_pose()
            self.red_bottle_initial_pose_on_table = self.red_bottle.get_pose()
            
    @property
    def left_target_pose(self):
        p = [0.5, 0.2, 1.2]
        (target_pose := Pose(p=p)).set_rpy(
            rpy=np.array([np.pi, np.pi, np.pi / 2], dtype=np.float32) + self.robot.left_ee_rpy_offset
        )
        return target_pose
    
    @property
    def right_target_pose(self):
        p = [0.5, -0.2, 1.2]
        (target_pose := Pose(p=p)).set_rpy(
            rpy=np.array([0, 0, np.pi / 2], dtype=np.float32)+ self.robot.right_ee_rpy_offset
        )
        return target_pose
        
    def solution(self):
        # 根据 enable_grasp_sample 决定是否使用智能抓取采样
        if self.enable_grasp_sample and self.planner is not None:
            # 采样最佳抓取角度
            left_grasp_angle, left_grasp_score = self._sample_best_grasp_angle(
                self.green_bottle, "left", self.green_bottle.get_pose().p
            )
            right_grasp_angle, right_grasp_score = self._sample_best_grasp_angle(
                self.red_bottle, "right", self.red_bottle.get_pose().p
            )
            
            # 使用采样的角度生成抓取姿态
            (left_grasp_ori := Pose()).set_rpy(
                rpy=(np.array([np.pi, -np.pi, left_grasp_angle], dtype=np.float32) + self.robot.left_ee_rpy_offset)
            )
            (right_grasp_ori := Pose()).set_rpy(
                rpy=(np.array([0, 0, right_grasp_angle], dtype=np.float32) + self.robot.right_ee_rpy_offset)
            )
            
            print(f"[Grasp Sampling] Left bottle: angle={left_grasp_angle:.2f}, score={left_grasp_score:.3f}")
            print(f"[Grasp Sampling] Right bottle: angle={right_grasp_angle:.2f}, score={right_grasp_score:.3f}")
        else:
            # 使用默认固定角度
            (left_grasp_ori := Pose()).set_rpy(rpy=(np.array([np.pi, -np.pi, 2.26], dtype=np.float32) + self.robot.left_ee_rpy_offset))
            (right_grasp_ori := Pose()).set_rpy(rpy=(np.array([0, 0, 0.88], dtype=np.float32) + self.robot.right_ee_rpy_offset))
        
        left_pose0 = Pose(p=self.green_bottle.get_pose().p+[-0.1096, 0.1164, 0.], q=left_grasp_ori.q)
        right_pose0 = Pose(p=self.red_bottle.get_pose().p+[-0.1096, -0.1164, 0.], q=right_grasp_ori.q)
        left_pose1 = Pose(p=self.green_bottle.get_pose().p+[-0.0196, 0.0164, 0.], q=left_grasp_ori.q)
        right_pose1 = Pose(p=self.red_bottle.get_pose().p+[-0.0196, -0.0164, 0.], q=right_grasp_ori.q)
        substeps = [
            ("move_to_pose", {"left_pose": deepcopy(left_pose0), "right_pose": deepcopy(right_pose0)}),
            ("open_gripper", {"action_mode": "both"}),
            ("move_to_pose", {"left_pose": deepcopy(left_pose1), "right_pose": deepcopy(right_pose1)}),
            ("close_gripper", {"action_mode": "both"}),
            ("move_to_pose", {"left_pose": deepcopy(self.left_target_pose), "right_pose": deepcopy(self.right_target_pose)}),
            ("move_to_pose", {"left_pose": deepcopy(self.left_target_pose), "right_pose": deepcopy(self.right_target_pose)}),
        ]
        for substep in substeps:
            yield substep
    
    def _get_info(self):
        left_distance = np.linalg.norm(self.green_bottle.get_pose().p - self.left_target_pose.p)
        right_distance = np.linalg.norm(self.red_bottle.get_pose().p - self.right_target_pose.p)
        left_height = self.green_bottle.get_pose().p[2].item()
        right_height = self.red_bottle.get_pose().p[2].item()
        left_success = left_distance < 0.15 and left_height >= self.left_target_pose.p[2].item() - 0.07
        right_success = right_distance < 0.15 and right_height >= self.right_target_pose.p[2].item() - 0.07
        
        return dict(
            left_success=left_success,
            right_success=right_success,
            success=left_success and right_success,
            left_distance=left_distance,
            right_distance=right_distance,
            left_height=left_height,
            right_height=right_height,
        )
    
    def _get_reset_info(self):
        return dict(
            red_bottle_initial_pos_on_table=copy(self.red_bottle_initial_pose_on_table.p),
            green_bottle_initial_pos_on_table=copy(self.green_bottle_initial_pose_on_table.p),
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "pick up the two bottles simultaneously"
    
    def get_object_dict(self):
        return dict(
            red_bottle=np.concatenate([self.red_bottle.get_pose().p, self.red_bottle.get_pose().q]),
            green_bottle=np.concatenate([self.green_bottle.get_pose().p, self.green_bottle.get_pose().q]),
        )
        
    def _sample_best_grasp_angle(self, bottle, arm, bottle_pos):
        """采样最佳抓取角度（6个均匀分布的候选角度）
        
        Args:
            bottle: 要抓取的瓶子对象
            arm: 使用的机械臂 ("left" 或 "right")
            bottle_pos: 瓶子位置
            
        Returns:
            tuple: (best_angle, best_score) 最佳角度和对应分数
        """
        # 生成6个均匀分布的候选角度（绕z轴）
        num_candidates = 6
        candidate_angles = [i * (2 * np.pi / num_candidates) for i in range(num_candidates)]
        
        # 如果没有planner，返回默认角度
        if self.planner is None:
            if arm == "left":
                return 2.26, 0.0  # 默认左臂角度
            else:
                return 0.88, 0.0  # 默认右臂角度
        
        best_angle = None
        best_score = float('-inf')
        valid_solutions = []
        
        # 获取当前机器人状态
        current_qpos = self.robot.get_qpos()
        
        # 测试每个候选角度
        from mplib.pymp import Pose as MPPose
        for angle in candidate_angles:
            # 构建测试抓取姿态
            if arm == "left":
                (test_ori := Pose()).set_rpy(
                    rpy=(np.array([np.pi, -np.pi, angle], dtype=np.float32) + self.robot.left_ee_rpy_offset)
                )
                # 左臂抓取位置稍微偏移
                test_pose = Pose(p=bottle_pos + [-0.0196, 0.0164, 0.], q=test_ori.q)
            else:
                (test_ori := Pose()).set_rpy(
                    rpy=(np.array([0, 0, angle], dtype=np.float32) + self.robot.right_ee_rpy_offset)
                )
                # 右臂抓取位置稍微偏移
                test_pose = Pose(p=bottle_pos + [-0.0196, -0.0164, 0.], q=test_ori.q)
            
            # 转换为 mplib Pose
            mp_pose = MPPose(p=test_pose.p, q=test_pose.q)
            
            # 测试IK可解性
            if arm == "left":
                result = self.planner.move_to_pose(
                    left_pose=mp_pose,
                    right_pose=None,
                    robot_qpos=current_qpos,
                    verbose=False
                )
            else:
                result = self.planner.move_to_pose(
                    left_pose=None,
                    right_pose=mp_pose,
                    robot_qpos=current_qpos,
                    verbose=False
                )
            
            # 如果IK有解，计算评分
            if result is not None:
                # 评分标准：
                score = 0.0
                
                # 1. 关节运动幅度评分（越小越好）
                if arm == "left" and result[0] is not None:
                    # 获取关节轨迹
                    joint_trajectory = result[0]['position']
                    if len(joint_trajectory) > 0:
                        # 计算关节运动总量
                        joint_motion = np.sum(np.abs(joint_trajectory[-1] - joint_trajectory[0]))
                        motion_score = 1.0 / (1.0 + joint_motion)  # 运动越小分数越高
                        score += motion_score * 0.4
                        
                        # 轨迹长度评分（越短越好）
                        traj_length_score = 1.0 / (1.0 + len(joint_trajectory) * 0.01)
                        score += traj_length_score * 0.3
                elif arm == "right" and result[1] is not None:
                    joint_trajectory = result[1]['position']
                    if len(joint_trajectory) > 0:
                        joint_motion = np.sum(np.abs(joint_trajectory[-1] - joint_trajectory[0]))
                        motion_score = 1.0 / (1.0 + joint_motion)
                        score += motion_score * 0.4
                        
                        traj_length_score = 1.0 / (1.0 + len(joint_trajectory) * 0.01)
                        score += traj_length_score * 0.3
                
                # 2. 抓取角度偏好评分（偏好某些角度）
                # 对于左臂，偏好 2.26 附近；对于右臂，偏好 0.88 附近
                if arm == "left":
                    preferred_angle = 2.26
                else:
                    preferred_angle = 0.88
                
                angle_diff = abs(angle - preferred_angle)
                # 归一化角度差到 [0, π]
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                angle_score = 1.0 - (angle_diff / np.pi)  # 角度差越小分数越高
                score += angle_score * 0.3
                
                valid_solutions.append((angle, score))
                
                # 更新最佳解
                if score > best_score:
                    best_score = score
                    best_angle = angle
        
        # 如果没有找到可行解，使用默认角度
        if best_angle is None:
            print(f"Warning: No valid grasp angle found for {arm} arm, using default")
            if arm == "left":
                return 2.26, 0.0
            else:
                return 0.88, 0.0
        
        # 打印所有可行解（调试用）
        if len(valid_solutions) > 0 and self.enable_visual:
            print(f"\n[{arm} arm] Found {len(valid_solutions)} valid grasp angles:")
            for angle, score in sorted(valid_solutions, key=lambda x: x[1], reverse=True):
                print(f"  Angle: {angle:.2f} rad ({np.degrees(angle):.1f}°), Score: {score:.3f}")
        
        return best_angle, best_score
    
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0