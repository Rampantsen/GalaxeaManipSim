import os

import mplib
import numpy as np
from mplib.pymp import Pose
from mplib.sapien_utils.conversion import SapienPlanner, SapienPlanningWorld
from sapien.wrapper.pinocchio_model import PinocchioModel
from loguru import logger
import transforms3d
from galaxea_sim import ASSETS_DIR
from galaxea_sim.planners.base import BasePlanner
from galaxea_sim.utils.mplib_utils import (
    get_planner_mask,
    get_sim2mplib_mapping,
    planner_attach_obj,
    planner_detach_obj,
    disable_table_collision,
)

GRIPPER_STEPS = 10


class BimanualPlanner(BasePlanner):
    def __init__(
        self,
        urdf_path,
        srdf_path,
        left_arm_move_group,
        right_arm_move_group,
        active_joint_names,
        control_freq,
        env=None,
    ):
        self.left_arm_planner = mplib.planner.Planner(
            urdf=ASSETS_DIR / urdf_path,
            srdf=ASSETS_DIR / srdf_path if srdf_path else None,
            move_group=left_arm_move_group,
        )
        self.right_arm_planner = mplib.planner.Planner(
            urdf=ASSETS_DIR / urdf_path,
            srdf=ASSETS_DIR / srdf_path if srdf_path else None,
            move_group=right_arm_move_group,
        )
        # ---------- 全身 Pinocchio FK 初始化 ----------
        urdf_str = (ASSETS_DIR / urdf_path).read_text()
        self._fk = PinocchioModel(urdf_str, [0.0, 0.0, -9.81])
        # FK 的关节顺序 = active_joint_names（全身顺序）
        self._fk.set_joint_order(active_joint_names)

        # 记录左右臂末端 link 名（直接指定正确的末端执行器）
        left_ee_link = "left_gripper_link"
        right_ee_link = "right_gripper_link"
        # set_link_order 的顺序我们约定：index=0 -> left, index=1 -> right
        self._fk.set_link_order([left_ee_link, right_ee_link])

        # 安全打印
        logger.debug(f"FK joint_order size: {len(active_joint_names)}")
        logger.debug(f"FK link_order: left='{left_ee_link}', right='{right_ee_link}'")

        self.time_step = 1 / control_freq
        num_dofs = len(active_joint_names)
        self.left_arm_planner_mask = get_planner_mask(
            num_dofs,
            self.left_arm_planner.move_group_joint_indices,
            self.right_arm_planner.move_group_joint_indices,
        )
        self.right_arm_planner_mask = get_planner_mask(
            num_dofs,
            self.right_arm_planner.move_group_joint_indices,
            self.left_arm_planner.move_group_joint_indices,
        )
        self.sim2mplib_mapping = get_sim2mplib_mapping(
            active_joint_names,
            self.left_arm_planner.user_joint_names,
        )
        self.action_dim = self.left_arm_action_dim + self.right_arm_action_dim + 2
        
        # 轨迹增强（执行噪声模式）相关参数
        self.accumulated_position_error = 0.0  # 末端位置最大误差（米）
        self.accumulated_rotation_error = 0.0  # 保留字段（未使用）
        self.traj_augmented_enabled = False   # 是否启用轨迹增强（执行噪声）
        self.planned_actions = []  # 存储规划的正确action
        self.executed_actions = []  # 存储带噪声的执行action
        
        logger.debug(
            f"Left arm move group joint indices: {self.left_arm_planner.move_group_joint_indices}"
        )
        logger.debug(
            f"Right arm move group joint indices: {self.right_arm_planner.move_group_joint_indices}"
        )
        logger.debug(f"Left arm planner mask: {self.left_arm_planner_mask}")
        logger.debug(f"Right arm planner mask: {self.right_arm_planner_mask}")
        logger.debug(f"action_dim: {self.action_dim}")

    @property
    def left_arm_action_dim(self):
        return np.sum((~self.left_arm_planner_mask).astype(np.int32))

    @property
    def right_arm_action_dim(self):
        return np.sum((~self.right_arm_planner_mask).astype(np.int32))

    def plan_pose(self, left_pose, right_pose, robot_qpos, verbose=False):
        left_result = None
        right_result = None
        if left_pose is not None:
            left_result = self.left_arm_planner.plan_pose(
                left_pose,
                robot_qpos,
                time_step=self.time_step,
                mask=self.left_arm_planner_mask,
                verbose=verbose,
            )

            if left_result["status"] != "Success":
                if verbose:
                    logger.error(left_result["status"])
                return None

        if right_pose is not None:
            right_result = self.right_arm_planner.plan_pose(
                right_pose,
                robot_qpos,
                time_step=self.time_step,
                mask=self.right_arm_planner_mask,
                verbose=verbose,
            )
            if right_result["status"] != "Success":
                if verbose:
                    logger.error(right_result["status"])
                return None

        return left_result, right_result

    def plan_screw(self, left_pose, right_pose, robot_qpos):
        left_result = (
            self.left_arm_planner.plan_screw(
                left_pose, robot_qpos, time_step=self.time_step
            )
            if left_pose
            else None
        )
        right_result = (
            self.right_arm_planner.plan_screw(
                right_pose, robot_qpos, time_step=self.time_step
            )
            if right_pose
            else None
        )

        if (left_result and left_result["status"] != "Success") or (
            right_result and right_result["status"] != "Success"
        ):
            return None
        return left_result, right_result

    def _to_pose(self, pose_like):
        if pose_like is None:
            return None
        if isinstance(pose_like, (list, np.ndarray)):
            return Pose(p=pose_like[:3], q=pose_like[3:])
        return pose_like

    def move_to_pose(
        self,
        left_pose=None,
        right_pose=None,
        robot_qpos=None,
        with_screw=False,
        verbose=True,
    ):
        """API to multiplex between the two planning methods"""
        left_pose = self._to_pose(left_pose)
        right_pose = self._to_pose(right_pose)
        if with_screw:
            return self.plan_screw(left_pose, right_pose, robot_qpos)
        else:
            return self.plan_pose(left_pose, right_pose, robot_qpos, verbose=verbose)


    def get_move_trajectory(self, init_pos, left_result=None, right_result=None):
        n_step_left = left_result["position"].shape[0] if left_result else 0
        n_step_right = right_result["position"].shape[0] if right_result else 0
        n_step = max(n_step_left, n_step_right)
        trajectory = (
            np.stack([init_pos] * n_step)
            if n_step > 0
            else np.zeros((1, self.action_dim))
        )
        left_arm_action_dim = self.left_arm_action_dim
        right_arm_action_dim = self.right_arm_action_dim
        if n_step_left > 0:
            trajectory[:n_step_left, :left_arm_action_dim] = left_result["position"][
                ..., -left_arm_action_dim:
            ]
            trajectory[n_step_left:, :left_arm_action_dim] = left_result["position"][
                -1
            ][-left_arm_action_dim:]
        if n_step_right > 0:
            trajectory[:n_step_right, -right_arm_action_dim - 1 : -1] = right_result[
                "position"
            ][..., -right_arm_action_dim:]
            trajectory[n_step_right:, -right_arm_action_dim - 1 : -1] = right_result[
                "position"
            ][-1][-right_arm_action_dim:]
        return trajectory

    def get_gripper_trajectory(self, init_pos, method, kwargs):
        trajectory = np.stack([init_pos] * kwargs.get("steps", GRIPPER_STEPS))
        if "gripper_target_state" in kwargs.keys():
            gripper_target_state = kwargs["gripper_target_state"]
        else:
            gripper_target_state = 0.05 if method == "open_gripper" else 0.0
        gripper_indices = []
        if kwargs["action_mode"] == "left":
            gripper_indices = [self.left_arm_action_dim]
        elif kwargs["action_mode"] == "right":
            gripper_indices = [-1]
        elif kwargs["action_mode"] == "both":
            gripper_indices = [self.left_arm_action_dim, -1]
        trajectory[:GRIPPER_STEPS, gripper_indices] = np.linspace(
            init_pos[gripper_indices], gripper_target_state, GRIPPER_STEPS
        )
        trajectory[GRIPPER_STEPS:, gripper_indices] = gripper_target_state
        return trajectory

    def enable_traj_augmented_mode(self, enable=True):
        """启用或禁用轨迹增强模式（执行噪声）"""
        self.traj_augmented_enabled = enable
        if enable:
            logger.info("轨迹增强模式（执行噪声）已启用")
        else:
            logger.info("轨迹增强模式（执行噪声）已禁用")
    
    def reset_accumulated_error(self):
        """重置累积误差"""
        self.accumulated_position_error = 0.0
        self.accumulated_rotation_error = 0.0
        self.planned_actions = []
        self.executed_actions = []
        #logger.debug("累积误差已重置")
    
    def add_execution_noise(self, actions, noise_probability=0.3, 
                           position_noise_std=0.005, rotation_noise_std=0.05):
        """
        给动作序列添加执行噪声，模拟机器人执行误差
        
        Args:
            actions: 规划的动作序列 (N, action_dim)
            noise_probability: 产生噪声的概率
            position_noise_std: 位置噪声标准差 (关节空间，弧度)
            rotation_noise_std: 旋转噪声标准差 (关节空间，弧度)
        
        Returns:
            noisy_actions: 带噪声的动作序列
            planned_actions: 原始规划的动作（用于记录）
        """
        if actions is None:
            return None, None
        
        planned_actions = actions.copy()
        noisy_actions = actions.copy()
        
        # 对每一步动作添加噪声
        for i in range(len(noisy_actions)):
            # 以一定概率添加噪声
            if np.random.random() < noise_probability:
                # 对关节位置添加高斯噪声（不包括夹爪关节）
                # 左臂关节噪声
                left_noise = np.random.normal(0, position_noise_std, self.left_arm_action_dim)
                noisy_actions[i, :self.left_arm_action_dim] += left_noise
                
                # 右臂关节噪声
                right_start = self.left_arm_action_dim + 1
                right_end = right_start + self.right_arm_action_dim
                right_noise = np.random.normal(0, position_noise_std, self.right_arm_action_dim)
                noisy_actions[i, right_start:right_end] += right_noise
                
                # 夹爪不添加噪声，保持精确控制
        
        return noisy_actions, planned_actions
    
    def compute_pose_error(self, qpos_planned, qpos_actual):
        """
        计算末端执行器位置误差（使用FK）
        
        Args:
            qpos_planned: 目标关节位置
            qpos_actual: 实际关节位置
        
        Returns:
            position_error: 末端执行器位置误差，取左右臂最大值 (m)
            rotation_error: 末端执行器旋转误差，取左右臂最大值 (rad)
        """
        # 使用FK计算末端执行器位姿
        # 注意：这里假设qpos是全身关节位置
        try:
            # 计算规划位姿的末端执行器位置
            self._fk.compute_forward_kinematics(qpos_planned)
            # get_link_pose需要link索引: 0=left_ee, 1=right_ee
            planned_left_pose = self._fk.get_link_pose(0)
            planned_right_pose = self._fk.get_link_pose(1)
            
            # 计算实际位姿的末端执行器位置
            self._fk.compute_forward_kinematics(qpos_actual)
            actual_left_pose = self._fk.get_link_pose(0)
            actual_right_pose = self._fk.get_link_pose(1)
            
            # 计算位置误差（取左右臂最大值）
            position_errors = []
            rotation_errors = []
            
            for planned_pose, actual_pose in [(planned_left_pose, actual_left_pose), 
                                               (planned_right_pose, actual_right_pose)]:
                # 位置误差
                pos_error = np.linalg.norm(planned_pose.p - actual_pose.p)
                position_errors.append(pos_error)
                
                # 旋转误差（使用四元数点积计算角度差）
                q1 = planned_pose.q
                q2 = actual_pose.q
                dot_product = np.abs(np.dot(q1, q2))
                dot_product = np.clip(dot_product, -1.0, 1.0)
                rot_error = 2 * np.arccos(dot_product)
                rotation_errors.append(rot_error)
            
            # 返回最大误差
            return max(position_errors), max(rotation_errors)
        except Exception as e:
            logger.warning(f"计算位姿误差失败: {e}")
            return 0.0, 0.0
    
    def check_replan_needed(self):
        """
        检查是否需要重新规划
        
        Returns:
            bool: 如果末端位置误差超过阈值返回True
        """
        if not self.traj_augmented_enabled:
            return False
        
        # 使用末端位置误差（单位：米）
        # 阈值设置为0.03m（3厘米）比较合理
        error_threshold = 0.04  # 米
        
        if self.accumulated_position_error > error_threshold:
            # logger.warning(
            #     f"末端位置误差超过阈值! "
            #     f"当前误差: {self.accumulated_position_error:.4f}m ({self.accumulated_position_error*1000:.1f}mm), "
            #     f"阈值: {error_threshold}m ({error_threshold*1000:.0f}mm)"
            # )
            return True
        return False

    def solve(self, substep, robot_qpos_in_sim, last_gripper_cmd, verbose=False):
        """
        规划并求解动作序列
        
        如果method是traj_augmented，会应用执行噪声：
        - 返回 (executed_actions, planned_actions)
        否则返回 actions
        """
        method, kwargs = substep
        robot_qpos = self.sim2mplib_mapping(robot_qpos_in_sim)
        init_pos = np.zeros(self.action_dim)
        init_pos[: self.left_arm_action_dim] = robot_qpos[~self.left_arm_planner_mask]
        init_pos[self.left_arm_action_dim] = last_gripper_cmd[0]
        init_pos[-self.right_arm_action_dim - 1 : -1] = robot_qpos[
            ~self.right_arm_planner_mask
        ]
        init_pos[-1] = last_gripper_cmd[1]

        if method == "move_to_pose":
            result = self.move_to_pose(**kwargs, robot_qpos=robot_qpos, verbose=verbose)
            if result is None:
                return None
            trajectory = self.get_move_trajectory(init_pos, *result)
        elif method == "move_to_pose_traj_augmented":
            # 轨迹增强：规划正确的轨迹，在执行时添加噪声
            result = self.move_to_pose(
                **kwargs, robot_qpos=robot_qpos, verbose=verbose
            )
            if result is None:
                return None
            trajectory = self.get_move_trajectory(init_pos, *result)
            
            # traj_augmented方法总是应用执行噪声
            # 返回 (noisy_trajectory, planned_trajectory)
            noisy_trajectory, planned_trajectory = self.add_execution_noise(
                trajectory,
                noise_probability=kwargs.get('noise_probability', 0.6),
                position_noise_std=kwargs.get('position_noise_std', 0.008),
            )
            return (noisy_trajectory, planned_trajectory)
        elif method in ["open_gripper", "close_gripper"]:
            trajectory = self.get_gripper_trajectory(init_pos, method, kwargs)
        else:
            raise NotImplementedError(
                f"Method {method} not implemented in BimanualPlanner"
            )
        
        return trajectory


# class SapienBimanualPlanner(BimanualPlanner):
#     def __init__(
#         self,
#         scene,
#         robot,
#         left_arm_move_group,
#         right_arm_move_group,
#         active_joint_names,
#         control_freq,
#     ):
#         planning_world = SapienPlanningWorld(scene, [robot])
#         self.left_arm_planner = SapienPlanner(planning_world, left_arm_move_group)
#         self.right_arm_planner = SapienPlanner(planning_world, right_arm_move_group)
#         self.num_dofs = len(active_joint_names)
#         num_dofs = len(active_joint_names)
#         self.left_arm_planner_mask = get_planner_mask(
#             num_dofs,
#             self.left_arm_planner.move_group_joint_indices,
#             self.right_arm_planner.move_group_joint_indices,
#         )
#         self.right_arm_planner_mask = get_planner_mask(
#             num_dofs,
#             self.right_arm_planner.move_group_joint_indices,
#             self.left_arm_planner.move_group_joint_indices,
#         )
#         self.sim2mplib_mapping = get_sim2mplib_mapping(
#             active_joint_names,
#             self.left_arm_planner.user_joint_names,
#         )
#         self.action_dim = self.left_arm_action_dim + self.right_arm_action_dim + 2
#         logger.debug(
#             f"Left arm move group joint indices: {self.left_arm_planner.move_group_joint_indices}"
#         )
#         logger.debug(
#             f"Right arm move group joint indices: {self.right_arm_planner.move_group_joint_indices}"
#         )
#         logger.debug(f"Left arm planner mask: {self.left_arm_planner_mask}")
#         logger.debug(f"Right arm planner mask: {self.right_arm_planner_mask}")
#         logger.debug(f"action_dim: {self.action_dim}")
#         self.time_step = 1 / control_freq

#         # Initialize PinocchioModel for forward kinematics
#         self._fk = robot.articulation.create_pinocchio_model()
#         self._fk.set_joint_order(active_joint_names)

#         # Get end-effector link names (直接指定正确的末端执行器)
#         left_ee_link = "left_gripper_link"
#         right_ee_link = "right_gripper_link"
#         self._fk.set_link_order([left_ee_link, right_ee_link])

#         disable_table_collision(self.left_arm_planner)
#         disable_table_collision(self.right_arm_planner)

#     def get_gripper_trajectory(self, init_pos, method, kwargs):
#         trajectory = np.stack([init_pos] * GRIPPER_STEPS)
#         gripper_target_state = 0.04 if method == "open_gripper" else 0.0
#         gripper_indices = []
#         use_left, use_right = False, False
#         if kwargs["action_mode"] == "left":
#             gripper_indices = [self.left_arm_action_dim]
#             use_left = True
#         elif kwargs["action_mode"] == "right":
#             gripper_indices = [-1]
#             use_right = True
#         elif kwargs["action_mode"] == "both":
#             gripper_indices = [self.left_arm_action_dim, -1]
#             use_left, use_right = True, True
#         if use_left:
#             if method == "close_gripper":
#                 planner_attach_obj(
#                     self.left_arm_planner,
#                     kwargs["cube_actor"],
#                     touch_links=[
#                         "left_gripper_finger_link1",
#                         "left_gripper_finger_link2",
#                     ],
#                 )
#             elif "cube_actor" in kwargs:
#                 planner_detach_obj(self.left_arm_planner, kwargs["cube_actor"])
#         if use_right:
#             if method == "close_gripper":
#                 planner_attach_obj(
#                     self.right_arm_planner,
#                     kwargs["cube_actor"],
#                     touch_links=[
#                         "right_gripper_finger_link1",
#                         "right_gripper_finger_link2",
#                     ],
#                 )
#             elif "cube_actor" in kwargs:
#                 planner_detach_obj(self.right_arm_planner, kwargs["cube_actor"])
#         trajectory[:, gripper_indices] = gripper_target_state
#         return trajectory
