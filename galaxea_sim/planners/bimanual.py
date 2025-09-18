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

        # 记录左右臂末端 link 名（从各自 move_group 的"最后一个关节对应 link"推断）
        left_ee_link = self.left_arm_planner.user_link_names[
            self.left_arm_planner.move_group_joint_indices[-1]
        ]
        right_ee_link = self.right_arm_planner.user_link_names[
            self.right_arm_planner.move_group_joint_indices[-1]
        ]
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
            # print(left_result)
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

    def move_to_pose_traj_augmented(
        self,
        left_pose=None,
        right_pose=None,
        robot_qpos=None,
        with_screw=False,
        verbose=True,
        num_waypoints=4,
        noise_std=0.8,
    ):
        """
        Enhanced trajectory planning with waypoint augmentation.

        This function:
        1. Plans initial trajectory from start to goal
        2. Randomly selects 1-3 waypoints along the trajectory
        3. Adds noise to joint positions at these waypoints
        4. Re-plans through the noisy waypoints to create augmented trajectory

        Args:
            left_pose: Target pose for left arm (can be None)
            right_pose: Target pose for right arm (can be None)
            robot_qpos: Current robot joint positions
            with_screw: Whether to use screw motion planning
            verbose: Whether to print debug info
            num_waypoints: Number of waypoints to add (1-3), randomly chosen if None
            noise_std: Standard deviation of noise to add to waypoints

        Returns:
            Tuple of (left_result, right_result) with augmented trajectories
        """
        # First get the original trajectory
        left_pose = self._to_pose(left_pose)
        right_pose = self._to_pose(right_pose)
        # print("use traj_augment")
        if with_screw:
            original_result = self.plan_screw(left_pose, right_pose, robot_qpos)
        else:
            original_result = self.plan_pose(
                left_pose, right_pose, robot_qpos, verbose=verbose
            )

        if original_result is None:
            return None

        left_result, right_result = original_result

        # Determine number of waypoints if not specified
        if num_waypoints is None:
            num_waypoints = np.random.randint(2, 4)  # 1 to 3 waypoints

        # Process left arm trajectory if exists
        if left_result is not None and left_result["position"].shape[0] > 0:
            left_result = self._augment_single_arm_trajectory(
                left_result,
                robot_qpos,
                self.left_arm_planner,
                self.left_arm_planner_mask,
                num_waypoints,
                noise_std,
                verbose,
            )

        # Process right arm trajectory if exists
        if right_result is not None and right_result["position"].shape[0] > 0:
            right_result = self._augment_single_arm_trajectory(
                right_result,
                robot_qpos,
                self.right_arm_planner,
                self.right_arm_planner_mask,
                num_waypoints,
                noise_std,
                verbose,
            )

        return left_result, right_result

    def _augment_single_arm_trajectory(
        self,
        arm_result,
        robot_qpos,
        arm_planner,
        arm_planner_mask,
        num_waypoints,
        noise_std,
        verbose,
    ):
        """Augment trajectory for a single arm by adding noisy waypoints in joint space.

        Notes:
        - We operate directly on arm_result["position"], which is the same shape the
          planner returns and which get_move_trajectory expects.
        - Only the last K dims (K = number of actionable joints for this arm) are
          perturbed to match how get_move_trajectory slices positions.
        - We preserve total trajectory length by interpolating each segment with the
          original segment length between waypoints.
        """
        positions = arm_result["position"]
        if positions is None or len(positions.shape) != 2:
            return arm_result

        num_steps, dim = positions.shape
        # Need at least start/mid/end to augment
        if num_steps < 1:
            return arm_result

        # Number of actionable joints (for this arm)
        arm_action_dim = int(np.sum((~arm_planner_mask).astype(np.int32)))
        arm_action_dim = max(0, min(arm_action_dim, dim))
        if arm_action_dim == 0:
            return arm_result

        # Choose waypoint indices (exclude 0 and last)
        max_mid = max(0, num_steps - 2)
        m = num_waypoints if num_waypoints is not None else np.random.randint(1, 4)
        m = max(1, min(m, max_mid)) if max_mid > 0 else 0
        if m == 0:
            return arm_result
        # Uniformly spaced waypoint indices (exclude start=0 and end=num_steps-1)
        mid_indices = np.linspace(1, num_steps - 2, num=m, dtype=int)
        mid_indices = np.unique(mid_indices)
        waypoint_indices = np.concatenate([[0], mid_indices, [num_steps - 1]])

        # Copy and perturb intermediate waypoints (only last K dims)
        noisy_positions = positions.copy()
        for idx in mid_indices:
            noise = np.random.normal(0.0, noise_std, size=(arm_action_dim,))
            noisy_positions[idx, -arm_action_dim:] = (
                noisy_positions[idx, -arm_action_dim:] + noise
            )

        # Rebuild trajectory by linear interpolation between noisy waypoints,
        # preserving each original segment length
        augmented_list = []
        for seg_i in range(len(waypoint_indices) - 1):
            start_idx = waypoint_indices[seg_i]
            end_idx = waypoint_indices[seg_i + 1]
            seg_len = end_idx - start_idx
            start_vec = noisy_positions[start_idx]
            end_vec = noisy_positions[end_idx]

            if seg_i == 0:
                # include the start point for the first segment
                if seg_len <= 0:
                    augmented_list.append(start_vec)
                else:
                    # seg_len steps between start (inclusive) and end (exclusive)
                    alphas = np.linspace(0.0, 1.0, seg_len + 1)[:-1]
                    for a in alphas:
                        augmented_list.append((1 - a) * start_vec + a * end_vec)
            else:
                if seg_len <= 0:
                    # no gap; skip
                    continue
                else:
                    # exclude the first point to avoid duplicate
                    alphas = np.linspace(0.0, 1.0, seg_len + 1)[1:-1]
                    # add interior points
                    for a in alphas:
                        augmented_list.append((1 - a) * start_vec + a * end_vec)

            # Always push the exact end waypoint for this segment (except for the very last end, which will be added by next segment or after loop)
            if seg_i < len(waypoint_indices) - 2:
                augmented_list.append(noisy_positions[end_idx])

        augmented_positions = np.stack(augmented_list, axis=0)
        # Safety: ensure same length as original
        if augmented_positions.shape[0] != num_steps:
            # Fallback: simple copy if shape mismatch (should be rare)
            if verbose:
                logger.warning(
                    f"Augmented length {augmented_positions.shape[0]} != original {num_steps}, skipping augmentation"
                )
            return arm_result

        new_result = dict(arm_result)
        new_result["position"] = augmented_positions
        return new_result

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

    def solve(self, substep, robot_qpos_in_sim, last_gripper_cmd, verbose=False):
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
            return self.get_move_trajectory(init_pos, *result)
        if method == "move_to_pose_traj_augmented":
            result = self.move_to_pose_traj_augmented(
                **kwargs, robot_qpos=robot_qpos, verbose=verbose
            )
            if result is None:
                return None
            return self.get_move_trajectory(init_pos, *result)
        elif method in ["open_gripper", "close_gripper"]:
            return self.get_gripper_trajectory(init_pos, method, kwargs)
        else:
            raise NotImplementedError(
                f"Method {method} not implemented in BimanualPlanner"
            )


class SapienBimanualPlanner(BimanualPlanner):
    def __init__(
        self,
        scene,
        robot,
        left_arm_move_group,
        right_arm_move_group,
        active_joint_names,
        control_freq,
    ):
        planning_world = SapienPlanningWorld(scene, [robot])
        self.left_arm_planner = SapienPlanner(planning_world, left_arm_move_group)
        self.right_arm_planner = SapienPlanner(planning_world, right_arm_move_group)
        self.num_dofs = len(active_joint_names)
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
        logger.debug(
            f"Left arm move group joint indices: {self.left_arm_planner.move_group_joint_indices}"
        )
        logger.debug(
            f"Right arm move group joint indices: {self.right_arm_planner.move_group_joint_indices}"
        )
        logger.debug(f"Left arm planner mask: {self.left_arm_planner_mask}")
        logger.debug(f"Right arm planner mask: {self.right_arm_planner_mask}")
        logger.debug(f"action_dim: {self.action_dim}")
        self.time_step = 1 / control_freq

        # Initialize PinocchioModel for forward kinematics
        self._fk = robot.articulation.create_pinocchio_model()
        self._fk.set_joint_order(active_joint_names)

        # Get end-effector link names
        left_ee_link = self.left_arm_planner.user_link_names[
            self.left_arm_planner.move_group_joint_indices[-1]
        ]
        right_ee_link = self.right_arm_planner.user_link_names[
            self.right_arm_planner.move_group_joint_indices[-1]
        ]
        self._fk.set_link_order([left_ee_link, right_ee_link])

        disable_table_collision(self.left_arm_planner)
        disable_table_collision(self.right_arm_planner)

    def get_gripper_trajectory(self, init_pos, method, kwargs):
        trajectory = np.stack([init_pos] * GRIPPER_STEPS)
        gripper_target_state = 0.04 if method == "open_gripper" else 0.0
        gripper_indices = []
        use_left, use_right = False, False
        if kwargs["action_mode"] == "left":
            gripper_indices = [self.left_arm_action_dim]
            use_left = True
        elif kwargs["action_mode"] == "right":
            gripper_indices = [-1]
            use_right = True
        elif kwargs["action_mode"] == "both":
            gripper_indices = [self.left_arm_action_dim, -1]
            use_left, use_right = True, True
        if use_left:
            if method == "close_gripper":
                planner_attach_obj(
                    self.left_arm_planner,
                    kwargs["cube_actor"],
                    touch_links=[
                        "left_gripper_finger_link1",
                        "left_gripper_finger_link2",
                    ],
                )
            elif "cube_actor" in kwargs:
                planner_detach_obj(self.left_arm_planner, kwargs["cube_actor"])
        if use_right:
            if method == "close_gripper":
                planner_attach_obj(
                    self.right_arm_planner,
                    kwargs["cube_actor"],
                    touch_links=[
                        "right_gripper_finger_link1",
                        "right_gripper_finger_link2",
                    ],
                )
            elif "cube_actor" in kwargs:
                planner_detach_obj(self.right_arm_planner, kwargs["cube_actor"])
        trajectory[:, gripper_indices] = gripper_target_state
        return trajectory
