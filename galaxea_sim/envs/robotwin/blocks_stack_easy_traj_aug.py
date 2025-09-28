import math
from copy import deepcopy
import random

import sapien
import numpy as np
import transforms3d

from galaxea_sim.utils.robotwin_utils import create_box
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv
import os

# 定义可选的抓取角度
GRASP_ANGLES = [0, math.pi / 6, math.pi / 4, math.pi / 3]

# 定义每个抓取角度对应的偏移量 (x, y, z)
GRASP_OFFSETS = {
    0: [-0.05, 0, 0.04],  # 0度抓取
    math.pi / 6: [-0.045, 0, 0.025],  # 30度抓取
    math.pi / 4: [-0.04, 0, 0.03],  # 45度抓取
    math.pi / 3: [-0.035, 0, 0.01],  # 60度抓取
}


class BlocksStackEasyTrajAugEnv(RoboTwinBaseEnv):
    place_pos_x_offset = -0.2
    block_half_size = 0.02

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_arm = None  # 跟踪当前使用的机械臂

    def _rand_pose(self):
        return rand_pose(
            xlim=[-0.12, -0.01],
            ylim=[-0.25, 0.25],
            zlim=[self.block_half_size],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi * 2],  # 绕z轴随机旋转-π到π弧度
            z_rotate_only=True,  # 只绕z轴旋转
        )

    def _setup_block1(self):
        rand_pos = self._rand_pose()
        while (
            abs(rand_pos.p[1]) < 0.05
            or np.sum(pow(rand_pos.p[:2] - np.array([self.place_pos_x_offset, 0]), 2))
            < 0.0225
        ):
            rand_pos = self._rand_pose()
        rand_pos.set_p(rand_pos.p + self.tabletop_center_in_world)

        self.block1 = create_box(
            scene=self._scene,
            pose=rand_pos,
            half_size=(
                self.block_half_size,
                self.block_half_size,
                self.block_half_size,
            ),
            color=(1, 0, 0),
            name="box",
        )

    def _setup_block2(self):
        rand_pos = self._rand_pose()
        while (
            abs(rand_pos.p[1]) < 0.05
            or np.sum(pow(rand_pos.p[:2] - np.array([self.place_pos_x_offset, 0]), 2))
            < 0.0225
            or np.linalg.norm(
                (rand_pos.p[:2] + self.tabletop_center_in_world[:2])
                - self.block1.get_pose().p[:2]
            )
            < 0.1
        ):
            rand_pos = self._rand_pose()
        rand_pos.set_p(rand_pos.p + self.tabletop_center_in_world)
        self.block2 = create_box(
            scene=self._scene,
            pose=rand_pos,
            half_size=(
                self.block_half_size,
                self.block_half_size,
                self.block_half_size,
            ),
            color=(0, 1, 0),
            name="box",
        )

    def reset_world(self, reset_info=None):
        if hasattr(self, "block1"):
            self._scene.remove_actor(self.block1)
        if hasattr(self, "block2"):
            self._scene.remove_actor(self.block2)
        self._setup_block1()
        self._setup_block2()
        if reset_info is not None:
            self.block1.set_pose(
                sapien.Pose(
                    reset_info["block1_pose"][:3], reset_info["block1_pose"][3:]
                )
            )
            self.block2.set_pose(
                sapien.Pose(
                    reset_info["block2_pose"][:3], reset_info["block2_pose"][3:]
                )
            )

    def rot_down_grip_pose(self, pose: sapien.Pose, grasp_angle: float = math.pi / 4):
        """根据抓取角度旋转抓取姿态

        Args:
            pose: 原始姿态
            grasp_angle: 抓取角度 (弧度)
        """
        pose_mat = pose.to_transformation_matrix()
        (lower_trans_quat := sapien.Pose()).set_rpy(rpy=(np.array([0, grasp_angle, 0])))
        lower_trans_mat = lower_trans_quat.to_transformation_matrix()

        new_pos = np.dot(pose_mat, lower_trans_mat)
        new_pose = sapien.Pose(matrix=new_pos)
        return new_pose

    def tf_to_grasp(self, pose: list, grasp_angle: float = math.pi / 4):
        """根据抓取角度应用对应的偏移量

        Args:
            pose: 原始姿态 [x, y, z, qx, qy, qz, qw]
            grasp_angle: 抓取角度 (弧度)
        """
        origin_pose = sapien.Pose(p=pose[:3], q=pose[3:])
        pose_mat = origin_pose.to_transformation_matrix()

        # 根据抓取角度获取对应的偏移量
        offset = GRASP_OFFSETS.get(grasp_angle, [-0.05, 0, 0.02])  # 默认偏移量
        tf_mat = np.array(
            [
                [1, 0, 0, offset[0]],
                [0, 1, 0, offset[1]],
                [0, 0, 1, offset[2]],
                [0, 0, 0, 1],
            ]
        )

        new_pos = np.dot(pose_mat, tf_mat)
        new_pose = sapien.Pose(matrix=new_pos)
        return list(new_pose.p) + list(new_pose.q)

    def move_block(self, actor: sapien.Entity, id, last_arm=None):
        # 随机选择一个抓取角度
        self.grasp_angle = random.choice(GRASP_ANGLES)

        actor_rpy = actor.get_pose().get_rpy()
        actor_pos = actor.get_pose().p
        actor_euler = math.fmod(actor_rpy[2], math.pi / 2)
        if actor_pos[1] < 0:
            # closer to right arm
            grasp_euler = actor_euler
            (grasp_qpose := sapien.Pose()).set_rpy(
                rpy=(np.array([0, 0, grasp_euler]) + self.robot.right_ee_rpy_offset)
            )
            grasp_qpose = self.rot_down_grip_pose(
                grasp_qpose, self.grasp_angle
            ).q.tolist()

            (target_qpose := sapien.Pose()).set_rpy(
                rpy=(np.array([0, 0, math.pi / 2]) + self.robot.right_ee_rpy_offset)
            )
            target_qpose = self.rot_down_grip_pose(
                target_qpose, self.grasp_angle
            ).q.tolist()
            target_pose = [
                self.tabletop_center_in_world[0] + self.place_pos_x_offset,
                0.01,
                self.table_height + id * self.block_half_size * 2 + 0.1,
            ] + target_qpose
        else:
            grasp_euler = actor_euler - math.pi / 2
            (grasp_qpose := sapien.Pose()).set_rpy(
                rpy=(np.array([0, 0, grasp_euler]) + self.robot.right_ee_rpy_offset)
            )
            grasp_qpose = self.rot_down_grip_pose(
                grasp_qpose, self.grasp_angle
            ).q.tolist()

            (target_qpose := sapien.Pose()).set_rpy(
                rpy=(np.array([0, 0, -math.pi / 2]) + self.robot.right_ee_rpy_offset)
            )
            target_qpose = self.rot_down_grip_pose(
                target_qpose, self.grasp_angle
            ).q.tolist()
            target_pose = [
                self.tabletop_center_in_world[0] + self.place_pos_x_offset,
                -0.01,
                self.table_height + id * self.block_half_size * 2 + 0.1,
            ] + target_qpose

        self.grasp_euler = grasp_euler
        target_pose = self.tf_to_grasp(target_pose, self.grasp_angle)
        substeps = []
        pre_grasp_pose = list(actor_pos + [0, 0, 0.2]) + grasp_qpose
        pre_grasp_pose = self.tf_to_grasp(pre_grasp_pose, self.grasp_angle)
        if actor_pos[1] < 0:
            now_arm = "right"
            if now_arm == last_arm or last_arm is None:
                if now_arm == last_arm:
                    pose0 = list(
                        self.robot.right_ee_link.get_entity_pose().p + [0, 0, 0.05]
                    ) + list(self.robot.right_ee_link.get_entity_pose().q)
                    substeps.append(
                        ("move_to_pose_traj_augmented", {"right_pose": pose0})
                    )
                substeps.append(
                    (
                        "move_to_pose_traj_augmented",
                        {"right_pose": deepcopy(pre_grasp_pose)},
                    )
                )
            else:
                substeps.append(
                    (
                        "move_to_pose_traj_augmented",
                        {
                            "right_pose": pre_grasp_pose,
                            "left_pose": self.robot.left_init_ee_pose,
                        },
                    )
                )
        else:
            now_arm = "left"
            if now_arm == last_arm or last_arm is None:
                if now_arm == last_arm:
                    pose0 = list(
                        self.robot.left_ee_link.get_entity_pose().p + [0, 0, 0.05]
                    ) + list(self.robot.left_ee_link.get_entity_pose().q)
                    substeps.append(
                        ("move_to_pose_traj_augmented", {"left_pose": pose0})
                    )
                substeps.append(
                    (
                        "move_to_pose_traj_augmented",
                        {"left_pose": deepcopy(pre_grasp_pose)},
                    )
                )
            else:
                substeps.append(
                    (
                        "move_to_pose_traj_augmented",
                        {
                            "left_pose": pre_grasp_pose,
                            "right_pose": self.robot.right_init_ee_pose,
                        },
                    )
                )

        substeps.append(("open_gripper", {"action_mode": now_arm}))
        pre_grasp_pose[2] -= 0.15
        substeps.append(
            (
                "move_to_pose_traj_augmented",
                {f"{now_arm}_pose": deepcopy(pre_grasp_pose)},
            )
        )
        substeps.append(("close_gripper", {"action_mode": now_arm}))
        pre_grasp_pose[2] += 0.15
        substeps.append(
            (
                "move_to_pose_traj_augmented",
                {f"{now_arm}_pose": deepcopy(pre_grasp_pose)},
            )
        )
        substeps.append(
            (
                "move_to_pose_traj_augmented",
                {f"{now_arm}_pose": deepcopy(target_pose)},
            )  # 长距离移动到目标位置使用轨迹增强
        )
        target_pose[2] -= 0.05
        substeps.append(
            (
                "move_to_pose_traj_augmented",
                {f"{now_arm}_pose": deepcopy(target_pose)},
            )
        )
        substeps.append(("open_gripper", {"action_mode": now_arm}))
        target_pose[2] += 0.1
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(target_pose)}))

        return substeps, now_arm

    def solution(self):
        # 移动第一个方块，带retry逻辑
        self.current_arm = None
        for step in self._move_block_with_retry(self.block1, 1, self.current_arm):
            yield step

        # 移动第二个方块，带retry逻辑
        for step in self._move_block_with_retry(self.block2, 2, self.current_arm):
            yield step

    def _move_block_with_retry(self, actor, id, last_arm, max_retries=2):
        """移动物体，带retry逻辑

        Args:
            actor: 要移动的物体
            id: 物体ID
            last_arm: 上次使用的机械臂
            max_retries: 最大重试次数，默认2次

        Returns:
            str: 使用的机械臂名称
        """
        retry_count = 0
        while retry_count <= max_retries:
            # 生成抓取动作步骤
            substeps, now_arm = self.move_block(actor, id, last_arm)

            # 执行抓取前的步骤
            for i, (action_name, action_params) in enumerate(substeps):
                if action_name == "close_gripper":
                    # 关闭夹爪
                    yield (action_name, action_params)

                    # 找到下一个抬起动作
                    lift_step_index = None
                    for j in range(i + 1, len(substeps)):
                        if substeps[j][0] == "move_to_pose_traj_augmented":
                            lift_step_index = j
                            break

                    if lift_step_index is not None:
                        # 执行抬起动作
                        yield substeps[lift_step_index]

                        # 在抬起后检测抓取是否成功
                        is_grasped = self.evaluate_grasp(actor, now_arm)

                        if is_grasped:
                            # 抓取成功，继续执行后续步骤
                            for remaining_step in substeps[lift_step_index + 1 :]:
                                yield remaining_step
                            self.current_arm = now_arm  # 更新当前使用的机械臂
                            return now_arm  # 成功完成，返回使用的机械臂
                        else:
                            # 抓取失败，如果还有重试机会，则重新开始
                            if retry_count < max_retries:
                                # 打开夹爪，准备重试
                                yield ("open_gripper", {"action_mode": now_arm})
                                # 移动到安全位置
                                safe_pose = list(
                                    actor.get_pose().p + [0, 0, 0.2]
                                ) + list(
                                    self.robot.left_ee_link.get_entity_pose().q
                                    if now_arm == "left"
                                    else self.robot.right_ee_link.get_entity_pose().q
                                )
                                yield ("move_to_pose", {f"{now_arm}_pose": safe_pose})
                                retry_count += 1
                                break  # 跳出当前循环，开始下一次重试
                            else:
                                # 没有重试机会了，直接返回失败
                                return now_arm
                    else:
                        # 没有找到抬起动作，直接检测
                        is_grasped = self.evaluate_grasp(actor, now_arm)
                        if is_grasped:
                            # 抓取成功，继续执行后续步骤
                            for remaining_step in substeps[i + 1 :]:
                                yield remaining_step
                            self.current_arm = now_arm  # 更新当前使用的机械臂
                            return now_arm
                        else:
                            # 抓取失败，如果还有重试机会，则重新开始
                            if retry_count < max_retries:
                                # 打开夹爪，准备重试
                                yield ("open_gripper", {"action_mode": now_arm})
                                # 移动到安全位置
                                safe_pose = list(
                                    actor.get_pose().p + [0, 0, 0.2]
                                ) + list(
                                    self.robot.left_ee_link.get_entity_pose().q
                                    if now_arm == "left"
                                    else self.robot.right_ee_link.get_entity_pose().q
                                )
                                yield ("move_to_pose", {f"{now_arm}_pose": safe_pose})
                                retry_count += 1
                                break  # 跳出当前循环，开始下一次重试
                            else:
                                # 没有重试机会了，直接返回失败
                                return now_arm
                else:
                    yield (action_name, action_params)

        return now_arm

    def _get_info(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        target_pose = [self.tabletop_center_in_world[0] + self.place_pos_x_offset, 0]
        eps = [0.025, 0.025, 0.01]
        success = np.all(
            np.abs(
                block1_pose
                - np.array(target_pose + [self.table_height + self.block_half_size])
            )
            < eps
        ) and np.all(
            np.abs(
                block2_pose
                - np.array(target_pose + [self.table_height + self.block_half_size * 3])
            )
            < eps
        )
        return dict(
            success=success,
        )

    def _get_reset_info(self):
        return dict(
            block1_pose=np.concatenate(
                [self.block1.get_pose().p, self.block1.get_pose().q]
            ),
            block2_pose=np.concatenate(
                [self.block2.get_pose().p, self.block2.get_pose().q]
            ),
        )

    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])

    @property
    def language_instruction(self):
        return "stack the blocks on the table."

    def get_object_dict(self):
        return dict(
            block1=np.concatenate([self.block1.get_pose().p, self.block1.get_pose().q]),
            block2=np.concatenate([self.block2.get_pose().p, self.block2.get_pose().q]),
        )

    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0

    def get_grasp_info(self):
        """获取当前抓取信息"""
        if hasattr(self, "grasp_angle"):
            return {
                "grasp_angle": self.grasp_angle,
                "grasp_angle_degrees": math.degrees(self.grasp_angle),
                "offset": GRASP_OFFSETS.get(self.grasp_angle, [-0.05, 0, 0.02]),
            }
        return None

    def evaluate_grasp(self, actor, arm):
        """评估物体是否被成功抓取

        Args:
            actor: 要检测的物体
            arm: 使用的机械臂 ("left" 或 "right")

        Returns:
            bool: 如果物体被成功抓取返回True，否则返回False
        """
        # 获取物体位置
        object_pos = actor.get_pose().p

        # 获取对应机械臂的末端执行器位置
        if arm == "left":
            ee_pos = self.robot.left_ee_link.get_entity_pose().p
        else:
            ee_pos = self.robot.right_ee_link.get_entity_pose().p

        # 检查物体是否在机械臂附近（水平距离小于5cm）
        horizontal_distance = np.linalg.norm(object_pos[:2] - ee_pos[:2])
        is_near_arm = horizontal_distance < 0.05

        # 检查物体是否被抬起（高度大于桌面+10cm）
        is_lifted = object_pos[2] > self.table_height + 0.1

        # 检查物体是否在机械臂上方（z轴距离小于10cm）
        vertical_distance = abs(object_pos[2] - ee_pos[2])
        is_above_arm = vertical_distance < 0.2

        # 综合判断：物体在机械臂附近、被抬起、且在机械臂上方
        is_grasped = is_near_arm and is_lifted and is_above_arm

        return is_grasped
