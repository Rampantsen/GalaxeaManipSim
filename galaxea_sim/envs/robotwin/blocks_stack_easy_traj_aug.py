import math
from copy import deepcopy
import random

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import create_box, create_visual_ee_link
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv
import os

# 定义可选的抓取角度
# GRASP_ANGLES = [0, math.pi / 6, math.pi / 4, math.pi / 5]
GRASP_ANGLES = [math.pi / 6, math.pi / 4, math.pi / 5]
# 定义每个抓取角度对应的偏移量 (x, y, z)
GRASP_OFFSETS = {
    math.pi / 6: [-0.055, 0, 0.025],  # 30度抓取
    math.pi / 4: [-0.05, 0, 0.02],  # 45度抓取
    math.pi / 5: [-0.052, 0, 0.023],  
}


class BlocksStackEasyTrajAugEnv(RoboTwinBaseEnv):
    place_pos_x_offset = -0.2
    block_half_size = 0.02

    def __init__(
        self,
        *args,
        enable_retry=False,
        enable_traj_augmented=False,
        enable_visual=True,
        enable_grasp_sample=True,
        table_type="white", # "redwood" or "whitewood"
        **kwargs,
    ):
        # 将 table_type 传递给父类
        super().__init__(*args, table_type=table_type, **kwargs)
        self.current_arm = None  # 跟踪当前使用的机械臂
        self.enable_retry = enable_retry  # 控制是否启用retry逻辑
        self.enable_visual = enable_visual  # 控制是否启用可视化
        self.enable_traj_augmented = enable_traj_augmented  # 控制是否启用轨迹增强（执行噪声）
        self.enable_grasp_sample = enable_grasp_sample  # 控制是否启用抓取角度采样
        # 创建可视化标记来显示目标位置（仅当enable_visual=True时）
        if self.enable_visual:
            self._setup_visual_markers()

    def _rand_pose(self):
        return rand_pose(
            xlim=[-0.12, -0.02],
            ylim=[-0.45, 0.45],
            zlim=[self.block_half_size],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi],  # 绕z轴随机旋转-π到π弧度
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
            color=(250/255, 0, 10/255),
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
            color=(0, 200/255, 30/255),
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

    def _get_move_method_name(self):
        """根据当前配置获取移动方法名称"""
        if self.enable_traj_augmented:
            return "move_to_pose_traj_augmented"
        else:
            return "move_to_pose"
    
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

    def select_best_grasp_angle(self, actor_rpy):
        # 生成多个候选抓取角度（包括±90度）
        base_euler = actor_rpy[2]  # 使用原始角度
        all_candidate_angles = [
            math.fmod(base_euler, math.pi / 2),  # 原始角度
            math.fmod(base_euler, math.pi / 2) + math.pi / 2,  # +90度
            math.fmod(base_euler, math.pi / 2) - math.pi / 2,  # -90度
        ]

        # 随机选择一个候选角度，提供多样性
        # 注意：这是一次性选择，不会在失败后改变（那是 retry 的职责）
        selected = random.choice(all_candidate_angles)
        return selected

    def select_grasp_angle_with_retry(self, actor_rpy, tried_angles):
        """根据物体姿态和已尝试的角度，选择下一个候选抓取角度（用于retry）

        Args:
            actor_rpy: 物体的姿态（roll, pitch, yaw）
            tried_angles: 已经尝试过的角度列表

        Returns:
            float: 选择的抓取角度，如果所有角度都尝试过则返回None
        """
        # 生成多个候选抓取角度（包括±90度）
        base_euler = actor_rpy[2]  # 使用原始角度
        all_candidate_angles = [
            math.fmod(base_euler, math.pi / 2),  # 原始角度
            math.fmod(base_euler, math.pi / 2) + math.pi / 2,  # +90度
            math.fmod(base_euler, math.pi / 2) - math.pi / 2,  # -90度
        ]

        # 过滤掉已经尝试过的角度
        candidate_angles = [
            angle
            for angle in all_candidate_angles
            if not any(abs(angle - tried) < 0.01 for tried in tried_angles)
        ]

        if not candidate_angles:
            print("Warning: All candidate angles have been tried")
            return None

        # 返回第一个未尝试的候选角度
        selected = candidate_angles[0]
        return selected

    def move_block(self, actor: sapien.Entity, id, last_arm=None, tried_angles=None):
        """移动方块到目标位置

        Args:
            actor: 要移动的物体
            id: 物体ID
            last_arm: 上次使用的机械臂（未使用，保留以兼容）
            tried_angles: 已经尝试过的抓取角度列表（用于retry时排除）
        """
        # 随机选择一个抓取角度（例如45度）
        self.grasp_angle = random.choice(GRASP_ANGLES)

        actor_rpy = actor.get_pose().get_rpy()
        actor_pos = actor.get_pose().p

        if self.enable_grasp_sample:
            # 使用抓取角度采样逻辑
            if tried_angles is not None:
                # Retry 模式：从剩余的候选角度中选择（排除已尝试过的）
                actor_euler = self.select_grasp_angle_with_retry(
                    actor_rpy, tried_angles
                )

                # 如果所有角度都尝试过了，使用第一个作为后备
                if actor_euler is None:
                    base_euler = actor_rpy[2]
                    actor_euler = math.fmod(base_euler, math.pi / 2)

                # 保存当前尝试的角度（用于retry记录）
                self.last_tried_angle = actor_euler
            else:
                # 非 Retry 模式：选择最佳抓取角度
                actor_euler = self.select_best_grasp_angle(actor_rpy)
        else:
            # 简单模式：只使用基础角度计算
            base_euler = actor_rpy[2]
            actor_euler = math.fmod(base_euler, math.pi / 2)

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
            now_arm = "right"
            init_pose = self.robot.right_init_ee_pose
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
            now_arm = "left"
            init_pose = self.robot.left_init_ee_pose

        self.grasp_euler = grasp_euler
        target_pose = self.tf_to_grasp(target_pose, self.grasp_angle)
        substeps = []
        pre_grasp_pose = list(actor_pos + [0, 0, 0.2]) + grasp_qpose
        pre_grasp_pose = self.tf_to_grasp(pre_grasp_pose, self.grasp_angle)

        # 更新可视化标记：显示预抓取位置（蓝色半透明夹爪）
        if self.enable_visual:
            pre_grasp_sapien_pose = sapien.Pose(p=pre_grasp_pose[:3], q=pre_grasp_pose[3:7])
            self._update_visual_gripper_pose(
                self.pre_grasp_marker_entities, pre_grasp_sapien_pose
            )

            # 计算并显示抓取位置（青色半透明夹爪）- 预抓取位置下降0.15m
            grasp_pose_vis = pre_grasp_pose.copy()
            grasp_pose_vis[2] -= 0.15
            grasp_sapien_pose = sapien.Pose(p=grasp_pose_vis[:3], q=grasp_pose_vis[3:7])
            self._update_visual_gripper_pose(self.grasp_marker_entities, grasp_sapien_pose)

            # 更新目标放置位置标记（黄色半透明夹爪）
            target_sapien_pose = sapien.Pose(p=target_pose[:3], q=target_pose[3:7])
            self._update_visual_gripper_pose(
                self.target_place_marker_entities, target_sapien_pose
            )

        # 单个手臂独自进行抓取
        move_method = self._get_move_method_name()
        
        # 1. 移动到预抓取位置
        substeps.append((move_method, {f"{now_arm}_pose": deepcopy(pre_grasp_pose)}))

        # 2. 打开夹爪
        substeps.append(("open_gripper", {"action_mode": now_arm}))

        # 3. 下降到抓取位置
        pre_grasp_pose[2] -= 0.15
        substeps.append((move_method, {f"{now_arm}_pose": deepcopy(pre_grasp_pose)}))

        # 4. 关闭夹爪
        substeps.append(("close_gripper", {"action_mode": now_arm}))

        # 5. 抬起到预抓取高度
        pre_grasp_pose[2] += 0.15
        substeps.append((move_method, {f"{now_arm}_pose": deepcopy(pre_grasp_pose)}))

        # 6. 移动到目标位置
        substeps.append((move_method, {f"{now_arm}_pose": deepcopy(target_pose)}))

        # 7. 下降放置
        target_pose[2] -= 0.05
        substeps.append((move_method, {f"{now_arm}_pose": deepcopy(target_pose)}))

        # 8. 打开夹爪释放物体
        substeps.append(("open_gripper", {"action_mode": now_arm}))

        # 9. 抬起一点
        target_pose[2] += 0.1
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(target_pose)}))

        # 10. 收回到初始位置
        substeps.append((move_method, {f"{now_arm}_pose": init_pose}))

        return substeps, now_arm

    def solution(self):
        if self.enable_retry:
            # 使用带retry逻辑的版本（物体掉落后重新抓取）
            self.current_arm = None
            for step in self._move_block_with_retry(self.block1, 1, self.current_arm):
                yield step

            # 移动第二个方块，带retry逻辑
            for step in self._move_block_with_retry(self.block2, 2, self.current_arm):
                yield step
        elif self.enable_grasp_sample:
            # 使用grasp_sample逻辑（尝试多个抓取角度）
            for step in self._move_block_with_grasp_sample(self.block1, 1):
                yield step
            for step in self._move_block_with_grasp_sample(self.block2, 2):
                yield step
        else:
            # 使用原始逻辑（不带retry，不带grasp_sample）
            substeps, last_arm = self.move_block(self.block1, 1)
            self.info = f"move block 1,{self.block1.get_pose().p}"
            for substep in substeps:
                yield substep
            substeps, last_arm = self.move_block(self.block2, 2, last_arm)
            self.info = f"move block 2,{self.block2.get_pose().p}"
            for substep in substeps:
                yield substep

    def _move_block_with_grasp_sample(self, actor, id, max_angle_tries=3):
        """移动物体，使用grasp_sample逻辑（尝试多个抓取角度）

        这个方法会从多个候选角度中选择最优的，如果抓取失败，会尝试其他角度。

        Args:
            actor: 要移动的物体
            id: 物体ID
            max_angle_tries: 最大尝试的角度数量，默认3个

        Returns:
            str: 使用的机械臂名称
        """
        tried_angles = []  # 记录已经尝试过的角度
        angle_try_count = 0

        while angle_try_count < max_angle_tries:
            # 生成抓取动作步骤，传递已尝试过的角度
            substeps, now_arm = self.move_block(actor, id, None, tried_angles)

            # 记录本次尝试的角度
            if hasattr(self, "last_tried_angle"):
                tried_angles.append(self.last_tried_angle)

            # 执行抓取前的步骤
            for i, (action_name, action_params) in enumerate(substeps):
                if action_name == "close_gripper":
                    # 在闭合夹爪之前，先检查物体是否在夹爪范围内
                    in_grasp_range = self.check_grasp_position(actor, now_arm)

                    if not in_grasp_range:
                        # 物体不在夹爪范围内，尝试下一个角度
                        if angle_try_count < max_angle_tries - 1:
                            # 移动到安全位置，准备尝试新角度
                            safe_pose = list(actor.get_pose().p + [0, 0, 0.2]) + list(
                                self.robot.left_ee_link.get_entity_pose().q
                                if now_arm == "left"
                                else self.robot.right_ee_link.get_entity_pose().q
                            )
                            yield ("move_to_pose", {f"{now_arm}_pose": safe_pose})
                            angle_try_count += 1
                            break  # 跳出当前循环，尝试新角度
                        else:
                            # 所有角度都尝试过了，抓取失败
                            return now_arm

                    # 物体在夹爪范围内，执行闭合夹爪
                    yield (action_name, action_params)

                    # 找到下一个抬起动作
                    lift_step_index = None
                    for j in range(i + 1, len(substeps)):
                        if substeps[j][0] in [
                            "move_to_pose_traj_augmented",
                            "move_to_pose",
                        ]:
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
                            self.current_arm = now_arm
                            return now_arm  # 成功完成
                        else:
                            # 抓取失败，尝试下一个角度
                            if angle_try_count < max_angle_tries - 1:
                                # 打开夹爪，准备尝试新角度
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
                                angle_try_count += 1
                                break  # 跳出当前循环，尝试新角度
                            else:
                                # 所有角度都尝试过了
                                return now_arm
                    else:
                        # 没有找到抬起动作，直接检测
                        is_grasped = self.evaluate_grasp(actor, now_arm)
                        if is_grasped:
                            # 抓取成功，继续执行后续步骤
                            for remaining_step in substeps[i + 1 :]:
                                yield remaining_step
                            self.current_arm = now_arm
                            return now_arm
                        else:
                            # 抓取失败，尝试下一个角度
                            if angle_try_count < max_angle_tries - 1:
                                # 打开夹爪，准备尝试新角度
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
                                angle_try_count += 1
                                break  # 跳出当前循环，尝试新角度
                            else:
                                # 所有角度都尝试过了
                                return now_arm
                else:
                    yield (action_name, action_params)

        return now_arm

    def _move_block_with_retry(self, actor, id, last_arm, max_retries=2):
        """移动物体，带retry逻辑（物体掉落后重新抓取）

        如果启用了 grasp_sample，每次 retry 都会重新执行完整的 grasp_sample 流程。

        Args:
            actor: 要移动的物体
            id: 物体ID
            last_arm: 上次使用的机械臂
            max_retries: 最大重试次数（物体掉落后的重试），默认2次

        Returns:
            str: 使用的机械臂名称
        """
        if self.enable_grasp_sample:
            # 如果启用了 grasp_sample，直接调用 grasp_sample 逻辑
            # grasp_sample 内部已经包含了多角度尝试的逻辑
            for step in self._move_block_with_grasp_sample(actor, id):
                yield step
            return

        # 以下是不使用 grasp_sample 的 retry 逻辑
        retry_count = 0
        tried_angles = []  # 记录已经尝试过的角度

        while retry_count <= max_retries:
            # 生成抓取动作步骤，传递已尝试过的角度
            substeps, now_arm = self.move_block(actor, id, last_arm, tried_angles)

            # 记录本次尝试的角度
            if hasattr(self, "last_tried_angle"):
                tried_angles.append(self.last_tried_angle)
                # print(
                #     f"Retry {retry_count}: Tried angles so far: {[f'{math.degrees(a):.1f}°' for a in tried_angles]}"
                # )

            # 执行抓取前的步骤
            for i, (action_name, action_params) in enumerate(substeps):
                if action_name == "close_gripper":
                    # 在闭合夹爪之前，先检查物体是否在夹爪范围内
                    in_grasp_range = self.check_grasp_position(actor, now_arm)

                    if not in_grasp_range:
                        # 物体不在夹爪范围内，不闭合夹爪，直接进入重试逻辑
                        # print(
                        #     f"Object not in grasp range, skipping close_gripper. Retry {retry_count}/{max_retries}"
                        # )
                        if retry_count < max_retries:
                            # 移动到安全位置，准备重试
                            safe_pose = list(actor.get_pose().p + [0, 0, 0.2]) + list(
                                self.robot.left_ee_link.get_entity_pose().q
                                if now_arm == "left"
                                else self.robot.right_ee_link.get_entity_pose().q
                            )
                            yield ("move_to_pose", {f"{now_arm}_pose": safe_pose})
                            retry_count += 1
                            break  # 跳出当前循环，开始下一次重试
                        else:
                            # 没有重试机会了，直接返回失败
                            # print("Max retries reached, grasp failed")
                            return now_arm

                    # 物体在夹爪范围内，执行闭合夹爪
                    yield (action_name, action_params)

                    # 找到下一个抬起动作（需要检查两种可能的动作名称）
                    lift_step_index = None
                    for j in range(i + 1, len(substeps)):
                        if substeps[j][0] in [
                            "move_to_pose_traj_augmented",
                            "move_to_pose",
                        ]:
                            lift_step_index = j
                            break

                    if lift_step_index is not None:
                        # 执行抬起动作
                        yield substeps[lift_step_index]

                        # 在抬起后检测抓取是否成功
                        is_grasped = self.evaluate_grasp(actor, now_arm)

                        if is_grasped:
                            # 抓取成功，继续执行后续步骤
                            # print("Grasp successful!")
                            for remaining_step in substeps[lift_step_index + 1 :]:
                                yield remaining_step
                            self.current_arm = now_arm  # 更新当前使用的机械臂
                            return now_arm  # 成功完成，返回使用的机械臂
                        else:
                            # 抓取失败，如果还有重试机会，则重新开始
                            # print(
                            #     f"Grasp failed after closing gripper. Retry {retry_count}/{max_retries}"
                            # )
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
                                # print("Max retries reached, grasp failed")
                                return now_arm
                    else:
                        # 没有找到抬起动作，直接检测
                        is_grasped = self.evaluate_grasp(actor, now_arm)
                        if is_grasped:
                            # 抓取成功，继续执行后续步骤
                            # print("Grasp successful!")
                            for remaining_step in substeps[i + 1 :]:
                                yield remaining_step
                            self.current_arm = now_arm  # 更新当前使用的机械臂
                            return now_arm
                        else:
                            # 抓取失败，如果还有重试机会，则重新开始
                            # print(
                            #     f"Grasp failed after closing gripper. Retry {retry_count}/{max_retries}"
                            # )
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
                                print("Max retries reached, grasp failed")
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

    def check_grasp_position(self, actor, arm):
        """检查物体是否在夹爪范围内（闭合夹爪前的检查）

        Args:
            actor: 要检测的物体
            arm: 使用的机械臂 ("left" 或 "right")

        Returns:
            bool: 如果物体在夹爪范围内返回True，否则返回False
        """
        # 获取物体位置
        object_pos = actor.get_pose().p

        # 获取对应机械臂的末端执行器位置
        if arm == "left":
            ee_pos = self.robot.left_ee_link.get_entity_pose().p
        else:
            ee_pos = self.robot.right_ee_link.get_entity_pose().p

        # 由于使用了 tf_to_grasp 偏移（x: -0.035~-0.045m, z: 0.01~0.025m）
        # 末端执行器中心会故意偏离物体位置，所以需要更宽松的阈值
        # 检查物体是否在机械臂附近（水平距离小于8cm，考虑偏移量）
        horizontal_distance = np.linalg.norm(object_pos[:2] - ee_pos[:2])
        is_near_arm = horizontal_distance < 0.08

        # 检查物体是否在夹爪高度范围内（z轴距离小于8cm，考虑偏移量）
        vertical_distance = abs(object_pos[2] - ee_pos[2])
        is_at_grasp_height = vertical_distance < 0.08

        # 综合判断：物体在机械臂附近且在合适的高度
        in_grasp_range = is_near_arm and is_at_grasp_height

        # if not in_grasp_range:
        #     print(
        #         f"Object NOT in grasp range - horizontal_dist: {horizontal_distance:.4f}m, vertical_dist: {vertical_distance:.4f}m"
        #     )
        # else:
        #     print(
        #         f"Object in grasp range - horizontal_dist: {horizontal_distance:.4f}m, vertical_dist: {vertical_distance:.4f}m"
        #     )

        return in_grasp_range

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

    def _setup_visual_markers(self):
        """创建可视化标记来显示夹爪目标位置"""
        # 初始位置在桌面下方（不可见）
        initial_pose = sapien.Pose(p=[0, 0, -1])

        # 创建预抓取位置夹爪可视化（蓝色半透明）
        self.pre_grasp_marker_entities = create_visual_ee_link(
            scene=self._scene,
            pose=initial_pose,
            color=(0.2, 0.4, 1.0, 0.5),  # 蓝色半透明
            name="pre_grasp_gripper",
            gripper_width=0.08,
            gripper_depth=0.04,
            finger_length=0.06,
        )

        # 创建抓取位置夹爪可视化（青色半透明）
        self.grasp_marker_entities = create_visual_ee_link(
            scene=self._scene,
            pose=initial_pose,
            color=(0.0, 0.8, 0.8, 0.6),  # 青色半透明
            name="grasp_gripper",
            gripper_width=0.08,
            gripper_depth=0.04,
            finger_length=0.06,
        )

        # 创建目标放置位置夹爪可视化（黄色半透明）
        self.target_place_marker_entities = create_visual_ee_link(
            scene=self._scene,
            pose=initial_pose,
            color=(1.0, 0.8, 0.0, 0.5),  # 黄色半透明
            name="target_place_gripper",
            gripper_width=0.08,
            gripper_depth=0.04,
            finger_length=0.06,
        )

    def _update_visual_gripper_pose(self, entities: list, pose: sapien.Pose):
        """更新虚拟夹爪的所有部件位置

        Args:
            entities: 夹爪实体列表 [base, left_finger, right_finger, connector]
            pose: 目标位姿
        """
        if len(entities) != 4:
            return

        gripper_width = 0.08
        finger_length = 0.06

        # 更新基座
        entities[0].set_pose(pose)

        # 更新左手指
        left_finger_pose = pose * sapien.Pose(
            p=[0, gripper_width / 2 - 0.008, -finger_length / 2]
        )
        entities[1].set_pose(left_finger_pose)

        # 更新右手指
        right_finger_pose = pose * sapien.Pose(
            p=[0, -gripper_width / 2 + 0.008, -finger_length / 2]
        )
        entities[2].set_pose(right_finger_pose)

        # 更新连接杆
        connector_pose = pose * sapien.Pose(p=[0, 0, -0.01])
        entities[3].set_pose(connector_pose)
