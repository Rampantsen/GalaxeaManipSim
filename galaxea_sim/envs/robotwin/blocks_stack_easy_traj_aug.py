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
GRASP_PITCH_ANGLES = [math.pi / 6, math.pi / 4, math.pi / 5]
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
        enable_traj_augmented=True,
        enable_visual=False,
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
        self.planner = None  # 用于grasp_sample的IK测试（需要外部设置）
        # 创建可视化标记来显示目标位置（仅当enable_visual=True时）
        if self.enable_visual:
            self._setup_visual_markers()
    
    def set_planner(self, planner):
        """设置planner引用，用于grasp_sample的IK测试"""
        self.planner = planner

    def _rand_pose(self):
        return rand_pose(
            xlim=[-0.12, -0.01],
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


    def _sample_best_grasp_angle(self, actor, now_arm):
        """在预抓取阶段采样最佳抓取角度（基于IK可解性）
        
        测试3个候选角度（物体朝向的0度、+90度、-90度），
        选择第一个IK可解的角度进行抓取。
        
        Args:
            actor: 要抓取的物体
            now_arm: 使用的机械臂 ("left" 或 "right")
            
        Returns:
            tuple: (selected_grasp_angle, selected_actor_euler) 或 (None, None) 如果所有角度都不可解
        """
        actor_rpy = actor.get_pose().get_rpy()
        actor_pos = actor.get_pose().p
        base_euler = actor_rpy[2]
        
        # 生成3个候选角度（物体朝向的0度、+90度、-90度）
        candidate_euler_angles = [
            math.fmod(base_euler, math.pi / 2),  # 原始角度
            math.fmod(base_euler, math.pi / 2) + math.pi / 2,  # +90度
            math.fmod(base_euler, math.pi / 2) - math.pi / 2,  # -90度
        ]
        
        # 如果planner未设置，使用第一个候选角度作为fallback
        if self.planner is None:
            print("Warning: Planner not set, using first candidate angle without IK test")
            return GRASP_PITCH_ANGLES[0], candidate_euler_angles[0]
        
        # 遍历每个候选角度，测试IK是否可解
        from mplib.pymp import Pose
        for actor_euler in candidate_euler_angles:
            for grasp_angle in GRASP_PITCH_ANGLES:
                # 计算抓取姿态
                if actor_pos[1] < 0:  # right arm
                    grasp_euler = actor_euler
                    (grasp_qpose := sapien.Pose()).set_rpy(
                        rpy=(np.array([0, 0, grasp_euler]) + self.robot.right_ee_rpy_offset)
                    )
                else:  # left arm
                    grasp_euler = actor_euler - math.pi / 2
                    (grasp_qpose := sapien.Pose()).set_rpy(
                        rpy=(np.array([0, 0, grasp_euler]) + self.robot.right_ee_rpy_offset)
                    )
                
                grasp_qpose = self.rot_down_grip_pose(grasp_qpose, grasp_angle).q.tolist()
                
                # 构建预抓取位置
                pre_grasp_pose = list(actor_pos + [0, 0, 0.2]) + grasp_qpose
                pre_grasp_pose = self.tf_to_grasp(pre_grasp_pose, grasp_angle)
                
                # 只测试抓取位置的IK（下降后的实际抓取位置）
                # 如果抓取位置可解，预抓取位置通常也可解（更高更安全）
                grasp_pose = pre_grasp_pose.copy()
                grasp_pose[2] -= 0.15  # 下降到实际抓取位置
                test_grasp_pose = Pose(p=grasp_pose[:3], q=grasp_pose[3:])
                
                if now_arm == "left":
                    result = self.planner.move_to_pose(
                        left_pose=test_grasp_pose,
                        right_pose=None,
                        robot_qpos=self.robot.get_qpos(),
                        verbose=False
                    )
                else:
                    result = self.planner.move_to_pose(
                        left_pose=None,
                        right_pose=test_grasp_pose,
                        robot_qpos=self.robot.get_qpos(),
                        verbose=False
                    )
                
                # 如果抓取位置IK可解，返回这个角度
                if result is not None:
                    #print(f"✓ 抓取位置IK可解 - pitch: {math.degrees(grasp_angle):.1f}°, euler: {math.degrees(actor_euler):.1f}°")
                    return grasp_angle, actor_euler
        
        # 所有角度都不可解
        print("Warning: No valid grasp angle found (all IK failed)")
        return None, None

    def move_block(self, actor: sapien.Entity, id, last_arm=None, 
                   specified_grasp_angle=None, specified_actor_euler=None):
        """移动方块到目标位置

        Args:
            actor: 要移动的物体
            id: 物体ID
            last_arm: 上次使用的机械臂（未使用，保留以兼容）
            specified_grasp_angle: 指定的抓取角度（如果为None则随机选择）
            specified_actor_euler: 指定的物体euler角度（如果为None则根据物体当前姿态计算）
        """
        actor_rpy = actor.get_pose().get_rpy()
        actor_pos = actor.get_pose().p

        # 使用指定的抓取角度，如果没有指定则随机选择
        if specified_grasp_angle is not None:
            self.grasp_angle = specified_grasp_angle
        else:
            self.grasp_angle = math.pi / 4

        # 使用指定的物体euler角度，如果没有指定则根据物体当前姿态计算
        if specified_actor_euler is not None:
            actor_euler = specified_actor_euler
        else:
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
            substeps, last_arm = self.move_block(self.block1, 1)
            self.info = f"move block 1,{self.block1.get_pose().p}"
            for substep in substeps:
                yield substep
            substeps, last_arm = self.move_block(self.block2, 2, last_arm)
            self.info = f"move block 2,{self.block2.get_pose().p}"
            for substep in substeps:
                yield substep

    def _move_block_with_grasp_sample(self, actor, id):
        """移动物体，使用grasp_sample逻辑（在预抓取阶段选择最优角度）

        在预抓取阶段，测试3个候选角度的IK可解性，选择第一个可解的角度，
        然后执行完整的抓取流程。不进行retry。

        Args:
            actor: 要移动的物体
            id: 物体ID

        Returns:
            str: 使用的机械臂名称
        """
        # 根据物体位置确定使用哪个机械臂
        actor_pos = actor.get_pose().p
        now_arm = "right" if actor_pos[1] < 0 else "left"
        
        # 在预抓取阶段采样最佳抓取角度（基于IK可解性）
        selected_grasp_angle, selected_actor_euler = self._sample_best_grasp_angle(actor, now_arm)
        
        if selected_grasp_angle is None:
            # 没有找到可行的抓取角度
            print(f"Warning: No valid grasp angle found for {now_arm} arm")
            return now_arm
        
        # 使用选定的角度生成抓取动作步骤
        substeps, now_arm = self.move_block(
            actor, id, 
            specified_grasp_angle=selected_grasp_angle,
            specified_actor_euler=selected_actor_euler
        )
        
        # 执行完整的抓取流程
        for action_name, action_params in substeps:
            yield (action_name, action_params)
        
        self.current_arm = now_arm
        return now_arm

    def _move_block_with_retry(self, actor, id, last_arm, max_retries=2):
        """移动物体，带retry逻辑（物体掉落后重新抓取）

        在闭合夹爪后检测物体是否被成功抓取，如果失败则回到安全位置，
        重新进行规划和grasp_sample（如果启用）。

        Args:
            actor: 要移动的物体
            id: 物体ID
            last_arm: 上次使用的机械臂
            max_retries: 最大重试次数（物体掉落后的重试），默认2次

        Returns:
            str: 使用的机械臂名称
        """
        retry_count = 0
        
        while retry_count <= max_retries:
            # 根据物体位置确定使用哪个机械臂
            actor_pos = actor.get_pose().p
            now_arm = "right" if actor_pos[1] < 0 else "left"
            
            # 每次重试都重新进行grasp_sample（如果启用）
            if self.enable_grasp_sample:
                # 在预抓取阶段采样最佳抓取角度（基于IK可解性）
                selected_grasp_angle, selected_actor_euler = self._sample_best_grasp_angle(actor, now_arm)
                
                if selected_grasp_angle is None:
                    # 没有找到可行的抓取角度
                    print(f"Warning: No valid grasp angle found for {now_arm} arm on retry {retry_count}")
                    if retry_count < max_retries:
                        retry_count += 1
                        continue
                    else:
                        return now_arm
                
                # 使用选定的角度生成抓取动作步骤
                substeps, now_arm = self.move_block(
                    actor, id,
                    specified_grasp_angle=selected_grasp_angle,
                    specified_actor_euler=selected_actor_euler
                )
            else:
                # 不使用grasp_sample，直接生成抓取动作步骤
                substeps, now_arm = self.move_block(actor, id)

            # 执行抓取步骤
            for i, (action_name, action_params) in enumerate(substeps):
                if action_name == "close_gripper":
                    # 执行闭合夹爪
                    yield (action_name, action_params)

                    # 找到下一个抬起动作
                    lift_step_index = None
                    for j in range(i + 1, len(substeps)):
                        if substeps[j][0] in ["move_to_pose_traj_augmented", "move_to_pose"]:
                            lift_step_index = j
                            break

                    if lift_step_index is not None:
                        # 执行抬起动作
                        yield substeps[lift_step_index]

                        # 在抬起后检测抓取是否成功
                        is_grasped = self.evaluate_grasp(actor, now_arm)

                        if is_grasped:
                            # 抓取成功，继续执行后续步骤
                            for remaining_step in substeps[lift_step_index + 1:]:
                                yield remaining_step
                            self.current_arm = now_arm
                            return now_arm
                        else:
                            # 抓取失败，回到安全位置准备retry
                            if retry_count < max_retries:
                                yield ("open_gripper", {"action_mode": now_arm})
                                safe_pose = list(actor.get_pose().p + [0, 0, 0.2]) + list(
                                    self.robot.left_ee_link.get_entity_pose().q
                                    if now_arm == "left"
                                    else self.robot.right_ee_link.get_entity_pose().q
                                )
                                yield ("move_to_pose", {f"{now_arm}_pose": safe_pose})
                                retry_count += 1
                                break
                            else:
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
        is_near_arm = horizontal_distance < 0.1

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
