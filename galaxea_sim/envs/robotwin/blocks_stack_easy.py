from gc import enable
import math
from copy import deepcopy

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import create_box
from galaxea_sim.utils.rand_utils import rand_pose, add_pose_noise
from .robotwin_base import RoboTwinBaseEnv


class BlocksStackEasyEnv(RoboTwinBaseEnv):
    place_pos_x_offset = -0.2
    block_half_size = 0.02

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

    def set_planner(self, planner):
        """设置planner引用，用于grasp_sample的IK测试"""
        self.planner = planner

    def _rand_pose(self):
        return rand_pose(
            xlim=[-0.12, -0.01],
            ylim=[-0.3,0.3],
            zlim=[self.block_half_size],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi],
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

    def rot_down_grip_pose(self, pose: sapien.Pose,grasp_angle: float = math.pi / 4):
        pose_mat = pose.to_transformation_matrix()
        (lower_trans_quat := sapien.Pose()).set_rpy(rpy=(np.array([0, grasp_angle, 0])))
        lower_trans_mat = lower_trans_quat.to_transformation_matrix()

        new_pos = np.dot(pose_mat, lower_trans_mat)
        new_pose = sapien.Pose(matrix=new_pos)
        return new_pose

    def tf_to_grasp(self, pose: list, grasp_angle: float = math.pi / 4):
        # 根据 grasp_angle 动态选择 offset
        grasp_offsets = {
            math.pi / 6: [-0.055, 0, 0.025], 
            math.pi / 5: [-0.052, 0, 0.023],   
            math.pi / 4: [-0.05, 0, 0.02],  
        }
        grasp_offset = grasp_offsets.get(grasp_angle, [-0.05, 0, 0.02])
        
        origin_pose = sapien.Pose(p=pose[:3], q=pose[3:])
        pose_mat = origin_pose.to_transformation_matrix()
        tf_mat = np.array(
            [[1, 0, 0, grasp_offset[0]], [0, 1, 0, grasp_offset[1]], [0, 0, 1, grasp_offset[2]], [0, 0, 0, 1]]
        )
        new_pos = np.dot(pose_mat, tf_mat)
        new_pose = sapien.Pose(matrix=new_pos)
        return list(new_pose.p) + list(new_pose.q)

    def move_block(self, actor: sapien.Entity, id, last_arm=None,enable_grasp_sample=False,enable_replan=False):
        actor_rpy = actor.get_pose().get_rpy()
        actor_pos = actor.get_pose().p
        
        # 确定使用哪个机械臂
        now_arm = "right" if actor_pos[1] < 0 else "left"
        
        # 根据 enable_grasp_sample 决定抓取角度
        if enable_grasp_sample:
            grasp_angle, actor_euler = self._sample_best_grasp_angle(actor, now_arm)
            if grasp_angle is None:
                # 如果没有找到可行角度，使用默认值
                print(f"Warning: No valid grasp angle found for {now_arm} arm, using default")
                grasp_angle = math.pi / 4
                actor_euler = math.fmod(actor_rpy[2], math.pi / 2)
        else:
            grasp_angle = math.pi / 4
            actor_euler = math.fmod(actor_rpy[2], math.pi / 2)
        
        if actor_pos[1] < 0:
            # closer to right arm
            grasp_euler = actor_euler
            (grasp_qpose := sapien.Pose()).set_rpy(
                rpy=(np.array([0, 0, grasp_euler]) + self.robot.right_ee_rpy_offset)
            )
            grasp_qpose = self.rot_down_grip_pose(grasp_qpose, grasp_angle).q.tolist()

            (target_qpose := sapien.Pose()).set_rpy(
                rpy=(np.array([0, 0, math.pi / 2]) + self.robot.right_ee_rpy_offset)
            )
            target_qpose = self.rot_down_grip_pose(target_qpose, grasp_angle).q.tolist()
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
            grasp_qpose = self.rot_down_grip_pose(grasp_qpose, grasp_angle).q.tolist()

            (target_qpose := sapien.Pose()).set_rpy(
                rpy=(np.array([0, 0, -math.pi / 2]) + self.robot.right_ee_rpy_offset)
            )
            target_qpose = self.rot_down_grip_pose(target_qpose, grasp_angle).q.tolist()
            target_pose = [
                self.tabletop_center_in_world[0] + self.place_pos_x_offset,
                -0.01,
                self.table_height + id * self.block_half_size * 2 + 0.1,
            ] + target_qpose

        self.grasp_euler = grasp_euler
        target_pose = self.tf_to_grasp(target_pose, grasp_angle)
        substeps = []
        pre_grasp_pose = list(actor_pos + [0, 0, 0.2]) + grasp_qpose
        print(f"pre_grasp_pose: {pre_grasp_pose}")
        pre_grasp_pose = self.tf_to_grasp(pre_grasp_pose, grasp_angle)
        
        # 获取初始位置（用于最后回到初始姿态）
        if now_arm == "right":
            init_pose = self.robot.right_init_ee_pose
        else:
            init_pose = self.robot.left_init_ee_pose
        
        # 单个手臂独自进行抓取，不再处理双臂协调
        # Replan方式1：在预抓取位置加噪声
        should_add_pregrasp_noise = enable_replan and np.random.random() < self.replan_prob
        if should_add_pregrasp_noise:
            # 移动到带噪声的预抓取位置（标记为replan_noise）
            noisy_pre_grasp_pose = add_pose_noise(
                deepcopy(pre_grasp_pose), 
                position_noise_range=self.replan_noise_range,
                orientation_noise_range=0.0,  # 不添加旋转噪声
                noise_axes=[True, True, False]  # 只在xy平面加噪声，保持z高度
            )
            substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(noisy_pre_grasp_pose), "_is_replan_noise": True}))
            # 从噪声位置replan到正确的预抓取位置
            substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(pre_grasp_pose), "_is_replan_noise": False}))
        else:
            # 1. 直接移动到预抓取位置
            substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(pre_grasp_pose), "_is_replan_noise": False}))
        
        # 2. 打开夹爪
        substeps.append(("open_gripper", {"action_mode": now_arm, "_is_replan_noise": False}))
        
        # 3. 下降到抓取位置
        grasp_pose = deepcopy(pre_grasp_pose)
        grasp_pose[2] -= 0.15  # 下降到物体位置
        
        # Replan方式2：在抓取位置加噪声（如果方式1没有触发）
        should_add_grasp_noise = enable_replan and not should_add_pregrasp_noise and np.random.random() < self.replan_prob
        if should_add_grasp_noise:
            # 移动到带噪声的抓取位置（可能抓不到物体）（标记为replan_noise）
            noisy_grasp_pose = add_pose_noise(
                deepcopy(grasp_pose),
                position_noise_range=self.replan_noise_range,
                orientation_noise_range=0.0,
                noise_axes=[True, True, True] 
            )
            substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(noisy_grasp_pose), "_is_replan_noise": True}))
            # 关闭夹爪（可能抓不到）（标记为replan_noise）
            substeps.append(("close_gripper", {"action_mode": now_arm, "_is_replan_noise": True}))
            # 打开夹爪准备重试（标记为replan_noise）
            substeps.append(("open_gripper", {"action_mode": now_arm, "_is_replan_noise": True}))
            # Replan到正确位置重新抓取
            substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(grasp_pose), "_is_replan_noise": False}))
            substeps.append(("close_gripper", {"action_mode": now_arm, "_is_replan_noise": False}))
        else:
            # 正常抓取
            substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(grasp_pose), "_is_replan_noise": False}))
            # 4. 关闭夹爪
            substeps.append(("close_gripper", {"action_mode": now_arm, "_is_replan_noise": False}))
        
        # 5. 抬起到固定安全高度（基于闭合夹爪后的位置）
        # 设置固定的安全高度：桌面上方 0.30m
        lift_pose = deepcopy(grasp_pose)
        lift_pose[2] = self.table_height + 0.2  # 固定的绝对高度
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(lift_pose), "_is_replan_noise": False}))
        
        # 6. 移动到目标位置
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(target_pose), "_is_replan_noise": False}))
        
        # 7. 下降放置
        target_pose[2] -= 0.05
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(target_pose), "_is_replan_noise": False}))
        
        # 8. 打开夹爪释放物体
        substeps.append(("open_gripper", {"action_mode": now_arm, "_is_replan_noise": False}))
        
        # 9. 抬起一点
        target_pose[2] += 0.1
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(target_pose), "_is_replan_noise": False}))
        
        # 10. 收回到初始位置
        substeps.append(("move_to_pose", {f"{now_arm}_pose": init_pose, "_is_replan_noise": False}))

        return substeps, now_arm

    def solution(self):
        substeps, last_arm = self.move_block(self.block1, 1,enable_grasp_sample=self.enable_grasp_sample,enable_replan=self.enable_replan)
        self.info = f"move block 1,{self.block1.get_pose().p}"
        for substep in substeps:
            yield substep
        substeps, last_arm = self.move_block(self.block2, 2, last_arm,enable_grasp_sample=self.enable_grasp_sample,enable_replan=self.enable_replan)
        self.info = f"move block 2,{self.block2.get_pose().p}"
        for substep in substeps:
            yield substep
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
        grasp_pitch_angles = [math.pi / 6, math.pi / 5, math.pi / 4]

        candidate_euler_angles = [
            math.fmod(base_euler, math.pi / 2),  
            math.fmod(base_euler, math.pi / 2) + math.pi / 2,  
            math.fmod(base_euler, math.pi / 2) - math.pi / 2, 
        ]
        # 如果planner未设置，使用第一个候选角度作为fallback
        if self.planner is None:
            print("Warning: Planner not set, using first candidate angle without IK test")
            return grasp_pitch_angles[0], candidate_euler_angles[0]
        
        # 遍历每个候选角度，测试IK是否可解
        from mplib.pymp import Pose
        for actor_euler in candidate_euler_angles:
            for grasp_angle in grasp_pitch_angles:
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
                grasp_pose = list(actor_pos + [0, 0, 0.05]) + grasp_qpose
                grasp_pose = self.tf_to_grasp(grasp_pose, grasp_angle)
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
                
                if result is not None:
                    return grasp_angle, actor_euler
        
        return grasp_pitch_angles[0], candidate_euler_angles[0]

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
