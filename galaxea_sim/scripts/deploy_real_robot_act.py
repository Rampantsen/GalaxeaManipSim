#!/usr/bin/env python3
"""
在真机上部署ACT策略
使用真机相机进行推理，通过ROS接口控制机器人
"""

from pathlib import Path
import numpy as np
import torch
import tyro
import cv2
from typing import Optional
import time

import rospy
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


class RealRobotACTDeployer:
    """真机ACT策略部署器"""
    
    def __init__(
        self,
        pretrained_policy_path: str,
        dataset_repo_id: str,
        device: str = "cuda",
        temporal_ensemble: bool = True,
        control_freq: int = 15,
    ):
        """
        初始化真机部署器
        
        Args:
            pretrained_policy_path: 预训练模型路径
            dataset_repo_id: 数据集ID（用于加载统计信息）
            device: 推理设备
            temporal_ensemble: 是否使用时序集成
            control_freq: 控制频率 (Hz)
        """
        self.device = device
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.bridge = CvBridge()
        
        # 加载策略
        print(f"📦 加载ACT策略: {pretrained_policy_path}")
        dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
        dataset_stats = dataset_metadata.stats
        
        self.policy = ACTPolicy.from_pretrained(
            pretrained_policy_path,
            dataset_stats=dataset_stats,
        )
        self.policy.eval()
        self.policy.to(device)
        
        # 启用时序集成
        if temporal_ensemble:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            self.policy.config.temporal_ensemble_coeff = 0.1
            self.policy.config.n_action_steps = 30
            self.policy.temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff=0.1, 
                chunk_size=self.policy.config.chunk_size
            )
            print("✅ 时序集成已启用")
        
        # 机器人状态
        self.current_joint_state = None
        self.camera_images = {}  # {camera_name: image}
        
        # ROS发布器
        self.joint_command_pub = rospy.Publisher(
            '/arm_joint_command_host',  # 关节控制话题
            JointState,
            queue_size=10
        )
        
        # 如果使用末端位姿控制
        self.left_ee_target_pub = rospy.Publisher(
            '/left_ee_target',
            PoseStamped,
            queue_size=10
        )
        self.right_ee_target_pub = rospy.Publisher(
            '/right_ee_target',
            PoseStamped,
            queue_size=10
        )
        
        print("✅ ROS发布器已初始化")
    
    def joint_state_callback(self, msg: JointState):
        """接收关节状态反馈"""
        self.current_joint_state = msg
    
    def camera_callback(self, msg: Image, camera_name: str):
        """接收相机图像"""
        try:
            # 将ROS图像转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.camera_images[camera_name] = cv_image
        except Exception as e:
            rospy.logerr(f"相机图像转换失败: {e}")
    
    def setup_subscribers(self, camera_topics: dict):
        """
        设置ROS订阅器
        
        Args:
            camera_topics: 相机话题字典，例如：
                {
                    'head_camera': '/camera/head/image_raw',
                    'left_wrist_camera': '/camera/left_wrist/image_raw',
                    'right_wrist_camera': '/camera/right_wrist/image_raw',
                }
        """
        # 订阅关节状态
        rospy.Subscriber(
            '/joint_states_host',  # 关节状态话题
            JointState,
            self.joint_state_callback
        )
        
        # 订阅各个相机
        for camera_name, topic in camera_topics.items():
            rospy.Subscriber(
                topic,
                Image,
                lambda msg, name=camera_name: self.camera_callback(msg, name)
            )
        
        print(f"✅ 订阅器已设置: 关节状态 + {len(camera_topics)}个相机")
        
        # 等待数据到达
        print("⏳ 等待机器人数据...")
        rate = rospy.Rate(10)
        timeout = 30  # 30秒超时
        start_time = time.time()
        
        while not rospy.is_shutdown():
            if self.current_joint_state is not None and len(self.camera_images) > 0:
                print("✅ 机器人数据已接收")
                break
            
            if time.time() - start_time > timeout:
                raise TimeoutError("等待机器人数据超时！请检查ROS话题是否正常发布")
            
            rate.sleep()
    
    def get_observation(self):
        """
        获取当前观测
        
        Returns:
            obs: 符合策略输入格式的观测字典
        """
        if self.current_joint_state is None:
            raise ValueError("关节状态未初始化")
        
        # 构建观测字典（需要与训练数据格式一致）
        obs = {}
        
        # 添加图像
        for camera_name, image in self.camera_images.items():
            # 调整图像大小（如果需要）
            # image = cv2.resize(image, (640, 480))
            obs[camera_name] = image
        
        # 添加关节状态
        joint_positions = np.array(self.current_joint_state.position)
        joint_velocities = np.array(self.current_joint_state.velocity)
        
        # 根据你的训练数据格式，分离左右臂
        # 这里需要根据实际的关节顺序调整
        num_joints_per_arm = len(joint_positions) // 2
        
        obs['left_arm_joint_position'] = joint_positions[:num_joints_per_arm]
        obs['right_arm_joint_position'] = joint_positions[num_joints_per_arm:]
        obs['left_arm_joint_velocity'] = joint_velocities[:num_joints_per_arm]
        obs['right_arm_joint_velocity'] = joint_velocities[num_joints_per_arm:]
        
        # TODO: 添加夹爪位置、末端位姿等其他观测
        
        return obs
    
    def execute_action(self, action: np.ndarray):
        """
        执行动作（发送到真机）
        
        Args:
            action: 策略输出的动作，形状为 (action_dim,)
                   例如：[left_joints(7), left_gripper(1), right_joints(7), right_gripper(1)]
        """
        # 构建关节命令消息
        joint_cmd = JointState()
        joint_cmd.header.stamp = rospy.Time.now()
        joint_cmd.header.frame_id = 'world'
        
        # 根据你的机器人配置，填充关节名称和目标位置
        # 这里需要根据实际情况调整
        joint_cmd.name = self.current_joint_state.name
        joint_cmd.position = action.tolist()
        
        # 发布关节命令
        self.joint_command_pub.publish(joint_cmd)
    
    def run_episode(self, max_steps: int = 1000):
        """
        运行一个episode
        
        Args:
            max_steps: 最大步数
            
        Returns:
            success: 是否成功
        """
        print("🚀 开始执行episode...")
        
        # 重置策略
        self.policy.reset()
        
        step = 0
        rate = rospy.Rate(self.control_freq)
        
        while not rospy.is_shutdown() and step < max_steps:
            # 1. 获取观测
            try:
                obs = self.get_observation()
            except Exception as e:
                rospy.logerr(f"获取观测失败: {e}")
                break
            
            # 2. 策略推理
            with torch.no_grad():
                # 转换为torch tensor
                obs_dict = {}
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        if value.dtype == np.uint8:  # 图像
                            value = torch.from_numpy(value).float() / 255.0
                        else:
                            value = torch.from_numpy(value).float()
                        obs_dict[key] = value.unsqueeze(0).to(self.device)
                
                # 策略输出
                action = self.policy.select_action(obs_dict)
                action = action.squeeze(0).cpu().numpy()
            
            # 3. 执行动作
            self.execute_action(action)
            
            step += 1
            
            # 打印进度
            if step % 50 == 0:
                print(f"📊 步数: {step}/{max_steps}")
            
            rate.sleep()
        
        print(f"✅ Episode完成，共执行 {step} 步")
        
        # TODO: 检查任务是否成功
        success = self.check_success()
        return success
    
    def check_success(self):
        """
        检查任务是否成功
        
        这个方法需要根据具体任务实现，例如：
        - 检测物体位置
        - 使用视觉识别
        - 人工确认
        
        Returns:
            bool: 是否成功
        """
        # 简单版本：询问用户
        user_input = input("任务是否成功？(y/n): ")
        return user_input.lower() == 'y'


def main(
    pretrained_policy_path: str,
    dataset_repo_id: str = "galaxea/R1ProBlocksStackEasy/traj_augmented",
    device: str = "cuda",
    temporal_ensemble: bool = True,
    control_freq: int = 15,
    num_episodes: int = 10,
    camera_topics: Optional[dict] = None,
):
    """
    在真机上部署ACT策略
    
    Args:
        pretrained_policy_path: 预训练模型路径
        dataset_repo_id: 数据集ID
        device: 推理设备
        temporal_ensemble: 是否使用时序集成
        control_freq: 控制频率
        num_episodes: 执行的episode数量
        camera_topics: 相机话题字典，例如：
            {
                'head_camera': '/camera/head/image_raw',
                'left_wrist_camera': '/camera/left_wrist/image_raw',
                'right_wrist_camera': '/camera/right_wrist/image_raw',
            }
    """
    # 初始化ROS节点
    rospy.init_node('galaxea_act_deployer', anonymous=True)
    print("✅ ROS节点已初始化")
    
    # 默认相机话题（需要根据实际情况修改）
    if camera_topics is None:
        camera_topics = {
            'head_camera': '/camera/head/color/image_raw',
            'left_wrist_camera': '/camera/left_wrist/color/image_raw',
            'right_wrist_camera': '/camera/right_wrist/color/image_raw',
        }
    
    # 创建部署器
    deployer = RealRobotACTDeployer(
        pretrained_policy_path=pretrained_policy_path,
        dataset_repo_id=dataset_repo_id,
        device=device,
        temporal_ensemble=temporal_ensemble,
        control_freq=control_freq,
    )
    
    # 设置订阅器
    deployer.setup_subscribers(camera_topics)
    
    # 运行多个episodes
    success_count = 0
    for episode_idx in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"📝 Episode {episode_idx + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        # 等待用户准备
        input("按Enter键开始执行...")
        
        # 运行episode
        success = deployer.run_episode(max_steps=1000)
        
        if success:
            success_count += 1
            print(f"✅ Episode {episode_idx + 1} 成功!")
        else:
            print(f"❌ Episode {episode_idx + 1} 失败")
        
        print(f"📊 当前成功率: {success_count}/{episode_idx + 1} = {success_count/(episode_idx + 1)*100:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"🎯 最终结果:")
    print(f"   成功: {success_count}/{num_episodes}")
    print(f"   成功率: {success_count/num_episodes*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    tyro.cli(main)

