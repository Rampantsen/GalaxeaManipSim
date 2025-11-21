import numpy as np
import torch
import collections
from typing import Dict
from loguru import logger
from tqdm import tqdm

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

# Galaxea环境导入
import gymnasium as gym


class GalaxeaImageRunner(BaseImageRunner):
    """
    Galaxea仿真环境评估器（Gymnasium兼容版）
    简化实现，不依赖AsyncVectorEnv等旧版gym工具
    """
    
    def __init__(
        self,
        output_dir: str,
        env_name: str = "R1ProBlocksStackEasy-v0",
        n_test: int = 10,
        n_test_vis: int = 0,  # 暂不支持视频（避免依赖）
        test_start_seed: int = 100000,
        max_steps: int = 300,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        fps: int = 15,
        past_action: bool = False,
        tqdm_interval_sec: float = 1.0,
        **kwargs  # 忽略其他参数以保持兼容性
    ):
        """
        Args:
            output_dir: 输出目录
            env_name: Galaxea环境名称 (如 R1ProBlocksStackEasy-v0)
            n_test: 测试集评估数量
            n_test_vis: 保存视频数量（暂不支持）
            test_start_seed: 测试集随机种子
            max_steps: 每个episode最大步数
            n_obs_steps: 观测历史步数
            n_action_steps: 动作预测步数
            fps: 控制频率
            past_action: 是否使用过去的动作
            tqdm_interval_sec: 进度条更新间隔
        """
        super().__init__(output_dir)
        
        self.env_name = env_name
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.test_start_seed = test_start_seed
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.past_action = past_action
        self.tqdm_interval_sec = tqdm_interval_sec
        
        logger.info(f"初始化Galaxea评估器（Gymnasium兼容版）: {env_name}, n_test={n_test}")
    
    def run(self, policy: BaseImagePolicy) -> Dict:
        """
        运行策略评估（简化实现，兼容gymnasium）
        
        Args:
            policy: 要评估的策略
            
        Returns:
            包含评估指标的字典（兼容WandB格式）
        """
        device = policy.device
        
        # 收集所有episode的结果
        all_rewards = []
        all_lengths = []
        all_success = []
        
        logger.info(f"开始评估 {self.n_test} 个episodes...")
        
        for episode_idx in tqdm(range(self.n_test), desc="Evaluating", mininterval=self.tqdm_interval_sec):
            seed = self.test_start_seed + episode_idx
            
            try:
                # 创建环境
                env = gym.make(self.env_name)
                
                # 重置环境
                obs_dict, info = env.reset(seed=seed)
                
                # 提取观测（Galaxea格式）
                obs_history = collections.deque(maxlen=self.n_obs_steps)
                
                episode_reward = 0.0
                episode_length = 0
                done = False
                truncated = False
                
                # 初始化观测历史
                obs_processed = self._process_obs(obs_dict, device)
                for _ in range(self.n_obs_steps):
                    obs_history.append(obs_processed)
                
                # 运行episode
                with torch.no_grad():
                    while not (done or truncated) and episode_length < self.max_steps:
                        # 准备策略输入（堆叠观测历史）
                        obs_seq = self._stack_obs_history(obs_history)
                        
                        # 预测动作
                        action_dict = policy.predict_action(obs_seq)
                        action = action_dict['action'][0].cpu().numpy()  # 取第一步动作
                        
                        # 执行动作
                        obs_dict, reward, done, truncated, info = env.step(action)
                        
                        # 更新历史
                        obs_processed = self._process_obs(obs_dict, device)
                        obs_history.append(obs_processed)
                        
                        episode_reward += reward
                        episode_length += 1
                
                # 记录结果
                all_rewards.append(episode_reward)
                all_lengths.append(episode_length)
                
                # 检查成功
                success = info.get('success', False) if isinstance(info, dict) else False
                all_success.append(1.0 if success else 0.0)
                
                logger.info(f"Episode {episode_idx+1}/{self.n_test}: "
                           f"reward={episode_reward:.2f}, length={episode_length}, success={success}")
                
                env.close()
                
            except Exception as e:
                logger.error(f"Episode {episode_idx} 失败: {e}")
                # 记录失败的episode为0
                all_rewards.append(0.0)
                all_lengths.append(0)
                all_success.append(0.0)
        
        # 计算聚合指标（兼容原版格式）
        log_data = {
            'test/mean_score': np.mean(all_rewards),
            'test/max_reward_mean': np.mean(all_rewards),
            'test/max_reward_std': np.std(all_rewards),
            'test/success_rate': np.mean(all_success),
            'test/avg_length': np.mean(all_lengths),
        }
        
        logger.info(f"📊 评估完成: "
                   f"平均奖励={log_data['test/mean_score']:.2f}, "
                   f"成功率={log_data['test/success_rate']:.1%}")
        
        return log_data
    
    def _process_obs(self, obs_dict: Dict, device) -> Dict[str, torch.Tensor]:
        """
        处理Galaxea环境的观测格式
        
        Args:
            obs_dict: Galaxea环境返回的观测字典
            device: torch device
            
        Returns:
            处理后的观测字典
        """
        # 提取upper_body_observations
        upper_body = obs_dict['upper_body_observations']
        
        # 提取三相机图像
        img_head = torch.from_numpy(upper_body['rgb_head']).float().to(device)
        img_left = torch.from_numpy(upper_body['rgb_left_hand']).float().to(device)
        img_right = torch.from_numpy(upper_body['rgb_right_hand']).float().to(device)
        
        # HWC -> CHW, normalize to [0,1]
        img_head = img_head.permute(2, 0, 1) / 255.0
        img_left = img_left.permute(2, 0, 1) / 255.0
        img_right = img_right.permute(2, 0, 1) / 255.0
        
        # 提取状态（16维qpos）
        state = np.concatenate([
            upper_body['left_arm_joint_position'],      # 7
            upper_body['left_arm_gripper_position'],    # 1
            upper_body['right_arm_joint_position'],     # 7
            upper_body['right_arm_gripper_position'],   # 1
        ], axis=0)
        state = torch.from_numpy(state).float().to(device)
        
        return {
            'img_head': img_head,
            'img_left': img_left,
            'img_right': img_right,
            'state': state,
        }
    
    def _stack_obs_history(self, obs_history: collections.deque) -> Dict[str, torch.Tensor]:
        """
        堆叠观测历史为序列
        
        Args:
            obs_history: 观测历史队列
            
        Returns:
            堆叠后的观测字典
        """
        # 每个obs是单帧，需要堆叠成序列
        obs_list = list(obs_history)
        
        obs_seq = {
            'img_head': torch.stack([o['img_head'] for o in obs_list], dim=0),  # (T, 3, H, W)
            'img_left': torch.stack([o['img_left'] for o in obs_list], dim=0),
            'img_right': torch.stack([o['img_right'] for o in obs_list], dim=0),
            'state': torch.stack([o['state'] for o in obs_list], dim=0),  # (T, state_dim)
        }
        
        # 添加batch维度
        obs_seq = {k: v.unsqueeze(0) for k, v in obs_seq.items()}  # (1, T, ...)
        
        return obs_seq

