#!/usr/bin/env python3
"""
轨迹增强功能使用示例

这个脚本展示了如何使用改进后的轨迹增强功能，包括：
1. 不同的增强策略
2. 自适应噪声调整
3. 任务复杂度感知
"""

import numpy as np
from galaxea_sim.planners.bimanual import BimanualPlanner

def demonstrate_trajectory_augmentation():
    """演示轨迹增强功能的使用方法"""
    
    print("=== 轨迹增强功能使用示例 ===\n")
    
    # 示例参数配置
    examples = [
        {
            "name": "基础关节空间增强",
            "augmentation_strategy": "joint_space",
            "noise_std": 0.8,
            "adaptive_noise": False,
            "description": "使用原始的关节空间噪声增强方法"
        },
        {
            "name": "自适应噪声增强",
            "augmentation_strategy": "joint_space", 
            "noise_std": 0.8,
            "adaptive_noise": True,
            "task_complexity": 1.5,
            "success_rate": 0.3,
            "description": "根据任务复杂度和成功率自适应调整噪声强度"
        },
        {
            "name": "笛卡尔空间增强",
            "augmentation_strategy": "cartesian_space",
            "noise_std": 0.6,
            "adaptive_noise": True,
            "task_complexity": 1.2,
            "success_rate": 0.7,
            "description": "在笛卡尔空间添加噪声，更符合实际任务需求"
        },
        {
            "name": "时间维度增强",
            "augmentation_strategy": "temporal",
            "noise_std": 0.4,
            "adaptive_noise": True,
            "task_complexity": 0.8,
            "success_rate": 0.9,
            "description": "通过改变轨迹时间特性来增强多样性"
        },
        {
            "name": "混合策略增强",
            "augmentation_strategy": "mixed",
            "noise_std": 1.0,
            "adaptive_noise": True,
            "task_complexity": 2.0,
            "success_rate": 0.2,
            "description": "结合多种增强策略，适用于复杂任务"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   策略: {example['augmentation_strategy']}")
        print(f"   基础噪声: {example['noise_std']}")
        print(f"   自适应噪声: {example['adaptive_noise']}")
        
        if 'task_complexity' in example:
            print(f"   任务复杂度: {example['task_complexity']}")
        if 'success_rate' in example:
            print(f"   历史成功率: {example['success_rate']}")
            
        print(f"   描述: {example['description']}")
        print()
    
    print("=== 参数调整建议 ===\n")
    
    print("1. 噪声强度调整 (noise_std):")
    print("   - 简单任务: 0.3-0.6")
    print("   - 中等任务: 0.6-1.0") 
    print("   - 复杂任务: 1.0-1.5")
    print()
    
    print("2. 任务复杂度 (task_complexity):")
    print("   - 简单任务 (如抓取): 0.5-0.8")
    print("   - 中等任务 (如装配): 1.0-1.5")
    print("   - 复杂任务 (如精细操作): 1.5-2.0")
    print()
    
    print("3. 成功率反馈 (success_rate):")
    print("   - 高成功率 (>0.8): 减少噪声强度")
    print("   - 中等成功率 (0.5-0.8): 保持当前设置")
    print("   - 低成功率 (<0.5): 增加噪声强度")
    print()
    
    print("4. 增强策略选择:")
    print("   - joint_space: 适用于关节空间任务")
    print("   - cartesian_space: 适用于末端执行器任务")
    print("   - temporal: 适用于时间敏感任务")
    print("   - mixed: 适用于复杂多模态任务")
    print()

def calculate_adaptive_noise(base_noise, task_complexity, success_rate):
    """计算自适应噪声强度"""
    complexity_factor = np.clip(task_complexity, 0.5, 2.0)
    success_factor = np.clip(2.0 - success_rate, 0.5, 2.0)
    adaptive_noise = base_noise * complexity_factor * success_factor
    return adaptive_noise

def demonstrate_adaptive_noise():
    """演示自适应噪声计算"""
    print("=== 自适应噪声计算示例 ===\n")
    
    base_noise = 0.8
    scenarios = [
        {"task_complexity": 0.5, "success_rate": 0.9, "description": "简单任务，高成功率"},
        {"task_complexity": 1.0, "success_rate": 0.7, "description": "中等任务，中等成功率"},
        {"task_complexity": 1.5, "success_rate": 0.5, "description": "复杂任务，中等成功率"},
        {"task_complexity": 2.0, "success_rate": 0.2, "description": "复杂任务，低成功率"},
    ]
    
    for scenario in scenarios:
        adaptive_noise = calculate_adaptive_noise(
            base_noise, 
            scenario["task_complexity"], 
            scenario["success_rate"]
        )
        print(f"场景: {scenario['description']}")
        print(f"  基础噪声: {base_noise}")
        print(f"  任务复杂度: {scenario['task_complexity']}")
        print(f"  成功率: {scenario['success_rate']}")
        print(f"  自适应噪声: {adaptive_noise:.3f}")
        print(f"  噪声倍数: {adaptive_noise/base_noise:.2f}x")
        print()

if __name__ == "__main__":
    demonstrate_trajectory_augmentation()
    demonstrate_adaptive_noise()
