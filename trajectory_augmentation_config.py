#!/usr/bin/env python3
"""
轨迹增强功能配置文件

这个文件包含了不同任务类型的推荐配置参数
"""

# 任务类型配置
TASK_CONFIGS = {
    "simple_grasp": {
        "name": "简单抓取任务",
        "augmentation_strategy": "joint_space",
        "noise_std": 0.5,
        "adaptive_noise": True,
        "task_complexity": 0.6,
        "num_waypoints": 3,
        "description": "适用于简单的抓取和放置任务"
    },
    
    "precision_manipulation": {
        "name": "精密操作任务", 
        "augmentation_strategy": "cartesian_space",
        "noise_std": 0.3,
        "adaptive_noise": True,
        "task_complexity": 1.8,
        "num_waypoints": 5,
        "description": "适用于需要精确定位的操作任务"
    },
    
    "assembly_task": {
        "name": "装配任务",
        "augmentation_strategy": "mixed",
        "noise_std": 0.8,
        "adaptive_noise": True,
        "task_complexity": 1.5,
        "num_waypoints": 4,
        "description": "适用于需要协调双臂的装配任务"
    },
    
    "dynamic_manipulation": {
        "name": "动态操作任务",
        "augmentation_strategy": "temporal",
        "noise_std": 0.6,
        "adaptive_noise": True,
        "task_complexity": 1.2,
        "num_waypoints": 6,
        "description": "适用于需要时间协调的动态操作"
    },
    
    "complex_bimanual": {
        "name": "复杂双臂任务",
        "augmentation_strategy": "mixed",
        "noise_std": 1.2,
        "adaptive_noise": True,
        "task_complexity": 2.0,
        "num_waypoints": 8,
        "description": "适用于复杂的双臂协调任务"
    }
}

# 自适应参数调整规则
ADAPTIVE_RULES = {
    "noise_scaling": {
        "high_success": {"threshold": 0.8, "factor": 0.7, "description": "高成功率时减少噪声"},
        "medium_success": {"threshold": 0.5, "factor": 1.0, "description": "中等成功率时保持噪声"},
        "low_success": {"threshold": 0.0, "factor": 1.5, "description": "低成功率时增加噪声"}
    },
    
    "complexity_scaling": {
        "simple": {"range": (0.0, 0.8), "factor": 0.8, "description": "简单任务减少噪声"},
        "medium": {"range": (0.8, 1.5), "factor": 1.0, "description": "中等任务保持噪声"},
        "complex": {"range": (1.5, 2.0), "factor": 1.3, "description": "复杂任务增加噪声"}
    }
}

def get_task_config(task_type):
    """获取指定任务类型的配置"""
    if task_type not in TASK_CONFIGS:
        raise ValueError(f"未知的任务类型: {task_type}")
    return TASK_CONFIGS[task_type]

def get_adaptive_noise_params(success_rate, task_complexity):
    """根据成功率和任务复杂度计算自适应噪声参数"""
    
    # 根据成功率调整
    if success_rate >= ADAPTIVE_RULES["noise_scaling"]["high_success"]["threshold"]:
        success_factor = ADAPTIVE_RULES["noise_scaling"]["high_success"]["factor"]
    elif success_rate >= ADAPTIVE_RULES["noise_scaling"]["medium_success"]["threshold"]:
        success_factor = ADAPTIVE_RULES["noise_scaling"]["medium_success"]["factor"]
    else:
        success_factor = ADAPTIVE_RULES["noise_scaling"]["low_success"]["factor"]
    
    # 根据任务复杂度调整
    if task_complexity <= ADAPTIVE_RULES["complexity_scaling"]["simple"]["range"][1]:
        complexity_factor = ADAPTIVE_RULES["complexity_scaling"]["simple"]["factor"]
    elif task_complexity <= ADAPTIVE_RULES["complexity_scaling"]["medium"]["range"][1]:
        complexity_factor = ADAPTIVE_RULES["complexity_scaling"]["medium"]["factor"]
    else:
        complexity_factor = ADAPTIVE_RULES["complexity_scaling"]["complex"]["factor"]
    
    return {
        "success_factor": success_factor,
        "complexity_factor": complexity_factor,
        "total_factor": success_factor * complexity_factor
    }

def recommend_config_for_task(task_description, current_success_rate=0.5):
    """根据任务描述推荐配置"""
    
    # 基于关键词匹配任务类型
    task_lower = task_description.lower()
    
    if any(keyword in task_lower for keyword in ["抓取", "grasp", "pick", "place"]):
        if any(keyword in task_lower for keyword in ["精密", "precision", "精细"]):
            return get_task_config("precision_manipulation")
        else:
            return get_task_config("simple_grasp")
    
    elif any(keyword in task_lower for keyword in ["装配", "assembly", "组装"]):
        return get_task_config("assembly_task")
    
    elif any(keyword in task_lower for keyword in ["动态", "dynamic", "时间", "timing"]):
        return get_task_config("dynamic_manipulation")
    
    elif any(keyword in task_lower for keyword in ["双臂", "bimanual", "协调", "coordination"]):
        return get_task_config("complex_bimanual")
    
    else:
        # 默认配置
        return get_task_config("simple_grasp")

def print_all_configs():
    """打印所有可用的配置"""
    print("=== 可用的任务配置 ===\n")
    
    for task_type, config in TASK_CONFIGS.items():
        print(f"任务类型: {task_type}")
        print(f"名称: {config['name']}")
        print(f"增强策略: {config['augmentation_strategy']}")
        print(f"噪声强度: {config['noise_std']}")
        print(f"自适应噪声: {config['adaptive_noise']}")
        print(f"任务复杂度: {config['task_complexity']}")
        print(f"路径点数量: {config['num_waypoints']}")
        print(f"描述: {config['description']}")
        print()

if __name__ == "__main__":
    print_all_configs()
    
    # 示例：根据任务描述推荐配置
    example_tasks = [
        "抓取红色方块",
        "精密装配小零件", 
        "双臂协调搬运",
        "动态抓取移动物体"
    ]
    
    print("=== 任务配置推荐示例 ===\n")
    for task in example_tasks:
        config = recommend_config_for_task(task)
        print(f"任务: {task}")
        print(f"推荐配置: {config['name']}")
        print(f"增强策略: {config['augmentation_strategy']}")
        print(f"噪声强度: {config['noise_std']}")
        print()
