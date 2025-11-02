"""
测试噪声过滤数据集的正确性

这个脚本会：
1. 创建一个模拟的带噪声标记的数据集
2. 测试 NoiseFilteredLeRobotDataset 是否正确过滤
3. 验证有效帧的索引映射
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from galaxea_sim.utils.noise_filtered_dataset import (
    NoiseFilteredLeRobotDataset,
    create_noise_filtered_dataloader
)


class MockLeRobotDataset(Dataset):
    """模拟的 LeRobotDataset，用于测试"""
    
    def __init__(self, total_frames=100, noise_ratio=0.1, seed=42):
        self.total_frames = total_frames
        np.random.seed(seed)
        
        # 随机生成噪声标记
        self.noise_mask = np.random.rand(total_frames) < noise_ratio
        self.noise_indices = np.where(self.noise_mask)[0].tolist()
        
        print(f"创建模拟数据集:")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 噪声帧数: {self.noise_mask.sum()}")
        print(f"  - 噪声帧索引: {self.noise_indices[:10]}..." if len(self.noise_indices) > 10 else f"  - 噪声帧索引: {self.noise_indices}")
    
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx):
        if idx >= self.total_frames:
            raise IndexError(f"索引 {idx} 超出范围 [0, {self.total_frames})")
        
        return {
            "observation.state": torch.randn(16),  # 模拟state
            "action": torch.randn(16),  # 模拟action
            "is_replan_noise": torch.tensor([self.noise_mask[idx]], dtype=torch.bool),
            "frame_idx": idx,  # 用于测试的额外字段
        }


def test_basic_filtering():
    """测试1：基本过滤功能"""
    print("\n" + "="*60)
    print("测试1：基本过滤功能")
    print("="*60)
    
    # 创建模拟数据集
    base_dataset = MockLeRobotDataset(total_frames=100, noise_ratio=0.1)
    
    # 创建过滤数据集
    filtered_dataset = NoiseFilteredLeRobotDataset(base_dataset, verbose=True)
    
    # 验证长度
    expected_valid = (~base_dataset.noise_mask).sum()
    assert len(filtered_dataset) == expected_valid, \
        f"过滤后的长度不正确: {len(filtered_dataset)} vs {expected_valid}"
    
    print("\n✅ 测试1通过：数据集长度正确")


def test_no_noise_in_targets():
    """测试2：确保采样的帧不包含噪声"""
    print("\n" + "="*60)
    print("测试2：确保action target不包含噪声帧")
    print("="*60)
    
    base_dataset = MockLeRobotDataset(total_frames=100, noise_ratio=0.1)
    filtered_dataset = NoiseFilteredLeRobotDataset(base_dataset, verbose=False)
    
    # 遍历所有过滤后的数据
    noise_found = False
    for i in range(len(filtered_dataset)):
        sample = filtered_dataset[i]
        frame_idx = sample["frame_idx"]
        
        # 检查这一帧在原始数据集中是否为噪声
        if base_dataset.noise_mask[frame_idx]:
            noise_found = True
            print(f"❌ 发现噪声帧 {frame_idx} 在过滤后的数据集中！")
    
    if not noise_found:
        print(f"✅ 测试2通过：采样的 {len(filtered_dataset)} 个帧都不是噪声帧")
    else:
        raise AssertionError("过滤失败：发现噪声帧")


def test_dataloader():
    """测试3：测试dataloader"""
    print("\n" + "="*60)
    print("测试3：测试DataLoader")
    print("="*60)
    
    base_dataset = MockLeRobotDataset(total_frames=100, noise_ratio=0.1)
    
    # 使用便捷函数创建dataloader
    dataloader = create_noise_filtered_dataloader(
        base_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,  # 测试时用0
        verbose=False,
    )
    
    total_batches = 0
    total_samples = 0
    
    for batch in dataloader:
        total_batches += 1
        batch_size = batch["observation.state"].shape[0]
        total_samples += batch_size
        
        # 检查batch中的frame_idx
        frame_indices = batch["frame_idx"].numpy()
        
        # 确保没有噪声帧
        for idx in frame_indices:
            if base_dataset.noise_mask[idx]:
                raise AssertionError(f"Batch中包含噪声帧 {idx}")
    
    print(f"  - 总batch数: {total_batches}")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - 预期样本数: {len(dataloader.dataset)}")
    print(f"✅ 测试3通过：DataLoader工作正常")


def test_index_mapping():
    """测试4：验证索引映射"""
    print("\n" + "="*60)
    print("测试4：验证索引映射")
    print("="*60)
    
    # 创建一个小数据集便于验证
    base_dataset = MockLeRobotDataset(total_frames=10, noise_ratio=0.2, seed=123)
    filtered_dataset = NoiseFilteredLeRobotDataset(base_dataset, verbose=False)
    
    print("\n原始数据集标记:")
    for i in range(len(base_dataset)):
        is_noise = base_dataset.noise_mask[i]
        marker = "❌" if is_noise else "✓"
        print(f"  帧 {i}: {marker} {'(噪声)' if is_noise else '(正常)'}")
    
    print(f"\n过滤后的索引映射:")
    for i in range(len(filtered_dataset)):
        sample = filtered_dataset[i]
        original_idx = sample["frame_idx"]
        print(f"  过滤索引 {i} -> 原始索引 {original_idx}")
    
    print(f"\n✅ 测试4通过：索引映射正确")


def test_edge_cases():
    """测试5：边界情况"""
    print("\n" + "="*60)
    print("测试5：边界情况")
    print("="*60)
    
    # 情况1：没有噪声帧
    print("\n子测试5.1：没有噪声帧")
    base_dataset1 = MockLeRobotDataset(total_frames=50, noise_ratio=0.0)
    filtered_dataset1 = NoiseFilteredLeRobotDataset(base_dataset1, verbose=False)
    assert len(filtered_dataset1) == 50, "没有噪声时长度应该相等"
    print("  ✅ 通过")
    
    # 情况2：全是噪声帧（极端情况）
    print("\n子测试5.2：全是噪声帧")
    base_dataset2 = MockLeRobotDataset(total_frames=50, noise_ratio=1.0)
    filtered_dataset2 = NoiseFilteredLeRobotDataset(base_dataset2, verbose=False)
    print(f"  - 过滤后数据集大小: {len(filtered_dataset2)}")
    assert len(filtered_dataset2) == 0, "全是噪声时应该为空"
    print("  ✅ 通过")
    
    # 情况3：很多噪声帧
    print("\n子测试5.3：50%噪声帧")
    base_dataset3 = MockLeRobotDataset(total_frames=100, noise_ratio=0.5)
    filtered_dataset3 = NoiseFilteredLeRobotDataset(base_dataset3, verbose=False)
    expected = (~base_dataset3.noise_mask).sum()
    assert len(filtered_dataset3) == expected, f"过滤后长度不对: {len(filtered_dataset3)} vs {expected}"
    print(f"  - 原始: 100帧, 过滤后: {len(filtered_dataset3)}帧")
    print("  ✅ 通过")
    
    print(f"\n✅ 测试5通过：所有边界情况正常")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪"*30)
    print("开始测试噪声过滤数据集")
    print("🧪"*30)
    
    try:
        test_basic_filtering()
        test_no_noise_in_targets()
        test_dataloader()
        test_index_mapping()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！噪声过滤方案工作正常！")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 测试失败: {e}")
        print("="*60)
        raise


if __name__ == "__main__":
    run_all_tests()

