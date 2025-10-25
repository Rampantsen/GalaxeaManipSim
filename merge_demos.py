#!/usr/bin/env python3
"""
将red和white目录中的demo合并到baseline目录
按照先red再white的顺序，同时合并meta_info.json
"""

import os
import shutil
import json
from pathlib import Path


def merge_demos():
    # 定义路径
    base_dir = Path("/home/sen/workspace/galaxea/GalaxeaManisim/datasets/R1ProBlocksStackEasy-traj_aug")
    red_dir = base_dir / "red" / "baseline" / "replayed"
    white_dir = base_dir / "white" / "baseline" / "replayed"
    output_dir = base_dir / "baseline" / "replayed"
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取meta_info.json
    print("正在读取meta_info.json文件...")
    with open(red_dir / "meta_info.json", "r") as f:
        red_meta = json.load(f)
    
    with open(white_dir / "meta_info.json", "r") as f:
        white_meta = json.load(f)
    
    print(f"Red meta info: {len(red_meta)} 条记录")
    print(f"White meta info: {len(white_meta)} 条记录")
    
    # 合并meta_info
    merged_meta = red_meta + white_meta
    print(f"合并后 meta info: {len(merged_meta)} 条记录")
    
    # 保存合并后的meta_info.json
    print("正在保存合并后的meta_info.json...")
    with open(output_dir / "meta_info.json", "w") as f:
        json.dump(merged_meta, f, indent=4)
    
    # 复制red目录的demo文件
    print("\n正在复制red目录的demo文件...")
    red_demos = [f for f in os.listdir(red_dir) if f.startswith("demo_") and f.endswith(".h5")]
    # 按照文件名中的数字排序，而不是字符串排序
    red_demos_sorted = sorted(red_demos, key=lambda x: int(x.split("_")[1].split(".")[0]))
    
    for idx, demo_file in enumerate(red_demos_sorted):
        src = red_dir / demo_file
        dst = output_dir / f"demo_{idx}.h5"
        print(f"复制 {src.name} -> {dst.name}")
        shutil.copy2(src, dst)
    
    red_count = len(red_demos_sorted)
    print(f"Red目录复制完成，共 {red_count} 个文件")
    
    # 复制white目录的demo文件（编号从red_count开始）
    print("\n正在复制white目录的demo文件...")
    white_demos = [f for f in os.listdir(white_dir) if f.startswith("demo_") and f.endswith(".h5")]
    # 按照文件名中的数字排序
    white_demos_sorted = sorted(white_demos, key=lambda x: int(x.split("_")[1].split(".")[0]))
    
    for idx, demo_file in enumerate(white_demos_sorted):
        src = white_dir / demo_file
        dst = output_dir / f"demo_{red_count + idx}.h5"
        print(f"复制 {src.name} -> {dst.name}")
        shutil.copy2(src, dst)
    
    white_count = len(white_demos_sorted)
    print(f"White目录复制完成，共 {white_count} 个文件")
    
    print(f"\n总计合并了 {red_count + white_count} 个demo文件")
    print(f"输出目录: {output_dir}")
    print("合并完成！")


if __name__ == "__main__":
    merge_demos()

