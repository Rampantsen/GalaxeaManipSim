#!/bin/bash
feature=$1
seed=$2
start_checkpoint=${3:-0}  # 可选参数：起始checkpoint，默认为0（从头开始）

base_dir="/home/sen/workspace/galaxea/GalaxeaManisim/outputs/ACT/R1ProBlocksStackEasy-traj_aug/${feature}/checkpoints"

echo "起始checkpoint: ${start_checkpoint}"
echo ""

# 获取所有checkpoint目录并按数值排序
checkpoint_dirs=$(find "$base_dir" -maxdepth 1 -type d -not -name "evaluations" | sort -V)

# 遍历 checkpoints 目录下的所有子目录（排除 evaluations）
for checkpoint_dir in $checkpoint_dirs; do
    # 跳过base_dir本身
    if [ "$checkpoint_dir" = "$base_dir" ]; then
        continue
    fi
    
    checkpoint_name=$(basename "$checkpoint_dir")
    
    # 跳过 evaluations 目录
    if [ "$checkpoint_name" = "evaluations" ]; then
        continue
    fi
    
    # 检查是否存在 pretrained_model
    if [ ! -d "${checkpoint_dir}/pretrained_model" ]; then
        echo "跳过 ${checkpoint_name}：未找到 pretrained_model"
        continue
    fi
    
    # 如果checkpoint名称是纯数字，则比较大小；否则跳过比较（如"last"）
    if [[ "$checkpoint_name" =~ ^[0-9]+$ ]]; then
        if [ "$checkpoint_name" -lt "$start_checkpoint" ]; then
            echo "跳过 ${checkpoint_name}：小于起始checkpoint ${start_checkpoint}"
            continue
        fi
    fi
    
    echo "=========================================="
    echo "正在评估模型: ${checkpoint_name}"
    echo "=========================================="
    
    python -m galaxea_sim.scripts.eval_lerobot_act_policy \
        --task R1ProBlocksStackEasy-traj_aug \
        --pretrained-policy-path "${checkpoint_dir}/pretrained_model" \
        --target_controller_type bimanual_relaxed_ik \
        --dataset_repo_id "galaxea/R1ProBlocksStackEasy-traj_aug/${feature}" \
        --seed ${seed}\
        --save-video
    
    echo "完成评估: ${checkpoint_name}"
    echo ""
done

echo "=========================================="
echo "所有模型评估完成！"
echo "=========================================="