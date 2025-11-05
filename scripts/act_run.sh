#!/bin/bash

# 默认参数值
FEATURE="grasp_sample_only"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --feature)
      FEATURE="$2"
      shift # 跳过参数名
      shift # 跳过参数值
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 运行数据转换
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot --robot r1_pro --task R1ProBlocksStackEasy --tag collected --feature $FEATURE

# 等待数据转换完成
wait

# 设置数据集变量，与特征相同
DATASET=$FEATURE

# 运行训练
python ../../lerobot/src/lerobot/scripts/train.py  \
    --policy.type=act     \
    --policy.dim_model=512    \
    --policy.n_action_steps=30    \
    --policy.chunk_size=30     \
    --policy.kl_weight=5    \
    --policy.vision_backbone=resnet18   \
    --dataset.repo_id=galaxea/R1ProBlocksStackEasy/${DATASET} \
    --dataset.image_transforms.enable=true    \
    --batch_size=64    \
    --num_workers=8    \
    --policy.optimizer_lr=2e-4  \
    --steps=150000    \
    --policy.use_vae=true    \
    --save_freq=30000    \
    --save_checkpoint=true   \
    --log_freq=5000 \
    --policy.push_to_hub=false   \
    --output_dir=../../outputs/ACT/R1ProBlocksStackEasy/${DATASET}