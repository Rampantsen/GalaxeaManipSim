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

# 等待数据转换完成
wait

# 设置数据集变量，与特征相同
DATASET=$FEATURE
# 运行训练
python ../lerobot/src/lerobot/scripts/train.py  \
    --policy.type=act     \
    --policy.dim_model=512    \
    --policy.n_action_steps=30    \
    --policy.chunk_size=30     \
    --policy.kl_weight=5    \
    --policy.vision_backbone=resnet18   \
    --dataset.repo_id=galaxea/R1ProBlocksStackEasy/${DATASET} \
    --dataset.image_transforms.enable=true    \
    --batch_size=32    \
    --num_workers=8    \
    --policy.optimizer_lr=1e-5  \
    --steps=300000    \
    --policy.use_vae=true    \
    --save_freq=30000    \
    --save_checkpoint=true   \
    --log_freq=5000 \
    --policy.push_to_hub=false   \
    --output_dir=./outputs/ACT/R1ProBlocksStackEasy/${DATASET}\
    --config_path=./outputs/ACT/R1ProBlocksStackEasy/${DATASET}/checkpoints/last/pretrained_model/train_config.json \
    --resume=true