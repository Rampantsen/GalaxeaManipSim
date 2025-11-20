#!/bin/bash
# Diffusion Policy 不同配置训练脚本

set -e

# 参数
TASK_NAME="${1:-R1ProBlocksStackEasy}"
MODE="${2:-all}"
CONFIG="${3:-original}"  # original, time_aligned, long
FILTER_NOISE="${4:-true}"

echo "=========================================="
echo "Diffusion Policy 配置化训练"
echo "=========================================="
echo "任务: ${TASK_NAME}"
echo "模式: ${MODE}"
echo "配置: ${CONFIG}"
echo "噪声过滤: ${FILTER_NOISE}"
echo "=========================================="

# 根据配置选择参数
case "${CONFIG}" in
    "original")
        echo "使用原版配置（16/8/2）"
        HORIZON=16
        N_ACTION_STEPS=8
        N_OBS_STEPS=2
        BATCH_SIZE=64
        EXP_NAME="original_16_8_2"
        ;;
    "time_aligned")
        echo "使用时间对齐配置（24/12/3）"
        HORIZON=24
        N_ACTION_STEPS=12
        N_OBS_STEPS=3
        BATCH_SIZE=32
        EXP_NAME="time_aligned_24_12_3"
        ;;
    "long")
        echo "使用长预测配置（32/20/4）"
        HORIZON=32
        N_ACTION_STEPS=20
        N_OBS_STEPS=4
        BATCH_SIZE=24
        EXP_NAME="long_32_20_4"
        ;;
    *)
        echo "错误：未知配置 ${CONFIG}"
        echo "可选：original, time_aligned, long"
        exit 1
        ;;
esac

# 时间计算（15Hz）
ACTION_TIME=$(echo "scale=2; ${N_ACTION_STEPS}/15" | bc)
HORIZON_TIME=$(echo "scale=2; ${HORIZON}/15" | bc)
OBS_TIME=$(echo "scale=2; ${N_OBS_STEPS}/15" | bc)

echo ""
echo "配置详情："
echo "  - Horizon: ${HORIZON}步 (${HORIZON_TIME}秒)"
echo "  - Action steps: ${N_ACTION_STEPS}步 (${ACTION_TIME}秒)"
echo "  - Obs steps: ${N_OBS_STEPS}步 (${OBS_TIME}秒)"
echo "  - Batch size: ${BATCH_SIZE}"
echo ""

# 检查数据集
ZARR_PATH="datasets_diffusion_policy/${TASK_NAME}_${MODE}_with_noise.zarr"
if [ ! -d "${ZARR_PATH}" ]; then
    echo "数据集不存在，请先运行："
    echo "./scripts/train_dp_with_noise.sh ${TASK_NAME} ${MODE} ${FILTER_NOISE}"
    exit 1
fi

# 检查任务配置文件
TASK_CONFIG="policy/dp/diffusion_policy/config/task/galaxea_${TASK_NAME}_${MODE}.yaml"
if [ ! -f "${TASK_CONFIG}" ]; then
    echo "任务配置不存在，请先运行："
    echo "./scripts/train_dp_with_noise.sh ${TASK_NAME} ${MODE} ${FILTER_NOISE}"
    exit 1
fi

echo "训练命令："
echo ""
echo "cd policy/dp && python train.py \\"
echo "  --config-name=train_galaxea_diffusion_unet_image_workspace \\"
echo "  task=galaxea_${TASK_NAME}_${MODE} \\"
echo "  horizon=${HORIZON} \\"
echo "  n_obs_steps=${N_OBS_STEPS} \\"
echo "  n_action_steps=${N_ACTION_STEPS} \\"
echo "  training.num_epochs=1000 \\"
echo "  training.device='cuda:0' \\"
echo "  dataloader.batch_size=${BATCH_SIZE} \\"
echo "  exp_name='${EXP_NAME}_filter_${FILTER_NOISE}'"
echo ""
echo "=========================================="

# 可选：直接运行
read -p "是否立即开始训练？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    cd policy/dp && python train.py \
        --config-name=train_galaxea_diffusion_unet_image_workspace \
        task=galaxea_${TASK_NAME}_${MODE} \
        horizon=${HORIZON} \
        n_obs_steps=${N_OBS_STEPS} \
        n_action_steps=${N_ACTION_STEPS} \
        training.num_epochs=1000 \
        training.device='cuda:0' \
        dataloader.batch_size=${BATCH_SIZE} \
        exp_name="${EXP_NAME}_filter_${FILTER_NOISE}"
fi
