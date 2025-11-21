#!/bin/bash

# Galaxea Diffusion Policy 噪声过滤训练脚本
# 功能：
# 1. 转换数据为Diffusion Policy格式（保留噪声标签）
# 2. 使用噪声过滤功能训练模型
# 3. 三相机输入: img_head, img_left, img_right

set -e

# 配置参数
TASK_NAME="${1:-R1ProBlocksStackEasy}"
MODE="${2:-all}"  # all, baseline, grasp_sample_only
FILTER_NOISE="${3:-true}"  # true or false

# 参数验证
if [[ "${FILTER_NOISE}" != "true" && "${FILTER_NOISE}" != "false" ]]; then
    echo "错误: FILTER_NOISE 必须是 true 或 false"
    exit 1
fi

# 路径设置
SRC_DIR="datasets/${TASK_NAME}/${MODE}/collected"
DST_PATH="datasets_diffusion_policy/${TASK_NAME}_${MODE}_with_noise.zarr"

# 检查源目录是否存在
if [ ! -d "${SRC_DIR}" ]; then
    echo "错误: 源目录不存在: ${SRC_DIR}"
    echo "请确保已经收集了数据"
    exit 1
fi

echo "=========================================="
echo "Galaxea Diffusion Policy 噪声过滤训练"
echo "=========================================="
echo "任务: ${TASK_NAME}"
echo "模式: ${MODE}"
echo "过滤噪声: ${FILTER_NOISE}"
echo "多相机: img_head, img_left, img_right"
echo "源目录: ${SRC_DIR}"
echo "目标文件: ${DST_PATH}"
echo "=========================================="

# 步骤1：转换数据（保留噪声标签）
echo ""
echo "[1/3] 转换数据集（三相机）..."
# 注意：.zarr是目录，不是文件，使用 -d 检查
if [ ! -d "${DST_PATH}" ]; then
    python -m galaxea_sim.scripts.convert_to_diffusion_policy_with_noise \
        --src-dir "${SRC_DIR}" \
        --dst-path "${DST_PATH}" \
        --use-multi-camera \
        --target-width 224 \
        --target-height 224
    echo "✅ 数据转换完成"
else
    echo "⚠️ 数据集已存在，跳过转换"
fi

# 步骤2：创建动态配置文件
echo ""
echo "[2/3] 创建配置文件..."

# 创建任务配置
TASK_CONFIG="policy/dp/diffusion_policy/config/task/galaxea_${TASK_NAME}_${MODE}.yaml"

cat > "${TASK_CONFIG}" << EOF
name: galaxea_${TASK_NAME}_${MODE}

# 图像和状态形状配置（三相机）
image_shape: &image_shape [3, 224, 224]  # 每个相机都是3通道
shape_meta: &shape_meta
  obs:
    img_head:  # 头部相机
      shape: *image_shape
      type: rgb
    img_left:  # 左手相机
      shape: *image_shape
      type: rgb
    img_right:  # 右手相机
      shape: *image_shape
      type: rgb
    state:
      shape: [16]
      type: low_dim
  action:
    shape: [16]

# 数据集配置
dataset:
  _target_: galaxea_sim.utils.dp_noise_filtered_dataset.GalaxeaImageDataset
  zarr_path: ../../${DST_PATH}  # 相对于policy/dp目录
  horizon: \${horizon}
  pad_before: \${eval:'\${n_obs_steps}-1'}
  pad_after: \${eval:'\${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.1
  max_train_episodes: null
  filter_noise: ${FILTER_NOISE}  # 是否过滤噪声

# 环境运行器（按照DP标准结构组织）
env_runner:
  _target_: diffusion_policy.env_runner.galaxea_image_runner.GalaxeaImageRunner
  output_dir: null
  env_name: ${TASK_NAME}-v0
  n_test: 5  # 每次评估5个episode
  n_test_vis: 0  # 暂不支持视频录制
  test_start_seed: 100000
  max_steps: 300
  n_obs_steps: \${n_obs_steps}
  n_action_steps: \${n_action_steps}
  fps: 15
  past_action: False
  tqdm_interval_sec: 1.0
EOF
echo "✅ 配置文件创建完成: ${TASK_CONFIG}"

# 显示训练配置总结
echo ""
echo "=========================================="
echo "💡 训练配置总结："
echo "=========================================="
echo "1. filter_noise=${FILTER_NOISE}: 噪声帧处理策略"
echo "2. Action chunk: 8步 (0.53秒@15Hz)"
echo "3. Horizon: 16步 (1.07秒@15Hz)"
echo "4. 三相机输入: 224x224 x 3个相机"
echo "5. Batch size: 16 (内存优化)"
echo "6. Workers: 2 (降低内存占用)"
echo "7. Epochs: 1000"
echo "=========================================="

# 步骤3：直接开始训练
echo ""
echo "[3/3] 启动训练..."
echo ""

# 进入目录并开始训练
# 注意：每50个epoch会在Galaxea环境中评估策略
cd policy/dp && python train.py \
  --config-name=train_galaxea_diffusion_unet_image_workspace \
  task=galaxea_${TASK_NAME}_${MODE} \
  training.num_epochs=1000 \
  training.device='cuda:0' \
  exp_name="filter_noise_${FILTER_NOISE}"

