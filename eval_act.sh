#!/bin/bash
set -e
set -u

# === 默认参数 ===
POLICY_REPO="${POLICY_REPO:-galaxea/ACT-R1ProBlocksStackEasy-traj_aug}"
DATASET_REPO="${DATASET_REPO:-galaxea/R1ProBlocksStackEasy-traj_aug/all}"
NUM_EVALS="${NUM_EVALS:-100}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/sen/workspace/galaxea/GalaxeaManipSim/outputs/EVAL}"
HEADLESS="${HEADLESS:-true}"
TEMPORAL_ENSEMBLE="${TEMPORAL_ENSEMBLE:-true}"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/R1ProBlocksStackEasy-${TIMESTAMP}}"

echo "=== Evaluating ACT Policy ==="
echo "POLICY_REPO = ${POLICY_REPO}"
echo "DATASET_REPO = ${DATASET_REPO}"
echo "NUM_EVALS = ${NUM_EVALS}"
echo "DEVICE = ${DEVICE}"
echo "OUTPUT_DIR = ${OUTPUT_DIR}"
echo "==================================="

python ../GalaxeaLeRobot/src/lerobot/scripts/eval.py \
    --policy.repo_id=${POLICY_REPO} \
    --dataset.repo_id=${DATASET_REPO} \
    --num_evaluations=${NUM_EVALS} \
    --policy.device=${DEVICE} \
    --temporal_ensemble=${TEMPORAL_ENSEMBLE} \
    --headless=${HEADLESS} \
    --output_dir=${OUTPUT_DIR} \
    --policy.path=${POLICY_REPO} \
    --env.type=blocks_stack_easy_traj_aug
