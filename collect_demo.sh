#!/bin/bash
# 默认参数
ENV_NAME=""
NUM_DEMOS=500
FEATURE="normal"
SEED=1
TABLE_TYPE="red"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --env-name)
      ENV_NAME="$2"
      shift; shift ;;
    --num-demos)
      NUM_DEMOS="$2"
      shift; shift ;;
    --feature)
      FEATURE="$2"
      shift; shift ;;
    --seed)
      SEED="$2"
      shift; shift ;;
    --table-type)
      TABLE_TYPE="$2"
      shift; shift ;;
    *)
      echo "Unknown argument: $1"
      exit 1 ;;
  esac
done

# 参数检查
if [[ -z "$ENV_NAME" ]]; then
  echo "❌ Error: --env-name is required!"
  exit 1
fi

# ============================
# 执行命令
# ============================
echo "🚀 Starting demo collection and replay..."
echo "Environment: $ENV_NAME"
echo "Num demos: $NUM_DEMOS"
echo "Feature: $FEATURE"
echo "Seed: $SEED"
echo "Table type: $TABLE_TYPE"
echo "=============================="

#1️⃣ 收集 demonstrations
python -m galaxea_sim.scripts.collect_demos \
  --env-name "$ENV_NAME" \
  --num-demos "$NUM_DEMOS" \
  --feature "$FEATURE" \
  --seed "$SEED" \
  --table_type "$TABLE_TYPE"\
  --obs_mode image
# 2️⃣ 回放 demonstrations
python -m galaxea_sim.scripts.replay_demos \
  --env-name "$ENV_NAME" \
  --num-demos "$((NUM_DEMOS-50))" \
  --feature "$FEATURE" \
  --table_type "$TABLE_TYPE"\

echo "✅ Finished!"
