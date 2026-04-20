#!/usr/bin/env bash
#
# 双卡训练启动脚本（2x RTX 3090, 24GB × 2）
#
# GPU 分配:
#   GPU 0: 训练（策略模型 + 参考模型 + LoRA + 优化器）
#   GPU 1: vLLM 推理服务器（生成完成回答）
#
# 用法:
#   ./scripts/train_distributed.sh --model configs/models/qwen3-4b.yaml --method configs/methods/grpo.yaml
#   ./scripts/train_distributed.sh --model configs/models/qwen3-4b.yaml --method configs/methods/rloo.yaml
#
# 可选参数:
#   --wandb          启用 WandB 日志
#   --output DIR     自定义输出目录
#   --vllm-port PORT vLLM 服务器端口（默认 8000）
#   --help           显示帮助信息

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_PORT=8000
WANDB_FLAG=""
OUTPUT_DIR=""

# 解析参数
MODEL_CONFIG=""
METHOD_CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --method)
            METHOD_CONFIG="$2"
            shift 2
            ;;
        --wandb)
            WANDB_FLAG="--wandb"
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --vllm-port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --help)
            head -15 "$0" | tail -12 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_CONFIG" || -z "$METHOD_CONFIG" ]]; then
    echo "用法: $0 --model <model_config> --method <method_config> [--wandb] [--output DIR]"
    exit 1
fi

echo "=========================================="
echo "  GRPO-Math 双卡训练启动"
echo "=========================================="
echo "模型配置: $MODEL_CONFIG"
echo "方法配置: $METHOD_CONFIG"
echo "vLLM 端口: $VLLM_PORT"
echo "GPU 0: 训练 | GPU 1: vLLM 推理"
echo "=========================================="

# 检查依赖
if ! command -v uv &> /dev/null; then
    echo "错误: 未找到 uv，请先安装: https://docs.astral.sh/uv/"
    exit 1
fi

# 提取模型 ID（从 YAML 中读取 huggingface_id 或 local_dir）
MODEL_ID=$(uv run python -c "
import yaml, sys
with open('$MODEL_CONFIG') as f:
    cfg = yaml.safe_load(f)
m = cfg['model']
print(m.get('local_dir', m['huggingface_id']))
")

echo "模型路径: $MODEL_ID"

# 清理可能残留的 vLLM 进程
pkill -f "trl vllm-serve" 2>/dev/null || true
sleep 2

# 启动 vLLM 服务器（GPU 1）
echo ""
echo "[1/3] 启动 vLLM 服务器（GPU 1）..."
CUDA_VISIBLE_DEVICES=1 uv run trl vllm-serve \
    --model "$MODEL_ID" \
    --tensor_parallel_size 1 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --log_level warning &

VLLM_PID=$!
echo "vLLM 服务器 PID: $VLLM_PID"

# 等待 vLLM 服务器就绪
echo "[2/3] 等待 vLLM 服务器就绪..."
MAX_RETRIES=300
RETRY=0
while ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
    RETRY=$((RETRY + 1))
    if [[ $RETRY -ge $MAX_RETRIES ]]; then
        echo "错误: vLLM 服务器启动超时（${MAX_RETRIES}s）"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "  等待中... (${RETRY}s)"
    sleep 1
done
echo "vLLM 服务器已就绪！"

# 启动训练（GPU 0）
echo ""
echo "[3/3] 启动训练（GPU 0）..."
TRAIN_CMD="CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
    --model $MODEL_CONFIG \
    --method $METHOD_CONFIG \
    $WANDB_FLAG"

if [[ -n "$OUTPUT_DIR" ]]; then
    TRAIN_CMD="$TRAIN_CMD --output $OUTPUT_DIR"
fi

echo "执行命令: $TRAIN_CMD"
echo "=========================================="

# 确保训练结束后清理 vLLM 进程
trap "echo '清理 vLLM 服务器...'; kill $VLLM_PID 2>/dev/null || true" EXIT

eval $TRAIN_CMD
TRAIN_EXIT=$?

echo ""
echo "=========================================="
if [[ $TRAIN_EXIT -eq 0 ]]; then
    echo "训练完成！"
else
    echo "训练失败（退出码: $TRAIN_EXIT）"
fi
echo "=========================================="

exit $TRAIN_EXIT
