set -euo pipefail

# 必要参数
export CUDA_VISIBLE_DEVICES=4,5
# 多卡训练使用 torchrun
export FORCE_TORCHRUN=1
export MASTER_PORT=11451

# 根据卡的情况而定
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export TOKENIZERS_PARALLELISM=false




llamafactory-cli train /home/wangxingjian/OneMLLM/LLaMA-Factory/examples/train_full/qwen3_vl_metaphor_emotion.yaml
