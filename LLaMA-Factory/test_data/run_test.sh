set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export FORCE_TORCHRUN=1

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export TOKENIZERS_PARALLELISM=false


llamafactory-cli train /home/wangxingjian/OneMLLM/LLaMA-Factory/examples/train_full/qwen3_full_sft_autotp.yaml