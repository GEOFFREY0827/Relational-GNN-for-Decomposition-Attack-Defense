#!/usr/bin/env bash
# MiniLM subtasks -> static-relation GNN -> GTR target space -> MLP -> Vec2Text.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

accelerate launch \
  --multi_gpu --num_processes 4 --num_machines 1 \
  --main_process_ip 127.0.0.1 --main_process_port "${MASTER_PORT:-29501}" \
  --mixed_precision no --dynamo_backend no \
  train.py --device cuda \
  --train data/decomposedharm_qa/train.jsonl \
  --val data/decomposedharm_qa/validation.jsonl \
  --encoder sentence-transformers/all-MiniLM-L6-v2 \
  --q-encoder vec2text/gtr-base \
  --static-relations --layers 2 \
  --lambda-generation 0.1 --adapter-hidden-dim 256 \
  --batch-size 10 --epochs 50 \
  --out runs/minilm-gtr-decoder-static \
  --wandb --wandb-name minilm-gtr-decoder-static \
  "$@"
