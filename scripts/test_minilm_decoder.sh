#!/usr/bin/env bash
# Usage: bash scripts/test_minilm_decoder.sh [experiment-directory] [decode-limit]
# decode-limit=0 means the entire test set. Default: first 20 examples.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
experiment="${1:-runs/minilm-gtr-decoder-static}"
limit="${2:-20}"

python eval.py \
  --ckpt "$experiment/best.pt" \
  --test data/decomposedharm_qa/test.jsonl \
  --batch-size 32 --wandb --wandb-name minilm-gtr-embedding-test

python decode_intent.py \
  --ckpt "$experiment/best.pt" \
  --test data/decomposedharm_qa/test.jsonl \
  --batch-size 4 --limit "$limit" \
  --out "$experiment/test-decode.jsonl" \
  --wandb --wandb-name minilm-gtr-generation-test
