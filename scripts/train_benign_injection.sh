#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python prepare_benign_injection.py --count 25 --seed "${INJECTION_SEED:-42}"
bash scripts/train_minilm_decoder.sh \
  --train data/benign_injection_25/train.jsonl \
  --val data/benign_injection_25/validation.jsonl \
  --detection-head \
  --out runs/minilm-defense-injection25 \
  --wandb-name minilm-defense-injection25 \
  "$@"
