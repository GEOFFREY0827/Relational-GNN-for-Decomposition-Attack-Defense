#!/usr/bin/env bash
# Compare both checkpoints on the exact same clean/injected test pair.
# Usage: bash scripts/test_benign_injection.sh [clean-run] [injection-trained-run]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
clean_run="${1:-runs/minilm-decoder-defense}"
injected_run="${2:-runs/minilm-defense-injection25}"

for experiment in "$clean_run" "$injected_run"; do
  for condition in clean injected25; do
    data=data/decomposedharm_qa/test.jsonl
    if [[ "$condition" == injected25 ]]; then
      data=data/benign_injection_25/test.jsonl
    fi
    python eval.py --ckpt "$experiment/best_detection.pt" --test "$data" \
      --batch-size 10 --out "$experiment/robustness-$condition.json" \
      --wandb --wandb-name "$(basename "$experiment")-$condition"
  done
done
