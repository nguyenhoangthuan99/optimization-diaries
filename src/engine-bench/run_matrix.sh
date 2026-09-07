#!/bin/bash
# Drive the full matrix on one GPU: all models for one engine+sweep, sequentially.
# Usage: run_matrix.sh <gpu> <engine> <sweep> [size-filter: 4b|9b|27b|all]
set -u
GPU=$1; ENGINE=$2; SWEEP=$3; FILTER=${4:-all}
ROOT=/root/thuan-engine-bench

declare -a ROWS=(
  "Qwen/Qwen3.5-4B|4b-bf16"
  "lovedheart/Qwen3.5-4B-FP8|4b-fp8"
  "AxionML/Qwen3.5-4B-NVFP4|4b-nvfp4"
  "Qwen/Qwen3.5-9B|9b-bf16"
  "lovedheart/Qwen3.5-9B-FP8|9b-fp8"
  "AxionML/Qwen3.5-9B-NVFP4|9b-nvfp4"
  "Qwen/Qwen3.8-27B|27b-bf16"
  "Qwen/Qwen3.8-27B-FP8|27b-fp8"
  "Inferact/Qwen3.8-27B-NVFP4|27b-nvfp4"
)

for ROW in "${ROWS[@]}"; do
  MODEL="${ROW%%|*}"; TAG="${ROW##*|}"
  if [ "$FILTER" != all ] && [[ "$TAG" != "$FILTER"* ]]; then continue; fi
  echo "########## $ENGINE $TAG $SWEEP (GPU $GPU) $(date) ##########"
  bash $ROOT/run_bench.sh "$ENGINE" "$MODEL" "$TAG" "$GPU" "$SWEEP"
done
echo "MATRIX DONE: $ENGINE $SWEEP $FILTER $(date)"
