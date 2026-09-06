#!/usr/bin/env bash
# Full ladder sweep -> data/ladder.csv (RESULT lines are already CSV-ish).
# Single-core rungs are pinned to one core for stable numbers.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
OUT=data/ladder_raw.log
: > "$OUT"

SIZE="${1:-2048}"
PIN="taskset -c 4"

run() { echo "+ $*" | tee -a "$OUT"; "$@" 2>>"$OUT" | tee -a "$OUT"; }

echo "== validation (odd shapes catch index bugs) ==" | tee -a "$OUT"
for k in $(./matmul -l | awk '{print $1}'); do
  t=1; [ "$k" = omp ] && t=8
  run ./matmul -k "$k" -t $t -v -n 1 -w 0 383 389 397
done

echo "== loop orders @ ${SIZE} (single core) ==" | tee -a "$OUT"
for k in ijk ikj jik jki kij kji; do
  n=3
  run $PIN ./matmul -k "$k" -n $n "$SIZE" "$SIZE" "$SIZE"
done

echo "== single-core rungs @ ${SIZE} ==" | tee -a "$OUT"
for k in tiled regblock avx2 avx512; do
  run $PIN ./matmul -k "$k" -n 5 "$SIZE" "$SIZE" "$SIZE"
done

echo "== blas single-core @ ${SIZE} ==" | tee -a "$OUT"
OPENBLAS_NUM_THREADS=1 run $PIN ./matmul -k blas -n 5 "$SIZE" "$SIZE" "$SIZE"

grep ^RESULT "$OUT" | sed 's/RESULT //' > data/ladder.csv
echo "wrote data/ladder.csv"
