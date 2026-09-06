#!/usr/bin/env bash
# Cache counters per kernel via perf stat. Single run, no warmup, so the
# counters are ~one kernel execution (the O(N^2) fill is noise next to O(N^3)).
# Size 1024 keeps the slow orders tolerable while still spilling L2.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
SIZE="${1:-1024}"
EVENTS="cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,cache-references,cache-misses"
OUT=data/perf_${SIZE}.log
: > "$OUT"

for k in ijk ikj jik jki kij kji tiled regblock avx2 avx512; do
  echo "=== kernel=$k size=$SIZE ===" | tee -a "$OUT"
  taskset -c 4 perf stat -e "$EVENTS" \
    ./matmul -k "$k" -n 1 -w 0 "$SIZE" "$SIZE" "$SIZE" >>"$OUT" 2>&1
done
echo "wrote $OUT"
