#!/usr/bin/env bash
# Packing at chip scale: omp (6x32, each thread re-streams B) vs omp_pack
# (8x16, all threads read one shared packed B panel). Same two kernels are
# ~4% apart at one core but ~25% apart at 64 threads, so the scale gap is a
# memory-traffic effect, not arithmetic. Writes data/multicore.csv + .log
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
SIZE="${1:-4096}"
OUT=data/multicore_raw.log
: > "$OUT"

for nt in 1 8 16 32 64; do
  for k in omp omp_pack; do
    # TI=0 -> default 6x32 register tile (omp); TI=8 -> 8x16 packed (omp_pack)
    TI=0; [ "$k" = omp_pack ] && TI=8
    echo "== k=$k nt=$nt ==" >&2
    taskset -c 0-$((nt-1)) \
      ./matmul -k "$k" -t "$nt" -T "$TI,256,256" -n 3 "$SIZE" "$SIZE" "$SIZE" \
      2>>"$OUT" | tee -a "$OUT"
  done
done
grep ^RESULT "$OUT" | sed 's/RESULT //' > data/multicore.csv
echo "wrote data/multicore.csv"
