#!/usr/bin/env bash
# Thread scaling: omp kernel, pinned (OMP_PROC_BIND) vs unpinned, plus BLAS.
# Writes data/scaling_pinned.log, data/scaling_unpinned.log, data/scaling_blas.log
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
SIZE="${1:-4096}"
THREADS="1 2 4 8 16 32 48 64 96 128 144"

: > data/scaling_pinned.log
for t in $THREADS; do
  echo "== pinned threads=$t ==" >&2
  OMP_PROC_BIND=close OMP_PLACES=cores \
    ./matmul -k omp -t "$t" -n 5 "$SIZE" "$SIZE" "$SIZE" \
    2>>data/scaling_pinned.log | tee -a data/scaling_pinned.log
done

: > data/scaling_unpinned.log
for t in $THREADS; do
  echo "== unpinned threads=$t ==" >&2
  OMP_PROC_BIND=false \
    ./matmul -k omp -t "$t" -n 5 "$SIZE" "$SIZE" "$SIZE" \
    2>>data/scaling_unpinned.log | tee -a data/scaling_unpinned.log
done

: > data/scaling_blas.log
for t in 1 16 64 144; do
  echo "== blas threads=$t ==" >&2
  OPENBLAS_NUM_THREADS=$t \
    ./matmul -k blas -t "$t" -n 5 "$SIZE" "$SIZE" "$SIZE" \
    2>>data/scaling_blas.log | tee -a data/scaling_blas.log
done
echo done
