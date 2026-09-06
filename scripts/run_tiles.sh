#!/usr/bin/env bash
# Tile-size sweep for the 'tiled' rung: square-ish tiles, TI=TK, TJ wide.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
SIZE="${1:-2048}"
OUT=data/tiles_raw.log
: > "$OUT"

for T in 8 16 32 64 128 256 512; do
  echo "== TI=TK=$T TJ=512 ==" >&2
  taskset -c 4 ./matmul -k tiled -T "$T,512,$T" -n 3 "$SIZE" "$SIZE" "$SIZE" \
    2>>"$OUT" | tee -a "$OUT"
done
grep ^RESULT "$OUT" | sed 's/RESULT //' > data/tiles.csv
echo "wrote data/tiles.csv"
