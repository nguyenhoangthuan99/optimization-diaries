#!/bin/bash
# Engine benchmark harness: vLLM vs TensorRT-LLM on 1x RTX Pro 6000 (SM120)
# Usage: run_bench.sh <engine: vllm|trtllm> <model_id> <tag> <gpu> [sweep: decode|prefill|smoke]
set -u
ENGINE=$1; MODEL=$2; TAG=$3; GPU=$4; SWEEP=${5:-smoke}
ROOT=/root/thuan-engine-bench
RESULTS=$ROOT/results/$TAG/$ENGINE
LOGS=$ROOT/logs
mkdir -p "$RESULTS" "$LOGS"
PORT=$((18000 + GPU))
NAME="bench-$ENGINE-gpu$GPU"

VLLM_IMG=vllm/vllm-openai:qwen38-flash-next
TRT_IMG=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc25
HF_MOUNT="-v /mnt/nas/hf_cache:/hf_cache -e HF_HOME=/hf_cache"
MAX_LEN=139264   # 136k: fits 128k prefill + output headroom

start_server() {
  docker rm -f "$NAME" >/dev/null 2>&1
  if [ "$ENGINE" = vllm ]; then
    docker run -d --name "$NAME" --gpus "device=$GPU" --ipc=host -p $PORT:8000 $HF_MOUNT \
      -e VLLM_USE_FLASHINFER_SAMPLER=0 -e VLLM_USE_DEEP_GEMM=0 \
      "$VLLM_IMG" \
      "$MODEL" --max-model-len $MAX_LEN --max-num-seqs 128 \
      --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
      --no-enable-prefix-caching >/dev/null
  else
    cat > /tmp/trtllm-extra-$GPU.yml <<EOF
kv_cache_config:
  free_gpu_memory_fraction: 0.90
enable_chunked_prefill: true
EOF
    docker run -d --name "$NAME" --gpus "device=$GPU" --ipc=host -p $PORT:8000 $HF_MOUNT \
      -v /tmp/trtllm-extra-$GPU.yml:/tmp/extra.yml \
      -v $ROOT/patches/qwen3_5_weight_mapper.py:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py:ro \
      "$TRT_IMG" \
      trtllm-serve "$MODEL" --host 0.0.0.0 --port 8000 \
      --max_batch_size 128 --max_num_tokens 8192 --max_seq_len $MAX_LEN \
      --extra_llm_api_options /tmp/extra.yml >/dev/null
  fi
}

wait_ready() {  # up to 30 min (weight load from NAS + warmup can be slow)
  for i in $(seq 1 360); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then return 0; fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
      echo "SERVER DIED during startup"; docker logs "$NAME" 2>&1 | tail -40; return 1
    fi
    sleep 5
  done
  echo "SERVER TIMEOUT"; return 1
}

bench_one() {  # in_len out_len concurrency
  local IN=$1 OUT=$2 CONC=$3
  local NP=$CONC
  local OUTFILE="$RESULTS/${SWEEP}_in${IN}_out${OUT}_c${CONC}.json"
  [ -s "$OUTFILE" ] && { echo "SKIP(exists) $OUTFILE"; return 0; }
  # feasibility caps: conc <= 8 at 64k/128k context; total prefill tokens <= 4.5M
  if [ $IN -ge 65536 ] && [ $CONC -gt 8 ]; then
    echo "SKIP(longctx-conc) in=$IN c=$CONC"; echo "{\"skipped\":\"longctx_conc_cap\",\"in\":$IN,\"conc\":$CONC}" > "$OUTFILE"; return 0
  fi
  local TOT=$((IN * NP))
  [ $TOT -gt 4500000 ] && { echo "SKIP(budget) in=$IN c=$CONC ($TOT tok)"; echo "{\"skipped\":\"token_budget\",\"in\":$IN,\"conc\":$CONC}" > "$OUTFILE"; return 0; }
  # server liveness pre-flight: a dead server must not burn the rest of the sweep
  if ! curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "SERVER DOWN before in=$IN out=$OUT c=$CONC — restarting"
    docker logs "$NAME" 2>&1 | tail -40 > "$RESULTS/server_crash_in${IN}_c${CONC}.log"
    start_server
    wait_ready || { echo "RESTART FAILED" | tee -a "$LOGS/failures.log"; return 1; }
  fi
  # distinct seed per cell so prompts never repeat across cells (defeats any
  # server-side prefix reuse; deterministic for reproducibility)
  local SEED=$((IN + OUT * 7 + CONC * 131))
  docker run --rm --network host $HF_MOUNT -v $ROOT/results:/results \
    --entrypoint vllm "$VLLM_IMG" bench serve \
    --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
    --dataset-name random --random-input-len $IN --random-output-len $OUT \
    --seed $SEED \
    --num-prompts $NP --max-concurrency $CONC --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "/results/$TAG/$ENGINE" \
    --result-filename "$(basename $OUTFILE)" 2>&1 | tail -3
}

# early exit if every result file for this sweep already exists
missing=0
check_missing() { [ -s "$RESULTS/${SWEEP}_in$1_out$2_c$3.json" ] || missing=1; }
case $SWEEP in
  smoke) check_missing 512 128 4 ;;
  decode) for C in 1 4 8 16 32 64 128; do for O in 512 1024 2048 4096 8192; do check_missing 128 $O $C; done; done ;;
  prefill) for C in 1 4 8 16 32 64 128; do for I in 2048 4096 8192 16384 32768 65536 131072; do check_missing $I 8 $C; done; done ;;
esac
[ $missing -eq 0 ] && { echo "ALL RESULTS EXIST, skipping $ENGINE $MODEL $SWEEP"; exit 0; }

start_server
if ! wait_ready; then
  echo "FAILED_TO_START $ENGINE $MODEL" | tee -a "$LOGS/failures.log"
  docker logs "$NAME" 2>&1 | tail -60 > "$RESULTS/server_start_fail.log"
  docker rm -f "$NAME" >/dev/null 2>&1
  exit 2
fi
echo "server ready: $ENGINE $MODEL on GPU $GPU"

# throwaway warmup: first real prefill triggers one-time lazy compile/autotune
# on both engines (measured ~34s on vLLM); pay it here, not in a measured cell
docker run --rm --network host $HF_MOUNT --entrypoint vllm "$VLLM_IMG" bench serve \
  --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
  --dataset-name random --random-input-len 2048 --random-output-len 64 \
  --num-prompts 8 --max-concurrency 8 --ignore-eos >/dev/null 2>&1
echo "warmup done"

case $SWEEP in
  smoke)
    bench_one 512 128 4
    ;;
  decode)
    for CONC in 1 4 8 16 32 64 128; do
      for OUT in 512 1024 2048 4096 8192; do
        bench_one 128 $OUT $CONC
      done
    done
    ;;
  prefill)
    for CONC in 1 4 8 16 32 64 128; do
      for IN in 2048 4096 8192 16384 32768 65536 131072; do
        bench_one $IN 8 $CONC
      done
    done
    ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1
echo "DONE $ENGINE $MODEL $SWEEP"
