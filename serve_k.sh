#!/usr/bin/env bash
# Start vLLM with k=0/1/2/3. k=0 means no speculative decoding.
#
# vLLM 0.27.1 cannot take stats_reporting_level. Do not add it unless you
# explicitly export ENABLE_SPEC_STATS_LEVEL=1 on a newer build.
set -euo pipefail

K="${1:?usage: ./serve_k.sh <0|1|2|3>}"
MODEL="${MODEL:-/root/autodl-tmp/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master}"
DRAFT="${DRAFT:-/root/autodl-tmp/models/Llama-3.1-8B-Instruct-eagle3}"
NAME="${NAME:-llama-3.1-8b}"
PORT="${PORT:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM="${GPU_MEM:-0.85}"

source /root/autodl-tmp/venv/bin/activate

cmd=(
  vllm serve "$MODEL"
  --served-model-name "$NAME"
  --tensor-parallel-size 1
  --max-model-len 4096
  --max-num-seqs "$MAX_NUM_SEQS"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEM"
)

if [[ "$K" != "0" ]]; then
  spec="{\"method\": \"eagle3\", \"model\": \"$DRAFT\", \"num_speculative_tokens\": $K, \"draft_tensor_parallel_size\": 1}"
  if [[ "${ENABLE_SPEC_STATS_LEVEL:-0}" == "1" ]]; then
    spec="{\"method\": \"eagle3\", \"model\": \"$DRAFT\", \"num_speculative_tokens\": $K, \"draft_tensor_parallel_size\": 1, \"stats_reporting_level\": \"detailed\"}"
  fi
  cmd+=(--speculative-config "$spec")
  echo "speculative-config: $spec"
fi

echo "starting vLLM  k=$K  port=$PORT"
exec "${cmd[@]}"
