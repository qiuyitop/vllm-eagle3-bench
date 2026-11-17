#!/usr/bin/env bash
# Run one vllm bench job. Example:
#   DATASET=mtbench K=3 T=0 C=16 ./run_bench.sh
set -euo pipefail

DATASET="${DATASET:?set DATASET=mtbench|humaneval|gsm8k|alpaca}"
K="${K:?set K=0|1|2|3}"
T="${T:?set T=0|0.7}"
C="${C:?set C=1|8|16}"

NAME="${NAME:-llama-3.1-8b}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
TOKENIZER="${TOKENIZER:-/root/autodl-tmp/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master}"
RESULT_DIR="${RESULT_DIR:-/root/autodl-tmp/datasets/week1_results}"

case "$DATASET" in
  mtbench)   DATA_PATH="${DATA_PATH:-/root/autodl-tmp/datasets/mt_bench/mtbench_chat_vllm.jsonl}" ;;
  humaneval) DATA_PATH="${DATA_PATH:-/root/autodl-tmp/datasets/humaneval/humaneval_vllm.jsonl}"; N="${N:-164}" ;;
  gsm8k)     DATA_PATH="${DATA_PATH:-/root/autodl-tmp/datasets/gsm8k/gsm8k_vllm.jsonl}"; N="${N:-160}" ;;
  alpaca)    DATA_PATH="${DATA_PATH:-/root/autodl-tmp/datasets/alpaca/alpaca_vllm.jsonl}"; N="${N:-160}" ;;
  *) echo "unknown DATASET=$DATASET  use mtbench|humaneval|gsm8k|alpaca"; exit 1 ;;
esac

T_TAG="$T"
if [[ "$T" == "0.7" ]]; then T_TAG="07"; fi
OUT="${RESULT_DIR}/${DATASET}_k${K}_t${T_TAG}_c${C}.json"
mkdir -p "$RESULT_DIR"

source /root/autodl-tmp/venv/bin/activate

echo "bench  dataset=$DATASET k=$K t=$T c=$C  -> $OUT"
vllm bench serve \
  --base-url "$BASE_URL" \
  --model "$NAME" \
  --endpoint /v1/completions \
  --dataset-name custom \
  --dataset-path "$DATA_PATH" \
  --tokenizer "$TOKENIZER" \
  --skip-chat-template \
  --num-prompts "$N" \
  --max-concurrency "$C" \
  --temperature "$T" \
  --save-result \
  --save-detailed \
  --result-filename "$OUT"

echo "done $OUT"
