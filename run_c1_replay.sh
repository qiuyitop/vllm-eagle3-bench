#!/usr/bin/env bash
# C=1 压测 + 转逐步日志 + 回放。服务必须已经在另一个终端用 ./serve_k.sh 3 起好。
#
# 用法:
#   ./run_c1_replay.sh alpaca
#   ./run_c1_replay.sh mtbench
#   T=0.7 ./run_c1_replay.sh gsm8k
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATASET="${1:?usage: $0 mtbench|humaneval|gsm8k|alpaca}"
export DATASET
export K="${K:-3}"
export T="${T:-0}"
export C=1
export RESULT_DIR="${RESULT_DIR:-/root/autodl-tmp/datasets/week1_results}"
OBSERVE="$RESULT_DIR/observe.jsonl"
BASE_URL="${BASE_URL:-http://localhost:8000}"

T_TAG="$T"
if [[ "$T" == "0.7" ]]; then T_TAG="07"; fi
TAG="${DATASET}_k${K}_t${T_TAG}_c${C}"
BENCH_JSON="$RESULT_DIR/${TAG}.json"
STEPS="$RESULT_DIR/steps_${TAG}.jsonl"
REPLAY_DIR="$RESULT_DIR/replay_${TAG}"

source /root/autodl-tmp/venv/bin/activate

if ! curl -sf "$BASE_URL/v1/models" >/dev/null; then
  echo "服务没起来。另开终端先执行:  cd $ROOT && ./serve_k.sh $K"
  exit 1
fi

mkdir -p "$RESULT_DIR"
rm -f "$OBSERVE"
echo "cleared $OBSERVE"

echo "===== 1/3 bench $TAG ====="
"$ROOT/run_bench.sh"

if [[ ! -s "$OBSERVE" ]]; then
  echo "没有 $OBSERVE。挂钩可能没打上。执行一次: python3 $ROOT/install_observe_hook.py"
  echo "然后重启 ./serve_k.sh $K，再重跑本脚本。"
  exit 1
fi

cp "$OBSERVE" "$RESULT_DIR/observe_${TAG}.jsonl"

echo "===== 2/3 observe -> steps ====="
python3 "$ROOT/observe_to_steps.py" \
  --observe "$OBSERVE" \
  --bench-json "$BENCH_JSON" \
  --out "$STEPS" \
  --dataset "$DATASET" \
  --temperature "$T" \
  --concurrency 1 \
  --run-id "$TAG"

echo "===== 3/3 replay ====="
python3 "$ROOT/replay.py" \
  --steps "$STEPS" \
  --out-dir "$REPLAY_DIR"

echo
echo "done $TAG"
echo "  bench   $BENCH_JSON"
echo "  observe $RESULT_DIR/observe_${TAG}.jsonl"
echo "  steps   $STEPS"
echo "  replay  $REPLAY_DIR/replay_summary.csv"
