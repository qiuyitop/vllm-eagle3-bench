#!/usr/bin/env bash
# Re-run replay on existing C=1 step logs. Does not start vLLM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULT="${RESULT_DIR:-/root/autodl-tmp/datasets/week1_results}"
source /root/autodl-tmp/venv/bin/activate

for ds in alpaca mtbench humaneval gsm8k; do
  steps="$RESULT/steps_${ds}_k3_t0_c1.jsonl"
  out="$RESULT/replay_${ds}_k3_t0_c1_r2fix"
  if [[ ! -f "$steps" ]]; then
    echo "missing $steps"
    exit 1
  fi
  echo "===== $ds ====="
  python3 "$ROOT/replay.py" --steps "$steps" --out-dir "$out"
  echo
done
