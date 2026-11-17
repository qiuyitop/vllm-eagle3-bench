#!/usr/bin/env python3
"""Convert C=1 observe.jsonl into steps.jsonl.

vLLM 0.27 only logs batch-level spec stats. With max-concurrency=1,
each observe line is usually one request's one draft/verify step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import iter_jsonl, write_jsonl


def accept_len_of(row: dict) -> tuple[int, int] | None:
    drafts = int(row.get("num_drafts") or 0)
    if drafts != 1:
        return None
    accepted = int(row.get("num_accepted_tokens") or 0)
    k_used = int(row.get("num_draft_tokens") or row.get("num_spec_tokens") or 3)
    if k_used <= 0:
        k_used = 3
    accepted = min(max(accepted, 0), k_used)
    return accepted, k_used


def split_by_output_lens(steps: list[dict], output_lens: list[int]) -> list[dict]:
    out: list[dict] = []
    i = 0
    for req_i, out_len in enumerate(output_lens):
        got = 0
        step = 0
        req_id = f"req-{req_i:04d}"
        while i < len(steps) and got < out_len:
            row = dict(steps[i])
            row["req_id"] = req_id
            row["step"] = step
            got += 1 + int(row["accept_len"])
            out.append(row)
            i += 1
            step += 1
    if i < len(steps):
        print(f"warning: {len(steps) - i} observe steps unused after packing output_lens")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bench-json", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    raw_steps: list[dict] = []
    skipped = 0
    for i, row in enumerate(iter_jsonl(args.observe)):
        parsed = accept_len_of(row)
        if parsed is None:
            skipped += 1
            continue
        acc, k_used = parsed
        raw_steps.append(
            {
                "run_id": args.run_id,
                "req_id": f"req-{i:04d}",
                "dataset": args.dataset,
                "temperature": args.temperature,
                "concurrency": args.concurrency,
                "step": 0,
                "k_used": k_used,
                "accept_len": acc,
                "pos0": int(acc >= 1),
                "pos1": int(acc >= 2),
                "pos2": int(acc >= 3),
            }
        )
    if not raw_steps:
        raise ValueError(
            f"{args.observe} has no num_drafts==1 rows. "
            "Re-collect with --max-concurrency 1 after install_observe_hook.py."
        )
    print(f"usable steps={len(raw_steps)}  skipped_batch_rows={skipped}")

    output_lens = None
    if args.bench_json and args.bench_json.exists():
        data = json.loads(args.bench_json.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("output_lens"), list):
            output_lens = [int(x) for x in data["output_lens"]]
            args.temperature = float(data.get("temperature", args.temperature) or args.temperature)
            args.concurrency = int(data.get("max_concurrency", args.concurrency) or args.concurrency)

    if output_lens:
        rows = split_by_output_lens(raw_steps, output_lens)
        print(f"packed into {len(output_lens)} requests using output_lens")
    else:
        rows = raw_steps
        print("no output_lens; each observe line is a separate req_id (ok for replay of k, weaker for per-request plots)")

    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
