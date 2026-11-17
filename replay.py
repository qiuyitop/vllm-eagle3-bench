#!/usr/bin/env python3
"""Replay week-1 k rules on a k=3 step log.

accept_len is the number of accepted *draft* tokens in that step, so 0..k_used.
Generated tokens in a step are 1 (bonus / target token) + accepted drafts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import iter_jsonl, mean


def next_k_r1(k: int, accept_len: int, k_used: int, k_max: int = 3) -> int:
    if accept_len == k_used:
        return min(k + 2, k_max)
    return max(k - 1, 1)


def next_k_r2old(
    k: int,
    accept_len: int,
    k_used: int,
    recent: list[int],
    temperature: float,
    k_max: int = 3,
    hot_temp: float = 0.5,
    recent_window: int = 8,
    low_accept: float = 1.2,
) -> int:
    """Old rule: turn off on a weak window even at temperature 0, then stay off."""
    k = next_k_r1(k, accept_len, k_used, k_max=k_max)
    if temperature >= hot_temp:
        k = min(k, 2)
    if len(recent) >= recent_window and mean([float(x) for x in recent[-recent_window:]]) < low_accept:
        k = 0
    return k


def next_k_r2(
    k: int,
    accept_len: int,
    k_used: int,
    recent: list[int],
    temperature: float,
    k_max: int = 3,
    hot_temp: float = 0.5,
    recent_window: int = 8,
    low_accept: float = 1.2,
) -> int:
    """New rule: at temp=0 behave like R1 (k>=1). Off only when temperature is high."""
    k = next_k_r1(k, accept_len, k_used, k_max=k_max)
    if temperature < hot_temp:
        return max(k, 1)
    k = min(k, 2)
    if len(recent) >= recent_window and mean([float(x) for x in recent[-recent_window:]]) < low_accept:
        return 0
    return max(k, 1)


def apply_try(k_try: int, m: int) -> tuple[int, int, int]:
    """Return (tokens_this_step, wasted_drafts, leftover_unused)."""
    if k_try <= 0:
        return 1, 0, 0
    accepted = min(k_try, m)
    wasted = max(k_try - m, 0)
    leftover = max(m - k_try, 0)
    return 1 + accepted, wasted, leftover


def step_time_ms(k_try: int, t_ar: float, t_draft: float, t_verify: float) -> float:
    if k_try <= 0:
        return t_ar
    return k_try * t_draft + t_verify


def load_steps(path: Path) -> list[dict[str, Any]]:
    steps = list(iter_jsonl(path))
    if not steps:
        raise ValueError(f"{path} has no steps")
    required = {"req_id", "step", "k_used", "accept_len"}
    missing = required - set(steps[0])
    if missing:
        raise ValueError(f"step log missing fields {sorted(missing)}")
    return steps


def group_requests(steps: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        grouped[str(row["req_id"])].append(row)
    for req_id, items in grouped.items():
        items.sort(key=lambda r: int(r["step"]))
        grouped[req_id] = items
    return grouped


def replay_request(
    steps: list[dict[str, Any]],
    rule: str,
    t_ar: float,
    t_draft: float,
    t_verify: float,
    k_init: int = 3,
    k_max: int = 3,
) -> dict[str, Any]:
    k = k_init
    recent: list[int] = []
    tokens = 0
    wasted = 0
    leftover = 0
    time_ms = 0.0
    k_hist: list[int] = []
    temperature = float(steps[0].get("temperature", 0.0))

    for row in steps:
        m = int(row["accept_len"])
        k_used = int(row["k_used"])
        if m < 0 or m > k_used:
            raise ValueError(
                f"req={row.get('req_id')} step={row.get('step')}: "
                f"accept_len={m} must be in 0..k_used={k_used}"
            )
        if rule == "B3":
            k_try = k_used
        elif rule == "U":
            k_try = m
        elif rule in {"R1", "R2", "R2old"}:
            k_try = k
        else:
            raise ValueError(f"unknown rule {rule}")

        got, waste, left = apply_try(k_try, m)
        tokens += got
        wasted += waste
        leftover += left
        time_ms += step_time_ms(k_try, t_ar, t_draft, t_verify)
        k_hist.append(k_try)
        observed = min(m, k_try) if k_try > 0 else 0
        recent.append(observed)

        if rule == "R1":
            k = next_k_r1(k_try, observed, k_try, k_max)
        elif rule == "R2old":
            if k_try <= 0:
                k = 0
            else:
                k = next_k_r2old(k_try, observed, k_try, recent, temperature, k_max=k_max)
        elif rule == "R2":
            k = next_k_r2(k_try, observed, k_try, recent, temperature, k_max=k_max)

    return {
        "req_id": steps[0]["req_id"],
        "dataset": steps[0].get("dataset", ""),
        "temperature": temperature,
        "steps": len(steps),
        "tokens": tokens,
        "wasted": wasted,
        "leftover": leftover,
        "time_ms": time_ms,
        "mean_k": mean([float(x) for x in k_hist]),
        "off_steps": sum(1 for x in k_hist if x == 0),
    }


def summarize(rows: list[dict[str, Any]], rule: str) -> dict[str, Any]:
    tokens = sum(r["tokens"] for r in rows)
    time_ms = sum(r["time_ms"] for r in rows)
    wasted = sum(r["wasted"] for r in rows)
    leftover = sum(r["leftover"] for r in rows)
    tok_s = tokens / (time_ms / 1000.0) if time_ms else 0.0
    return {
        "rule": rule,
        "requests": len(rows),
        "steps": sum(r["steps"] for r in rows),
        "tokens": tokens,
        "wasted_drafts": wasted,
        "leftover_unused": leftover,
        "est_tok_s": round(tok_s, 2),
        "mean_k": round(mean([r["mean_k"] for r in rows]), 3),
        "off_steps": sum(r["off_steps"] for r in rows),
        "waste_per_step": round(wasted / max(sum(r["steps"] for r in rows), 1), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay k controllers on a step log")
    parser.add_argument("--steps", type=Path, required=True, help="k=3 steps.jsonl")
    parser.add_argument("--out-dir", type=Path, default=Path("replay_out"))
    parser.add_argument("--t-ar-ms", type=float, default=17.5, help="plain decode one-step ms")
    parser.add_argument("--t-draft-ms", type=float, default=2.1, help="one draft-token ms")
    parser.add_argument("--t-verify-ms", type=float, default=17.5, help="one target verify ms")
    parser.add_argument("--k-init", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument(
        "--rules",
        default="B3,R1,R2old,R2,U",
        help="B3 forever-k3, R1 +/-, R2old weak-window off, R2 hot-only off, U ideal",
    )
    args = parser.parse_args()

    steps = load_steps(args.steps)
    grouped = group_requests(steps)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for rule in [x.strip() for x in args.rules.split(",") if x.strip()]:
        per_req = [
            replay_request(
                req_steps,
                rule,
                args.t_ar_ms,
                args.t_draft_ms,
                args.t_verify_ms,
                k_init=args.k_init,
                k_max=args.k_max,
            )
            for req_steps in grouped.values()
        ]
        summary = summarize(per_req, rule)
        summary_rows.append(summary)
        req_path = args.out_dir / f"replay_{rule}_requests.jsonl"
        with req_path.open("w", encoding="utf-8") as f:
            for row in per_req:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"{rule:<6} tok/s={summary['est_tok_s']:<8}  "
            f"waste={summary['wasted_drafts']:<8}  "
            f"leftover={summary['leftover_unused']:<6}  "
            f"mean_k={summary['mean_k']:<5}  off_steps={summary['off_steps']}"
        )

    csv_path = args.out_dir / "replay_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.out_dir / "replay_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
