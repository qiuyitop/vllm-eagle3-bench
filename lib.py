"""Shared helpers for week-1 mix / replay / summarize scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_prompt(row: dict[str, Any]) -> str:
    for key in ("prompt", "text", "input"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"]
    raise ValueError(f"cannot find prompt in keys={list(row)}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
