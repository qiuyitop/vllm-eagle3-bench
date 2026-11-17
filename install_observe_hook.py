#!/usr/bin/env python3
"""Patch vLLM 0.27.x SpecDecodingLogging.observe to append one json line per decode step.

This version has no stats_reporting_level / per_step_accepted API.
The hook writes /root/autodl-tmp/datasets/week1_results/observe.jsonl
Restart vLLM after installing. Use C=1 when collecting logs for replay.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "EAGLE3_OBSERVE_HOOK"
INSERT = """
        # EAGLE3_OBSERVE_HOOK
        try:
            import json, time, os
            from pathlib import Path as _P
            _p = _P(os.environ.get(
                "EAGLE3_STEP_LOG",
                "/root/autodl-tmp/datasets/week1_results/observe.jsonl",
            ))
            _p.parent.mkdir(parents=True, exist_ok=True)
            _d = {"ts": time.time()}
            for _name in (
                "num_drafts",
                "num_draft_tokens",
                "num_accepted_tokens",
                "num_accepted_tokens_per_pos",
                "num_spec_tokens",
            ):
                if hasattr(spec_decoding_stats, _name):
                    _v = getattr(spec_decoding_stats, _name)
                    if hasattr(_v, "tolist"):
                        _v = _v.tolist()
                    elif isinstance(_v, tuple):
                        _v = list(_v)
                    _d[_name] = _v
            with _p.open("a", encoding="utf-8") as _f:
                _f.write(json.dumps(_d) + "\\n")
        except Exception:
            pass
"""


def find_metrics() -> Path:
    import vllm.v1.spec_decode.metrics as metrics

    path = Path(metrics.__file__).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def install(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"already patched: {path}")
        return
    needle = None
    for line in text.splitlines():
        if "def observe(" in line and "spec_decoding_stats" in line:
            needle = line
            break
    if needle is None:
        raise RuntimeError(f"could not find observe(spec_decoding_stats) in {path}")
    text = text.replace(needle, needle + INSERT, 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")
    print("restart: ./serve_k.sh 3")
    print("then collect with C=1, log -> /root/autodl-tmp/datasets/week1_results/observe.jsonl")


def uninstall(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER not in text:
        print("not patched")
        return
    start = text.find("        # " + MARKER)
    if start < 0:
        raise RuntimeError("marker found but block cannot be removed cleanly")
    end = text.find("            pass\n", start)
    if end < 0:
        raise RuntimeError("cannot find end of hook block")
    end = end + len("            pass\n")
    path.write_text(text[:start] + text[end:], encoding="utf-8")
    print(f"removed hook from {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--undo", action="store_true")
    args = parser.parse_args()
    path = find_metrics()
    print("metrics.py:", path)
    if args.undo:
        uninstall(path)
    else:
        install(path)


if __name__ == "__main__":
    main()
