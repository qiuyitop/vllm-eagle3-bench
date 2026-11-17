#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一数据集格式转换脚本（输出 vLLM 所需的 jsonl）

用法示例：
    python convert_datasets.py alpaca
    python convert_datasets.py alpaca gsm8k humaneval
    python convert_datasets.py mtbench_chat
    python convert_datasets.py all
"""

import argparse
import json
import os

from datasets import load_from_disk

BASE_DIR = "/root/autodl-tmp/datasets"
TOKENIZER_PATH = (
    "/root/autodl-tmp/models/"
    "LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master"
)


# ---------------------------------------------------------------------------
# 各数据集转换函数
# ---------------------------------------------------------------------------
def convert_alpaca():
    """Alpaca: raw/ -> alpaca_vllm.jsonl"""
    input_path = os.path.join(BASE_DIR, "alpaca/raw")
    output_path = os.path.join(BASE_DIR, "alpaca/alpaca_vllm.jsonl")

    ds = load_from_disk(input_path)
    print(f"[alpaca] Loaded {len(ds)} prompts")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in ds:
            if item.get("input"):
                prompt = f"{item['instruction']}\n{item['input']}"
            else:
                prompt = item["instruction"]
            fout.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            count += 1

    print(f"[alpaca] Converted {count} prompts -> {output_path}")


def convert_gsm8k():
    """GSM8K: raw/ -> gsm8k_vllm.jsonl"""
    input_path = os.path.join(BASE_DIR, "gsm8k/raw")
    output_path = os.path.join(BASE_DIR, "gsm8k/gsm8k_vllm.jsonl")

    ds = load_from_disk(input_path)
    print(f"[gsm8k] Loaded {len(ds)} problems")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in ds:
            prompt = (
                f"Solve the following math problem step by step:\n"
                f"{item['question']}\nAnswer:"
            )
            fout.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")

    print(f"[gsm8k] Converted -> {output_path}")
    print(f"[gsm8k] Total lines: {len(ds)}")


def convert_humaneval():
    """HumanEval: HumanEval.jsonl -> humaneval_vllm.jsonl"""
    input_path = os.path.join(BASE_DIR, "humaneval/HumanEval.jsonl")
    output_path = os.path.join(BASE_DIR, "humaneval/humaneval_vllm.jsonl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            prompt = data["prompt"]
            if not prompt.endswith("\n"):
                prompt += "\n"
            fout.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            count += 1

    print(f"[humaneval] Converted {count} prompts -> {output_path}")


def convert_mtbench_chat():
    """
    MT-Bench 两阶段转换（合并原 convert_mtbench_chat.py + convert_data.py）：
      step1: mt_bench_prompts.jsonl -> mtbench_chat.jsonl   （把每轮拆成 messages）
      step2: mtbench_chat.jsonl     -> mtbench_chat_vllm.jsonl （套 chat template）
    """
    from transformers import AutoTokenizer  # 延迟导入：只有这个数据集才需要

    base = os.path.join(BASE_DIR, "mt_bench")
    input_path = os.path.join(base, "mt_bench_prompts.jsonl")
    mid_path = os.path.join(base, "mtbench_chat.jsonl")
    output_path = os.path.join(base, "mtbench_chat_vllm.jsonl")
    os.makedirs(base, exist_ok=True)

    # ---------- step 1: 拆轮次 ----------
    total = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(mid_path, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            for turn_text in data["prompt"]:
                item = {"messages": [{"role": "user", "content": turn_text}]}
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                total += 1
    print(f"[mtbench_chat] step1: {total} turns -> {mid_path}")

    # ---------- step 2: apply_chat_template ----------
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    count = 0
    with open(mid_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                data["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            fout.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            count += 1
    print(f"[mtbench_chat] step2: {count} prompts -> {output_path}")


# ---------------------------------------------------------------------------
# 注册表 & CLI
# ---------------------------------------------------------------------------
CONVERTERS = {
    "alpaca": convert_alpaca,
    "gsm8k": convert_gsm8k,
    "humaneval": convert_humaneval,
    "mtbench_chat": convert_mtbench_chat,
    "mt_bench": convert_mtbench_chat,  # 别名，两种写法都能用
}

ALL_DATASETS = ["alpaca", "gsm8k", "humaneval", "mtbench_chat"]


def main():
    parser = argparse.ArgumentParser(
        description="将数据集转换为 vLLM 推理所需的 jsonl 格式",
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help=f"数据集名称，可选: {', '.join(ALL_DATASETS)}；传 'all' 表示全部转换",
    )
    args = parser.parse_args()

    # 'all' 展开为全部数据集
    if "all" in args.datasets:
        targets = list(ALL_DATASETS)
    else:
        targets = args.datasets

    # 去重且保持顺序
    seen, uniq = set(), []
    for name in targets:
        if name not in seen:
            seen.add(name)
            uniq.append(name)
    targets = uniq

    unknown = [t for t in targets if t not in CONVERTERS]
    if unknown:
        parser.error(
            f"未知数据集: {unknown}；可选: {', '.join(ALL_DATASETS)} 或 all"
        )

    print(f"将转换 {len(targets)} 个数据集: {targets}\n")
    for name in targets:
        print(f"===== 开始: {name} =====")
        CONVERTERS[name]()
        print(f"===== 完成: {name} =====\n")
    print("全部完成 ✅")


if __name__ == "__main__":
    main()