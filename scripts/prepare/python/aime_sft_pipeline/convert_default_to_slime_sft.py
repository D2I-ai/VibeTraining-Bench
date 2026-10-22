#!/usr/bin/env python3
"""Convert OpenR1-Math-220k default rows to Qwen3.5/slime SFT JSONL."""

import argparse
import glob
import json
import os
from pathlib import Path

REASONING_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def append_reasoning_instruction(messages):
    converted = []
    for message in messages:
        if message.get("role") == "user" and not message.get("content", "").rstrip().endswith(REASONING_INSTRUCTION):
            message = dict(message)
            message["content"] = f"{message['content'].rstrip()}\n{REASONING_INSTRUCTION}"
        converted.append(message)
    return converted


def split_thinking_content(content):
    text = content.lstrip()
    if not text.startswith("<think>"):
        raise ValueError("assistant content does not start with <think>")
    payload = text[len("<think>") :]
    if "</think>" not in payload:
        return payload.lstrip("\n"), ""
    reasoning, answer = payload.rsplit("</think>", 1)
    reasoning = reasoning.lstrip("\n")
    answer = answer.lstrip("\n")
    if answer.startswith("<think>"):
        reasoning = f"{reasoning}\n\n{answer[len('<think>') :].lstrip(chr(10))}".strip()
        answer = ""
    return reasoning, answer


def convert_messages(messages):
    converted = []
    for message in messages:
        if message.get("role") != "assistant":
            converted.append(message)
            continue
        reasoning_content, content = split_thinking_content(message.get("content") or "")
        converted.append({"role": "assistant", "content": content, "reasoning_content": reasoning_content})
    return converted


def build_record(row, source_file, index):
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing messages")
    return {
        "messages": append_reasoning_instruction(convert_messages(messages)),
        "metadata": {
            "trajectory_id": row["uuid"],
            "source_file": source_file,
            "index": index,
            "source_dataset": "open-r1/OpenR1-Math-220k",
            "config": "default",
            "answer": row.get("answer"),
            "source": row.get("source"),
            "problem_type": row.get("problem_type"),
            "question_type": row.get("question_type"),
            "correctness_count": row.get("correctness_count"),
        },
    }


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_files = sorted(glob.glob(str(args.input_dir / "train-*.parquet")))
    if not source_files:
        raise SystemExit(f"No Parquet shards found under {args.input_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = args.output.with_suffix(args.output.suffix + ".partial")
    partial_output.unlink(missing_ok=True)
    scanned = kept = 0
    columns = ["uuid", "answer", "source", "problem_type", "question_type", "correctness_count", "messages"]
    with partial_output.open("w", encoding="utf-8") as output:
        for source_path in source_files:
            source_name = Path(source_path).name
            for batch in pq.ParquetFile(source_path).iter_batches(columns=columns, batch_size=4096):
                for row in batch.to_pylist():
                    output.write(json.dumps(build_record(row, source_name, kept), ensure_ascii=False) + "\n")
                    scanned += 1
                    kept += 1
                    if kept % 10_000 == 0:
                        print(f"kept={kept}", flush=True)
    os.replace(partial_output, args.output)
    print(f"scanned={scanned}")
    print(f"kept={kept}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
