#!/usr/bin/env python3
"""Convert Nemotron-SFT-Math-v4 COT rows to Qwen3.5/slime SFT JSONL."""

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


def build_record(row, source_file, index):
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing messages")
    metadata = {
        "trajectory_id": row["uuid"],
        "source_file": source_file,
        "index": index,
        "source_dataset": "nvidia/Nemotron-SFT-Math-v4",
        "subset": row["subset"],
        "expected_answer": row["expected_answer"],
        "source": row["source"],
        "license": row["license"],
    }
    for key in ("url", "user_url", "username", "dataset", "used_in"):
        if row.get(key) is not None:
            metadata[key] = row[key]
    return {"messages": append_reasoning_instruction(messages), "metadata": metadata}


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
    with partial_output.open("w", encoding="utf-8") as output:
        for source_path in source_files:
            source_name = Path(source_path).name
            for batch in pq.ParquetFile(source_path).iter_batches(batch_size=4096):
                for row in batch.to_pylist():
                    scanned += 1
                    if row.get("subset") != "cot":
                        continue
                    output.write(json.dumps(build_record(row, source_name, kept), ensure_ascii=False) + "\n")
                    kept += 1
                    if kept % 10_000 == 0:
                        print(f"kept={kept} scanned={scanned}", flush=True)
    os.replace(partial_output, args.output)
    print(f"scanned={scanned}")
    print(f"kept={kept}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
