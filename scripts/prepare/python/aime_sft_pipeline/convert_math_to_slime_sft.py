#!/usr/bin/env python3
"""Convert OpenThoughts3 math rows to Qwen3.5/slime SFT JSONL."""

import argparse
import glob
import json
import os
from pathlib import Path

ROLE_MAP = {"human": "user", "gpt": "assistant"}
REASONING_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def append_reasoning_instruction(content):
    """Append the shared instruction to a user prompt without duplication."""
    if content.rstrip().endswith(REASONING_INSTRUCTION):
        return content
    return f"{content.rstrip()}\n{REASONING_INSTRUCTION}"


def split_thinking_content(content):
    """Separate one leading inline thinking block from an assistant reply."""
    text = content.lstrip()
    if not text.startswith("<think>"):
        return None, content
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


def convert_conversations(conversations):
    """Map OpenInstruct conversations to slime messages."""
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("missing conversations")
    messages = []
    for turn in conversations:
        source_role = turn.get("from")
        role = ROLE_MAP.get(source_role)
        if role is None:
            raise ValueError(f"unsupported conversation role: {source_role!r}")
        content = turn.get("value")
        if not isinstance(content, str):
            raise ValueError(f"non-string content for role {source_role!r}")
        if role == "assistant":
            reasoning_content, final_content = split_thinking_content(content)
            message = {"role": role, "content": final_content}
            if reasoning_content is not None:
                message["reasoning_content"] = reasoning_content
            messages.append(message)
        else:
            messages.append({"role": role, "content": append_reasoning_instruction(content)})
    return messages


def build_record(row, source_file, index):
    return {
        "messages": convert_conversations(row["conversations"]),
        "metadata": {
            "trajectory_id": f"{source_file}:{index}",
            "source_file": source_file,
            "index": index,
            "source_dataset": "open-thoughts/OpenThoughts3-1.2M",
            "domain": row["domain"],
            "source": row.get("source"),
            "difficulty": row.get("difficulty"),
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
    scanned = kept = skipped_non_math = 0
    with partial_output.open("w", encoding="utf-8") as output:
        for source_path in source_files:
            source_file = Path(source_path).name
            for batch in pq.ParquetFile(source_path).iter_batches(
                columns=["difficulty", "source", "domain", "conversations"], batch_size=4096
            ):
                for row in batch.to_pylist():
                    scanned += 1
                    if row.get("domain") != "math":
                        skipped_non_math += 1
                        continue
                    output.write(json.dumps(build_record(row, source_file, kept), ensure_ascii=False) + "\n")
                    kept += 1
                    if kept % 10_000 == 0:
                        print(f"kept={kept} scanned={scanned}", flush=True)
    os.replace(partial_output, args.output)
    print(f"scanned={scanned}")
    print(f"kept_math={kept}")
    print(f"skipped_non_math={skipped_non_math}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
