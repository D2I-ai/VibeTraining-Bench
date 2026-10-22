#!/usr/bin/env python3
"""Materialize pinned opencompass/AIME2025 data for direct evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _aime_materialize import (  # noqa: E402
    MaterializeError,
    materialize_atomically,
    output_default,
    read_json,
    write,
)

BENCHMARK = "aime__aime"
DATASET = "opencompass/AIME2025"
REVISION = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
ANSWER = re.compile(r"^\s*\$?\s*([+-]?\d+)\s*(?:\^\\?circ|°)?\s*\$?\s*$")
SOURCES = (("I", "aime2025-I.jsonl"), ("II", "aime2025-II.jsonl"))


def _answer(value: Any) -> int:
    if isinstance(value, bool):
        raise MaterializeError("boolean answer is invalid")
    if isinstance(value, int):
        answer = value
    elif isinstance(value, str) and ANSWER.match(value):
        match = ANSWER.match(value)
        assert match is not None
        answer = int(match.group(1))
    else:
        raise MaterializeError(f"invalid answer: {value!r}")
    if not 0 <= answer <= 999:
        raise MaterializeError(f"answer outside AIME range: {answer}")
    return answer


def _fetch(timeout: float) -> list[dict]:
    rows = []
    for exam, filename in SOURCES:
        url = f"https://huggingface.co/datasets/{DATASET}/raw/{REVISION}/{filename}"
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "vibetrainbench-aime-materializer"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                lines = response.read().decode("utf-8").splitlines()
        except OSError as exc:
            raise MaterializeError(f"could not fetch {url}: {exc}") from exc
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise MaterializeError(f"invalid JSON in {filename}:{number}") from exc
            if not isinstance(row, dict):
                raise MaterializeError(f"invalid row in {filename}:{number}")
            rows.append({**row, "exam": exam, "problem_number": number})
    return rows


def _input(path: Path) -> list[dict]:
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = next(
            (
                payload[key]
                for key in ("rows", "data", "items", "samples")
                if isinstance(payload.get(key), list)
            ),
            payload,
        )
    if not isinstance(payload, list):
        raise MaterializeError("offline input must be a list or contain a list")
    return [
        item["row"]
        if isinstance(item, dict) and isinstance(item.get("row"), dict)
        else item
        for item in payload
    ]


def materialize(rows: list[dict], output: Path, overwrite: bool, source: str) -> dict:
    if len(rows) != 30:
        raise MaterializeError(f"expected 30 rows, got {len(rows)}")
    normal = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MaterializeError(f"row {index} is not an object")
        exam = str(row.get("exam") or ("I" if index < 15 else "II"))
        exam = exam.upper().replace("AIME2025-", "")
        number = row.get("problem_number") or (index + 1 if exam == "I" else index - 14)
        question = row.get("question") or row.get("problem")
        if (
            exam not in {"I", "II"}
            or not isinstance(number, int)
            or isinstance(number, bool)
            or not 1 <= number <= 15
            or not isinstance(question, str)
            or not question.strip()
            or (exam, number) in seen
        ):
            raise MaterializeError(f"invalid row {index}")
        seen.add((exam, number))
        normal.append(
            {
                "exam": exam,
                "problem_number": number,
                "question": question.strip(),
                "answer": _answer(row.get("answer")),
                "raw_answer": row.get("answer"),
            }
        )
    normal.sort(key=lambda row: (row["exam"], row["problem_number"]))

    def populate(stage: Path) -> dict:
        tasks = []
        for row in normal:
            task_id = f"aime2025_{row['exam'].lower()}_{row['problem_number']:02d}"
            relative = Path("tasks") / task_id
            write(stage / relative / "instruction.md", row["question"] + "\n")
            write(
                stage / relative / "answer.json",
                {
                    "answer": row["answer"],
                    "raw_answer": row["raw_answer"],
                    "exam": row["exam"],
                    "problem_number": row["problem_number"],
                    "source": DATASET,
                },
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "path": str(relative),
                    "exam": row["exam"],
                    "problem_number": row["problem_number"],
                }
            )
        jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in normal)
        digest = hashlib.sha256(jsonl.encode("utf-8")).hexdigest()
        write(stage / "data.jsonl", jsonl)
        write(
            stage / "manifest.json",
            {
                "tasks": tasks,
                "dataset": DATASET,
                "configs": ["AIME2025-I", "AIME2025-II"],
                "split": "test",
                "revision": REVISION,
                "dataset_total": len(tasks),
            },
        )
        manifest = {
            "dataset": DATASET,
            "revision": REVISION,
            "source": source,
            "license": "mit",
            "dataset_total": len(tasks),
            "data_jsonl_sha256": digest,
            "materialized_at": datetime.now(timezone.utc).isoformat(),
        }
        write(stage / "source_manifest.json", manifest)
        return manifest

    return materialize_atomically(output, overwrite, populate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(output_default(Path(__file__).resolve().parents[2], BENCHMARK)),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--input-json")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    try:
        source = str(Path(args.input_json).resolve()) if args.input_json else DATASET
        rows = _input(Path(args.input_json)) if args.input_json else _fetch(args.timeout_seconds)
        result = materialize(rows, Path(args.output_dir), args.overwrite, source)
    except MaterializeError as exc:
        print(f"AIME 2025 materialization failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "dataset_total": result["dataset_total"],
                "data_jsonl_sha256": result["data_jsonl_sha256"],
                "revision": REVISION,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
