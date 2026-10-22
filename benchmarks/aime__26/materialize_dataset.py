#!/usr/bin/env python3
"""Materialize pinned MathArena/aime_2026 data for direct evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
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

BENCHMARK = "aime__26"
DATASET = "MathArena/aime_2026"
REVISION = "d2de22f3c656b4f56cf8981212186377d1e23bc3"


def _rows_url() -> str:
    return "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": 0,
            "length": 100,
            "revision": REVISION,
        }
    )


def _fetch(timeout: float) -> Any:
    try:
        request = urllib.request.Request(
            _rows_url(),
            headers={"User-Agent": "vibetrainbench-aime-materializer"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise MaterializeError(f"could not fetch dataset rows: {exc}") from exc


def materialize(payload: Any, output: Path, overwrite: bool, source: str) -> dict:
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or len(rows) != 30:
        raise MaterializeError("expected a 30-row payload")
    normalized = []
    seen = set()
    for index, item in enumerate(rows):
        row = item.get("row") if isinstance(item, dict) and isinstance(item.get("row"), dict) else item
        if not isinstance(row, dict):
            raise MaterializeError(f"row {index} is invalid")
        number = row.get("problem_idx")
        answer = row.get("answer")
        problem = row.get("problem")
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number in seen
            or not isinstance(answer, int)
            or isinstance(answer, bool)
            or not 0 <= answer <= 999
            or not isinstance(problem, str)
            or not problem.strip()
        ):
            raise MaterializeError(f"row {index} is invalid")
        seen.add(number)
        normalized.append(
            {"problem_idx": number, "answer": answer, "problem": problem.strip()}
        )
    normalized.sort(key=lambda row: row["problem_idx"])

    def populate(stage: Path) -> dict:
        tasks = []
        for row in normalized:
            task_id = f"aime2026_{row['problem_idx']:02d}"
            relative = Path("tasks") / task_id
            write(stage / relative / "instruction.md", row["problem"] + "\n")
            write(
                stage / relative / "answer.json",
                {
                    "answer": row["answer"],
                    "problem_idx": row["problem_idx"],
                    "source": DATASET,
                },
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "path": str(relative),
                    "problem_idx": row["problem_idx"],
                }
            )
        jsonl = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in normalized
        )
        digest = hashlib.sha256(jsonl.encode("utf-8")).hexdigest()
        write(stage / "data.jsonl", jsonl)
        write(
            stage / "manifest.json",
            {
                "tasks": tasks,
                "dataset": DATASET,
                "config": "default",
                "split": "train",
                "revision": REVISION,
                "dataset_total": len(tasks),
            },
        )
        manifest = {
            "dataset": DATASET,
            "revision": REVISION,
            "source": source,
            "license": "cc-by-nc-sa-4.0",
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
        source = str(Path(args.input_json).resolve()) if args.input_json else _rows_url()
        payload = read_json(Path(args.input_json)) if args.input_json else _fetch(args.timeout_seconds)
        result = materialize(payload, Path(args.output_dir), args.overwrite, source)
    except MaterializeError as exc:
        print(f"AIME 2026 materialization failed: {exc}", file=sys.stderr)
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
