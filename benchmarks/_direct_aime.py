#!/usr/bin/env python3
"""Shared direct evaluator for locally materialized AIME datasets."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

BOXED_INTEGER = re.compile(r"\\boxed\s*\{\s*([+-]?\d+)\s*\}")
INTEGER = re.compile(r"(?<![\w.])[+-]?\d+(?![\w.])")


class EvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    answer: int


@dataclass(frozen=True)
class Completion:
    task_id: str
    sample_index: int
    seed: int
    response: str
    parsed_answer: Optional[int]
    parse_method: str
    correct: bool
    elapsed_seconds: float


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _config(benchmark_dir: Path) -> dict[str, Any]:
    try:
        value = _read_json(benchmark_dir / "benchmark.json").get("direct_eval")
    except EvaluationError:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _default_dataset_dir(repo: Path, benchmark: str) -> Path:
    vtb_root = Path(os.environ.get("VTB_ROOT") or (repo / "vtb-env")).expanduser()
    hub_root = Path(
        os.environ.get("VTB_HUB_TASK_SPECS")
        or os.environ.get("VTB_HARBOR_DATASET_DIR")
        or (vtb_root / "hub_task_specs")
    ).expanduser()
    return hub_root / benchmark


def _manifest_tasks(dataset_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(dataset_dir / "manifest.json")
    if isinstance(payload, dict):
        payload = payload.get("tasks") or payload.get("items") or payload.get("samples")
    if not isinstance(payload, list):
        raise EvaluationError(f"unsupported manifest format in {dataset_dir / 'manifest.json'}")
    return payload


def load_tasks(
    dataset_dir: Path,
    limit: int,
    only: Optional[list[str]],
) -> tuple[list[Task], int]:
    entries = _manifest_tasks(dataset_dir)
    total = len(entries)
    if total != 30:
        raise EvaluationError(f"AIME package must contain 30 tasks, found {total}")

    wanted: Optional[set[str]] = None
    if only:
        cleaned = [item.strip() for item in only if item.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise EvaluationError("--only contains duplicate task ids")
        wanted = set(cleaned)

    all_tasks: list[Task] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise EvaluationError(f"manifest entry {index} is not an object")
        task_id = entry.get("task_id") or entry.get("id") or entry.get("name")
        relative = entry.get("path")
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise EvaluationError(f"manifest entry {index} has invalid or duplicate task id")
        if not isinstance(relative, str) or not relative:
            relative = str(Path("tasks") / task_id)
        seen.add(task_id)
        task_dir = dataset_dir / relative
        try:
            question = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise EvaluationError(f"cannot read question for {task_id}: {exc}") from exc
        answer_payload = _read_json(task_dir / "answer.json")
        answer = answer_payload.get("answer") if isinstance(answer_payload, dict) else None
        if (
            not question
            or isinstance(answer, bool)
            or not isinstance(answer, int)
            or not 0 <= answer <= 999
        ):
            raise EvaluationError(f"invalid materialized task {task_id}")
        all_tasks.append(Task(task_id, question, answer))

    if wanted is not None:
        missing = sorted(wanted - seen)
        if missing:
            raise EvaluationError("--only includes unknown task ids: " + ", ".join(missing[:5]))
        tasks = [task for task in all_tasks if task.task_id in wanted]
    else:
        tasks = all_tasks if limit < 0 else all_tasks[:limit]
    return tasks, total


def parse_answer(text: str) -> tuple[Optional[int], str]:
    boxed = BOXED_INTEGER.findall(text)
    if boxed:
        return int(boxed[-1]), "boxed"
    integers = INTEGER.findall(text)
    return (int(integers[-1]), "last_integer") if integers else (None, "parse_error")


def _seed(benchmark: str, task_id: str, sample_index: int, base_seed: int) -> int:
    raw = f"{base_seed}:{benchmark}:{task_id}:{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _chat_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    return (base if base.endswith("/v1") else base + "/v1") + "/chat/completions"


def _response_text(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EvaluationError("chat completion has no choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    raise EvaluationError("chat completion content is not text")


def request_completion(args: argparse.Namespace, model: str, question: str, seed: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": question
                + "\n\nPlease reason step by step, and put your final answer within \\boxed{}",
            }
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }
    data = json.dumps(payload).encode("utf-8")
    failure: Optional[BaseException] = None
    for attempt in range(args.request_retries + 1):
        request = urllib.request.Request(
            _chat_url(args.api_base),
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {args.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
                return _response_text(json.loads(response.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            failure = EvaluationError(
                f"chat completion HTTP {exc.code}: "
                + exc.read().decode("utf-8", errors="replace")[:500]
            )
            retryable = exc.code == 429 or 500 <= exc.code < 600
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            failure, retryable = exc, True
        if not retryable or attempt == args.request_retries:
            break
        time.sleep(min(2**attempt, 8))
    raise EvaluationError(
        f"chat completion failed after {args.request_retries + 1} attempts: {failure}"
    )


def _complete(
    task: Task,
    sample_index: int,
    args: argparse.Namespace,
    benchmark: str,
    model: str,
) -> Completion:
    seed = _seed(benchmark, task.task_id, sample_index, args.seed)
    started = time.monotonic()
    response = request_completion(args, model, task.question, seed)
    parsed, method = parse_answer(response)
    return Completion(
        task.task_id,
        sample_index,
        seed,
        response,
        parsed,
        method,
        parsed == task.answer,
        round(time.monotonic() - started, 6),
    )


def _trajectory(
    path: Path,
    task_id: str,
    samples_per_task: int,
    completions: list[Completion],
) -> float:
    ordered = sorted(completions, key=lambda item: item.sample_index)
    if len(ordered) != samples_per_task:
        raise EvaluationError(f"task {task_id} has incomplete completion set")
    score = sum(item.correct for item in ordered) / samples_per_task
    _write_json(
        path / f"{task_id}.json",
        {
            "task_id": task_id,
            "samples_per_task": samples_per_task,
            "correct_completions": sum(item.correct for item in ordered),
            "avg_at_n": score,
            "completions": [
                {
                    "sample_index": item.sample_index,
                    "seed": item.seed,
                    "response": item.response,
                    "parsed_answer": item.parsed_answer,
                    "parse_method": item.parse_method,
                    "correct": item.correct,
                    "elapsed_seconds": item.elapsed_seconds,
                }
                for item in ordered
            ],
        },
    )
    return score


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise EvaluationError(f"invalid boolean value: {value!r}")


def _value(config: dict[str, Any], env: str, key: str, default: Any, cast):
    try:
        return cast(os.environ.get(env, str(config.get(key, default))))
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{env} has invalid value") from exc


def parse_args(
    benchmark: str,
    benchmark_dir: Path,
    dataset_env: str,
    argv: Optional[list[str]] = None,
) -> argparse.Namespace:
    config = _config(benchmark_dir)
    parser = argparse.ArgumentParser(description=f"Direct AIME evaluator for {benchmark}")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--json-output-file", required=True)
    parser.add_argument("--templates-dir", default=None)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get(
            dataset_env,
            str(_default_dataset_dir(benchmark_dir.parents[1], benchmark)),
        ),
    )
    parser.add_argument("--model", "--served-model-name", dest="model", default=None)
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=_value(
            config,
            "VTB_AIME_DIRECT_SAMPLES_PER_TASK",
            "samples_per_task",
            1,
            int,
        ),
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=_value(config, "VTB_AIME_DIRECT_CONCURRENCY", "n_concurrent", 8, int),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_value(config, "VTB_AIME_DIRECT_TEMPERATURE", "temperature", 1.0, float),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=_value(config, "VTB_AIME_DIRECT_TOP_P", "top_p", 1.0, float),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_value(config, "VTB_AIME_DIRECT_MAX_TOKENS", "max_tokens", 16384, int),
    )
    thinking = (
        config.get("chat_template_kwargs", {}).get("enable_thinking", False)
        if isinstance(config.get("chat_template_kwargs"), dict)
        else False
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=_bool(os.environ.get("VTB_AIME_DIRECT_ENABLE_THINKING", thinking)),
    )
    parser.add_argument("--disable-thinking", action="store_false", dest="enable_thinking")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_value(
            config,
            "VTB_AIME_DIRECT_TIMEOUT_SECONDS",
            "timeout_seconds",
            600.0,
            float,
        ),
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=_value(config, "VTB_AIME_DIRECT_REQUEST_RETRIES", "request_retries", 2, int),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_value(config, "VTB_AIME_DIRECT_SEED", "seed", 0, int),
    )
    args = parser.parse_args(argv)
    args.api_key = os.environ.get("VTB_API_KEY") or os.environ.get("OPENAI_API_KEY") or "inspectai"
    if args.limit < -1 or (args.only and args.limit >= 0):
        parser.error("--limit must be -1 or non-negative and cannot be combined with --only")
    if (
        args.samples_per_task < 1
        or args.n_concurrent < 1
        or args.max_tokens < 1
        or args.request_retries < 0
    ):
        parser.error("invalid samples, concurrency, token, or retry value")
    if args.timeout_seconds <= 0 or args.temperature < 0 or not 0 <= args.top_p <= 1:
        parser.error("invalid decoding parameter")
    return args


def main(
    benchmark: str,
    dataset: str,
    dataset_revision: str,
    dataset_env: str,
    argv: Optional[list[str]] = None,
) -> int:
    benchmark_dir = Path(__file__).resolve().parent / benchmark
    try:
        args = parse_args(benchmark, benchmark_dir, dataset_env, argv)
        tasks, total = load_tasks(
            Path(args.dataset_dir).expanduser().resolve(),
            args.limit,
            args.only,
        )
        model = args.model or Path(args.model_path).name
        completions: dict[str, list[Completion]] = {task.task_id: [] for task in tasks}
        trajectory_dir = (
            Path(args.work_dir).expanduser().resolve()
            / "backend_output"
            / "direct_trajectories"
        )
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_concurrent) as pool:
            futures = {
                pool.submit(_complete, task, index, args, benchmark, model): task.task_id
                for task in tasks
                for index in range(args.samples_per_task)
            }
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                completions[task_id].append(future.result())
                if len(completions[task_id]) == args.samples_per_task:
                    _trajectory(
                        trajectory_dir,
                        task_id,
                        args.samples_per_task,
                        completions[task_id],
                    )
        scores = {
            task.task_id: _trajectory(
                trajectory_dir,
                task.task_id,
                args.samples_per_task,
                completions[task.task_id],
            )
            for task in tasks
        }
        accuracy = sum(scores.values()) / len(scores) if scores else 0.0
        correct_completions = sum(
            item.correct
            for task in tasks
            for item in completions[task.task_id]
        )
        elapsed = round(time.monotonic() - started, 6)
        metadata = {
            "benchmark": benchmark,
            "dataset": dataset,
            "dataset_revision": dataset_revision,
            "runner": "direct-aime",
            "backend": "direct",
            "model": model,
            "api_base": args.api_base,
            "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
            "dataset_total": total,
            "task_count": len(tasks),
            "samples_per_task": args.samples_per_task,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "enable_thinking": args.enable_thinking,
            "n_concurrent": args.n_concurrent,
            "base_seed": args.seed,
            "elapsed_seconds": elapsed,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        work_dir = Path(args.work_dir).expanduser().resolve()
        _write_json(work_dir / "backend_output" / "run.json", metadata)
        metrics = {
            **metadata,
            "accuracy": accuracy,
            "mean_reward": accuracy,
            "solved": sum(score >= 1.0 for score in scores.values()),
            "graded": len(tasks),
            "attempted": len(tasks),
            "ungraded": 0,
            "correct_completions": correct_completions,
            "total_completions": len(tasks) * args.samples_per_task,
            "environment_type": "host",
            "agent": "none",
            "backend_returncode": 0,
        }
        out_json = Path(args.json_output_file).expanduser().resolve()
        _write_json(out_json, metrics)
        _write_json(
            out_json.with_suffix(".per_sample.json"),
            {
                "dataset_total": total,
                "score_mode": "avg_at_n",
                "sample_ids": [task.task_id for task in tasks],
                "correct": scores,
            },
        )
        print(json.dumps(metrics, sort_keys=True))
        return 0
    except EvaluationError as exc:
        print(f"direct AIME evaluation failed: {exc}", file=sys.stderr)
        return 2
