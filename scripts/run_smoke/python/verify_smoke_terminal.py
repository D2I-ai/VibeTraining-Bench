#!/usr/bin/env python3
"""Verify one completed Terminal smoke validation and final evaluation."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BENCHMARK = "terminal-bench__terminal-bench-2-1"
VAL_BENCHMARK = "openthoughts__openthoughts-tblite-val50"
SUBMIT = re.compile(
    r"\[submit\] queued job=(\w+).*? role=(agent|official) "
    r"task=([^ ]+) eval_task=([^ ]+) mode=(val|final) limit=(\d+)(?: |$)"
)


class VerificationError(ValueError):
    pass


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected a JSON object: {path}")
    return value


def verify(state: Path, result_dir: Path) -> dict[str, float | None]:
    try:
        log = (state / "service.log").read_text(encoding="utf-8")
    except OSError as exc:
        raise VerificationError(f"cannot read service log: {exc}") from exc
    submissions = SUBMIT.findall(log)
    if len(submissions) != 2:
        raise VerificationError(f"expected exactly two smoke submissions, found {submissions}")
    expected = {
        "agent": (BENCHMARK, VAL_BENCHMARK, "val", "1"),
        "official": (BENCHMARK, BENCHMARK, "final", "1"),
    }
    scores: dict[str, float | None] = {}
    seen_roles = set()
    for job_id, role, task, eval_task, mode, limit in submissions:
        if role in seen_roles or (task, eval_task, mode, limit) != expected.get(role):
            raise VerificationError(
                f"unexpected {role} submission: {(task, eval_task, mode, limit)}"
            )
        seen_roles.add(role)
        finish = re.search(
            rf"\[worker\] finish job={re.escape(job_id)} state=done score=(\S+)", log
        )
        if finish is None:
            raise VerificationError(f"job {job_id} did not finish successfully")
        archives = list((state / "archive").glob(f"**/{mode}_{job_id}"))
        if len(archives) != 1:
            raise VerificationError(f"job {job_id} archive count is {len(archives)}")
        metrics_paths = [
            path
            for path in archives[0].glob("metrics_p*_a*.json")
            if not path.name.endswith(".per_sample.json")
        ]
        commands = list(archives[0].glob("**/backend_command.txt"))
        if len(metrics_paths) != 1 or len(commands) != 1:
            raise VerificationError(f"job {job_id} archive is incomplete")
        metrics = _load_json(metrics_paths[0])
        counts = (
            metrics.get("graded"),
            metrics.get("attempted"),
            metrics.get("ungraded"),
        )
        if metrics.get("backend_returncode") != 0 or counts != (1, 1, 0):
            raise VerificationError(
                f"job {job_id} metrics are not one clean verdict: {metrics}"
            )
        dataset = (
            "openthoughts/openthoughts-tblite"
            if role == "agent"
            else "terminal-bench/terminal-bench-2-1"
        )
        command = commands[0].read_text(encoding="utf-8")
        if f"--dataset {dataset}" not in command:
            raise VerificationError(f"job {job_id} used the wrong dataset command")
        scores[role] = metrics.get("accuracy")

    final_metrics = _load_json(result_dir / "metrics.json")
    if (
        final_metrics.get("graded") != 1
        or final_metrics.get("attempted") != 1
        or final_metrics.get("ungraded") != 0
        or final_metrics.get("backend_returncode") != 0
    ):
        raise VerificationError(f"invalid official metrics: {final_metrics}")
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    try:
        scores = verify(args.state.expanduser().resolve(), args.result_dir.expanduser().resolve())
    except VerificationError as exc:
        parser.error(str(exc))
    print(
        "TERMINAL_EXPERIMENT_SMOKE_OK",
        "val_score=", scores.get("agent"),
        "final_score=", scores.get("official"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
