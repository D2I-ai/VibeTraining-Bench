#!/usr/bin/env python3
"""Verify bounded Tau3 validation and final smoke jobs."""
import argparse
import json
import re
from pathlib import Path

BENCHMARK = "sierra-research__tau3-bench"
SUBMIT = re.compile(
    r"\[submit\] queued job=(\w+).*? role=(agent|official) task=([^ ]+) "
    r"eval_task=([^ ]+) mode=(val|final) limit=(\d+)(?: |$)"
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(state, result_dir):
    state, result_dir = Path(state), Path(result_dir)
    log = (state / "service.log").read_text(encoding="utf-8")
    rows = SUBMIT.findall(log)
    if len(rows) != 2:
        raise ValueError(f"expected two Tau3 submissions, found {rows}")
    scores = {}
    for job, role, task, eval_task, mode, limit in rows:
        expected_mode = "val" if role == "agent" else "final"
        if task != BENCHMARK or eval_task != BENCHMARK or mode != expected_mode or limit != "1":
            raise ValueError(f"unexpected Tau3 submission: {(job, role, task, eval_task, mode, limit)}")
        if not re.search(rf"\[worker\] finish job={job} state=done score=\S+", log):
            raise ValueError(f"job {job} did not finish")
        archives = list((state / "archive").glob(f"**/{mode}_{job}"))
        metrics_files = [
            path for path in archives[0].glob("metrics_p*_a*.json")
            if not path.name.endswith(".per_sample.json")
        ] if len(archives) == 1 else []
        if len(metrics_files) != 1:
            raise ValueError(f"job {job} archive is incomplete")
        metrics = load(metrics_files[0])
        verdict = metrics.get("graded"), metrics.get("attempted"), metrics.get("ungraded")
        if metrics.get("backend_returncode") != 0 or verdict != (1, 1, 0):
            raise ValueError(f"job {job} is not one clean verdict")
        scores[role] = metrics.get("accuracy")
    final = load(result_dir / "metrics.json")
    verdict = final.get("graded"), final.get("attempted"), final.get("ungraded")
    if final.get("backend_returncode") != 0 or verdict != (1, 1, 0):
        raise ValueError("official Tau3 metrics are invalid")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    try:
        scores = verify(args.state, args.result_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("TAU3_EXPERIMENT_SMOKE_OK", "val_score=", scores.get("agent"),
          "final_score=", scores.get("official"))


if __name__ == "__main__":
    main()
