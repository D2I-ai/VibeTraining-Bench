#!/usr/bin/env python3
"""Verify bounded SWE validation and final smoke jobs."""
import argparse, json, re
from pathlib import Path

BENCHMARK = "swe-bench__swe-bench-verified"
SUBMIT = re.compile(r"\[submit\] queued job=(\w+).*? role=(agent|official) task=([^ ]+) eval_task=([^ ]+) mode=(val|final) limit=(\d+)(?: |$)")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(state, result_dir):
    state, result_dir = Path(state), Path(result_dir)
    log = (state / "service.log").read_text(encoding="utf-8")
    rows = SUBMIT.findall(log)
    if len(rows) != 2:
        raise ValueError(f"expected two SWE submissions, found {rows}")
    scores = {}
    for job, role, task, eval_task, mode, limit in rows:
        if task != BENCHMARK or eval_task != BENCHMARK or limit != "1" or mode != ("val" if role == "agent" else "final"):
            raise ValueError(f"unexpected SWE submission: {(job, role, task, eval_task, mode, limit)}")
        if not re.search(rf"\[worker\] finish job={job} state=done score=\S+", log):
            raise ValueError(f"job {job} did not finish")
        archives = list((state / "archive").glob(f"**/{mode}_{job}"))
        metrics_files = [p for p in archives[0].glob("metrics_p*_a*.json") if not p.name.endswith(".per_sample.json")] if len(archives) == 1 else []
        if len(metrics_files) != 1:
            raise ValueError(f"job {job} archive is incomplete")
        metrics = load(metrics_files[0])
        if metrics.get("backend_returncode") != 0 or (metrics.get("graded"), metrics.get("attempted"), metrics.get("ungraded")) != (1, 1, 0):
            raise ValueError(f"job {job} is not one clean verdict")
        scores[role] = metrics.get("accuracy")
    final = load(result_dir / "metrics.json")
    if final.get("backend_returncode") != 0 or final.get("graded") != 1:
        raise ValueError("official SWE metrics are invalid")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path); parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    try:
        scores = verify(args.state, args.result_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print("SWE_EXPERIMENT_SMOKE_OK", "val_score=", scores.get("agent"), "final_score=", scores.get("official"))


if __name__ == "__main__":
    main()
