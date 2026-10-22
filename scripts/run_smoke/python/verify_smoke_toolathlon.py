#!/usr/bin/env python3
"""Verify one ToolGym validation and one Toolathlon Verified final smoke."""
import argparse, json, re
from pathlib import Path

TARGET = "eigent-ai__toolathlon_gym"; VAL = TARGET; FINAL = "hkust-nlp__toolathlon_verified"
SUBMIT = re.compile(
    r"\[submit\] queued job=(\w+).*? role=(agent|official) task=([^ ]+) "
    r"eval_task=([^ ]+) mode=(val|final) limit=(-?\d+) position=\d+"
    r"(?: samples=(\d+))?"
)

def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def verify(state, result_dir):
    state, result_dir = Path(state), Path(result_dir); log = (state / "service.log").read_text(encoding="utf-8")
    rows = SUBMIT.findall(log)
    if len(rows) != 2: raise ValueError(f"expected two Toolathlon submissions, found {rows}")
    expected = {"agent": (VAL, "val"), "official": (FINAL, "final")}; scores = {}
    for job, role, task, eval_task, mode, limit, samples in rows:
        expected_limit = "-1" if role == "agent" else "1"
        expected_samples = "1" if role == "agent" else ""
        if task != TARGET or (eval_task, mode) != expected.get(role) or limit != expected_limit or samples != expected_samples:
            raise ValueError(f"unexpected submission: {(job, role, task, eval_task, mode, limit, samples)}")
        if not re.search(rf"\[worker\] finish job={job} state=done score=\S+", log): raise ValueError(f"job {job} did not finish")
        archives = list((state / "archive").glob(f"**/{mode}_{job}")); files = [p for p in archives[0].glob("metrics_p*_a*.json") if not p.name.endswith(".per_sample.json")] if len(archives) == 1 else []
        if len(files) != 1: raise ValueError(f"job {job} archive incomplete")
        metrics = load(files[0]); counts = (metrics.get("graded"), metrics.get("attempted"), metrics.get("ungraded"))
        if counts != (1, 1, 0): raise ValueError(f"job {job} is not one clean verdict")
        scores[role] = metrics.get("accuracy")
    final = load(result_dir / "metrics.json")
    if (final.get("graded"), final.get("attempted"), final.get("ungraded")) != (1, 1, 0): raise ValueError("official Toolathlon metrics invalid")
    return scores

def main():
    p = argparse.ArgumentParser(); p.add_argument("state", type=Path); p.add_argument("result_dir", type=Path); a = p.parse_args()
    try: scores = verify(a.state, a.result_dir)
    except ValueError as exc: p.error(str(exc))
    print("TOOLATHLON_EXPERIMENT_SMOKE_OK", "val_score=", scores.get("agent"), "final_score=", scores.get("official"))
if __name__ == "__main__": main()
