"""
Toolathlon-GYM evaluation service.

API contract:
  POST /evaluate            submit a model for evaluation (async job)
  GET  /jobs/{job_id}       job status + per-task results
  GET  /jobs                all jobs
  GET  /tasks               the task subset used for evaluation
  GET  /health              docker/image prerequisite check

Input  : model platform / name / api key / api url
Output : pass rate + per-task pass/fail details

Tasks inside one job run CONCURRENTLY (default 12 at a time): every task gets
its own postgres + docker network via runner.run_task_isolated, so nothing is
shared between tasks. Jobs themselves are still processed one at a time by the
single worker thread, which caps total container pressure on the host.
"""

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:  # package import (uvicorn eval.service.toolathlon_gym.app:app)
    from . import paths, runner
except ImportError:  # script import (run from the service dir)
    import paths
    import runner

HERE = Path(__file__).parent
TASKS_FILE = HERE / "tasks.json"
DEFAULT_TASKS = json.loads(TASKS_FILE.read_text())["tasks"]

# How many tasks of one job run at once. 12 = the whole default subset in one
# wave, which is what a <1h wall-clock budget needs; each concurrent slot costs
# one digest-pinned postgres 15 image (~400MB) plus one agent container.
MAX_CONCURRENT = int(os.environ.get("TOOLATHLON_MAX_CONCURRENT", "12"))

# One fresh-container retry for outcomes that say nothing about the model
# (agent_crash, PG_NOT_READY, docker errors -- pass is None and the outcome is
# none of the model-behaviour markers below). gave_up / max_steps /
# claimed_done are graded verdicts upstream (the VTB adapter's
# _MODEL_ZERO_OUTCOMES), so retrying those would re-roll a finished
# measurement. run_task_isolated removes same-name leftovers before creating
# its containers, so a retry is idempotent.
TASK_RETRIES = int(os.environ.get("TOOLATHLON_TASK_RETRIES", "1"))
_MODEL_DONE_OUTCOMES = frozenset({"gave_up", "max_steps", "claimed_done"})

app = FastAPI(title="Toolathlon-GYM Evaluation Service", version="1.1")

# ---------------------------------------------------------------- job store
JOBS: Dict[str, Dict] = {}
JOBS_LOCK = threading.Lock()
JOB_QUEUE: Queue = Queue()


class EvalRequest(BaseModel):
    model_platform: str = Field(..., description="openai_compatible | openai | anthropic | gemini")
    model_name: str = Field(..., description="e.g. gpt-4o, qwen-max, claude-sonnet-4")
    model_api_key: str
    model_api_url: Optional[str] = Field(None, description="required for openai_compatible")
    tasks: Optional[List[str]] = Field(
        None,
        description="subset override; default = 12-task validation set from tasks.json",
    )
    max_steps: int = Field(100, ge=1, le=200)
    enable_thinking: Optional[bool] = Field(
        None,
        description="per-job thinking switch; None falls back to the service env / model native")
    max_tokens: Optional[int] = Field(
        None, ge=1,
        description="per-job generation cap; None -> model_provider default (8192)")
    temperature: Optional[float] = None
    top_p: Optional[float] = Field(None, gt=0, le=1)
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = Field(None, gt=0)
    model_timeout: Optional[float] = Field(
        None, gt=0,
        description="per-job model HTTP timeout in seconds; None -> model_provider default (900)")
    model_retries: Optional[int] = Field(
        None, ge=0,
        description="per-job model retry count; None -> model_provider default (2)")


def _worker():
    while True:
        job_id = JOB_QUEUE.get()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            continue
        try:
            _execute_job(job)
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)[:500]
        finally:
            job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if job["status"] == "running":
                job["status"] = "done"


def _execute_job(job: Dict):
    job["status"] = "running"
    model_cfg = {
        "MODEL_PLATFORM": job["model_platform"],
        "MODEL_PROVIDER": job["model_platform"],
        "MODEL_NAME": job["model_name"],
        "MODEL_API_KEY": job["model_api_key_full"],
        "MODEL_API_URL": job["model_api_url"] or "",
    }
    # Per-job thinking wins over the service env; absent (None) leaves the
    # service-env fallback in runner untouched, so old callers are unchanged.
    if job.get("enable_thinking") is not None:
        model_cfg["TOOLATHLON_ENABLE_THINKING"] = "1" if job["enable_thinking"] else "0"
    # max_tokens / timeout / retries flow the same way: benchmark.json declares
    # them, evaluate.py posts them, and runner forwards them into the container.
    # Absent (None) leaves the runner's service-env fallback / model_provider
    # default in place, so old callers are unchanged.
    if job.get("max_tokens") is not None:
        model_cfg["TOOLATHLON_MAX_TOKENS"] = str(job["max_tokens"])
    for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
        if job.get(key) is not None:
            model_cfg[f"TOOLATHLON_{key.upper()}"] = str(job[key])
    if job.get("model_timeout") is not None:
        model_cfg["TOOLATHLON_MODEL_TIMEOUT"] = str(job["model_timeout"])
    if job.get("model_retries") is not None:
        model_cfg["TOOLATHLON_MODEL_RETRIES"] = str(job["model_retries"])
    total = len(job["tasks"])
    done = {"n": 0}
    lock = threading.Lock()

    def _one(task: str) -> Dict:
        res = runner.run_task_isolated(
            task, model_cfg, job["id"], max_steps=job["max_steps"],
            output_root=str(HERE / "eval_output"),
        )
        retries = TASK_RETRIES
        while (retries > 0 and res.get("pass") is None
               and res.get("agent_outcome") not in _MODEL_DONE_OUTCOMES):
            retries -= 1
            # Keep the crashed attempt on disk for diagnosis instead of
            # letting the retry overwrite its run.log.
            job_out = HERE / "eval_output" / job["id"]
            src = job_out / task
            if src.exists():
                n = 1
                while (job_out / f"{task}.attempt{n}").exists():
                    n += 1
                src.rename(job_out / f"{task}.attempt{n}")
            first = {k: res.get(k) for k in
                     ("agent_outcome", "status", "details", "duration_sec")}
            res = runner.run_task_isolated(
                task, model_cfg, job["id"], max_steps=job["max_steps"],
                output_root=str(HERE / "eval_output"),
            )
            res["retried"] = True
            res["first_attempt"] = first
            res["duration_sec"] = round(
                (first.get("duration_sec") or 0) + (res.get("duration_sec") or 0), 1)
        with lock:
            done["n"] += 1
            job["results"].append(res)
            # No single "current" task under concurrency; report the count.
            job["progress"] = {"done": done["n"], "total": total, "current": None}
        return res

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        list(pool.map(_one, job["tasks"]))

    passed = sum(1 for r in job["results"] if r["pass"] is True)
    failed = sum(1 for r in job["results"] if r["pass"] is False)
    graded = passed + failed
    job["summary"] = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "unknown": total - graded,
        # pass_rate keeps the historical meaning (unknown counts against you);
        # pass_rate_graded is the accuracy over tasks that reached a verdict,
        # which is the number VTB's metrics contract wants.
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "pass_rate_graded": round(passed / graded, 4) if graded else 0.0,
        "max_concurrent": MAX_CONCURRENT,
        "total_duration_sec": round(sum(r["duration_sec"] for r in job["results"]), 1),
    }
    # The full key was only needed to launch containers; never keep it around.
    job.pop("model_api_key_full", None)


threading.Thread(target=_worker, daemon=True).start()

# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return runner.check_prerequisites()


@app.get("/tasks")
def tasks():
    return {"count": len(DEFAULT_TASKS), "tasks": DEFAULT_TASKS}


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    pre = runner.check_prerequisites()
    if not pre["ok"]:
        raise HTTPException(503, f"environment not ready: {pre['problems']}")
    if req.model_platform == "openai_compatible" and not req.model_api_url:
        raise HTTPException(400, "model_api_url is required for openai_compatible platform")

    # Unknown-task validation resolves against the official checkout's task set
    # (the read-only upstream this service drives), not the service dir.
    finalpool = paths.official_repo() / "tasks" / "finalpool"
    req_tasks = req.tasks or DEFAULT_TASKS
    bad = [t for t in req_tasks if t not in DEFAULT_TASKS and
           not (finalpool / t).is_dir()]
    if bad:
        raise HTTPException(400, f"unknown tasks: {bad}")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "model_platform": req.model_platform,
        "model_name": req.model_name,
        "model_api_key": req.model_api_key[:4] + "****",  # never expose the full key
        "model_api_key_full": req.model_api_key,          # dropped after the run
        "model_api_url": req.model_api_url,
        "max_steps": req.max_steps,
        "enable_thinking": req.enable_thinking,
        "max_tokens": req.max_tokens,
        "model_timeout": req.model_timeout,
        "model_retries": req.model_retries,
        "tasks": req_tasks,
        "status": "queued",
        "progress": {"done": 0, "total": len(req_tasks), "current": None},
        "results": [],
        "summary": None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    JOB_QUEUE.put(job_id)
    return {"job_id": job_id, "status": "queued", "task_count": len(req_tasks)}


def _public(job: Dict) -> Dict:
    return {k: v for k, v in job.items() if k != "model_api_key_full"}


@app.get("/jobs")
def list_jobs():
    with JOBS_LOCK:
        return [{"job_id": j["id"], "model_name": j["model_name"], "status": j["status"],
                 "progress": j["progress"], "summary": j["summary"],
                 "created_at": j["created_at"]} for j in JOBS.values()]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return _public(job)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)
