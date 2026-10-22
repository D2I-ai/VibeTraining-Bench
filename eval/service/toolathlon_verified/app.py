"""
Toolathlon-VERIFIED evaluation service (final/held-out test set).

API contract -- identical to the gym service's so the VTB adapter only swaps
the URL:
  POST /evaluate            submit a model for evaluation (async job)
  GET  /jobs/{job_id}       job status + per-task results
  GET  /jobs                all jobs
  GET  /tasks               task count only; final task IDs remain hidden
  GET  /health              docker/image prerequisite check

Tasks inside one job run CONCURRENTLY (default 12 at a time): every task gets
its own docker network + container via runner.run_task_isolated. That isolates
the TASK CONTAINERS from each other, but NOT the host-side app services
(poste/canvas/woocommerce/... are shared single instances): tasks whose
preprocess resets the same server-side state must not overlap. Like the
official run_parallel.py, tasks in a task_conflict.json group are serialized
with a shared lock; additionally, canvas tasks sharing an admin_canvas_token
are detected and grouped dynamically (the official file lists only one of
those pairs -- empirically token2 is shared by 4 tasks, token1 by 2).
Jobs are still processed one at a time by the single worker thread,
which caps total container pressure on the host.

The service evaluates the final-task subset declared by the benchmark manifest.
Task definitions come from the external official checkout selected by
VTB_TOOLATHLON_VERIFIED_REPO via paths.py. Kubernetes tasks use the isolated
Kind execution path in runner.py.
"""

import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:  # package import (uvicorn eval.service.toolathlon_verified.app:app)
    from . import paths, runner
except ImportError:  # script import (run from the service dir)
    import paths
    import runner

HERE = Path(__file__).parent

# Kind-based tasks run through the official containerized-mode mounts:
# --network host + docker.sock. The exclusion is empty on a host with Kind;
# the constant remains as one auditable switch for deployments without Kind.
K8S_EXCLUDED = frozenset()

MAX_CONCURRENT = int(os.environ.get("TOOLATHLON_VERIFIED_MAX_CONCURRENT", "12"))

# One fresh-container retry for outcomes that say nothing about the model
# (agent_crash, preprocess/stash failures, docker errors -- pass is None and
# the outcome is none of the model-behaviour markers below). claimed_done /
# gave_up are graded verdicts upstream (the VTB adapter's
# _MODEL_ZERO_OUTCOMES), so retrying those would re-roll a finished
# measurement. run_task_isolated removes same-name leftovers before creating
# its containers, so a retry is idempotent.
TASK_RETRIES = int(os.environ.get("TOOLATHLON_TASK_RETRIES", "1"))
_MODEL_DONE_OUTCOMES = frozenset({"gave_up", "max_steps", "claimed_done"})


# ---------------------------------------------------------------- task set
# The finalpool, default task list and conflict groups all live in the external
# official checkout, so they are resolved lazily via paths.official_repo() the
# first time a request needs them -- never at import (importing this module,
# e.g. to print paths, must not require the repo env to be set).
_FINALPOOL: Optional[Path] = None
_DEFAULT_TASKS: Optional[List[str]] = None
_CONFLICT_LOCKS: Optional[Dict[str, threading.Lock]] = None
_TASKSET_LOCK = threading.Lock()


def _finalpool() -> Path:
    global _FINALPOOL
    if _FINALPOOL is None:
        _FINALPOOL = paths.official_repo() / "tasks" / "finalpool"
    return _FINALPOOL


def _default_tasks() -> List[str]:
    global _DEFAULT_TASKS
    if _DEFAULT_TASKS is None:
        fp = _finalpool()
        _DEFAULT_TASKS = sorted(
            p.name for p in fp.iterdir()
            if p.is_dir() and p.name not in K8S_EXCLUDED and not p.name.startswith(".")
        )
    return _DEFAULT_TASKS


# ---------------------------------------------------------------- conflicts
def _conflict_groups() -> List[List[str]]:
    """Task groups that must never run concurrently.

    Source 1 (official): tasks/finalpool/task_conflict.json -- run_parallel.py
    gives each group one shared lock (e.g. canvas-do-quiz + canvas-submit-late-work
    share mcpcanvasadmin2@'s Canvas workspace).
    Source 2 (derived): Canvas tasks whose token_key_session declares the
    same admin_canvas_token or canvas_api_token operate on shared courses
    and must be serialized. Derive these groups at service startup so upstream
    token changes are reflected automatically.
    """
    finalpool = _finalpool()
    groups: List[List[str]] = []
    try:
        data = json.loads(
            (finalpool / "task_conflict.json").read_text(encoding="utf-8"))
        groups.extend(g for g in data.get("conflict_groups", []) if len(g) > 1)
    except Exception:
        pass  # absent/unreadable -> official groups unknown; derived ones below still apply
    by_token: Dict[str, List[str]] = {}
    for tk in sorted(finalpool.glob("canvas-*/token_key_session.py")):
        src = tk.read_text(encoding="utf-8", errors="replace")
        for field in ("admin_canvas_token", "canvas_api_token"):
            m = re.search(field + r'\s*=\s*"([^"]+)"', src)
            if m:
                by_token.setdefault(f"{field}:{m.group(1)}", []).append(
                    tk.parent.name)
    groups.extend(g for g in by_token.values() if len(g) > 1)
    return groups


def _conflict_locks() -> Dict[str, threading.Lock]:
    """task -> shared Lock, built once. Stable objects across requests."""
    global _CONFLICT_LOCKS
    if _CONFLICT_LOCKS is None:
        with _TASKSET_LOCK:
            if _CONFLICT_LOCKS is None:
                locks: Dict[str, threading.Lock] = {}
                for g in _conflict_groups():
                    lk = threading.Lock()
                    for t in g:
                        locks[t] = lk
                _CONFLICT_LOCKS = locks
    return _CONFLICT_LOCKS


def _spread(tasks: List[str]) -> List[str]:
    """Interleave same-group tasks so they rarely occupy workers at the same
time. The conflict locks guarantee correctness; spreading only avoids worker
threads starving on a lock while its holder runs a full task."""
    from collections import defaultdict, deque
    conflict_locks = _conflict_locks()
    by_lock = defaultdict(deque)
    for t in tasks:
        by_lock[conflict_locks.get(t)].append(t)  # None -> the solo queue
    solo = by_lock.pop(None, deque())
    # a lone conflicted task leaves `solo` empty -- keep it out or popleft
    # below raises IndexError (hit by --only <one conflicted task> jobs)
    queues = [q for q in [*by_lock.values(), solo] if q]
    out: List[str] = []
    while queues:
        nxt = []
        for q in queues:
            out.append(q.popleft())
            if q:
                nxt.append(q)
        queues = nxt
    return out


app = FastAPI(title="Toolathlon-VERIFIED Evaluation Service", version="1.1")

# ---------------------------------------------------------------- job store
JOBS: Dict[str, Dict] = {}
JOBS_LOCK = threading.Lock()
JOB_QUEUE: Queue = Queue()


class EvalRequest(BaseModel):
    model_platform: str = Field("openai_compatible",
                                description="only openai_compatible is wired")
    model_name: str = Field(..., description="served model name, e.g. final_model")
    model_api_key: str = "fake-key"
    model_api_url: Optional[str] = Field(None, description="required; e.g. http://172.17.0.1:20993/v1")
    tasks: Optional[List[str]] = Field(None, description="subset override; default = finalpool minus k8s tasks")
    max_steps: int = Field(200, ge=1, le=400)
    enable_thinking: Optional[bool] = Field(
        None,
        description="per-job thinking switch; None falls back to the service env / model native")
    max_tokens: Optional[int] = Field(
        None, ge=1,
        description="per-job agent generation cap; None keeps the verified default (8192)")
    temperature: Optional[float] = None
    top_p: Optional[float] = Field(None, gt=0, le=1)
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = Field(None, gt=0)
    model_retries: Optional[int] = Field(
        None, ge=1,
        description="per-job model request attempts; None keeps the provider default (10)")


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
        "MODEL_PROVIDER": "unified",
        "MODEL_NAME": job["model_name"],
        "MODEL_API_KEY": job["model_api_key_full"],
        "MODEL_API_URL": job["model_api_url"] or "",
    }
    # Per-job thinking wins over the service env; absent (None) leaves the
    # service-env fallback in runner untouched, so old callers are unchanged.
    if job.get("enable_thinking") is not None:
        model_cfg["TOOLATHLON_ENABLE_THINKING"] = "1" if job["enable_thinking"] else "0"
    if job.get("max_tokens") is not None:
        model_cfg["TOOLATHLON_MAX_TOKENS"] = str(job["max_tokens"])
    for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
        if job.get(key) is not None:
            model_cfg[f"TOOLATHLON_{key.upper()}"] = str(job[key])
    if job.get("model_retries") is not None:
        model_cfg["TOOLATHLON_MODEL_RETRIES"] = str(job["model_retries"])
    total = len(job["tasks"])
    done = {"n": 0}
    lock = threading.Lock()
    conflict_locks = _conflict_locks()

    def _one(task: str) -> Dict:
        # Hold the group lock (if any) across the WHOLE attempt+retry sequence:
        # a retry re-runs preprocess, so it clobbers shared server-side state
        # exactly like the first attempt.
        group_lock = conflict_locks.get(task)
        if group_lock is None:
            return _run_task(task)
        with group_lock:
            return _run_task(task)

    def _run_task(task: str) -> Dict:
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
            job["progress"] = {"done": done["n"], "total": total, "current": None}
        return res

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        list(pool.map(_one, _spread(job["tasks"])))

    passed = sum(1 for r in job["results"] if r["pass"] is True)
    failed = sum(1 for r in job["results"] if r["pass"] is False)
    graded = passed + failed
    job["summary"] = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "unknown": total - graded,
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
    # Privacy: this is the held-out FINAL test set. Task ids are deliberately
    # NOT enumerated here -- the gym service's /tasks answers with its
    # validation set because those ids are the agent's to see; these are not.
    # The count and the k8s deviation stay public for sanity checks; the
    # adapter gets its ordered list from its own manifest.json instead.
    return {"count": len(_default_tasks()), "excluded": sorted(K8S_EXCLUDED)}


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    pre = runner.check_prerequisites()
    if not pre["ok"]:
        raise HTTPException(503, f"environment not ready: {pre['problems']}")
    if req.model_platform != "openai_compatible":
        raise HTTPException(400, "this service only speaks to OpenAI-compatible endpoints")
    if not req.model_api_url:
        raise HTTPException(400, "model_api_url is required")

    default_tasks = _default_tasks()
    finalpool = _finalpool()
    req_tasks = req.tasks or default_tasks
    bad = [t for t in req_tasks if t not in default_tasks and
           not (finalpool / t).is_dir()]
    if bad:
        raise HTTPException(400, f"unknown tasks: {bad}")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "model_platform": req.model_platform,
        "model_name": req.model_name,
        "model_api_key": (req.model_api_key[:4] + "****") if req.model_api_key else "****",
        "model_api_key_full": req.model_api_key,
        "model_api_url": req.model_api_url,
        "max_steps": req.max_steps,
        "enable_thinking": req.enable_thinking,
        "max_tokens": req.max_tokens,
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
    uvicorn.run(app, host="0.0.0.0", port=8061)
