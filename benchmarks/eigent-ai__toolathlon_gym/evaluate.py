#!/usr/bin/env python3
"""toolathlon-gym: MCP tool-use benchmark, scored through its own HTTP service.

An adapter, not a shim. The grading service only speaks the command-line
contract (--model-path/--api-base/--limit/--json-output-file); the toolathlon
service only speaks HTTP (POST /evaluate, GET /jobs/{id}). This file translates
between the two, so neither side changes for the other.

The service typically runs on ANOTHER host (the toolathlon host), set through:

    VTB_TOOLATHLON_URL=http://<gym-ip>:8060

Because the service is remote, a loopback --api-base (which is what the grading
worker hands over after serving the model locally) is unreachable from there.
It is rewritten to this host's routable address below -- same problem and same
fix as eval/runner/task_backends.py:host_endpoint for rootless docker.

Task order comes from manifest.json in this directory and is load-bearing:
--limit N means the first N tasks of that order, which is what makes the
validation split (a prefix) the same set on every submission.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

SERVICE = (os.environ.get("VTB_TOOLATHLON_URL") or "http://127.0.0.1:8060").rstrip("/")
POLL_SECONDS = int(os.environ.get("VTB_TOOLATHLON_POLL") or 15)
# The whole job: 12 tasks in one concurrent wave, ~35 min ceiling per task,
# plus pg startup and result extraction. 55 min keeps the strict <1h budget.
JOB_TIMEOUT = int(os.environ.get("VTB_TOOLATHLON_TIMEOUT") or 3300)


def _call(method: str, path: str, payload: dict | None = None, timeout: int = 60) -> dict:
    url = f"{SERVICE}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:400]
        raise SystemExit(f"toolathlon service answered {exc.code} for {path}: {body}")
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"cannot reach the toolathlon service at {url}: {exc.reason}. "
            "Is the service running on the toolathlon host? Set VTB_TOOLATHLON_URL."
        )


def _host_address() -> str | None:
    """This host's routable address (same trick as eval/runner/task_backends.py)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 53))  # no packet sent; fixes the local addr
            return sock.getsockname()[0]
    except OSError:
        return None


def _reachable_api_base(api_base: str) -> str:
    """Rewrite a loopback model endpoint into one the task containers can reach.

    Per-task bridge containers resolve loopback to themselves, even when the
    service shares the model host. Rewrite loopback endpoints unconditionally
    using VTB_MODEL_ADVERTISE_HOST or the detected routable host address.
    The model must listen on 0.0.0.0, as the grading worker's served_model() does.
    """
    for loopback in ("127.0.0.1", "localhost", "[::1]"):
        if loopback in api_base:
            host = os.environ.get("VTB_MODEL_ADVERTISE_HOST") or _host_address()
            if not host:
                raise SystemExit(
                    "the model endpoint is on loopback and no routable address was "
                    "found to reach it from the toolathlon task containers. Set "
                    "VTB_MODEL_ADVERTISE_HOST to this host's IP."
                )
            return api_base.replace(loopback, host)
    return api_base


def ordered_task_ids() -> list[str]:
    payload = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    ids = [t["task_id"] for t in payload.get("tasks") or [] if t.get("task_id")]
    if not ids:
        raise SystemExit(f"{HERE / 'manifest.json'} lists no tasks")
    return ids


def _model_runtime() -> dict:
    """Runtime parameters this benchmark declares under agent_kwargs.

    max_steps is consumed by this adapter; model-call settings are forwarded by
    the gym service to each task container. Missing keys retain the established
    service/model-provider fallbacks for compatibility.
    """
    out: dict = {}
    try:
        bench = json.loads((HERE / "benchmark.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    ak = bench.get("agent_kwargs") or {}
    if ak.get("max_steps") is not None:
        out["max_steps"] = ak["max_steps"]
    llm = ak.get("llm_call_kwargs")
    if isinstance(llm, str):
        try:
            llm = json.loads(llm)
        except ValueError:
            llm = {}
    llm = llm or {}
    for key in ("max_tokens", "temperature", "top_p", "top_k", "repetition_penalty"):
        if key in llm:
            out[key] = llm[key]
    et = ((llm.get("extra_body") or {}).get("chat_template_kwargs") or {}).get("enable_thinking")
    if et is not None:
        out["enable_thinking"] = et
    if ak.get("timeout") is not None:
        out["model_timeout"] = ak["timeout"]
    if ak.get("max_retries") is not None:
        out["model_retries"] = ak["max_retries"]
    return out


# Outcomes that are a statement about the MODEL, so they belong in the score as a
# zero. "Did not finish the workflow" is a failure to do the task, and the checks
# would have failed on the state it left behind -- the upstream evaluator just
# never bothers to run them (utils/evaluation/evaluator.py refuses anything but
# SUCCESS), which is why these arrive with pass=None instead of pass=False.
_MODEL_ZERO_OUTCOMES = frozenset({"gave_up", "max_steps"})

# Statuses that are a statement about the MODEL even without an agent_outcome.
_MODEL_ZERO_STATUSES = frozenset({"max_turns_reached"})


def _classify(task_id: str, r: dict | None) -> dict:
    """Turn one service result into a graded verdict or an explicit non-result.

    Three outcomes stay apart, as in eval/runner/trial_results.py -- but the line
    is drawn by CAUSE, not by whether the upstream evaluator happened to run:

      graded 1.0   the task's checks passed
      graded 0.0   the model did not accomplish the task (gave up, ran out of
                   steps, or the checks failed)
      ungraded     nothing was learned about the model: the agent crashed, the
                   postgres never came up, docker failed, the run was interrupted

    The distinction matters because `failed` is overloaded upstream: task_agent.py
    presets status=FAILED and its `except Exception` branch only prints, so a
    crash and a give-up are indistinguishable by status alone. `agent_outcome`
    (added in service/runner.py) carries the difference.
    """
    if r is None:
        return {"task_id": task_id, "graded": False, "reward": None,
                "reason": "the service produced no result for this task"}

    base = {"task_id": task_id, "status": r.get("status"),
            "agent_outcome": r.get("agent_outcome"),
            "duration_sec": r.get("duration_sec")}

    verdict = r.get("pass")
    if verdict is not None:  # the evaluator ran, or the runner already decided
        return {**base, "graded": True, "reward": 1.0 if verdict else 0.0}

    outcome = r.get("agent_outcome")
    if outcome in _MODEL_ZERO_OUTCOMES or r.get("status") in _MODEL_ZERO_STATUSES:
        return {**base, "graded": True, "reward": 0.0,
                "reason": f"model did not complete the task ({outcome or r.get('status')})"}

    if outcome == "agent_crash":
        return {**base, "graded": False, "reward": None,
                "reason": f"the agent crashed, so the model was never really "
                          f"tested: {r.get('details') or 'see run.log'}"}

    # No agent_outcome at all: either an infrastructure status (PG_NOT_READY /
    # ERROR / interrupted) or a service that predates the field. Ungraded on
    # purpose -- guessing here is what would hide a broken environment behind a
    # plausible-looking zero.
    return {**base, "graded": False, "reward": None,
            "reason": r.get("details") or "no verdict and no agent outcome reported"}


# Where a task's transcript can be read from. The service reports an absolute
# output_dir on ITS OWN host, so that path only resolves here in the single-host
# layout (service on 127.0.0.1) or when its output tree is mounted at the same
# path. The override covers the mounted case, where the two hosts disagree about
# the prefix; without a readable tree the trials are still written, carrying an
# explicit reason instead of a transcript.
TRAJ_ROOT_OVERRIDE = os.environ.get("VTB_TOOLATHLON_EVAL_OUTPUT") or ""


def _trajectory_source(job_id: str, task_id: str, result: dict | None) -> Path | None:
    """The service-side traj.json for one task, when this host can read it.

    camel writes {"messages": [...], "tool_calls": [...]} -- already the shape
    eval/runner/trial_results.trajectory_tail() parses into (action, observation)
    steps, so the file is copied verbatim rather than converted. The nesting under
    the reported directory (<platform>_<model>/SingleUserTurn-<task>/) is the
    upstream dumps layout and is searched rather than reconstructed.
    """
    roots = []
    if TRAJ_ROOT_OVERRIDE:
        roots.append(Path(TRAJ_ROOT_OVERRIDE) / job_id / task_id)
    reported = (result or {}).get("output_dir")
    if isinstance(reported, str) and reported.strip():
        roots.append(Path(reported.strip()))
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in sorted(root.rglob("traj.json")):
            if candidate.is_file():
                return candidate
    return None


def _materialize_trials(fo: Path, records: list[dict], by_task: dict, job_id: str) -> dict:
    """Per-task trial directories, in the layout the grading service reads.

    eval/runner/trial_results.index_trials() maps a task id to a trial directory by
    finding a result.json that names it, and the transcript is read from
    <trial>/agent/trajectory.json (AGENT_TRAJECTORY_NAMES). Writing that layout is
    what lets an agent see what its own model DID on each validation task --
    disclosure.py serves it as per_sample_outcomes[].trajectory_excerpt and through
    /sample -- instead of a bare score with no way to tell a give-up from a loop.
    Before this, a toolathlon job wrote only the service's job.json, so every
    per-sample trajectory_excerpt was null.

    The per-task result.json must NOT carry a "stats" key: _latest_job_dir() uses
    exactly that key to recognise a job subdirectory inside a jobs-dir, so a
    "stats" here would make framework_output/ look like a jobs-dir and collapse
    twelve trials into whichever task sorted last.
    """
    counts = {"copied": 0, "unavailable": 0}
    for record in records:
        task_id = record["task_id"]
        trial = fo / task_id
        (trial / "agent").mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in record.items() if k != "stats"}
        payload["task_name"] = task_id
        source = _trajectory_source(job_id, task_id, by_task.get(task_id))
        payload["trajectory"] = None
        if source is not None:
            try:
                shutil.copyfile(source, trial / "agent" / "trajectory.json")
                payload["trajectory"] = "agent/trajectory.json"
                counts["copied"] += 1
            except OSError as exc:
                payload["trajectory_unavailable"] = f"could not copy {source}: {exc}"
        if payload["trajectory"] is None:
            counts["unavailable"] += 1
            payload.setdefault("trajectory_unavailable",
                               "the toolathlon service's output tree is not readable "
                               "from this host; set VTB_TOOLATHLON_EVAL_OUTPUT when it "
                               "is mounted here")
        (trial / "result.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="toolathlon-gym: 12 MCP tool-use tasks, evaluated concurrently "
        "by the toolathlon service"
    )
    p.add_argument("--model-path", type=str, default=None,
                   help="Checkpoint directory; only its name is used (the model is "
                   "reached over --api-base, never loaded here).")
    p.add_argument("--api-base", type=str,
                   default=os.environ.get("VTB_SWE_API_BASE") or None,
                   help="OpenAI-compatible endpoint the tasks drive.")
    p.add_argument("--served-model-name", type=str, default=None)
    # A positive literal, so eval/grading/config.py's default_limit() regex can
    # read the subset size for progress percentages.
    p.add_argument("--limit", type=int, default=12,
                   help="First N tasks in manifest order. -1 means all.")
    p.add_argument("--step-limit", type=int, default=None,
                   help="max_steps per task; the retry ladder lowers this.")
    p.add_argument("--only", nargs="*", default=None, help="Explicit task ids.")
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--json-output-file", type=str, default=None)
    p.add_argument("--templates-dir", type=str, default="templates/",
                   help="Accepted for contract compatibility; unused.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.api_base:
        raise SystemExit(
            "--api-base is required: toolathlon drives the model over HTTP and "
            "this adapter never loads weights itself."
        )

    all_ids = ordered_task_ids()
    if args.only:
        wanted = set(args.only)
        ids = [t for t in all_ids if t in wanted]
        missing = wanted - set(ids)
        if missing:
            raise SystemExit(f"no such task(s): {', '.join(sorted(missing))}")
    elif args.limit is not None and args.limit >= 0:
        ids = all_ids[: args.limit]
    else:
        ids = all_ids

    served = args.served_model_name or (
        Path(args.model_path).name if args.model_path else "model"
    )
    api_base = _reachable_api_base(args.api_base)
    if api_base != args.api_base:
        print(f"[toolathlon] rewrote model endpoint for the remote service: "
              f"{args.api_base} -> {api_base}", flush=True)

    rt = _model_runtime()
    _env_think = os.environ.get("VTB_TOOLATHLON_ENABLE_THINKING")
    body = {
        "model_platform": "openai_compatible",
        "model_name": served,
        "model_api_key": os.environ.get("VTB_API_KEY", "inspectai"),
        "model_api_url": api_base,
        "tasks": ids,
        # benchmark.json explicitly records toolathlon's 100-step default. The retry
        # ladder can still lower it through --step-limit without changing the registered
        # default used by ordinary runs.
        "max_steps": args.step_limit or int(rt.get("max_steps", 100)),
        # SFT checkpoints are trained on thinking data; their native mode is
        # thinking-on. benchmark.json (agent_kwargs.llm_call_kwargs) is the
        # source of truth here; VTB_TOOLATHLON_ENABLE_THINKING still overrides
        # it for ad-hoc runs (=0 for a base, non-thinking model). The service
        # defaults to nothink when unset, which is out-of-distribution for
        # these models and collapses the score, so always send an explicit
        # per-job switch.
        "enable_thinking": (
            _env_think not in ("0", "false", "no", "")
            if _env_think is not None
            else bool(rt.get("enable_thinking", True))
        ),
    }
    # max_tokens / timeout / max_retries: declared in benchmark.json, forwarded
    # to the gym service which injects them per task container as TOOLATHLON_*.
    # Omitted keys leave the service / model_provider defaults untouched.
    for key in ("max_tokens", "temperature", "top_p", "top_k", "repetition_penalty"):
        if key in rt:
            body[key] = rt[key]
    if "model_timeout" in rt:
        body["model_timeout"] = rt["model_timeout"]
    if "model_retries" in rt:
        body["model_retries"] = rt["model_retries"]
    submitted = _call("POST", "/evaluate", body)
    job_id = submitted["job_id"]
    print(f"[toolathlon] job={job_id} tasks={len(ids)} service={SERVICE} "
          f"model={served} max_steps={body['max_steps']}", flush=True)

    deadline = time.time() + JOB_TIMEOUT
    state: dict = {}
    last = ""
    while time.time() < deadline:
        state = _call("GET", f"/jobs/{job_id}", timeout=30)
        if state.get("status") in ("done", "failed"):
            break
        prog = state.get("progress") or {}
        line = f"[toolathlon] {state.get('status')} {prog.get('done')}/{prog.get('total')}"
        if line != last:
            print(line, flush=True)
            last = line
        time.sleep(POLL_SECONDS)
    else:
        raise SystemExit(f"toolathlon job {job_id} did not finish within {JOB_TIMEOUT}s")

    if state.get("status") == "failed":
        raise SystemExit(f"toolathlon job failed: {state.get('error')}")

    results = state.get("results") or []
    by_task = {r.get("task"): r for r in results if r.get("task")}

    records = [_classify(tid, by_task.get(tid)) for tid in ids]

    graded = [r for r in records if r["graded"]]
    solved = [r for r in graded if r["reward"] == 1.0]
    summary = {
        # accuracy over tasks that reached a verdict; infrastructure trouble is
        # counted in `ungraded`, never as a failure.
        "accuracy": (len(solved) / len(graded)) if graded else 0.0,
        "solved": len(solved),
        "graded": len(graded),
        "attempted": len(records),
        "ungraded": len(records) - len(graded),
        "dataset_total": len(all_ids),
        "pass_rate_including_unknown": (state.get("summary") or {}).get("pass_rate"),
        "benchmark": "eigent-ai__toolathlon_gym",
        "runner": f"toolathlon-gym service @ {SERVICE}",
        "agent": "camel-toolathlon",
        "model": f"openai/{served}",
        "job_id": job_id,
    }

    work_dir = Path(args.work_dir) if args.work_dir else HERE / "toolathlon_run"
    # The grading archiver scans <work-dir>/framework_output/, so the full job
    # record lands where _archive_full_record() already looks -- no server change.
    fo = work_dir / "framework_output"
    fo.mkdir(parents=True, exist_ok=True)
    (fo / "job.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    trials = _materialize_trials(fo, records, by_task, job_id)
    print(f"trajectories: {trials['copied']} copied, "
          f"{trials['unavailable']} unavailable", flush=True)
    (work_dir / "records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    per_sample = {
        # Dict shape, as eval/grading/valsplit.py:read_per_sample_file requires.
        # Ungraded tasks are left out of `correct` so the reconstruction check
        # can reproduce `accuracy` exactly.
        "dataset_total": len(all_ids),
        "sample_ids": [r["task_id"] for r in records],
        "correct": {r["task_id"]: bool(r["reward"]) for r in graded},
    }

    targets = [Path(args.json_output_file)] if args.json_output_file else []
    targets.append(work_dir / "metrics.json")
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        target.with_suffix(".per_sample.json").write_text(
            json.dumps(per_sample, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"accuracy={summary['accuracy']:.4f} solved={summary['solved']}/"
          f"{summary['graded']} graded ({summary['ungraded']} ungraded)", flush=True)
    if not graded and records:
        # Every task ungraded is an infrastructure verdict, not a model one.
        raise SystemExit(
            f"no task reached a verdict; {work_dir / 'records.json'} says why"
        )


if __name__ == "__main__":
    main()
