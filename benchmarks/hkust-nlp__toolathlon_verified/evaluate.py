#!/usr/bin/env python3
"""toolathlon_verified: the official held-out FINAL test set, scored through
its own HTTP service.

An adapter, not a shim, and the twin of benchmarks/eigent-ai__toolathlon_gym/evaluate.py:
the grading service only speaks the command-line contract
(--model-path/--api-base/--limit/--json-output-file); the toolathlon-verified
service only speaks HTTP (POST /evaluate, GET /jobs/{id}). This file translates
between the two, so neither side changes for the other.

The service runs on the same host as this adapter, reached through:

    VTB_TOOLATHLON_VERIFIED_URL=http://127.0.0.1:8061

On this single-host deployment the model endpoint a grading worker serves on
loopback is unreachable from the per-task bridge containers, so both the
service URL and the advertised model host must be the docker0 gateway
(172.17.0.1) -- the same pattern the gym service already uses:

    VTB_TOOLATHLON_VERIFIED_URL=http://172.17.0.1:8061
    VTB_MODEL_ADVERTISE_HOST=172.17.0.1

Task order comes from manifest.json in this directory and is load-bearing:
--limit N means the first N tasks of that order. The service itself never
enumerates the task ids (GET /tasks answers with a count only -- this is a
held-out set), so the manifest is the one ordered authority.

Outcome vocabulary is the gym adapter's, because the verified runner maps the
official TaskStatus onto it before reporting (service/runner.py
_classify_outcome): claimed_done / gave_up / agent_crash. The three-state line
is identical -- graded 1.0 (checks passed), graded 0.0 (model did not
accomplish the task: gave up, ran out of steps, checks failed), ungraded
(nothing was learned about the model: agent crashed, infrastructure failed).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

SERVICE = (os.environ.get("VTB_TOOLATHLON_VERIFIED_URL")
           or "http://127.0.0.1:8061").rstrip("/")
POLL_SECONDS = int(os.environ.get("VTB_TOOLATHLON_VERIFIED_POLL") or 15)
# The whole job: 88 tasks at 12-way concurrency is ~8 waves; the per-task
# wall clock ceiling is TASK_TIMEOUT=2700s (45 min, service default) plus
# container startup and one crash retry. 3.75 h covers the slow tail without
# sitting on the attempt forever; official runs may raise it.
JOB_TIMEOUT = int(os.environ.get("VTB_TOOLATHLON_VERIFIED_TIMEOUT") or 13500)


def _call(method: str, path: str, payload: dict | None = None,
          timeout: int = 60) -> dict:
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
        raise SystemExit(
            f"toolathlon-verified service answered {exc.code} for {path}: {body}")
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"cannot reach the toolathlon-verified service at {url}: {exc.reason}. "
            "Is it running (port 8061)? Set VTB_TOOLATHLON_VERIFIED_URL."
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

    Per-task bridge containers resolve loopback to themselves. Rewrite loopback
    endpoints using VTB_MODEL_ADVERTISE_HOST or a detected routable address.
    On a single-host Docker deployment, the advertised host is typically the
    bridge gateway, for example 172.17.0.1.
    """
    for loopback in ("127.0.0.1", "localhost", "[::1]"):
        if loopback in api_base:
            host = (os.environ.get("VTB_MODEL_ADVERTISE_HOST")
                    or _host_address())
            if not host:
                raise SystemExit(
                    "the model endpoint is on loopback and no routable address "
                    "was found to reach it from the toolathlon-verified task "
                    "containers. Set VTB_MODEL_ADVERTISE_HOST to this host's IP."
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
    """Read runtime parameters declared by this benchmark registration."""
    out: dict = {}
    try:
        bench = json.loads((HERE / "benchmark.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    agent_kwargs = bench.get("agent_kwargs") or {}
    if agent_kwargs.get("max_steps") is not None:
        out["max_steps"] = agent_kwargs["max_steps"]
    llm_kwargs = agent_kwargs.get("llm_call_kwargs")
    if isinstance(llm_kwargs, str):
        try:
            llm_kwargs = json.loads(llm_kwargs)
        except ValueError:
            llm_kwargs = {}
    llm_kwargs = llm_kwargs or {}
    for key in ("max_tokens", "temperature", "top_p", "top_k", "repetition_penalty"):
        if key in llm_kwargs:
            out[key] = llm_kwargs[key]
    enable_thinking = (
        ((llm_kwargs.get("extra_body") or {}).get("chat_template_kwargs") or {})
        .get("enable_thinking")
    )
    if enable_thinking is not None:
        out["enable_thinking"] = enable_thinking
    if agent_kwargs.get("max_retries") is not None:
        out["model_retries"] = agent_kwargs["max_retries"]
    return out


# Outcomes that are a statement about the MODEL, so they belong in the score
# as a zero. Same vocabulary the gym adapter draws its line with; the verified
# runner (service/runner.py _classify_outcome) already reports in it.
_MODEL_ZERO_OUTCOMES = frozenset({"gave_up", "max_steps"})

# Statuses that are a statement about the MODEL even without an agent_outcome.
# The verified runner maps max_turns_reached to gave_up itself, but the status
# field still carries the raw value; belt and braces.
_MODEL_ZERO_STATUSES = frozenset({"max_turns_reached", "timeout"})


def _classify(task_id: str, r: dict | None) -> dict:
    """Turn one service result into a graded verdict or an explicit non-result.

    Three outcomes stay apart, as in eval/runner/trial_results.py -- but the
    line is drawn by CAUSE, not by whether the upstream evaluator happened to
    run:

      graded 1.0   the task's checks passed
      graded 0.0   the model did not accomplish the task (gave up, ran out of
                   steps/turns, wall-clocked, or the checks failed)
      ungraded     nothing was learned about the model: the agent crashed, the
                   container never came up, docker failed
    """
    if r is None:
        return {"task_id": task_id, "graded": False, "reward": None,
                "reason": "the service produced no result for this task"}

    base = {"task_id": task_id, "status": r.get("status"),
            "agent_outcome": r.get("agent_outcome"),
            "duration_sec": r.get("duration_sec")}
    if r.get("retried"):
        base["retried"] = True
        base["first_attempt"] = r.get("first_attempt")

    verdict = r.get("pass")
    if verdict is not None:  # the evaluator ran, or the runner already decided
        return {**base, "graded": True, "reward": 1.0 if verdict else 0.0}

    outcome = r.get("agent_outcome")
    if outcome in _MODEL_ZERO_OUTCOMES or r.get("status") in _MODEL_ZERO_STATUSES:
        return {**base, "graded": True, "reward": 0.0,
                "reason": f"model did not complete the task "
                          f"({outcome or r.get('status')})"}

    if outcome == "agent_crash":
        return {**base, "graded": False, "reward": None,
                "reason": f"the agent crashed, so the model was never really "
                          f"tested: {r.get('details') or 'see run.log'}"}

    # No agent_outcome at all: an infrastructure status (PREPROCESS_FAILED /
    # STASH_FAILED / ERROR / interrupted) from a service that still reports
    # one. Ungraded on purpose -- guessing here is what would hide a broken
    # environment behind a plausible-looking zero.
    return {**base, "graded": False, "reward": None,
            "reason": r.get("details") or "no verdict and no agent outcome reported"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="toolathlon_verified: the official held-out final test set "
        "(88 MCP tool-use tasks, including Kubernetes/Kind tasks), evaluated "
        "concurrently by the toolathlon-verified service"
    )
    p.add_argument("--model-path", type=str, default=None,
                   help="Checkpoint directory; only its name is used (the model "
                        "is reached over --api-base, never loaded here).")
    p.add_argument("--api-base", type=str,
                   default=os.environ.get("VTB_SWE_API_BASE") or None,
                   help="OpenAI-compatible endpoint the tasks drive.")
    p.add_argument("--served-model-name", type=str, default=None)
    # A positive literal, so eval/grading/config.py's default_limit() regex can
    # read the subset size for progress percentages. 88 matches manifest.json.
    p.add_argument("--limit", type=int, default=88,
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
        print(f"[toolathlon-verified] rewrote model endpoint for the task "
              f"containers: {args.api_base} -> {api_base}", flush=True)

    runtime = _model_runtime()
    env_thinking = os.environ.get("VTB_TOOLATHLON_ENABLE_THINKING")
    body = {
        "model_platform": "openai_compatible",
        "model_name": served,
        "model_api_key": os.environ.get("VTB_API_KEY", "inspectai"),
        "model_api_url": api_base,
        "tasks": ids,
        # benchmark.json explicitly records the official formal_run_v0 default;
        # the retry ladder can still lower it through --step-limit.
        "max_steps": args.step_limit or int(runtime.get("max_steps", 200)),
        # benchmark.json is the source of truth; the environment remains an
        # explicit per-run override for base (non-thinking) models.
        "enable_thinking": (
            env_thinking not in ("0", "false", "no", "")
            if env_thinking is not None
            else bool(runtime.get("enable_thinking", True))
        ),
    }
    for key in ("max_tokens", "temperature", "top_p", "top_k", "repetition_penalty"):
        if key in runtime:
            body[key] = runtime[key]
    if "model_retries" in runtime:
        body["model_retries"] = runtime["model_retries"]
    submitted = _call("POST", "/evaluate", body)
    job_id = submitted["job_id"]
    print(f"[toolathlon-verified] job={job_id} tasks={len(ids)} "
          f"service={SERVICE} model={served} "
          f"max_steps={body['max_steps']}", flush=True)

    deadline = time.time() + JOB_TIMEOUT
    state: dict = {}
    last = ""
    while time.time() < deadline:
        state = _call("GET", f"/jobs/{job_id}", timeout=30)
        if state.get("status") in ("done", "failed"):
            break
        prog = state.get("progress") or {}
        line = (f"[toolathlon-verified] {state.get('status')} "
                f"{prog.get('done')}/{prog.get('total')}")
        if line != last:
            print(line, flush=True)
            last = line
        time.sleep(POLL_SECONDS)
    else:
        raise SystemExit(
            f"toolathlon-verified job {job_id} did not finish within "
            f"{JOB_TIMEOUT}s")

    if state.get("status") == "failed":
        raise SystemExit(
            f"toolathlon-verified job failed: {state.get('error')}")

    results = state.get("results") or []
    by_task = {r.get("task"): r for r in results if r.get("task")}

    records = [_classify(tid, by_task.get(tid)) for tid in ids]

    graded = [r for r in records if r["graded"]]
    solved = [r for r in graded if r["reward"] == 1.0]
    summary = {
        # accuracy over tasks that reached a verdict; infrastructure trouble
        # is counted in `ungraded`, never as a failure.
        "accuracy": (len(solved) / len(graded)) if graded else 0.0,
        "solved": len(solved),
        "graded": len(graded),
        "attempted": len(records),
        "ungraded": len(records) - len(graded),
        "dataset_total": len(all_ids),
        "pass_rate_including_unknown": (state.get("summary") or {}).get("pass_rate"),
        "benchmark": "hkust-nlp__toolathlon_verified",
        "runner": f"toolathlon-verified service @ {SERVICE}",
        "agent": "openai-agents-sdk (official verified harness)",
        "model": f"openai/{served}",
        "job_id": job_id,
    }

    work_dir = Path(args.work_dir) if args.work_dir else HERE / "hkust-nlp__toolathlon_verified_run"
    # The grading archiver scans <work-dir>/framework_output/, so the full job
    # record lands where _archive_full_record() already looks -- no server
    # change.
    fo = work_dir / "framework_output"
    fo.mkdir(parents=True, exist_ok=True)
    (fo / "job.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
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
            json.dumps(per_sample, indent=2, ensure_ascii=False),
            encoding="utf-8")

    print(f"accuracy={summary['accuracy']:.4f} solved={summary['solved']}/"
          f"{summary['graded']} graded ({summary['ungraded']} ungraded)",
          flush=True)
    if not graded and records:
        # Every task ungraded is an infrastructure verdict, not a model one.
        raise SystemExit(
            f"no task reached a verdict; {work_dir / 'records.json'} says why"
        )


if __name__ == "__main__":
    main()
