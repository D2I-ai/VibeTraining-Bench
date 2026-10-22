"""Thin CLI over the grading service HTTP API.

run_task.sh drives the service through this instead of curl, so JSON quoting
stays out of shell scripts. stdlib only, so it runs on the host python.

Invocable two ways, and both must keep working:

    python3 -m eval.grading.client health          # from the repo root
    python3 /abs/path/to/eval/grading/client.py health   # from ANY directory

The second form is what the agent uses from inside its container, where the
repo root is not the working directory and not on sys.path. That is why this
module imports nothing from its own package.

    python3 -m eval.grading.client register --run-id R --task aime__aime \
        --base-model Qwen/Qwen3-1.7B-Base --sandbox-root DIR --eval-dir DIR
    python3 -m eval.grading.client submit --token T --model-path final_model \
        --mode final --write-metrics --wait
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

# Connection settings are re-read from the environment rather than imported
# from .config, so that running this file directly as a script cannot fail on a
# relative import. Keep the defaults in step with config.py.
_HOST = os.environ.get("VTB_GRADING_HOST") or "127.0.0.1"
_PORT = os.environ.get("VTB_GRADING_PORT") or "5800"
DEFAULT_BASE = f"http://{'127.0.0.1' if _HOST in ('0.0.0.0', '') else _HOST}:{_PORT}"

# secrets.token_urlsafe(24) always renders as exactly 32 url-safe characters.
TOKEN_LENGTH = 32
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{%d}$" % TOKEN_LENGTH)


def _call(base: str, method: str, path: str, payload: Optional[dict] = None, timeout: int = 60) -> dict:
    url = f"{base.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = {"error": body.strip() or exc.reason}
        parsed["_http_status"] = exc.code
        return parsed
    except urllib.error.URLError as exc:
        return {"error": f"cannot reach grading service at {url}: {exc.reason}", "_http_status": 0}


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _failed(payload: dict) -> bool:
    status = payload.get("_http_status")
    return bool(payload.get("error")) or (status is not None and status >= 400) or status == 0


def _resolve_token(args: argparse.Namespace) -> str:
    """Token from a file if given, else from --token.

    --token-file exists because capturing a token through `TOKEN=$(... register)`
    is easy to get wrong: if the command fails, the shell variable silently ends
    up holding an error blob, and every later request then reports
    `403 unknown or expired token` while pointing nowhere near the real cause.
    A file also happens to be how run_task.sh hands the token to the agent.
    """
    token_file = getattr(args, "token_file", None)
    if token_file:
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(f"[grading] cannot read token file {token_file}: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        token = (getattr(args, "token", None) or "").strip()

    if not token:
        print("[grading] no token supplied: pass --token or --token-file", file=sys.stderr)
        sys.exit(1)
    if not TOKEN_RE.match(token):
        # Refuse to send obvious garbage, so the failure names the real problem
        # instead of surfacing as an authentication error from the service.
        preview = token if len(token) <= 60 else token[:57] + "..."
        print(
            f"[grading] {'--token-file contents' if token_file else '--token'} does not look like a token "
            f"({len(token)} chars; expected {TOKEN_LENGTH}). Got: {preview!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def cmd_health(args: argparse.Namespace) -> int:
    payload = _call(args.base, "GET", "/health", timeout=10)
    _emit(payload)
    return 1 if _failed(payload) else 0


def _admin_token(args: argparse.Namespace) -> str:
    """Read the operator secret that /register requires.

    Defaults to <state_dir>/admin.token, which is where the service writes it. An
    agent cannot reach that file: the service state directory is never bound into
    its container. Read here rather than accepted on the command line so the value
    never lands in a process list.
    """
    if args.admin_token_file:
        path = Path(args.admin_token_file).expanduser()
    elif os.environ.get("VTB_ADMIN_TOKEN_FILE"):
        path = Path(os.environ["VTB_ADMIN_TOKEN_FILE"]).expanduser()
    else:
        state = os.environ.get("VTB_GRADING_STATE")
        repo = Path(os.environ.get("VTB_REPO") or Path(__file__).resolve().parents[2]).expanduser()
        root = Path(os.environ.get("VTB_ROOT") or repo / "vtb-env").expanduser()
        path = Path(state).expanduser() if state else root / "grading_states" / "direct"
        path = path / "admin.token"
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SystemExit(
            f"[grading] cannot read the admin token at {path}: {exc}\n"
            "registration is operator-only; start the service first, or point"
            " --admin-token-file at the file it wrote."
        )
    if not token:
        raise SystemExit(f"[grading] the admin token at {path} is empty")
    return token


def cmd_register(args: argparse.Namespace) -> int:
    payload = _call(
        args.base,
        "POST",
        "/register",
        {
            "admin_token": _admin_token(args),
            "run_id": args.run_id,
            "task": args.task,
            "base_model": args.base_model,
            "sandbox_root": args.sandbox_root,
            "eval_dir": args.eval_dir,
            "role": args.role,
            "log_dir": args.log_dir,
            "disclosure_dir": args.disclosure_dir,
            "validation_source": args.validation_source,
            "editable_harness": args.editable_harness,
        },
    )
    if _failed(payload):
        _emit(payload)
        return 1
    token = payload["token"]
    if args.token_file:
        path = Path(args.token_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token, encoding="utf-8")
        os.chmod(path, 0o600)
        print(f"[grading] token written to {path}", file=sys.stderr, flush=True)
    # Bare token on stdout so shell callers can capture it directly.
    print(token)
    return 0


def cmd_unregister(args: argparse.Namespace) -> int:
    payload = _call(args.base, "POST", "/unregister", {"token": _resolve_token(args)})
    _emit(payload)
    return 1 if _failed(payload) else 0


def cmd_baseline(args: argparse.Namespace) -> int:
    token = _resolve_token(args)
    if args.from_results:
        # One tree, three facts: the score, the validation split's membership, and the
        # base model's transcripts for the agent to read. Adopting them separately is
        # how they end up disagreeing.
        payload = _call(
            args.base,
            "POST",
            "/baseline",
            {
                "admin_token": _admin_token(args),
                "token": token,
                "results_dir": args.from_results,
            },
            timeout=1800,
        )
    elif args.score is None:
        payload = _call(args.base, "GET", f"/baseline?token={urllib.parse.quote(token)}")
    else:
        # Writing a baseline rewrites every improvement measured against it, so the
        # service treats it as an operator action.
        payload = _call(
            args.base,
            "POST",
            "/baseline",
            {"admin_token": _admin_token(args), "token": token, "score": args.score},
        )
    _emit(payload)
    return 1 if _failed(payload) else 0


def cmd_queue(args: argparse.Namespace) -> int:
    token = _resolve_token(args)
    payload = _call(args.base, "GET", f"/queue?token={urllib.parse.quote(token)}")
    _emit(payload)
    return 1 if _failed(payload) else 0


def cmd_cancel(args: argparse.Namespace) -> int:
    """Stop a submitted evaluation and take the shared GPUs back.

    Blocks until the cards report free memory, because the answer is what the caller
    uses to decide it may start training.
    """
    payload = _call(
        args.base,
        "POST",
        "/cancel",
        {
            "token": _resolve_token(args),
            "job_id": args.job_id,
            "reason": args.reason,
            "wait": not args.no_wait,
        },
        timeout=max(60, args.wait_timeout),
    )
    _emit(payload)
    return 1 if _failed(payload) else 0


def cmd_sample(args: argparse.Namespace) -> int:
    """One evaluated sample's problem statement, transcript and agent counts."""
    token = _resolve_token(args)
    query = "&".join(
        f"{key}={urllib.parse.quote(value)}"
        for key, value in (
            ("token", token),
            ("job_id", args.job_id),
            ("sample_id", args.sample_id),
        )
    )
    payload = _call(args.base, "GET", f"/sample?{query}", timeout=60)
    _emit(payload)
    return 1 if _failed(payload) else 0


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _progress_line(job_id: str, state: str, prog: dict) -> str:
    """One compact human-readable line, suitable for reprinting on each poll."""
    parts = [f"[grading] {job_id[:8]} {state}"]
    label = prog.get("phase_label") or prog.get("phase")
    if label and state not in ("done", "failed", "cancelled"):
        parts.append(str(label))
    if prog.get("queue_position"):
        parts.append(f"queue #{prog['queue_position']}")
    done = prog.get("completions_observed")
    if done:
        total = prog.get("target_samples")
        pct = prog.get("percent")
        if total and pct is not None:
            parts.append(f"{done}/{total} ({pct}%)")
        elif total:
            parts.append(f"{done}/{total}")
        else:
            parts.append(f"{done} completions")
    if prog.get("elapsed_seconds") is not None:
        parts.append(f"elapsed {_fmt_duration(prog['elapsed_seconds'])}")
    if prog.get("eta_seconds") is not None:
        parts.append(f"eta ~{_fmt_duration(prog['eta_seconds'])} ({prog.get('eta_basis')})")
    return " | ".join(parts)


def _advice_line(advice: Optional[dict]) -> str:
    if not advice:
        return ""
    parts = [f"[grading] advice: {advice.get('message') or advice.get('kind')}"]
    if advice.get("eval_gpus"):
        parts.append(f"eval_gpus={','.join(str(g) for g in advice['eval_gpus'])}")
    if advice.get("required_mib"):
        parts.append(f"need {advice['required_mib']} MiB free each")
    if advice.get("retry_after_seconds"):
        parts.append(f"retry in ~{advice['retry_after_seconds']}s")
    return " | ".join(parts)


def cmd_submit(args: argparse.Namespace) -> int:
    token = _resolve_token(args)
    body: dict[str, Any] = {"token": token, "model_path": args.model_path}
    # Agent tokens may also send a positive --limit; the service caps it at the
    # configured validation upper bound. Operator tokens keep the historical override.
    if args.mode is not None:
        body["mode"] = args.mode
    if args.limit is not None:
        body["limit"] = args.limit
    if args.write_metrics:
        body["write_metrics"] = True
    if args.no_cache:
        # Bypass the result cache: the key is (task, mode, limit, model) and cannot
        # see service-side knobs like the thinking mode, so a mode sweep needs the
        # same key evaluated twice.
        body["no_cache"] = True
    if args.sample_ids:
        # Named samples instead of the split's first N. Only samples the service has
        # already established are inside the validation split are accepted.
        body["sample_ids"] = args.sample_ids

    payload = _call(args.base, "POST", "/submit", body)
    if _failed(payload):
        _emit(payload)
        return 1

    job_id = payload["job_id"]
    if not args.wait:
        _emit(payload)
        return 0

    result_path = f"/result?token={urllib.parse.quote(token)}&job_id={urllib.parse.quote(job_id)}"
    deadline = time.time() + args.wait_timeout
    last_line = ""
    last_advice = ""
    while time.time() < deadline:
        result = _call(args.base, "GET", result_path, timeout=30)
        if _failed(result):
            _emit(result)
            return 1
        state = result.get("state", "")
        line = _progress_line(job_id, state, result.get("progress") or {})
        advice = _advice_line(result.get("resource_advice"))
        if line != last_line:
            # Flush progress immediately so redirected logs remain usable for
            # monitoring while evaluation is running.
            print(line, file=sys.stderr, flush=True)
            last_line = line
        if advice and advice != last_advice:
            print(advice, file=sys.stderr, flush=True)
            last_advice = advice
        if state in ("done", "failed", "cancelled"):
            _emit(result)
            return 0 if state == "done" else 1
        time.sleep(args.poll)

    print(f"[grading] timed out after {args.wait_timeout}s waiting for job {job_id}", file=sys.stderr, flush=True)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VibeTrainBench grading service client")
    parser.add_argument("--base", default=DEFAULT_BASE, help=f"service base URL (default {DEFAULT_BASE})")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_token_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--token", default=None, help="run token (exactly 32 url-safe chars)")
        p.add_argument("--token-file", default=None, help="read the token from this file instead")

    sub.add_parser("health").set_defaults(func=cmd_health)

    p = sub.add_parser("queue")
    add_token_args(p)
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("register")
    p.add_argument("--run-id", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--sandbox-root", required=True, help="agent-writable root; submissions must resolve inside it")
    p.add_argument("--eval-dir", required=True, help="where submissions.json and metrics.json are written")
    p.add_argument(
        "--role",
        choices=["agent", "official"],
        default="official",
        help="agent tokens may only submit for validation and see only what disclosure allows",
    )
    p.add_argument(
        "--validation-source",
        default=None,
        help="the val source this run declares (e.g. split:150); "
        "the service refuses registration if it disagrees with its effective source for the task",
    )
    p.add_argument(
        "--editable-harness",
        default=None,
        help='"off" opts this run out of a benchmark-declared editable harness (a model-only ablation)',
    )
    p.add_argument(
        "--log-dir",
        default=None,
        help="keep each evaluation's inspect log here; refused if inside the sandbox",
    )
    p.add_argument(
        "--disclosure-dir",
        default=None,
        help="write each validation's full per-sample results here for the agent to "
             "read; refused if inside the sandbox, so mount it read-only instead",
    )
    p.add_argument(
        "--admin-token-file",
        default=None,
        help="operator secret; defaults to <state_dir>/admin.token",
    )
    p.add_argument("--token-file", default=None, help="also write the issued token here, mode 0600")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("unregister")
    add_token_args(p)
    p.set_defaults(func=cmd_unregister)

    p = sub.add_parser("submit")
    add_token_args(p)
    p.add_argument("--model-path", required=True)
    p.add_argument("--mode", choices=["val", "final"], default=None,
                   help="operator tokens only; an agent submission is always validation")
    p.add_argument("--limit", type=int, default=None,
                   help="agent validation cap or operator sample-count override")
    p.add_argument("--write-metrics", action="store_true", help="official run: also write metrics.json")
    p.add_argument("--no-cache", action="store_true", default=False,
                   help="bypass the result cache and force a fresh evaluation "
                        "(needed when service-side knobs like the thinking mode "
                        "changed but the cache key did not)")
    p.add_argument(
        "--sample-ids",
        nargs="+",
        action="extend",
        default=None,
        metavar="ID",
        help="evaluate these validation samples instead of the split's first N; "
             "only ids the service knows are inside the split are accepted. Repeatable, "
             "and accepts several ids per flag",
    )
    p.add_argument("--wait", action="store_true", help="poll until the job finishes")
    p.add_argument("--wait-timeout", type=int, default=36000)
    p.add_argument("--poll", type=int, default=15)
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser(
        "cancel",
        help="stop a submitted evaluation, keep what it graded, and release the GPUs",
    )
    add_token_args(p)
    p.add_argument("--job-id", required=True)
    p.add_argument(
        "--reason",
        required=True,
        help="why this evaluation is being given up, for the run's record",
    )
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="return without confirming the evaluation GPUs are free again",
    )
    p.add_argument("--wait-timeout", type=int, default=300)
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("sample", help="one evaluated sample's problem, transcript and counts")
    add_token_args(p)
    p.add_argument("--job-id", required=True)
    p.add_argument("--sample-id", required=True)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("baseline")
    add_token_args(p)
    p.add_argument("--score", type=float, default=None, help="omit to read the cached baseline")
    p.add_argument(
        "--from-results",
        default=None,
        metavar="BACKEND_OUTPUT",
        help="adopt a finished baseline validation: score it, record which samples the "
             "split contains, and keep its transcripts where runs can read them",
    )
    p.add_argument(
        "--admin-token-file",
        default=None,
        help="operator secret, needed only when writing; defaults to <state_dir>/admin.token",
    )
    p.set_defaults(func=cmd_baseline)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
