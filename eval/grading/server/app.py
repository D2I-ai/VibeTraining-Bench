"""HTTP front end and process entry for the grading service.

Handler routes requests onto the single GradingService instance; main()
builds the service and serves until signalled.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse
from .. import config, executor, valsplit

from .jobs import Conflict
from .service import GradingService


SERVICE: Optional[GradingService] = None


class Handler(BaseHTTPRequestHandler):
    server_version = "VTBGrading/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
        pass

    # ------------------------------------------------------------------ helpers

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8")) or {}
        except ValueError:
            raise ValueError("request body is not valid JSON")

    def _query(self) -> dict:
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    # ------------------------------------------------------------------ routes

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        svc = SERVICE
        assert svc is not None
        try:
            if path == "/health":
                # Reachable without the admin token so an agent can check the
                # service is up, but then it says only that. The resolved host
                # paths are how a caller would construct a sandbox_root to
                # register against, so they are operator-only.
                gpu_coordination = {
                    "mode": config.GPU_COORDINATION_MODE,
                    "eval_gpus": config.EVAL_GPUS,
                    "requires_training_gpu_release": config.GPU_COORDINATION_MODE == "shared",
                }
                if config.GPU_COORDINATION_MODE == "shared":
                    gpu_coordination["agent_advice"] = (
                        "Release training/serving GPU processes before validation and do CPU-only work while waiting."
                    )
                harbor_adapter = config.harbor_repo_status()
                harbor_adapter_public = {k: v for k, v in harbor_adapter.items() if k != "path"}
                payload = {
                    "status": "ok",
                    "eval_gpu": config.EVAL_GPU or None,
                    "eval_gpus": config.EVAL_GPUS,
                    "gpu_coordination": gpu_coordination,
                    "harbor_adapter": harbor_adapter_public,
                    "tasks": config.known_tasks(),
                    "queue": svc.queue_snapshot()["pending_count"],
                    "validation": {
                        **svc.val_source.to_json(),
                        "describes": svc.val_source.describe(),
                        "resolved_counts": svc._validation_counts(),
                        # Whether naming individual samples will be accepted. It
                        # unlocks only after one evaluation has covered a whole split,
                        # and without saying so the only way to find out is to submit
                        # and be refused -- which costs a round trip to learn something
                        # the service already knows.
                        "named_samples_available": sorted(svc.task_stats.tasks_with_val_ids()),
                    },
                    "agent_disclosure": sorted(svc.disclosure),
                    "agent_disclosure_limits": {
                        # Per-sample rows are complete: every sample an evaluation
                        # graded comes back. max_samples is the ceiling on what may
                        # be disclosed at all, not a page size, and an agent's limit
                        # is already capped at the validation split.
                        "per_sample_rows": "every sample the evaluation graded",
                        "max_samples": config.AGENT_DISCLOSURE_MAX_SAMPLES,
                        "max_chars": config.AGENT_DISCLOSURE_MAX_CHARS,
                        "detail_endpoint": "/sample?token=&job_id=&sample_id=",
                    },
                    # Shared cards need a way to be handed back that is not "take the
                    # memory anyway". The per-run count needs a token, so it comes
                    # back from /submit and /cancel; this is the policy.
                    "cancellation": {
                        "endpoint": "POST /cancel {token, job_id, reason}",
                        "max_agent_cancels": config.MAX_AGENT_CANCELS or None,
                        "keeps_partial_results": True,
                        "partial_results_comparable": False,
                    },
                }
                if self._query().get("admin_token"):
                    svc.require_admin(self._query().get("admin_token"))
                    payload.update(
                        {
                            "serve": {
                                "backend": config.SERVE_BACKEND,
                                "runtime": config.SERVE_RUNTIME,
                                "image": config.serve_image(),
                            },
                            "harbor_adapter": harbor_adapter,
                            "active_runs": len(svc.registry.active()),
                            "dataset_totals": svc.task_stats.snapshot(),
                            # Resolved, not the defaults: the whole point of
                            # keeping every path derived from VTB_ROOT is being
                            # able to see where this instance actually put things.
                            "paths": {
                                "vtb_root": str(config.VTB_ROOT),
                                "repo_root": str(config.REPO_ROOT),
                                "state_dir": str(config.STATE_DIR),
                                "hf_home": str(config.HF_HOME),
                            },
                        }
                    )
                self._send(200, payload)
            elif path == "/queue":
                self._send(200, svc.queue_snapshot(self._query().get("token")))
            elif path == "/result":
                params = self._query()
                self._send(200, svc.job(params.get("token", ""), params.get("job_id", "")))
            elif path == "/sample":
                params = self._query()
                self._send(
                    200,
                    svc.sample_detail(
                        params.get("token", ""),
                        params.get("job_id", ""),
                        params.get("sample_id", ""),
                    ),
                )
            elif path == "/baseline":
                ctx = svc.registry.lookup(self._query().get("token", ""))
                self._send(
                    200,
                    {
                        "task": ctx.task,
                        "base_model": ctx.base_model,
                        "baseline_score": svc.baselines.get(ctx.task, ctx.base_model),
                    },
                )
            else:
                self._send(404, {"error": f"no such endpoint: {path}"})
        except KeyError as exc:
            self._send(404, {"error": str(exc)})
        except PermissionError as exc:
            self._send(403, {"error": str(exc)})
        except ValueError as exc:
            self._send(400, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        svc = SERVICE
        assert svc is not None
        try:
            body = self._body()
            if path == "/register":
                svc.require_admin(body.get("admin_token"))
                missing = [k for k in ("run_id", "task", "base_model", "sandbox_root", "eval_dir") if not body.get(k)]
                if missing:
                    raise ValueError(f"missing fields: {', '.join(missing)}")
                # A log_dir under the agent's own sandbox would let a validation
                # submission's preserved log leak back to the agent -- it can read
                # anything beneath sandbox_root. Refuse that rather than trust the
                # caller, since the whole val/held-out split depends on the agent not
                # seeing per-sample outcomes. The full record is archived separately
                # under grading_state, which no sandbox_root covers.
                _sandbox_root = Path(str(body["sandbox_root"])).resolve()
                _log_dir = Path(str(body["log_dir"])).resolve() if body.get("log_dir") else None
                if _log_dir is not None and (
                    _log_dir == _sandbox_root or _sandbox_root in _log_dir.parents
                ):
                    raise ValueError(
                        "log_dir is inside the run sandbox: a path the agent can read "
                        "cannot hold preserved evaluation logs"
                    )
                # disclosure_dir is the opposite intent -- it exists to be read by the
                # agent -- but it still may not sit inside the sandbox, because there
                # the agent could also rewrite it. The run harness mounts it read-only.
                _disclosure_dir = (
                    Path(str(body["disclosure_dir"])).resolve() if body.get("disclosure_dir") else None
                )
                if _disclosure_dir is not None and (
                    _disclosure_dir == _sandbox_root or _sandbox_root in _disclosure_dir.parents
                ):
                    raise ValueError(
                        "disclosure_dir is inside the run sandbox: mount it read-only "
                        "instead so the agent cannot rewrite its own evaluation results"
                    )
                # A run whose recipe declares a validation source is scored against
                # that source; the service has its own. If the two disagree, the
                # run's baseline reference would be measured on a different split
                # than its candidates -- refuse the pairing at registration rather
                # than discover it in the numbers.
                _validation_source = body.get("validation_source")
                _task = str(body["task"])
                if svc._validation_source(_task).kind == "json" and _task != "sierra-research__tau3-bench":
                    raise ValueError("VTB_VAL_SOURCE=json: is only valid for sierra-research__tau3-bench")
                if _validation_source:
                    # Grammar-check first, with the same parser the service used at
                    # startup, so a typo'd declaration fails registration with the
                    # same error rather than mid-evaluation.
                    declared = valsplit.parse_val_source(str(_validation_source))
                    if declared.kind == "json" and _task != "sierra-research__tau3-bench":
                        raise ValueError("VTB_VAL_SOURCE=json: is only valid for sierra-research__tau3-bench")
                    # Compare against the run-task's effective source (operator env
                    # override, else the benchmark's own validation_source, else the
                    # default) -- not the raw env value, which an unset VTB_VAL_SOURCE
                    # would leave empty while the service actually runs its default.
                    effective_source = svc._validation_source(_task).raw
                    if declared.raw != effective_source:
                        raise ValueError(
                            f"run declares val_source={declared.raw!r}, but this service runs "
                            f"val_source={effective_source!r} for {body['task']}; restart the service "
                            "with a matching source or fix the recipe's validation block"
                        )
                # A run may opt out of its benchmark's editable harness (a model-only
                # ablation). Only "off" carries meaning; anything else is a typo.
                _editable_harness = body.get("editable_harness")
                if _editable_harness is not None and str(_editable_harness) != "off":
                    raise ValueError('editable_harness must be "off" when given')
                token = svc.registry.register(
                    run_id=str(body["run_id"]),
                    task=str(body["task"]),
                    base_model=str(body["base_model"]),
                    sandbox_root=Path(str(body["sandbox_root"])),
                    eval_dir=Path(str(body["eval_dir"])),
                    role=str(body.get("role") or "official"),
                    log_dir=_log_dir,
                    disclosure_dir=_disclosure_dir,
                    validation_source=str(_validation_source) if _validation_source else None,
                    editable_harness=str(_editable_harness) if _editable_harness else None,
                )
                svc.log(f"[register] run={body['run_id']} task={body['task']} role={body.get('role') or 'official'}")
                self._send(200, {"token": token, "run_id": body["run_id"], "task": body["task"]})
            elif path == "/unregister":
                svc.registry.unregister(str(body.get("token", "")))
                self._send(200, {"status": "unregistered"})
            elif path == "/submit":
                if not body.get("model_path"):
                    raise ValueError("missing model_path")
                result = svc.submit(
                    token=str(body.get("token", "")),
                    model_path=str(body["model_path"]),
                    mode=str(body["mode"]) if body.get("mode") is not None else None,
                    limit=int(body["limit"]) if body.get("limit") is not None else None,
                    write_metrics=bool(body.get("write_metrics")),
                    sample_ids=list(body["sample_ids"]) if body.get("sample_ids") else None,
                    no_cache=bool(body.get("no_cache")),
                )
                self._send(202, result)
            elif path == "/cancel":
                if not body.get("job_id"):
                    raise ValueError("missing job_id")
                self._send(
                    200,
                    svc.cancel(
                        token=str(body.get("token", "")),
                        job_id=str(body["job_id"]),
                        reason=str(body.get("reason") or ""),
                        # Waiting is the default: the answer is what the caller uses to
                        # decide the cards are its own.
                        wait=bool(body.get("wait", True)),
                    ),
                )
            elif path == "/baseline":
                # Rewriting a baseline rewrites every improvement computed against
                # it, so this is operator-only too.
                svc.require_admin(body.get("admin_token"))
                ctx = svc.registry.lookup(str(body.get("token", "")))
                # Adopting a finished baseline run derives the score from the tree
                # instead of taking one on faith, and records the split membership and
                # the readable transcripts from that same tree.
                if body.get("results_dir"):
                    self._send(200, svc.import_baseline(ctx, str(body["results_dir"])))
                    return
                score = body.get("score")
                if not isinstance(score, (int, float)):
                    raise ValueError("score must be a number")
                svc.baselines.put(ctx.task, ctx.base_model, float(score))
                self._send(200, {"task": ctx.task, "base_model": ctx.base_model, "baseline_score": float(score)})
            else:
                self._send(404, {"error": f"no such endpoint: {path}"})
        except KeyError as exc:
            self._send(403, {"error": str(exc)})
        except PermissionError as exc:
            self._send(403, {"error": str(exc)})
        except Conflict as exc:
            self._send(409, {"error": str(exc)})
        except ValueError as exc:
            self._send(400, {"error": str(exc)})


def main() -> None:
    global SERVICE
    parser = argparse.ArgumentParser(description="VibeTrainBench shared grading service")
    parser.add_argument("--host", default=config.BIND_HOST)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--pid-file", default=None, help="write the service pid here for a clean stop")
    parser.add_argument(
        "--serve-backend",
        choices=("sglang",),
        default=config.SERVE_BACKEND,
        help="Local serving backend for backend-managed benchmarks.",
    )
    parser.add_argument(
        "--serve-runtime",
        choices=("docker", "host"),
        default=config.SERVE_RUNTIME,
        help="Where the serving backend runs. docker by default; host uses "
        "whatever sglang is installed on this machine.",
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=config.SERVE_PORT,
        help="Fixed local serving port; 0 means choose a free port per job.",
    )
    parser.add_argument(
        "--serve-image",
        default=None,
        help="Override the serving image.",
    )
    parser.add_argument(
        "--serve-tool-call-parser",
        default=None,
        help="Tool-call parser for the local serving endpoint.",
    )
    parser.add_argument(
        "--serve-reasoning-parser",
        default=None,
        help="Reasoning parser for backends that support one, such as sglang.",
    )
    args = parser.parse_args()

    config.SERVE_BACKEND = args.serve_backend
    config.SERVE_RUNTIME = args.serve_runtime
    config.SERVE_PORT = args.serve_port
    if args.serve_image:
        config.SERVE_SGLANG_IMAGE = args.serve_image
    if args.serve_tool_call_parser:
        config.SERVE_TOOL_CALL_PARSER = args.serve_tool_call_parser
        config.TOOL_CALL_PARSER = args.serve_tool_call_parser
    if args.serve_reasoning_parser is not None:
        config.SERVE_REASONING_PARSER = args.serve_reasoning_parser

    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    config.BASELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    SERVICE = GradingService()
    SERVICE.log(
        f"grading service starting on {args.host}:{args.port} | "
        f"eval_gpus={config.EVAL_GPU_SPEC or 'inherit'} | "
        f"gpu_mode={config.GPU_COORDINATION_MODE} | "
        f"serve={config.SERVE_BACKEND}/{config.SERVE_RUNTIME}:{config.serve_image()}:{config.SERVE_PORT or 'auto'} | "
        f"parser={config.SERVE_TOOL_CALL_PARSER}{('/' + config.SERVE_REASONING_PARSER) if config.SERVE_REASONING_PARSER else ''} | "
        f"repo={config.REPO_ROOT} | state={config.STATE_DIR} | pid={os.getpid()}"
    )
    if config.SERVE_RUNTIME == "docker" and not executor.serve_image_present():
        # Said at startup rather than at the first submission: a missing image is an
        # operator's problem, and finding out at grading time costs a whole run.
        SERVICE.log(
            f"WARNING: serving image not present: {config.serve_image()} "
            f"-- build it with `bash containers/build_image.sh {config.SERVE_BACKEND}`"
        )
    if not config.EVAL_GPUS:
        SERVICE.log("WARNING: VTB_EVAL_GPUS/VTB_EVAL_GPU is unset -- evaluations will inherit whatever card is visible")
    harbor_adapter = config.harbor_repo_status()
    if harbor_adapter["required_by"] and not harbor_adapter["ready"]:
        SERVICE.log(
            "WARNING: VTB_HARBOR_REPO is unset or does not contain adapters/ -- "
            f"adapter-local Harbor benchmarks {harbor_adapter['required_by']} will fail before evaluation. "
            "Set VTB_HARBOR_REPO to a Harbor checkout to run Tau adapter grading."
        )

    # A task with no ladder entry retries with an unchanged token budget, so all
    # nine attempts are the same attempt. Nothing failed and nothing said so,
    # which is the worst shape for a benchmark added later to fail in.
    unladdered = config.tasks_without_ladder()
    if unladdered:
        SERVICE.log(
            f"WARNING: no TOKEN_LADDER entry for {unladdered} -- their retries will repeat the "
            "same token budget instead of lowering it, which is the documented remedy for the "
            "GPU-memory failures the ladder exists for. Add one in eval/grading/config.py."
        )

    # A SIGKILL, OOM kill or host reboot cannot be handled in-process, so look
    # for the debris of a previous unclean exit before accepting work.
    for action in executor.sweep_stale_overlays(SERVICE.log):
        SERVICE.log(f"[sweep] {action}")

    pid_file = Path(args.pid_file) if args.pid_file else config.STATE_DIR / "service.pid"
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    def _shutdown(signum, _frame):
        # An evaluation child runs in its own session and would survive this
        # process, holding the evaluation card; the fuse-overlayfs mount would
        # also leak, because a default SIGTERM never runs `finally` blocks.
        name = signal.Signals(signum).name
        SERVICE.log(f"received {name}, shutting down")
        for action in executor.cleanup_active(SERVICE.log):
            SERVICE.log(f"[cleanup] {action}")
        interrupted = SERVICE.mark_interrupted()
        if interrupted:
            SERVICE.log(f"[cleanup] marked in-flight job {interrupted} as failed")
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        # Belt and braces: covers an exit that did not come through a signal.
        executor.cleanup_active(SERVICE.log)
        pid_file.unlink(missing_ok=True)
        SERVICE.log("grading service stopped")
