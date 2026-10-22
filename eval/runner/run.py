"""The entry point every benchmark's evaluate.py shim calls.

Generic middleware. It serves the model, hands the task set to an execution backend,
collects the trial tree and writes the metrics contract. What the benchmarks *are* lives in
benchmarks/registry.py; nothing benchmark-specific belongs here.

The argument surface is the same one the grading service passes to every benchmark in
this repo -- --model-path / --api-base / --limit / --json-output-file -- so a SWE
benchmark is scored through exactly the machinery a multiple-choice one is, and the
service needs no special case.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from . import model_server, runtime_paths, task_backends, trial_results
from .tau3_assets import TAU3_BENCHMARK, Tau3AssetError, validate_export


def build_parser(spec, default_limit: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{spec.name}: {spec.notes or 'a backend-managed benchmark'}"
    )
    parser.add_argument("--model-path", type=str, default=None, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--limit",
        type=int,
        default=default_limit,
        help="Number of tasks to evaluate. -1 means the whole set.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        # VTB_API_BASE, not VTB_SWE_API_BASE: this entry point runs every benchmark in
        # benchmarks/, and the endpoint it talks to has never been SWE-specific. The old
        # name still works so existing shell history does not break.
        default=os.environ.get("VTB_API_BASE") or os.environ.get("VTB_SWE_API_BASE") or None,
        help="An already-running OpenAI-compatible endpoint. When set, no model is served here.",
    )
    parser.add_argument(
        "--serve-backend",
        choices=("sglang",),
        default=os.environ.get("VTB_SERVE_BACKEND") or "sglang",
        help="Local serving backend to launch when --model-path is used and --api-base is absent.",
    )
    parser.add_argument(
        "--serve-runtime",
        choices=("docker", "host"),
        default=os.environ.get("VTB_SERVE_RUNTIME") or None,
        help="Local serving runtime. Defaults to docker; host runs "
        "whatever sglang is installed here.",
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=int(os.environ.get("VTB_SERVE_PORT") or 0),
        help="Fixed local serving port when --model-path is used; 0 chooses a free port.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The name the endpoint advertises. Defaults to the model directory's name.",
    )
    parser.add_argument("--only", nargs="*", default=None, help="Explicit task ids, for debugging.")
    parser.add_argument("--n-concurrent", type=int, default=runtime_paths.eval_concurrency())
    parser.add_argument("--step-limit", type=int, default=None, help="Cap on agent steps per task.")
    parser.add_argument(
        "--agent-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Backend agent kwarg to pass through; repeatable. Overrides benchmark.json.",
    )
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--json-output-file", type=str, default=None)
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Override the spec's task-container backend (docker).",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="templates/",
        help="Accepted for contract compatibility; these tasks carry their own environments.",
    )
    return parser


def _agent_kwarg_overrides(items: list[str]) -> dict:
    """Parse repeatable KEY=VALUE agent kwarg overrides from the CLI."""
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--agent-kwarg must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--agent-kwarg has an empty key: {item!r}")
        overrides[key] = value
    return overrides


def _require_environment(spec) -> None:
    """Fail before serving a model when a benchmark's required configuration is absent."""
    missing = [name for name in spec.required_env if not os.environ.get(name, "").strip()]
    if missing:
        names = ", ".join(missing)
        raise SystemExit(
            f"{spec.name}: missing required environment variable(s): {names}. "
            "Set them before starting this benchmark."
        )


class _Task:
    """Just enough of a task to name it and to collect its trial."""

    __slots__ = ("task_id", "language")

    def __init__(self, task_id: str, language: Optional[str] = None) -> None:
        self.task_id = task_id
        self.language = language


def _read_manifest(source: Path) -> list:
    """Task ids from a Harbor dataset manifest, in the order it lists them.

    Read here rather than through a per-benchmark helper: `manifest.json` with a
    `tasks` array of objects carrying `task_id` is the Harbor dataset format, so one
    reader serves any in-repo task tree. Order is the manifest's own, which is what
    makes a `--limit N` subset the same set from run to run rather than whatever a
    directory listing happened to return.
    """
    manifest = source / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(
            f"no manifest at {manifest}. An in-repo task tree needs one to name its "
            "tasks; a registry dataset does not, and should set dataset= instead."
        )
    payload = json.loads(manifest.read_text(encoding="utf-8")) or {}
    tasks = []
    for entry in payload.get("tasks") or []:
        task_id = entry.get("task_id") if isinstance(entry, dict) else None
        if task_id:
            tasks.append(_Task(str(task_id), (entry.get("language") if isinstance(entry, dict) else None)))
    if not tasks:
        raise SystemExit(f"{manifest} lists no tasks")
    return tasks


def _expected_tasks(spec, task_dir: Path, args, task_source: Optional[Path] = None) -> tuple:
    """(task objects, explicit ids, dataset total) when a task tree is locally available.

    Returns (None, None, None) for a registry dataset that the backend resolves itself,
    except when ids were named: the backend can select by name even for a dataset it
    downloads, so those are passed straight through. That is what lets an evaluation be
    pointed at particular samples instead of a prefix -- without it, the samples at the
    back of a long run are unreachable however often it is re-run.
    """
    if task_source is None:
        if spec.task_ids:
            # A curated registry-backed subset declares its IDs in benchmark.json; treat
            # them like a local manifest so the backend gets explicit include filters in
            # declared order and cannot substitute a sampled subset.
            dataset = [_Task(task_id) for task_id in spec.task_ids]
        elif not spec.tasks_subdir:
            return (None, list(args.only), None) if args.only else (None, None, None)
        else:
            task_source = task_dir / spec.tasks_subdir
            dataset = _read_manifest(task_source)
    else:
        dataset = _read_manifest(task_source)
    if args.only:
        wanted = set(args.only)
        selected = [t for t in dataset if t.task_id in wanted]
        missing = wanted - {t.task_id for t in selected}
        if missing:
            raise SystemExit(f"no such task(s): {', '.join(sorted(missing))}")
    elif args.limit is not None and args.limit >= 0:
        selected = dataset[: args.limit]
    else:
        selected = dataset
    if spec.task_ids and not selected:
        raise SystemExit("a curated task_ids benchmark cannot run an empty subset")
    return selected, [t.task_id for t in selected], len(dataset)


def main(*, benchmark: str, task_dir: Path, argv: Optional[list] = None) -> None:
    # The benchmarks/ dir, for registry. A shim's task_dir sits one level below it;
    # the generic entry point below has no shim dir, so fall back on this file's own
    # location.
    benchmarks_dir = task_dir.parents[0]
    if not (benchmarks_dir / "registry.py").is_file():
        benchmarks_dir = Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path.insert(0, str(benchmarks_dir))
    import registry  # noqa: PLC0415

    spec = registry.get(benchmark)
    args = build_parser(spec, spec.default_limit).parse_args(argv)
    _require_environment(spec)
    if args.serve_runtime:
        os.environ["VTB_SERVE_RUNTIME"] = args.serve_runtime
    agent_kwarg_overrides = _agent_kwarg_overrides(args.agent_kwarg)
    editable_agent = os.environ.get("VTB_EDITABLE_HARNESS_AGENT")
    editable_root = os.environ.get("VTB_EDITABLE_HARNESS_ROOT")
    if bool(editable_agent) != bool(editable_root):
        raise SystemExit("editable harness root and import path must be provided together")
    if editable_agent:
        # Validated against the declaring benchmark at submit time; here just check the
        # shape and that the snapshot carries the module the path names. The declaration
        # cannot be re-read off the eval task's spec: a run registered on benchmark A may
        # be routed to evaluate on benchmark B, and the harness belongs to A's workspace.
        if not editable_agent.startswith("harness.") or ":" not in editable_agent:
            raise SystemExit(f"unsupported editable harness import path: {editable_agent!r}")
        root = Path(editable_root).resolve()
        module_name = editable_agent.split(":", 1)[0].split(".", 1)[1]
        if not (root / "harness" / (module_name + ".py")).is_file():
            raise SystemExit(f"editable harness is incomplete: {root / 'harness'}")
    if args.environment or agent_kwarg_overrides or editable_agent:
        updated = {**spec.__dict__}
        if args.environment:
            updated["environment"] = args.environment
        if agent_kwarg_overrides:
            updated["agent_kwargs"] = {**spec.agent_kwargs, **agent_kwarg_overrides}
        if editable_agent:
            # The server set this from an immutable submission snapshot. It applies
            # equally when a TB2.1 validation is routed to the TBLite benchmark.
            updated["agent"] = editable_agent
        spec = type(spec)(**updated)

    label = spec.name
    work_dir = Path(args.work_dir) if args.work_dir else runtime_paths.SCRATCH_DIR / f"{spec.name}_run"
    work_dir.mkdir(parents=True, exist_ok=True)

    version = task_backends.version(spec.backend)
    if spec.name == TAU3_BENCHMARK:
        task_source = runtime_paths.TAU3_TASK_SPECS_DIR
        try:
            validate_export(task_source)
        except Tau3AssetError as exc:
            raise SystemExit(
                f"{exc}\nPrepare the Tau3 environment first:\n"
                "  bash scripts/prepare/prepare_tau3.sh all"
            ) from exc
    else:
        task_source = (task_dir / spec.tasks_subdir) if spec.tasks_subdir else None
    tasks, task_ids, dataset_total = _expected_tasks(spec, task_dir, args, task_source)
    source_label = str(task_source) if task_source is not None else (spec.dataset or spec.tasks_subdir)
    print(
        f"[{label}] {spec.backend} {version} | {source_label} | "
        f"env={spec.environment} | agent={spec.agent}"
        f"{' (non-agentic)' if not spec.agentic else ''}",
        flush=True,
    )

    served_name = args.served_model_name or (
        Path(args.model_path).name if args.model_path else "model"
    )
    server = None
    api_base = args.api_base
    if api_base:
        print(f"[{label}] using existing model endpoint {api_base}", flush=True)
    elif args.model_path:
        server = model_server.ModelServer(
            args.model_path,
            served_name=served_name,
            label=label,
            backend=args.serve_backend,
            fixed_port=args.serve_port or None,
        )
        api_base = server.start()
    else:
        raise SystemExit("pass --model-path to serve a model, or --api-base to use an endpoint")

    # A loopback endpoint is not reachable from inside a rootless docker container, so
    # it is rewritten to this host's own address. Said out loud rather than done
    # quietly: if the endpoint is only listening on 127.0.0.1 the rewrite will not save
    # it, and the run will fail on every request with nothing pointing at why.
    reachable = task_backends.host_endpoint(api_base, spec.environment)
    if reachable != api_base:
        print(
            f"[{label}] rewrote the endpoint for a {spec.environment} container: "
            f"{api_base} -> {reachable}. The model must be listening on 0.0.0.0.",
            flush=True,
        )
        api_base = reachable

    output_dir = work_dir / "backend_output"
    subset = None if args.limit is not None and args.limit < 0 else args.limit
    command = task_backends.build_harbor(
        spec,
        task_source=task_source,
        task_ids=task_ids,
        output_dir=output_dir,
        model=f"openai/{served_name}",
        n_concurrent=args.n_concurrent,
        step_limit=args.step_limit,
        limit=subset,
    )
    (work_dir / "backend_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")

    try:
        returncode = task_backends.run(
            command,
            log_path=work_dir / f"{spec.backend}.log",
            env=task_backends.child_environment(
                api_base=api_base,
                api_key=model_server.api_key(),
            ),
        )
    finally:
        if server is not None:
            server.stop()

    # Did the step ceiling land? It travels as a backend kwarg that the harness is free to
    # ignore without a word, so the request is compared against what a trajectory says was
    # resolved. Reported rather than enforced: a run whose limit was dropped still produced
    # trials, and the number is the thing that makes them readable.
    if args.step_limit is not None:
        applied = task_backends.effective_step_limit(output_dir)
        if applied is None:
            print(f"[step-limit] requested {args.step_limit}; no trajectory recorded one", flush=True)
        elif applied != args.step_limit:
            print(
                f"[step-limit] requested {args.step_limit} but the agent resolved "
                f"{applied} -- the limit did not reach the harness",
                flush=True,
            )
        else:
            print(f"[step-limit] in effect: {applied} steps per task", flush=True)

    # task_ids without task objects is the named-sample case on a registry dataset: the
    # ids are all that was resolved, and they are still the expected set. Passing them
    # keeps a sample the backend skipped visible as ungraded instead of dropping it and
    # scoring the rest.
    records = trial_results.collect(output_dir, expected=tasks or task_ids)
    result = trial_results.summarise(
        records,
        benchmark=spec.name,
        agent=spec.agent,
        model=f"openai/{served_name}",
        backend=spec.backend,
        backend_version=version,
        environment=spec.environment,
    )
    result["backend_returncode"] = returncode
    trial_results.write_outputs(
        result,
        metrics_path=Path(args.json_output_file) if args.json_output_file else None,
        work_dir=work_dir,
        dataset_total=dataset_total,
    )

    graded, attempted = result["graded"], result["attempted"]
    print(
        f"accuracy={result['accuracy']:.4f} solved={result['solved']}/{graded} graded "
        f"({result['ungraded']} ungraded) partial_mean={result['partial_mean']}",
        flush=True,
    )
    print(f"[{label}] work dir: {work_dir}", flush=True)
    if graded == 0 and (attempted or returncode != 0):
        # Every task ungraded is an infrastructure verdict, not a model one, and it
        # must not read as a score of zero. This also covers the backend crashing
        # before it attempted any task (returncode != 0, attempted == 0), e.g. a
        # transient task-registry outage during package resolution.
        raise SystemExit(
            f"no task reached a verdict (backend rc={returncode}); "
            f"{work_dir / 'records.json'} / {work_dir / (spec.backend + '.log')} says why"
        )


if __name__ == "__main__":
    # Generic entry point for a benchmark with no shim directory of its own. The
    # benchmark must already be visible to benchmarks/registry.py, normally via
    # benchmarks/<name>/benchmark.json. scripts/prepare/register_harbor_benchmark.sh creates that
    # registration for raw Harbor Hub owner/name datasets.
    #
    #   python3 -m eval.runner.run --benchmark aime__aime --api-base http://.../v1 ...
    #
    # --benchmark is peeled off here; everything else is the same argument surface a
    # shim's evaluate.py accepts.
    _peel = argparse.ArgumentParser(add_help=False)
    _peel.add_argument("--benchmark", required=True)
    _known, _rest = _peel.parse_known_args()
    main(
        benchmark=_known.benchmark,
        # No task tree exists for a hub dataset; _expected_tasks ignores the path
        # whenever the spec resolves its tasks from the registry.
        task_dir=Path.cwd() / "benchmarks" / _known.benchmark.replace("/", "__"),
        argv=_rest,
    )
