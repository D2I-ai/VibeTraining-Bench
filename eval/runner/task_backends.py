"""Harbor command builders and process environment.

This module is deliberately about *task execution backends*: it translates a
BenchmarkSpec into a `harbor run` command, chooses the task-container backend,
rewrites host endpoints for containers, and prepares the environment passed to the
backend process.

It does not know what any benchmark is; benchmark identity and source live in
benchmarks/<name>/benchmark.json and are loaded by benchmarks/registry.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from . import runtime_paths


def _binary(backend: str = "harbor") -> str:
    """Which executable to run, preferring the one installed for this checkout.

    scripts/prepare/install_backend_tools.sh puts harbor under vtb-env/tools/bin at the
    containers/requirements.txt pins. That wins over whatever the name resolves to
    on PATH, so a score cannot come from a version this repo never chose. An explicit
    VTB_HARBOR still overrides both.
    """
    override = os.environ.get("VTB_HARBOR")
    if override:
        return override
    local = runtime_paths.TOOLS_BIN / backend
    if local.is_file():
        return str(local)
    return backend


def safe_alias(value: str) -> str:
    """Filesystem-safe benchmark alias, matching scripts/prepare/register_harbor_benchmark.sh."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value.strip().strip("/"))


# What to pass to `harbor run --env`. That flag takes either one of harbor's built-in
# environment type names or a `module.path:Class` import path -- it replaces
# --environment-import-path, which harbor now warns is deprecated.
#
# docker is the only wired-up backend. This map is the extension point: a new
# task-container environment is one entry here, resolved to the name or import path
# harbor should run. A known-but-unwired name gets None and a refusal at backend_arg.
_BACKENDS: dict[str, Optional[str]] = {
    # Harbor's own default, and the one this repo runs. It builds a task's environment/
    # Dockerfile when task.toml names no image, and it honours a task's no-network
    # declaration -- two things a task set is entitled to assume.
    #
    # Which daemon it talks to is runtime_paths.docker_host()'s decision, and whether
    # container root is host root follows from that. See host_endpoint below for the one
    # thing a rootless daemon changes that this code has to care about.
    "docker": "docker",
}

_UNIMPLEMENTED: dict[str, str] = {}


def _harbor_repo_path() -> Optional[Path]:
    value = os.environ.get("VTB_HARBOR_REPO")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not (path / "adapters").is_dir():
        raise BackendUnavailable(
            f"VTB_HARBOR_REPO={path} does not look like a Harbor checkout: "
            "expected an adapters/ directory."
        )
    return path


def _requires_harbor_repo(agent: str) -> bool:
    return agent.startswith("adapters.") or ".adapters." in agent


def _require_harbor_repo_for_agent(agent: str) -> None:
    if _requires_harbor_repo(agent) and _harbor_repo_path() is None:
        raise BackendUnavailable(
            f"agent {agent!r} is an adapter-local Harbor import path; set "
            "VTB_HARBOR_REPO to the Harbor checkout that contains adapters/."
        )


class BackendToolMissing(RuntimeError):
    pass


class BackendUnavailable(RuntimeError):
    pass


def backend_arg(environment: str) -> str:
    """What `--env` should be given for a named backend, or a refusal that explains."""
    if environment not in _BACKENDS:
        raise BackendUnavailable(
            f"unknown environment {environment!r}; known: {', '.join(sorted(_BACKENDS))}"
        )
    value = _BACKENDS[environment]
    if value is None:
        raise BackendUnavailable(_UNIMPLEMENTED.get(environment, f"{environment} is not implemented"))
    return value


def is_docker(environment: str) -> bool:
    return environment == "docker"


def host_endpoint(api_base: Optional[str], environment: str) -> Optional[str]:
    """Rewrite a loopback endpoint into one a task container can actually reach.

    The model is served on the host, and under rootless docker a container cannot reach
    the host's loopback at all: slirp4netns runs with --disable-host-loopback, so
    127.0.0.1 is the container itself, 172.17.0.1 does not answer, and
    `--add-host=host.docker.internal:host-gateway` resolves but the address refuses the
    connection. Measured, all three. What does work is the host's own LAN address, so
    that is what gets substituted -- and the endpoint has to be listening on 0.0.0.0
    rather than 127.0.0.1 for it to help.

    A rootful daemon reaches the host's loopback over the bridge, so the rewrite is not
    needed there; it is done for every docker environment anyway, because a LAN address
    the model is listening on works under both and this code cannot tell which daemon a
    given task container came from.
    """
    if not api_base or not is_docker(environment):
        return api_base
    for loopback in ("127.0.0.1", "localhost", "[::1]"):
        if loopback in api_base:
            host_ip = _host_address()
            if host_ip is None:
                raise BackendUnavailable(
                    "the model endpoint is on loopback and no host address was found to "
                    "reach it from a rootless container. Serve the model on 0.0.0.0 and "
                    "pass --api-base with this host's address explicitly."
                )
            return api_base.replace(loopback, host_ip)
    return api_base


def _host_address() -> Optional[str]:
    """This host's own address on the network a container can route to.

    Found by asking the kernel which source address it would use to reach outside,
    rather than by parsing `hostname -I` -- that returns every address including ones a
    container cannot use, and the first is not reliably the routable one.
    """
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # No packet is sent; connect on UDP only fixes the local address.
            sock.connect(("8.8.8.8", 53))
            return sock.getsockname()[0]
    except OSError:
        return None


def version(backend: str = "harbor") -> str:
    """The backend tool's version, and a readable failure when it is not installed."""
    binary = _binary(backend)
    if shutil.which(binary) is None and not Path(binary).is_file():
        raise BackendToolMissing(
            f"{binary} is not available; install it with: bash scripts/prepare/install_backend_tools.sh"
        )
    out = subprocess.run([binary, "--version"], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return "unknown"
    text = (out.stdout or out.stderr).strip()
    return text.splitlines()[0] if text else "unknown"


# Which harnesses take a step ceiling through mini-swe-agent's own config rather than as a
# backend kwarg. See step_limit_args.
_MINI_SWE_AGENTS = frozenset({"mini-swe-agent"})


def _is_mini_swe_agent(agent) -> bool:
    if not isinstance(agent, str):
        return False
    if agent in _MINI_SWE_AGENTS:
        return True
    # An editable-harness import path ("harness.<module>:<Class>") built on the
    # mini-swe-agent shim speaks the same config language as the registered name.
    return agent.startswith("harness.") and agent.split(":", 1)[0].rsplit(".", 1)[-1] == "mini_swe_agent"


def _mini_swe_config_kwarg(agent, step_limit: Optional[int], declared_config: Optional[dict]) -> list:
    """The mini-swe-agent `config` kwarg: one mapping carrying the benchmark's declared
    config layer (e.g. fixed sampling under model.model_kwargs) and the step ceiling.

    One kwarg, not two: repeated keys on the harbor command line overwrite each other,
    so a declared config and a step limit sent separately would leave only one standing.
    The shim yaml-dumps the mapping into the task container as a -c layer over the
    builtin mini config.
    """
    if not _is_mini_swe_agent(agent):
        return []
    payload: dict = dict(declared_config or {})
    if step_limit is not None:
        agent_cfg = payload.setdefault("agent", {})
        if not isinstance(agent_cfg, dict):
            raise BackendUnavailable(
                f"{agent}: declared config 'agent' must be a mapping, got {type(agent_cfg).__name__}"
            )
        agent_cfg["step_limit"] = int(step_limit)
    if not payload:
        return []
    return ["--agent-kwarg", f"config={json.dumps(payload, separators=(',', ':'))}"]


def _pop_declared_config(spec, agent_kwargs: dict) -> Optional[dict]:
    """Extract the benchmark's declared mini-swe-agent config layer from agent_kwargs.

    Popped because the `config` kwarg must not reach the command line twice: a step
    limit shares the one key with it, and repeated keys overwrite each other on the
    harbor CLI, so the later one would silently drop the earlier. No-op for other
    agent families, where `config` means nothing special.
    """
    if not _is_mini_swe_agent(spec.agent) or "config" not in agent_kwargs:
        return None
    raw_config = agent_kwargs.pop("config")
    try:
        declared_config = json.loads(raw_config)
    except ValueError as exc:
        raise BackendUnavailable(f"{spec.name}: agent_kwargs.config is not valid JSON: {exc}")
    if not isinstance(declared_config, dict):
        raise BackendUnavailable(f"{spec.name}: agent_kwargs.config must decode to a mapping")
    return declared_config


def step_limit_args(agent, step_limit: Optional[int]) -> list:
    """How to spell a per-task step ceiling for one harness.

    `--agent-kwarg step_limit=N` is the obvious form and it is silently useless for
    mini-swe-agent: harbor's adapter declares only cost_limit, and everything else lands in
    AgentBase(**kwargs), which stores nothing and complains about nothing. The retry
    ladder's lowered limits went out on every attempt and never reached an agent.

    For mini-swe-agent the limit is `agent.step_limit` in its own config, and harbor's
    `config` kwarg reaches it: the adapter yaml-dumps the mapping into the task container
    and appends it to the `-c` spec list after the builtin `-c mini`, so a one-key mapping
    layers over the packaged config. A whole config file would replace it instead --
    reverting system_template and friends to the dataclass defaults and quietly changing
    what is being measured -- which is why only the single key is sent.

    mini-swe-agent's cost_limit cannot replace a step ceiling: a self-served model has
    no price in litellm, so instance_cost stays 0.0 and the cost limit never trips.
    An explicit step limit bounds repeated actions and context growth.
    """
    if step_limit is None:
        return []
    mini = _mini_swe_config_kwarg(agent, step_limit, None)
    if mini:
        return mini
    return ["--agent-kwarg", f"step_limit={step_limit}"]


def effective_step_limit(output_dir: Path) -> Optional[int]:
    """The step ceiling a finished mini-swe-agent run actually applied, if it recorded one.

    Read back from a trajectory's own `info.config`, which is the agent's resolved config
    rather than what the command line asked for. Worth the read precisely because the kwarg
    route fails silently: a limit that was accepted and dropped looks identical to one that
    took, until the trials come back 900 steps long.
    """
    for path in sorted(output_dir.glob("*/*/agent/mini-swe-agent.trajectory.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                info = (json.load(handle) or {}).get("info") or {}
        except (OSError, ValueError):
            continue
        limit = ((info.get("config") or {}).get("agent") or {}).get("step_limit")
        if isinstance(limit, int):
            return limit
    return None


def build_harbor(
    spec,
    *,
    task_source: Optional[Path],
    task_ids: Optional[list],
    output_dir: Path,
    model: str,
    n_concurrent: int,
    step_limit: Optional[int] = None,
    limit: Optional[int] = None,
) -> list:
    """`harbor run` for one spec.

    Task source is either `-d owner/name` (registry; harbor downloads and caches) or
    `-p <dir>` (a task tree in this repo). Selection within a set is repeated
    `--include-task-name`; repeating `--path` per task is rejected. The output
    directory is `-o/--jobs-dir`, not --output-path.

    `limit` is honoured only for a registry dataset with no explicit ids, where the task
    ids are not known until harbor has downloaded it: harbor's own `--n-tasks` picks the
    subset. For an in-repo tree the caller has already read the manifest and passes
    explicit ids, which is reproducible in a way a count is not -- and a caller that
    names ids for a registry dataset means those samples, not the first N of them, so
    the two selections are never sent together.
    """
    command = [_binary("harbor"), "run", "--jobs-dir", str(output_dir)]
    # A validated local source wins even when benchmark metadata retains its upstream
    # dataset name. Tau3 relies on this to consume the prefetched patched export instead
    # of redownloading an unpatched registry package into an implicit user cache.
    if task_source is not None:
        command += ["--path", str(task_source)]
    elif spec.dataset:
        command += ["--dataset", spec.dataset]
    else:
        raise BackendUnavailable(f"{spec.name}: no dataset or local task source")
    for task_id in task_ids or []:
        command += ["--include-task-name", task_id]
        if task_source is None and spec.dataset and "/" not in task_id:
            # A registry dataset's task names carry their org: harbor filters on
            # PackageTaskId.get_name(), which is "<org>/<name>", so a bare id matches
            # nothing and harbor refuses the whole filter. The ids this repo passes
            # around are bare -- that is the form the trial trees and the scoring report
            # -- so the org-qualified shape is added alongside rather than guessed at,
            # and the filter is a union, so a task matching both is still run once.
            command += ["--include-task-name", f"*/{task_id}"]
    if task_source is None and spec.dataset and limit is not None and limit > 0 and not task_ids:
        command += ["--n-tasks", str(limit)]

    _require_harbor_repo_for_agent(spec.agent)

    command += [
        # spec.agent names the harness outright. The spec's agentic flag is declarative
        # metadata for records and readers; the command line uses the selected Harbor
        # agent or custom import path directly.
        "--agent", spec.agent,
        "--model", model,
        "--n-concurrent", str(n_concurrent),
        # Non-interactive: a registry dataset otherwise prompts before downloading.
        "--yes",
    ]
    command += ["--env", backend_arg(spec.environment)]

    # A benchmark-declared config layer (e.g. fixed sampling for mini-swe-agent) and a
    # ladder step limit share the one `config` kwarg -- merge them here and keep the raw
    # key out of the generic pass-through below. Keyed on the agent family (the config
    # channel's only consumer), never on a benchmark.
    agent_kwargs = dict(spec.agent_kwargs or {})
    declared_config = _pop_declared_config(spec, agent_kwargs)
    config_kwarg = _mini_swe_config_kwarg(spec.agent, step_limit, declared_config)
    command += config_kwarg if config_kwarg else step_limit_args(spec.agent, step_limit)
    for key, value in agent_kwargs.items():
        command += ["--agent-kwarg", f"{key}={value}"]
    command += list(spec.extra_args)
    return command


def child_environment(*, api_base: Optional[str], api_key: Optional[str]) -> dict:
    """Environment for the backend process.

    The endpoint goes to harbor, not into the task container: the backend passes on
    what the installed agent needs. Nothing about the model is injected into the task
    container by us.
    """
    env = dict(os.environ)
    env.setdefault("XDG_CACHE_HOME", str(runtime_paths.CACHE_DIR))
    # Custom agents may be named by import path on the harbor command line, so this
    # checkout root must be importable from the backend process. An editable harness
    # comes only from the grading service's immutable snapshot; put its package parent
    # first so the declared `harness.<module>:<Class>` resolves to that submission.
    repo_root = str(Path(__file__).resolve().parents[2])
    python_paths = []
    editable_root = env.get("VTB_EDITABLE_HARNESS_ROOT")
    if editable_root:
        root = Path(editable_root).resolve()
        agent = env.get("VTB_EDITABLE_HARNESS_AGENT", "")
        module_name = agent.split(":", 1)[0].split(".", 1)[1] if agent.startswith("harness.") and ":" in agent else ""
        if not module_name or not (root / "harness" / (module_name + ".py")).is_file():
            raise BackendUnavailable(f"editable harness root is incomplete: {root}")
        python_paths.append(str(root))
    python_paths.append(repo_root)
    harbor_repo = _harbor_repo_path()
    if harbor_repo is not None:
        python_paths.append(str(harbor_repo))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    docker_host = runtime_paths.docker_host()
    if docker_host:
        # Assigned, not defaulted: an inherited DOCKER_HOST is one of the inputs
        # docker_host() already weighed, and a repo-level override has to be able to
        # replace it rather than lose to it.
        env["DOCKER_HOST"] = docker_host
    if api_base:
        env["OPENAI_BASE_URL"] = api_base
        env["OPENAI_API_BASE"] = api_base
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    return env


def run(command: list, *, log_path: Path, env: dict, cwd: Optional[Path] = None) -> int:
    """Run a backend command, streaming its output to a log and to stdout.

    The command line is written to the log first: a run's own log should say exactly
    what was executed, so a failure can be reproduced by hand.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()
