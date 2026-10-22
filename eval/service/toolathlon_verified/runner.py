"""
Task runner for the Toolathlon-VERIFIED evaluation service.

The code that runs inside the container is *official verified + overlay +
per-host secrets*: paths.py stages the external read-only official checkout
(VTB_TOOLATHLON_VERIFIED_REPO) into a throwaway tree, layers this package's
overlay/ on top, then the per-host secrets (VTB_TOOLATHLON_VERIFIED_SECRETS).
The runner drives that staged tree; it never touches the official checkout.

Runs one finalpool task end-to-end in a per-task container, replicating the
official phased containerized flow (scripts/run_single_containerized.sh):

  1. preprocess   (container)  scripts.decoupled.container_preprocess
  2. stash        (host)       hide evaluator/ground-truth from the agent
  3. agent loop   (container)  scripts.containerized.container_agent
  4. restore      (host)       put the evaluator back, drop agent-forged results
  5. eval         (container)  scripts.decoupled.container_eval

The runner uses the official bind mounts for persistent OAuth state, the
pinned Notion OpenAPI patch, and per-task dumps. k8s tasks (k8s-*) additionally
get the official containerized-mode setup: --network host +
/var/run/docker.sock, so their in-container preprocess can `kind create`
against the host daemon and the generated kubeconfig's 127.0.0.1 API endpoint
resolves.

Each task gets its own docker network and container names derived from the
job id + task hash, so any number of tasks (and jobs) run concurrently without
touching each other. No postgres is needed -- verified tasks are
self-contained.

Outcome semantics, aligned with VTB's three-state contract (same line the gym
adapter draws, by CAUSE not by who happened to run):
  pass=True/False   the evaluator reached a verdict
  pass=None         infrastructure trouble (agent crashed before reporting a
                    status, docker failed) -- ungraded upstream, never a 0.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

try:  # package import (uvicorn eval.service.toolathlon_verified.app:app)
    from . import paths
except ImportError:  # script import (run from the service dir)
    import paths

IMAGE = os.environ.get(
    "TOOLATHLON_VERIFIED_IMAGE", "lockon0927/toolathlon-task-image:1016beta@sha256:4d04fe4e0a6fdb4946f51bb05120cb44a0eef980231c11252f93b62897afcb9f")

# Wall clock per task. Official default budget is max_steps=200 single-turn
# steps; observed healthy tasks finish well inside 30 min, but verified tasks
# are longer-horizon than gym's, so the default here is 45 min with the whole
# five-phase flow (preprocess + agent + eval + container startup) inside it.
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", "2700"))

CONTAINER_PREFIX = "tvt-ag-"
NETWORK_PREFIX = "tvt-net-"

# The staged tree (official + overlay + secrets) is built once and reused for
# every task; it is what all the PROJECT_ROOT-relative reads below resolve
# against, and the source for the per-task docker cp / bind mounts.
_PROJECT_ROOT: Optional[Path] = None


def _project_root() -> Path:
    """The staged (official + overlay + secrets) tree; the container's source."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = paths.build_staged_root()
    return _PROJECT_ROOT


# The code directories the official script copies into the container
# (FILES_TO_COPY in run_single_containerized.sh). configs/.mcp-auth is copied
# separately so the host copy is never touched by container-side token
# rotation.
FILES_TO_COPY = [
    "configs",
    "scripts",
    "utils",
    "main.py",
    "deployment/k8s",
    "deployment/canvas/logs",
    "global_preparation/check_installation.py",
    "local_binary/github-mcp-server",
    # The image bakes local_servers/, but at an older revision than configs/:
    # 1016beta ships arxiv-latex-mcp 0.1.0, whose pyproject declares no
    # [project.scripts], and no wandb-mcp-server at all -- while
    # configs/mcp_servers/{arxiv-latex-mcp,wandb}.yaml spawn them as
    # `uv run ... arxiv-latex-mcp` / `uv run ... wandb_mcp_server`. Both die
    # with "Failed to spawn: ... No such file or directory (os error 2)", the
    # MCP handshake then times out and the task aborts with "Only N servers
    # connected, expected N+1" -- seen on latex-prompt-box, wandb-best-score
    # and wandb-shortest-length, which grade as ungraded crashes rather than 0.
    # Shipping the repo's own copies (arxiv-latex-mcp 0.2.2 and wandb-mcp-server,
    # both declaring the entry points the configs call) realigns code with
    # config. Only these two are listed: the other baked servers work as-is and
    # carry prebuilt .venvs plus ~314MB that would be copied per task for
    # nothing. docker cp merges into the existing directory, so the baked .venv
    # survives and uv re-syncs it against the newer pyproject on first run
    # (measured cold start: arxiv-latex 2s, wandb 10s, both inside their
    # client_session_timeout_seconds of 10s and 60s).
    "local_servers/arxiv-latex-mcp",
    "local_servers/wandb-mcp-server",
]

# TaskStatus values that mean the MODEL finished (or exhausted) the task.
# Everything else on the agent path (interrupted / no status line at all)
# is treated as infrastructure -> ungraded + retried by the service layer.
_AGENT_DONE_STATUSES = frozenset({
    "success", "failed", "timeout", "max_turns_reached", "interrupted",
})
# Of those, this subset is a MODEL failure that must grade 0 (never retried);
# "interrupted" is environmental and stays eligible for a retry.
_MODEL_DONE_STATUSES = frozenset({
    "success", "failed", "timeout", "max_turns_reached",
})

# Every localhost:<port> the official finalpool code dials (grep over
# tasks/finalpool): canvas 10001/20001 (token_key_session's canvas_domain and
# its per-task other_key.py overrides), woocommerce 10003, poste
# 10005/1143/1587/2525, canvas redis 6379 (unpublished, lives inside the
# canvas container), webhook 18080 (argparse help text), k8s NodePorts
# 30123/30124 (k8s tasks run --network host anyway). The forwarder probes
# each upstream against the gateway and only binds the reachable ones, so
# the rest keep the official "connection refused" semantics instead of
# gaining a fake listener.
_LOCALHOST_FWD_PORTS = (10001, 20001, 10003, 10005, 1143, 1587, 2525,
                        6379, 18080, 30123, 30124)


def _run(cmd: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _docker(*args: str, timeout: Optional[int] = None, check: bool = True) -> subprocess.CompletedProcess:
    res = _run(["docker", *args], timeout=timeout)
    if check and res.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:2])} failed: {res.stderr.strip()[:300]}")
    return res


def _k8s_cluster_name(task: str) -> Optional[str]:
    """The kind cluster name a k8s-* task creates, derived so the runner can
    tear it down even when the task aborts before its own `stop` ever runs.
    Each task's scripts/<one>.sh sets cluster_name="cluster-XXX" and appends
    instance_suffix from configs/ports_config.yaml (e.g. "-inst-alpha")."""
    project_root = _project_root()
    scripts_dir = project_root / "tasks" / "finalpool" / task / "scripts"
    base = None
    for sh in sorted(scripts_dir.glob("*.sh")):
        m = re.search(r'cluster_name="([^"]+)"', sh.read_text())
        if m:
            base = m.group(1)
            break
    if not base:
        return None
    suffix = ""
    ports = project_root / "configs" / "ports_config.yaml"
    if ports.is_file():
        m = re.search(r'^instance_suffix:\s*"?([^"\n]*)"?', ports.read_text(),
                      re.M)
        if m:
            suffix = m.group(1).strip()
    return f"{base}{suffix}"


def _py(*args: str, timeout: Optional[int] = None, check: bool = True,
        cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a host-side helper module (task_artifact_guard) from the repo root."""
    import sys
    res = subprocess.run([sys.executable, "-m", *args], capture_output=True,
                         text=True, timeout=timeout, cwd=cwd)
    if check and res.returncode != 0:
        raise RuntimeError(
            f"Python module {' '.join(args[:2])} failed: {(res.stderr or res.stdout).strip()[:300]}")
    return res


def _start_localhost_forwarder(agent_c: str, mcp_host: str,
                                logs: List[str]) -> None:
    """Relay the container's loopback to the docker0 gateway (best-effort).

    Official preprocess/eval code dials http://localhost:<port> straight
    from per-task other_key.py / setup scripts -- strings the
    TOOLATHLON_MCP_HOST remap never rewrites (its consumption points are
    task text, token sessions, canvas api_client and email configs). The
    official containerized run hides this because its task container runs
    --network host; VTB's per-task bridge containers die on it
    ("Connection refused ... localhost:10001", canvas-art-quiz et al.,
    17 PREPROCESS_FAILED in formal 20260922-002126). utils/
    localhost_forward.py -- shipped into the container with the rest of
    utils/ via FILES_TO_COPY -- restores the localhost view; it probes each
    port against mcp_host first, so host-unpublished ports are skipped. A
    missing forwarder file only logs: it degrades to the old behaviour,
    never fails the task.
    """
    fwd = "/workspace/utils/localhost_forward.py"
    if _docker("exec", agent_c, "test", "-f", fwd,
               check=False).returncode != 0:
        logs.append(f"== localhost forwarder absent ({fwd}); "
                    "hardcoded-localhost preprocess may fail ==")
        return
    _docker("exec", "-d", agent_c, "bash", "-c",
            f"TOOLATHLON_MCP_HOST={mcp_host} python3 {fwd} "
            f"{' '.join(str(p) for p in _LOCALHOST_FWD_PORTS)} "
            "> /run/localhost_forward.log 2>&1")
    time.sleep(1.0)  # let it probe+bind before preprocess dials localhost
    r = _docker("exec", agent_c, "cat", "/run/localhost_forward.log",
                check=False)
    logs.append(f"== localhost forwarder: "
                f"{(r.stdout or r.stderr or '').strip()} ==")


def check_prerequisites() -> Dict:
    problems = []
    warnings = []
    if _run(["docker", "image", "inspect", IMAGE]).returncode != 0:
        problems.append(f"image {IMAGE} not found")
    else:
        # The official image ships the runtime (uv, node, mcp servers) but no
        # project code -- the runner docker-cp's it in per task.
        r = _run(["docker", "run", "--rm", "--entrypoint", "/bin/bash", IMAGE,
                  "-c", "which uv && which node"])
        if r.returncode != 0:
            problems.append("image missing uv/node runtime")
    # Resolve the staged tree (official + overlay + secrets) the tasks will
    # actually run; a missing/invalid VTB_TOOLATHLON_VERIFIED_REPO surfaces
    # here as a problem instead of at the first task.
    try:
        project_root = _project_root()
    except RuntimeError as exc:
        problems.append(str(exc))
        return {"ok": False, "problems": problems, "warnings": warnings}
    if not (project_root / "tasks" / "finalpool").is_dir():
        problems.append(f"tasks/finalpool missing under {project_root}")
    # Google Cloud tasks require a real service-account key generated by the
    # upstream setup. All eight are outside the curated 88-task benchmark, so
    # report a missing key as a warning for explicit --only runs. Supply this
    # per-host secret through VTB_TOOLATHLON_VERIFIED_SECRETS.
    if not (project_root / "configs" / "gcp-service_account.keys.json").is_file():
        warnings.append(
            "configs/gcp-service_account.keys.json missing (provide via "
            "VTB_TOOLATHLON_VERIFIED_SECRETS): the google-cloud server cannot "
            "start; all 8 google-cloud tasks sit outside the 88-task benchmark, "
            "so only explicit --only runs of them would fail")
    return {"ok": not problems, "problems": problems, "warnings": warnings}


def _eval_config(model_cfg: Dict, max_steps: int) -> Dict:
    """The per-job eval config: official formal_run_v0.json shape, provider
    pinned to `unified` (OpenAI-compatible endpoint via env) and the model
    pointed at the request's name. The user-side model is the same endpoint --
    single-turn tasks do not consult an external user model."""
    return {
        "global_task_config": {
            "max_turns": 50,
            "max_steps_under_single_turn_mode": max_steps,
            "dump_path": "./dumps",
            "direct_to_dumps": True,
        },
        "mcp": {"server_config_path": "configs/mcp_servers"},
        "agent": {
            "model": {"short_name": model_cfg.get("MODEL_NAME", "final_model"),
                      "provider": "unified"},
            "generation": {
                "max_tokens": int(model_cfg.get("TOOLATHLON_MAX_TOKENS", 8192)),
                "temperature": float(model_cfg.get("TOOLATHLON_TEMPERATURE", 1.0)),
                "top_p": float(model_cfg.get("TOOLATHLON_TOP_P", 1.0)),
                # Unknown Generation fields are preserved as OpenAI request
                # parameters by the official harness.
                "top_k": int(model_cfg.get("TOOLATHLON_TOP_K", -1)),
                "repetition_penalty": float(
                    model_cfg.get("TOOLATHLON_REPETITION_PENALTY", 1.0)
                ),
            },
            "tool": {"tool_choice": "auto", "parallel_tool_calls": True,
                     "max_inner_turns": 2000},
        },
        "user": {
            "model": {"short_name": model_cfg.get("MODEL_NAME", "final_model"),
                      "provider": "unified"},
            "generation": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 1024},
        },
    }


def _parse_agent_status(agent_stdout: str) -> Optional[str]:
    """container_agent prints 'Agent phase finished with status: <value>'."""
    m = re.search(r"Agent phase finished with status:\s*(\w+)", agent_stdout)
    return m.group(1).lower() if m else None


def _classify_outcome(agent_status: Optional[str],
                      agent_stdout: str = "") -> str:
    """Map the agent phase's TaskStatus onto the gym adapter's vocabulary.

      claimed_done  the model terminated the task itself (success or failed)
      gave_up       the model ran out of steps/turns without finishing
      agent_crash   the agent process died or hit an exception -- infra

    The trap specific to verified: task_agent.py's except branch maps ANY
    agent-loop exception onto TaskStatus.FAILED (L1049), so a dead model
    endpoint (after the provider's own 10 retries) reports the SAME status as
    "the model tried and did not accomplish the task". The except branch's
    one distinctive stdout line ("Error when running agent - ...", printed
    because this runner always passes --debug) is the line between infra
    (ungraded + retried) and a model failure (graded 0):

      "Failed to get agent response within N inner steps"  MODEL failure
        (task_agent L822: result still None after max_inner_steps tries --
        the model produced no usable response; graded 0, never retried)
      anything else ("Failed to get response after N retries, error:
      Connection error.", ...)  infrastructure (ungraded, retried)
    """
    if "Error when running agent" in agent_stdout:
        if "Failed to get agent response within" in agent_stdout:
            return "gave_up"  # model failure: graded 0, not retried
        return "agent_crash"
    if agent_status is None:
        return "agent_crash"
    if agent_status in ("success", "failed"):
        return "claimed_done"
    if agent_status in ("timeout", "max_turns_reached"):
        return "gave_up"
    return "agent_crash"  # interrupted / unknown -> infra, retried upstream


def run_task_isolated(task: str, model_cfg: Dict, job_id: str,
                      max_steps: int = 200,
                      output_root: str = "eval_output") -> Dict:
    """Run one task with its own network + container. Never raises."""
    tag = f"{job_id[:8]}-{hashlib.md5(task.encode()).hexdigest()[:8]}"
    agent_c = f"{CONTAINER_PREFIX}{tag}"
    net = f"{NETWORK_PREFIX}{tag}"
    task_rel = f"finalpool/{task}"
    container_task_path = f"/workspace/tasks/{task_rel}"
    output_dir = Path(output_root) / job_id / task
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = time.time() + TASK_TIMEOUT
    logs: List[str] = []

    def _remain() -> int:
        return max(60, int(deadline - time.time()))

    def _result(passed, status, details, agent_outcome=None):
        (output_dir / "run.log").write_text(
            stdout_all + "\n".join(logs), encoding="utf-8", errors="replace")
        return {"task": task, "pass": passed, "status": status,
                "details": details, "agent_outcome": agent_outcome,
                "duration_sec": round(time.time() - started, 1),
                "output_dir": str(output_dir)}

    # filled in as the phases run
    stdout_all = ""

    # The staged tree (official + overlay + secrets); source for every cp below.
    project_root = _project_root()

    # k8s tasks replicate the official containerized mode (run_single_containerized.sh
    # L248-294): --network host so the kubeconfig's `server: 127.0.0.1:<port>` written
    # by in-container `kind create` resolves to the HOST daemon that actually runs the
    # kind node, plus the docker.sock mount kind itself talks to. The task image ships
    # kind/kubectl/helm (kind v0.20.0, kubectl v1.34.1) and each task's
    # preprocess builds and tears down its own named cluster; cluster names are
    # per-task and the kind API port is host-random, so concurrent k8s tasks stay
    # collision-free without the per-task bridge the other tasks keep.
    is_k8s = task.startswith("k8s-")

    try:
        # Idempotent against leftovers of a crashed earlier attempt.
        _docker("rm", "-f", agent_c, check=False)
        if not is_k8s:
            _docker("network", "rm", net, check=False)
            # Transient docker address-pool exhaustion ("all predefined address
            # pools have been fully subnetted"): the default 172.17-31/192.168
            # pool is shared with other workloads on this host, so under
            # concurrency it can momentarily have no free block. Back off and
            # retry so a finishing task's `network rm` frees one, instead of
            # hard-failing the task as ERROR.
            for _attempt in range(6):
                _res = _docker("network", "create", net, check=False)
                if _res.returncode == 0:
                    break
                logs.append(f"== network create retry {_attempt + 1}: "
                            f"{(_res.stderr or _res.stdout or '').strip()[-200:]} ==")
                _docker("network", "rm", net, check=False)
                time.sleep(3 * (_attempt + 1))
            else:
                return _result(None, "ERROR",
                               "docker network create failed after retries "
                               f"(address-pool exhaustion): {net}")

        env_args = [
            "-e", f"TOOLATHLON_OPENAI_BASE_URL={model_cfg['MODEL_API_URL']}",
            "-e", f"TOOLATHLON_OPENAI_API_KEY={model_cfg.get('MODEL_API_KEY') or 'fake-key'}",
        ]
        # Optional upstream provider adapter; normal grading remains provider=unified.
        tool_base = os.environ.get("VTB_TOOL_MODEL_BASE_URL")
        if tool_base:
            env_args += ["-e", f"VTB_TOOL_MODEL_BASE_URL={tool_base}"]
        _think = model_cfg.get("TOOLATHLON_ENABLE_THINKING") or os.environ.get("TOOLATHLON_ENABLE_THINKING")
        if _think:
            env_args += ["-e", f"TOOLATHLON_ENABLE_THINKING={_think}"]
        model_retries = (model_cfg.get("TOOLATHLON_MODEL_RETRIES")
                         or os.environ.get("TOOLATHLON_MODEL_RETRIES"))
        if model_retries:
            env_args += ["-e", f"TOOLATHLON_MODEL_RETRIES={model_retries}"]
        # Split-MCP host remap (utils/mcp_host.py): task containers live on
        # per-task bridge networks, so "localhost" app services (poste IMAP/
        # SMTP, woocommerce, canvas -- all bound 0.0.0.0 on the host) are only
        # reachable via the docker0 gateway. The framework rewrites localhost
        # -> TOOLATHLON_MCP_HOST at every consumption point (token sessions,
        # email configs, task text). Same trick the official decoupled script
        # uses (run_single_decoupled.sh L264). VTB_TOOLATHLON_MCP_HOST=""
        # disables the remap (upstream single-machine semantics).
        mcp_host = os.environ.get("VTB_TOOLATHLON_MCP_HOST", "172.17.0.1")
        if mcp_host:
            env_args += ["-e", f"TOOLATHLON_MCP_HOST={mcp_host}"]
        # Overlay utils files absolute-import "from utils.xxx" (official
        # data_structures does too). uv-run entrypoints get /workspace on
        # sys.path via cwd, but task preprocess helper scripts spawned as a
        # plain "python <script>" (canvas-homework-grader-python's
        # extract_student_ids.py) only get the script dir plus their
        # hand-added /workspace/utils on sys.path, so the overlay's
        # "from utils.mcp_host import remap_localhost" inside
        # app_specific/canvas/api_client.py dies with "No module named
        # 'utils'" -- a second layer the old localhost Connection-refused
        # used to hide (88-task run: same task died 8x on Connection
        # refused before ever reaching this import). PYTHONPATH=/workspace
        # puts the project root on every python subprocess's path, which is
        # exactly what the official uv-run entrypoints already see.
        env_args += ["-e", "PYTHONPATH=/workspace"]

        # Official bind mounts (run_single_containerized.sh L305-338):
        # OAuth refreshes and task dumps persist on the host, while the pinned
        # Notion OpenAPI schema replaces the image's bundled schema.
        mcp_auth = project_root / "configs" / ".mcp-auth"
        mcp_auth.mkdir(exist_ok=True)
        mount_args: List[str] = [
            "-v", f"{mcp_auth}:/workspace/configs/.mcp-auth",
            "-v", f"{output_dir}:/workspace/dumps",
        ]
        notion_patch = (project_root / "configs" / "notion-mcp-patches"
                        / "notion-openapi.json")
        if notion_patch.is_file():
            mount_args += [
                "-v", f"{notion_patch}:/workspace/node_modules/"
                "@notionhq/notion-mcp-server/scripts/notion-openapi.json:ro"]

        net_args = (["--network", "host",
                     "-v", "/var/run/docker.sock:/var/run/docker.sock"]
                    if is_k8s else ["--network", net])
        _docker("create", "--name", agent_c, *net_args, *env_args,
                *mount_args, "-w", "/workspace", IMAGE,
                "sleep", str(TASK_TIMEOUT + 600))
        _docker("start", agent_c)
        for _ in range(60):
            if _docker("exec", agent_c, "true", check=False).returncode == 0:
                break
            time.sleep(1)

        # --- project code -------------------------------------------------
        for item in FILES_TO_COPY:
            src = project_root / item
            if not src.exists():
                continue
            if src.is_dir():
                _docker("exec", agent_c, "mkdir", "-p", f"/workspace/{item}")
                if item == "configs" and (project_root / "configs/.mcp-auth").is_dir():
                    # cp the directory contents EXCLUDING .mcp-auth (it goes in
                    # separately below), matching the official staging logic.
                    stage = Path(tempfile.mkdtemp())
                    try:
                        for child in src.iterdir():
                            if child.name == ".mcp-auth":
                                continue
                            if child.is_dir():
                                shutil.copytree(child, stage / child.name,
                                                dirs_exist_ok=True)
                            else:
                                shutil.copy2(child, stage / child.name)
                        # Docker cp with SRC/ nests the source directory in the
                        # destination; use SRC/. to copy only its contents.
                        _docker("cp", f"{stage}/.", f"{agent_c}:/workspace/{item}")
                    finally:
                        shutil.rmtree(stage, ignore_errors=True)
                else:
                    # SRC/. = copy contents (see the staging branch note).
                    _docker("cp", f"{src}/.", f"{agent_c}:/workspace/{item}")
            else:
                # Docker cp does not create parent directories for a single-file
                # copy; create the destination parent before copying.
                _docker("exec", agent_c, "mkdir", "-p",
                        f"/workspace/{Path(item).parent}")
                _docker("cp", str(src), f"{agent_c}:/workspace/{item}")

        # --- task directory (rm first: docker cp merges directories) ------
        _docker("exec", agent_c, "rm", "-rf", "--", container_task_path)
        _docker("exec", agent_c, "mkdir", "-p", "/workspace/tasks/finalpool")
        _docker("cp", f"{project_root / 'tasks' / task_rel}",
                f"{agent_c}:/workspace/tasks/finalpool/")

        # --- gcp oauth keys for gmail/calendar MCP (official copy_config) -
        _docker("exec", agent_c, "bash", "-c",
                "for dir in ~/.gmail-mcp ~/.calendar-mcp; do mkdir -p $dir && "
                "cp ./configs/gcp-oauth.keys.json $dir/ 2>/dev/null; "
                "cp ./configs/google_credentials.json $dir/credentials.json 2>/dev/null; done",
                check=False)

        # --- loopback relay for hardcoded-localhost official code --------
        # k8s tasks run --network host: their localhost IS the host, and
        # the remap-env gate (mcp_host empty) keeps upstream semantics.
        if not is_k8s and mcp_host:
            _start_localhost_forwarder(agent_c, mcp_host, logs)

        # --- per-job eval config ------------------------------------------
        cfg_path = output_dir / "service_eval_config.json"
        cfg_path.write_text(json.dumps(_eval_config(model_cfg, max_steps), indent=2))
        _docker("cp", str(cfg_path), f"{agent_c}:/workspace/service_eval_config.json")

        def _phase(args: List[str]) -> subprocess.CompletedProcess:
            # The container name goes in here: without it, "docker exec uv run ..."
            # treats the FIRST phase argument (uv) as the container.
            return _docker("exec", agent_c, *args,
                           timeout=_remain(), check=False)

        def _mktemp_bundle() -> str:
            r = _docker("exec", agent_c, "mktemp",
                        "/run/toolathlon-task-bundle.XXXXXX.json")
            return r.stdout.strip()

        # --- phase 1: preprocess ------------------------------------------
        p1_bundle = _mktemp_bundle()
        res = _phase(["uv", "run", "python3", "-m",
                      "scripts.decoupled.container_preprocess",
                      "--eval_config", "/workspace/service_eval_config.json",
                      "--task_dir", task_rel,
                      "--max_steps_under_single_turn_mode", str(max_steps),
                      "--model_short_name", model_cfg.get("MODEL_NAME", "final_model"),
                      "--provider", "unified",
                      "--bundle_file", p1_bundle, "--debug"])
        logs.append(f"== preprocess (exit {res.returncode}) ==")
        if res.returncode != 0:
            logs.append((res.stdout or "") + (res.stderr or ""))
            return _result(None, "PREPROCESS_FAILED",
                           f"preprocess exited {res.returncode}: "
                           f"{(res.stderr or res.stdout or '')[-300:]}")

        # Trusted bundle comes off-container before the model runs.
        host_bundle = output_dir / "task_bundle.json"
        _docker("cp", f"{agent_c}:{p1_bundle}", str(host_bundle))
        _docker("exec", agent_c, "rm", "-f", "--", p1_bundle, check=False)
        try:
            bundle = json.loads(host_bundle.read_text())
            task_root = (bundle.get("container_paths") or {}).get("task_root")
            assert bundle.get("schema_version") == 2 and task_root \
                and str(task_root).startswith("/workspace/"), "bad bundle"
        except Exception as e:
            return _result(None, "BAD_BUNDLE", f"preprocess bundle invalid: {e}")

        # --- phase 2: hide the evaluator from the agent -------------------
        stash_root = Path(tempfile.mkdtemp(prefix="tvt-stash-"))
        try:
            r = _py("scripts.containerized.task_artifact_guard", "stash",
                    "--runtime", "docker", "--container", agent_c,
                    "--task-path", container_task_path,
                    "--stash-root", str(stash_root),
                    timeout=_remain(), check=False, cwd=str(project_root))
            if r.returncode != 0:
                logs.append(f"== stash failed ==\n{(r.stderr or r.stdout or '')[-500:]}")
                return _result(None, "STASH_FAILED",
                               "could not hide evaluator artifacts; agent not started")
            # The stash CLI prints the artifact directory (manifest + payload)
            # on stdout. Restore's --stash-dir must refer to that directory, not
            # its --stash-root parent, to satisfy artifact-directory validation
            # and restore the evaluator files before scoring.
            stash_dir = ""
            for line in (r.stdout or "").splitlines():
                if line.strip().startswith("/"):
                    stash_dir = line.strip()
            logs.append("== evaluator hidden ==")

            # --- phase 3: the agent itself --------------------------------
            a_bundle = _mktemp_bundle()
            _docker("cp", str(host_bundle), f"{agent_c}:{a_bundle}")
            try:
                ares = _phase(["uv", "run", "python3", "-m",
                               "scripts.containerized.container_agent",
                               "--bundle_file", a_bundle, "--debug"])
            finally:
                _docker("exec", agent_c, "rm", "-f", "--", a_bundle, check=False)
            agent_exit = ares.returncode
            agent_status = _parse_agent_status(ares.stdout or "")
            stdout_all = (ares.stdout or "") + "\n== STDERR ==\n" + (ares.stderr or "")
            logs.append(f"== agent (exit {agent_exit}, status {agent_status}) ==")

            # --- phase 4: restore evaluator, drop forged results ----------
            r = _py("scripts.containerized.task_artifact_guard", "restore",
                    "--runtime", "docker", "--container", agent_c,
                    "--task-path", container_task_path,
                    "--stash-dir", stash_dir,
                    timeout=_remain(), check=False, cwd=str(project_root))
            restore_failed = r.returncode != 0
            if restore_failed:
                # Without the restore the eval phase cannot import the evaluator
                # module; quarantine the task as ungraded instead of letting the
                # import crash masquerade as pass=False.
                logs.append(f"== restore failed ==\n{(r.stderr or r.stdout or '')[-500:]}")
            _docker("exec", agent_c, "rm", "-rf", "--", f"{task_root}/eval_res.json",
                    check=False)

            # --- phase 5: trusted evaluation ------------------------------
            e_bundle = _mktemp_bundle()
            _docker("cp", str(host_bundle), f"{agent_c}:{e_bundle}")
            try:
                eres = _phase(["uv", "run", "python3", "-m",
                               "scripts.decoupled.container_eval",
                               "--bundle_file", e_bundle,
                               "--require_resolved_task_config",
                               "--boolean_evaluation_status",
                               "--consume_bundle",
                               "--agent_exit_code", str(agent_exit)])
            finally:
                _docker("exec", agent_c, "rm", "-f", "--", e_bundle, check=False)
            eval_exit = eres.returncode
            logs.append(f"== eval (exit {eval_exit}) ==")
            stdout_all += "\n== EVAL ==\n" + (eres.stdout or "")[-4000:]

            # Dumps land in output_dir through the bind mount; copy the
            # evaluator's verdict file out of the hidden task directory.
            _docker("cp", f"{agent_c}:{task_root}/eval_res.json",
                    f"{output_dir}/eval_res.json", check=False)

            outcome = _classify_outcome(agent_status, ares.stdout or "")
            # The evaluator's boolean exit code is the verdict -- except when
            # the agent process died without reporting any status: then the
            # fail says nothing about the model (same crash the gym service
            # quarantines as ungraded, so the retry layer gets its chance).
            passed: Optional[bool] = (eval_exit == 0)
            if outcome == "agent_crash" or restore_failed:
                passed = None

            details = None
            ev_file = output_dir / "eval_res.json"
            if ev_file.is_file():
                try:
                    ev = json.loads(ev_file.read_text())
                    details = ev.get("details") or ev.get("failure")
                    if ev.get("pass") is not None:
                        passed = bool(ev["pass"]) if outcome != "agent_crash" else None
                except Exception:
                    pass
            return _result(passed, agent_status or "EVALUATED", details, outcome)
        finally:
            shutil.rmtree(stash_root, ignore_errors=True)

    except subprocess.TimeoutExpired:
        logs.append(f"== TIMEOUT after {TASK_TIMEOUT}s ==")
        return _result(False, "TIMEOUT", f"task exceeded {TASK_TIMEOUT}s")
    except Exception as e:
        # Docker/network trouble says nothing about the model: ungraded, not 0.
        logs.append(f"== ERROR ==\n{e}")
        return _result(None, "ERROR", str(e)[:300])
    finally:
        _docker("rm", "-f", agent_c, check=False)
        if not is_k8s:
            _docker("network", "rm", net, check=False)
        else:
            # This runner drives preprocess/agent/eval directly and never calls
            # the task's own `stop`, so a k8s task's kind cluster (created via
            # docker.sock against the HOST daemon) outlives the task and leaks.
            # Each kind node holds ~25 of the host's 128
            # fs.inotify.max_user_instances -- a per-uid, cross-namespace kernel
            # limit -- so a handful of leaked clusters exhaust it and the next
            # `kind create` dies at `docker exec ... setns: exit status 1`
            # ("Writing configuration"), which is exactly how k8s-safety-audit
            # failed while cluster-{cleanup,mysql,pr-preview,redis-helm} sat
            # leaked from earlier tasks. Tear the cluster down here (best-effort,
            # off the host daemon via a throwaway container) so inotify capacity
            # is reclaimed no matter where the task aborted; eval has already run
            # by now, so the cluster is no longer needed. Raising the host
            # sysctl to e.g. 512 remains the complementary fix for genuinely
            # concurrent k8s tasks, but requires root the service lacks.
            cname = _k8s_cluster_name(task)
            if cname:
                _docker("run", "--rm", "--network", "host",
                        "-v", "/var/run/docker.sock:/var/run/docker.sock",
                        "--entrypoint", "bash", IMAGE, "-lc",
                        "KIND_EXPERIMENTAL_PROVIDER=docker kind delete cluster "
                        f"--name {cname}",
                        timeout=180, check=False)
