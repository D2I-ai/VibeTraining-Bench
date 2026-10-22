"""
Task runner for the Toolathlon-GYM evaluation service.

The code that runs inside the container is *official gym + overlay*: paths.py
stages the external read-only official checkout (VTB_TOOLATHLON_GYM_REPO) into a
throwaway tree and layers this package's overlay/ on top. The runner drives
that staged tree; it never touches the official checkout.

Two container strategies, picked by a one-time bind-mount probe:

  mounts allowed (normal hosts)  the upstream `run_parallel.sh` shape: the
                                 staged project tree is bind-mounted at
                                 /workspace and the per-task output dir at
                                 /workspace/dumps, so results land on the host
                                 with no `docker cp` at all; init.sql.gz rides
                                 into postgres as a read-only mount.
  mounts blocked (authZ hosts)   the fallback for mount-restricted hosts:
                                 code pre-baked into the image
                                 (scripts/bake_code.sh), `docker create` ->
                                 `docker cp` -> `docker start/exec`, and results
                                 extracted with `docker cp` after the run.

Every task gets its own postgres container, Docker network, and globally unique
names, so tasks and jobs can run concurrently without sharing mutable state.
"""

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

try:  # package import (uvicorn eval.service.toolathlon_gym.app:app)
    from . import paths
except ImportError:  # script import (run from the service dir)
    import paths

IMAGE = os.environ.get("TOOLATHLON_IMAGE", "toolathlon-pack:latest")
POSTGRES_IMAGE = os.environ.get(
    "POSTGRES_IMAGE",
    "postgres:15@sha256:9b1d34adbce1dd07ee6e94b4a2cf698884b89bd44a6c9c12f5da8f3acbfe4957",
)
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", "2100"))  # 35 min per task (1h budget)
# First start of a per-task postgres imports the 60 MB init.sql; allow for 12
# containers doing that at once on a loaded disk.
PG_READY_TIMEOUT = int(os.environ.get("PG_READY_TIMEOUT", "300"))

# The staged tree (official + overlay) is built once and reused for every task.
_PROJECT_ROOT: Optional[Path] = None


def _project_root() -> Path:
    """The staged (official + overlay) tree; the container's /workspace source."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = paths.build_staged_root()
    return _PROJECT_ROOT


def _init_sql() -> Path:
    return _project_root() / "db" / "init.sql.gz"


def _run(cmd: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _docker(*args: str, timeout: Optional[int] = None, check: bool = True) -> subprocess.CompletedProcess:
    res = _run(["docker", *args], timeout=timeout)
    if check and res.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:2])} failed: {res.stderr.strip()[:300]}")
    return res


_MOUNT_PROBE: Dict = {}


def _bind_mounts_allowed() -> bool:
    """Whether this docker daemon accepts -v bind mounts, probed once.

    Some managed hosts' docker authZ plugin denies every volume mount, which is what the
    bake/docker-cp fallback exists for; on a normal host the upstream
    bind-mount flow is used instead. Same probe pattern as VTB's
    eval/grading/executor.py. The pack image is the probe image because it is
    a hard prerequisite anyway.
    """
    if "allowed" not in _MOUNT_PROBE:
        probe = _run(["docker", "run", "--rm", "-v", "/etc/hostname:/vtb-mount-probe",
                      "--entrypoint", "/bin/bash", IMAGE, "-c", "true"])
        _MOUNT_PROBE["allowed"] = probe.returncode == 0
    return _MOUNT_PROBE["allowed"]


def check_prerequisites() -> Dict:
    """Verify image, init.sql.gz and the postgres base image.

    The parallel mode creates a network and a postgres per task, so no shared
    `toolathlon_net` / `toolathlon_pg` is required any more; what must exist up
    front is everything a per-task setup pulls from. Baked code is only a
    requirement of the no-mount fallback -- with bind mounts the live project
    tree is mounted over /workspace and baking is a no-op.
    """
    problems = []
    if _run(["docker", "image", "inspect", IMAGE]).returncode != 0:
        problems.append(f"image {IMAGE} not found")
    elif not _bind_mounts_allowed():
        r = _run(["docker", "run", "--rm", "--entrypoint", "/bin/bash", IMAGE,
                  "-c", "test -f /workspace/main.py"])
        if r.returncode != 0:
            problems.append("code not baked into image; run scripts/bake_code.sh")
    if _run(["docker", "image", "inspect", POSTGRES_IMAGE]).returncode != 0:
        problems.append(f"{POSTGRES_IMAGE} image not pulled; run: docker pull {POSTGRES_IMAGE}")
    try:
        init_sql = _init_sql()
    except RuntimeError as exc:
        return {"ok": False, "problems": [str(exc)]}
    if not init_sql.is_file():
        problems.append(f"init sql missing: {init_sql}")
    return {"ok": not problems, "problems": problems}


def _fix_sent_log_fk(container: str) -> None:
    script = (
        "import psycopg2, os\n"
        "conn = psycopg2.connect(host=os.environ['PGHOST'], database=os.environ['PGDATABASE'],"
        " user=os.environ['PGUSER'], password=os.environ['PGPASSWORD'])\n"
        "conn.autocommit = True\ncur = conn.cursor()\n"
        "try:\n"
        "    cur.execute('ALTER TABLE email.sent_log DROP CONSTRAINT sent_log_message_id_fkey')\n"
        "    cur.execute('ALTER TABLE email.sent_log ADD CONSTRAINT sent_log_message_id_fkey"
        " FOREIGN KEY (message_id) REFERENCES email.messages(id) ON DELETE CASCADE')\n"
        "except Exception: pass\n"
        "conn.close()\n"
    )
    _docker("exec", container, "/opt/venv/bin/python3", "-c", script, check=False)


def _parse_result(stdout: str, output_dir: Path) -> Dict:
    """Extract pass/details/outcome from main.py stdout and the copied eval_res.json."""
    result = {"pass": None, "details": None, "status": None, "agent_outcome": None}

    # Why `agent_outcome` exists: task_agent.py initialises status to FAILED and its
    # `except Exception` branch only PRINTS, so a crashed agent reports the same
    # "failed" as one that simply never called claim_done. Downstream that difference
    # is the difference between a model's zero and an infrastructure non-result, so
    # it has to be recovered here -- from the markers the agent prints on each path.
    if "[agent] Error:" in stdout:
        result["agent_outcome"] = "agent_crash"
    elif "Done (claim_done called)" in stdout:
        result["agent_outcome"] = "claimed_done"
    elif "Max iterations reached" in stdout:
        result["agent_outcome"] = "max_steps"
    elif "Ended without claim_done" in stdout:
        result["agent_outcome"] = "gave_up"

    m = re.search(r"Status:\s*(\w+)", stdout)
    if m:
        result["status"] = m.group(1)
    m = re.search(r"Pass:\s*(True|False|None)", stdout)
    if m:
        result["pass"] = {"True": True, "False": False, "None": None}[m.group(1)]
    m = re.search(r"Details:\s*(.+)", stdout)
    if m:
        result["details"] = m.group(1).strip()

    # eval_res.json (written by the evaluator next to the log file) wins if present
    for p in output_dir.rglob("eval_res.json"):
        try:
            ev = json.loads(p.read_text())
            result["pass"] = ev.get("pass", result["pass"])
            result["details"] = ev.get("details") or ev.get("failure") or result["details"]
            break
        except Exception:
            continue
    return result


def run_task_isolated(task: str, model_cfg: Dict, job_id: str, max_steps: int = 50,
                      output_root: str = "eval_output") -> Dict:
    """Run one task with its own postgres + network; safe to call concurrently.

    Ports are never published to the host: the agent container reaches its
    postgres by container name over the per-task network, so N tasks never
    fight over host :5432. init.sql.gz reaches initdb either as a read-only
    bind mount (normal hosts) or via `docker cp` BEFORE the first start
    (authZ fallback).

    Outcome semantics, aligned with VTB's three-state contract:
      pass=True/False   the verifier reached a verdict
      pass=None         infrastructure trouble (pg never ready, docker error)
                        -- reported as ungraded upstream, never as a failure.
    Never raises.
    """
    tag = f"{job_id[:8]}-{hashlib.md5(task.encode()).hexdigest()[:8]}"
    pg_c = f"tge-pg-{tag}"
    agent_c = f"tge-ag-{tag}"
    net = f"tge-net-{tag}"
    output_dir = Path(output_root) / job_id / task
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stdout = ""

    def _result(passed, status, details, agent_outcome=None):
        return {"task": task, "pass": passed, "status": status, "details": details,
                "agent_outcome": agent_outcome,
                "duration_sec": round(time.time() - started, 1),
                "output_dir": str(output_dir)}

    try:
        init_sql = _init_sql()
        # Leftovers from a crashed earlier run with the same names would wedge
        # `docker create`; clearing them first makes retries idempotent.
        for c in (agent_c, pg_c):
            _docker("rm", "-f", c, check=False)
        _docker("network", "rm", net, check=False)

        # Retry transient docker address-pool exhaustion (the default pool is
        # shared with other workloads on this host and can momentarily have no
        # free block under concurrency); back off so a finishing task's
        # `network rm` frees one rather than hard-failing as ERROR.
        for _attempt in range(6):
            _res = _docker("network", "create", net, check=False)
            if _res.returncode == 0:
                break
            stdout += (f"== network create retry {_attempt + 1}: "
                       f"{(_res.stderr or _res.stdout or '').strip()[-200:]} ==\n")
            _docker("network", "rm", net, check=False)
            time.sleep(3 * (_attempt + 1))
        else:
            return _result(None, "ERROR",
                           "docker network create failed after retries "
                           f"(address-pool exhaustion): {net}")

        mounts = _bind_mounts_allowed()
        project_root = _project_root()

        # Per-task postgres. With mounts, init.sql.gz is bind-mounted into
        # /docker-entrypoint-initdb.d so initdb imports it on first boot (the
        # upstream run_parallel.sh shape); without mounts it is docker-cp'd
        # in before the first start (same trick as start.sh).
        pg_args = ["--name", pg_c, "--network", net,
                   "-e", "POSTGRES_DB=toolathlon_gym",
                   "-e", "POSTGRES_USER=eigent",
                   "-e", "POSTGRES_PASSWORD=camel",
                   "--health-cmd", "pg_isready -U eigent -d toolathlon_gym",
                   "--health-interval=3s", "--health-retries=60"]
        if mounts:
            pg_args += ["-v", f"{init_sql}:/docker-entrypoint-initdb.d/init.sql.gz:ro"]
        pg_args += [POSTGRES_IMAGE]
        _docker("create", *pg_args)
        if not mounts:
            _docker("cp", str(init_sql),
                    f"{pg_c}:/docker-entrypoint-initdb.d/init.sql.gz")
        _docker("start", pg_c)

        deadline = time.time() + PG_READY_TIMEOUT
        while True:
            r = _run(["docker", "inspect", "--format",
                      "{{.State.Health.Status}}", pg_c])
            if r.returncode == 0 and r.stdout.strip() == "healthy":
                break
            if time.time() > deadline:
                return _result(None, "PG_NOT_READY",
                               f"postgres not healthy within {PG_READY_TIMEOUT}s")
            time.sleep(3)

        # Agent container on the same per-task network; PGHOST is the pg
        # container's NAME, resolved by docker's embedded DNS.
        env_args = ["-e", f"PGHOST={pg_c}", "-e", f"PG_HOST={pg_c}",
                    "-e", "PGPORT=5432", "-e", "PG_PORT=5432",
                    "-e", "PGUSER=eigent", "-e", "PG_USER=eigent",
                    "-e", "PGPASSWORD=camel", "-e", "PG_PASSWORD=camel",
                    "-e", "PGDATABASE=toolathlon_gym", "-e", "PG_DATABASE=toolathlon_gym",
                    "-e", "LOCAL_SERVERS_PATH=/opt/local_servers",
                    "-e", "PYTHON_BIN=/opt/venv/bin/python3"]
        for k in ("MODEL_PLATFORM", "MODEL_PROVIDER", "MODEL_NAME",
                  "MODEL_API_KEY", "MODEL_API_URL"):
            if model_cfg.get(k):
                env_args += ["-e", f"{k}={model_cfg[k]}"]
        # The thinking switch is read inside the container, so forward the
        # operator's service setting when present.
        _think = model_cfg.get("TOOLATHLON_ENABLE_THINKING") or os.environ.get("TOOLATHLON_ENABLE_THINKING")
        if _think:
            env_args += ["-e", f"TOOLATHLON_ENABLE_THINKING={_think}"]
        _maxtok = model_cfg.get("TOOLATHLON_MAX_TOKENS") or os.environ.get("TOOLATHLON_MAX_TOKENS")
        if _maxtok:
            env_args += ["-e", f"TOOLATHLON_MAX_TOKENS={_maxtok}"]
        for _key in ("TEMPERATURE", "TOP_P", "TOP_K", "REPETITION_PENALTY"):
            _value = model_cfg.get(f"TOOLATHLON_{_key}") or os.environ.get(f"TOOLATHLON_{_key}")
            if _value:
                env_args += ["-e", f"TOOLATHLON_{_key}={_value}"]
        _tmo = model_cfg.get("TOOLATHLON_MODEL_TIMEOUT") or os.environ.get("TOOLATHLON_MODEL_TIMEOUT")
        if _tmo:
            env_args += ["-e", f"TOOLATHLON_MODEL_TIMEOUT={_tmo}"]
        _retries = model_cfg.get("TOOLATHLON_MODEL_RETRIES") or os.environ.get("TOOLATHLON_MODEL_RETRIES")
        if _retries:
            env_args += ["-e", f"TOOLATHLON_MODEL_RETRIES={_retries}"]

        agent_args = ["--name", agent_c, "--network", net, *env_args,
                      "-w", "/workspace"]
        if mounts:
            # Upstream run_parallel.sh shape: the staged project tree at
            # /workspace and this task's output dir at /workspace/dumps, so
            # results (traj_log/eval_res.json/camel_logs) land on the host as
            # the task runs -- no post-hoc docker cp. The dumps mount shadows
            # the tree's own dumps/ and keeps concurrent tasks from writing
            # into each other's outputs.
            agent_args += ["-v", f"{project_root}:/workspace",
                           "-v", f"{output_dir}:/workspace/dumps"]
        agent_args += [IMAGE, "sleep", str(TASK_TIMEOUT + 600)]
        _docker("create", *agent_args)
        _docker("start", agent_c)
        for _ in range(30):
            if _docker("exec", agent_c, "true", check=False).returncode == 0:
                break
            time.sleep(1)

        # The agent environment points this helper at the task's own postgres.
        _fix_sent_log_fk(agent_c)

        res = _docker("exec", agent_c, "/opt/venv/bin/python3", "main.py",
                      "--task_dir", task, "--max_steps", str(max_steps), "--debug",
                      timeout=TASK_TIMEOUT, check=False)
        stdout = res.stdout or ""
        (output_dir / "run.log").write_text(stdout + "\n== STDERR ==\n" + (res.stderr or ""))

        # Bind-mount mode: dumps already landed in output_dir live.
        if not mounts:
            _docker("cp", f"{agent_c}:/workspace/dumps/.", f"{output_dir}/", check=False)
        parsed = _parse_result(stdout, output_dir)
        return _result(parsed["pass"], parsed["status"], parsed["details"],
                       parsed["agent_outcome"])

    except subprocess.TimeoutExpired:
        try:
            (output_dir / "run.log").write_text(stdout + "\n== TIMEOUT ==\n")
        except OSError:
            pass
        return _result(False, "TIMEOUT", f"task exceeded {TASK_TIMEOUT}s")
    except Exception as e:
        # Docker/network trouble says nothing about the model: ungraded, not 0.
        return _result(None, "ERROR", str(e)[:300])
    finally:
        for c in (agent_c, pg_c):
            _docker("rm", "-f", c, check=False)
        _docker("network", "rm", net, check=False)
