"""Evaluation executor: the task's own evaluate.py, with the retry ladder.

Every evaluation is a subprocess. The service never imports inspect-ai, never touches
evaluate.py, and never scores anything itself -- it only decides *when* to evaluate and
records what came back. That is what keeps the scoring authority in the benchmark's own
file rather than here.

What runs in a container is the delivered model (served over HTTP, see served_model)
and each task's own environment (started by harbor). The orchestration between
them runs on this host, because that is where the tools it needs are installed.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

from . import config
from .store import RunContext
from ..runner import model_server
from ..runner import runtime_paths as runner_paths

# evaluate.py lives in <repo>/benchmarks/<task>/ and is invoked with that as cwd,
# so the shared templates directory is two levels up.
TEMPLATES_REL = "../../eval/templates"

# Share port allocation, flag probing, and SGLang gRPC offsets with model_server.
# Grading retains its own Docker command for delivery mounts, cancellation,
# and the service's reserved GPU allocation.
SGLANG_GRPC_PORT_OFFSET = model_server.SGLANG_GRPC_PORT_OFFSET
SGLANG_MAX_HTTP_PORT = model_server.SGLANG_MAX_HTTP_PORT


Logger = Callable[[str], None]

WaitCallback = Callable[[Optional[dict]], None]


# ------------------------------------------------------- interruption safety
# An evaluation owns three things that outlive a plain process exit: a child process
# in its own session (so a timeout can kill the whole tree without killing the
# service), a fuse-overlayfs mount, and -- when the model is served in docker -- a
# container, which does not stop just because the client that started it was killed.
# None of them is released by a default SIGTERM, which does not run `finally` blocks,
# so all three are registered here for a signal handler to tear down explicitly.
#
# Left behind, an orphaned server keeps ~25 GB of the evaluation card allocated,
# which then traps every later submission behind the GPU free-memory gate.
_ACTIVE_LOCK = threading.Lock()
_ACTIVE: dict = {"proc": None, "mount": None, "serve": None, "serve_container": None}

# The job whose evaluation has been asked to stop, if any. Kept beside _ACTIVE and
# under the same lock, because the two are always read together: one says what is
# running, the other says not to start it again.
_CANCEL: Optional[str] = None


def _set_active(key: str, value) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE[key] = value


def request_cancel(job_id: str, log: Logger) -> list:
    """Stop the named evaluation and release everything it is holding.

    The flag matters as much as the teardown. Killing the evaluation process on its
    own is indistinguishable from an evaluation that died for an infrastructure
    reason, and the retry ladder answers that by starting the next of its nine
    attempts -- serving the model again and taking the cards straight back, seconds
    after the caller was told they were free.

    Set before the teardown, so the ladder cannot slip an attempt in between.
    """
    global _CANCEL
    with _ACTIVE_LOCK:
        _CANCEL = job_id
    log(f"[cancel] job={job_id} requested; releasing the evaluation")
    return cleanup_active(log)


def cancel_requested(job_id: Optional[str]) -> bool:
    """Whether this job -- or any job, when job_id is None -- was cancelled."""
    with _ACTIVE_LOCK:
        if _CANCEL is None:
            return False
        return job_id is None or _CANCEL == job_id


def clear_cancel() -> None:
    """Forget a cancellation once its job has finished being wound up.

    Without this the next submission would find the flag still set and stop before
    its first attempt.
    """
    global _CANCEL
    with _ACTIVE_LOCK:
        _CANCEL = None


def cleanup_active(log: Logger) -> list:
    """Tear down whatever an interrupted evaluation left running.

    Idempotent, and safe to call from a signal handler: it only kills a process
    tree and unmounts a path, both of which tolerate already being gone.
    """
    with _ACTIVE_LOCK:
        proc = _ACTIVE.get("proc")
        mount = _ACTIVE.get("mount")
        serve = _ACTIVE.get("serve")
        serve_container = _ACTIVE.get("serve_container")
        _ACTIVE["proc"] = None
        _ACTIVE["mount"] = None
        _ACTIVE["serve"] = None
        _ACTIVE["serve_container"] = None

    actions = []
    if proc is not None and proc.poll() is None:
        log(f"[cleanup] killing evaluation process tree pid={proc.pid}")
        _kill_tree(proc)
        actions.append(f"killed evaluation pid={proc.pid}")
    if serve is not None and serve.poll() is None:
        # A model server holds the GPU just like the evaluation itself; left running it
        # would trap every later submission behind the free memory gate.
        log(f"[cleanup] killing model server tree pid={serve.pid}")
        _kill_tree(serve)
        actions.append(f"killed model server pid={serve.pid}")
    if serve_container:
        # And the container it started, which the kill above does not reach: `docker
        # run` is a client, and the server is a child of the daemon.
        log(f"[cleanup] removing serving container {serve_container}")
        _remove_container(serve_container)
        actions.append(f"removed container {serve_container}")
    if mount is not None:
        log(f"[cleanup] unmounting {mount}")
        subprocess.run(["fusermount", "-u", str(mount)], capture_output=True)
        shutil.rmtree(mount, ignore_errors=True)
        actions.append(f"unmounted {mount}")
    actions += report_backend_task_containers(log)
    return actions


# Harbor runs each task in its own docker compose project, named after the trial with
# an `__env` suffix (harbor/environments/docker/docker.py). The project name embeds a
# random per-trial id, so the service cannot know it, and nothing else distinguishes
# those containers from any other compose project on the host.
_BACKEND_PROJECT_SUFFIX = "__env"


def report_backend_task_containers(log: Logger) -> list:
    """Name the backend's per-task containers still running after a teardown.

    Deliberately only a report. Killing the evaluation's process tree does not reach
    the containers the backend asked the daemon for -- the same gap that makes the
    serving container a separate step above -- but unlike that one there is no name to
    remove: the project name carries a random per-trial id. A filter loose enough to
    match them would also match any other compose project on this host, and removing
    a container that belongs to somebody else is worse than leaking one.

    So the leak is made visible instead of guessed at. If these show up, harbor's own
    SIGTERM handling is not finishing, and that is a fix for harbor's invocation
    rather than for a broader filter here.
    """
    try:
        out = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "label=com.docker.compose.project",
                "--format",
                '{{.Label "com.docker.compose.project"}}',
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=_docker_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    projects = sorted(
        {line.strip() for line in out.stdout.splitlines() if line.strip().endswith(_BACKEND_PROJECT_SUFFIX)}
    )
    if not projects:
        return []
    log(
        f"[cleanup] {len(projects)} backend task container project(s) still running "
        f"after teardown: {', '.join(projects[:5])}"
        + (" ..." if len(projects) > 5 else "")
    )
    return [f"reported {len(projects)} orphaned backend task project(s)"]


def sweep_stale_overlays(log: Logger) -> list:
    """Release overlay mounts and scratch left by a previous unclean exit.

    SIGKILL, an OOM kill or a host reboot cannot be handled in-process, so the
    only way to recover is to look for the debris at startup. A stale mount on a
    work directory would otherwise make that path unusable and leave a
    fuse-overlayfs process running indefinitely.
    """
    actions = []
    work_root = config.STATE_DIR / "work"
    if not work_root.is_dir():
        return actions

    try:
        mounts = subprocess.run(["mount"], capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError):
        mounts = ""

    for overlay_dir in sorted(work_root.glob("*/merged_huggingface")):
        if str(overlay_dir) in mounts:
            log(f"[sweep] unmounting stale overlay {overlay_dir}")
            subprocess.run(["fusermount", "-u", str(overlay_dir)], capture_output=True)
            actions.append(f"unmounted stale {overlay_dir}")

    # The three overlay directories are pure scratch; per-attempt logs and
    # metrics in the same work dir are evidence and are deliberately kept.
    for name in ("merged_huggingface", "upper_huggingface", "fuse_workdir"):
        for leftover in sorted(work_root.glob(f"*/{name}")):
            shutil.rmtree(leftover, ignore_errors=True)
            if not leftover.exists():
                actions.append(f"removed stale {leftover}")
    return actions


class AttemptResult:
    __slots__ = ("ok", "score", "metrics", "phase", "attempt", "elapsed", "error", "log_path", "extra_args")

    def __init__(
        self,
        ok: bool,
        score: Optional[float],
        metrics: Optional[dict],
        phase: int,
        attempt: int,
        elapsed: float,
        error: Optional[str],
        log_path: Path,
        extra_args: list,
    ) -> None:
        self.ok = ok
        self.score = score
        self.metrics = metrics
        self.phase = phase
        self.attempt = attempt
        self.elapsed = elapsed
        self.error = error
        self.log_path = log_path
        self.extra_args = extra_args

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "score": self.score,
            "metrics": self.metrics,
            "phase": self.phase,
            "attempt": self.attempt,
            "elapsed_seconds": round(self.elapsed, 2),
            "error": self.error,
            "log": str(self.log_path),
            "token_override": " ".join(self.extra_args) if self.extra_args else None,
        }


# ------------------------------------------------------------------ gpu gate


def gpu_free_mib(gpu: str) -> Optional[int]:
    """Free MiB on the given GPU index, or None if nvidia-smi is unusable."""
    values = gpu_free_mib_map([gpu] if gpu else [])
    if values is None:
        return None
    return min(values.values()) if values else None


def gpu_free_mib_map(gpus: list[str]) -> Optional[dict[str, int]]:
    """Free MiB by host GPU index, or None if nvidia-smi is unusable."""
    cmd = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    if gpus:
        cmd += ["-i", ",".join(gpus)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None

    values: dict[str, int] = {}
    for line in out.stdout.split("\n"):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] and parts[1].isdigit():
            values[parts[0]] = int(parts[1])
    return values


def wait_until_eval_gpus_free(log: Logger, timeout: Optional[int] = None) -> dict:
    """Block until the evaluation cards report free memory again.

    The other side of the free-memory gate, and the reason a cancellation can be
    acted on: a caller that has just been told the cards are its own will start a
    training job immediately, and a serving container that is still shutting down
    would meet it as an out-of-memory failure with no visible cause.
    """
    gpus = list(config.EVAL_GPUS)
    if not gpus:
        return {"released": True, "reason": "no evaluation cards are configured"}
    deadline = time.time() + (config.GPU_RELEASE_WAIT_SECONDS if timeout is None else timeout)
    while True:
        free_by_gpu = gpu_free_mib_map(gpus)
        if free_by_gpu is None:
            return {"released": True, "reason": "nvidia-smi unavailable, so no reading to wait for"}
        busy = {
            gpu: free_by_gpu[gpu]
            for gpu in gpus
            if gpu in free_by_gpu and free_by_gpu[gpu] < config.GPU_FREE_MIB_REQUIRED
        }
        if not busy:
            log("[cancel] evaluation cards are free")
            return {"released": True, "free_mib": free_by_gpu}
        if time.time() >= deadline:
            log(f"[cancel] evaluation cards still busy after waiting: {busy}")
            return {
                "released": False,
                "free_mib": free_by_gpu,
                "required_mib": config.GPU_FREE_MIB_REQUIRED,
                "reason": "the cards had not reported free memory before the wait ran out",
            }
        time.sleep(config.GPU_POLL_SECONDS)


def wait_for_gpu(
    log: Logger,
    on_gpu_wait: Optional[WaitCallback] = None,
    job_id: Optional[str] = None,
) -> bool:
    """Block until every evaluation card has room, or give up.

    In isolated deployments the operator keeps these cards out of every run's training
    allocation. In shared mode the agent may have used the same card pool for training;
    the callback lets the service report that validation is waiting for GPU memory.

    A cancellation ends the wait immediately. Queued behind a gate with a half-hour
    budget is exactly where a submission sits when the run has decided it wants its
    cards back, and making it wait out the timeout would defeat the point.
    """
    gpus = list(config.EVAL_GPUS)
    if not gpus:
        log("[gpu] VTB_EVAL_GPUS/VTB_EVAL_GPU unset -- skipping the free-memory gate")
        if on_gpu_wait is not None:
            on_gpu_wait(None)
        return True

    deadline = time.time() + config.GPU_WAIT_SECONDS
    while True:
        if cancel_requested(job_id):
            log("[gpu] cancelled while waiting for the evaluation cards")
            return False
        free_by_gpu = gpu_free_mib_map(gpus)
        if free_by_gpu is None:
            log("[gpu] nvidia-smi unavailable -- proceeding without the gate")
            if on_gpu_wait is not None:
                on_gpu_wait(None)
            return True
        missing = [gpu for gpu in gpus if gpu not in free_by_gpu]
        busy = {gpu: free_by_gpu[gpu] for gpu in gpus if gpu in free_by_gpu and free_by_gpu[gpu] < config.GPU_FREE_MIB_REQUIRED}
        if not missing and not busy:
            detail = ", ".join(f"card {gpu}: {free_by_gpu[gpu]} MiB free" for gpu in gpus)
            log(f"[gpu] eval GPUs ready ({detail}) -- proceeding")
            if on_gpu_wait is not None:
                on_gpu_wait(None)
            return True
        remaining = int(deadline - time.time())
        wait_detail = {
            "reason": "gpu_busy",
            "eval_gpus": gpus,
            "free_mib": free_by_gpu,
            "missing_gpus": missing,
            "required_mib": config.GPU_FREE_MIB_REQUIRED,
            "remaining_seconds": max(0, remaining),
            "retry_after_seconds": config.GPU_POLL_SECONDS,
        }
        if on_gpu_wait is not None:
            on_gpu_wait(wait_detail)
        if remaining <= 0:
            status = ", ".join(
                f"card {gpu}: {free_by_gpu.get(gpu, 'unknown')} MiB free" for gpu in gpus
            )
            log(
                f"[gpu] gave up waiting: eval GPUs busy ({status}), "
                f"need {config.GPU_FREE_MIB_REQUIRED} MiB each"
            )
            return False
        status = ", ".join(f"card {gpu}: {free_by_gpu.get(gpu, 'unknown')} MiB free" for gpu in gpus)
        log(f"[gpu] eval GPUs busy ({status}), waiting {remaining}s more")
        time.sleep(config.GPU_POLL_SECONDS)


# -------------------------------------------------------------- hf overlay


@contextlib.contextmanager
def huggingface_overlay(work_dir: Path, log: Logger) -> Iterator[Path]:
    """fuse-overlayfs the shared HF cache read-only under a throwaway upper.

    Same construction as run_task.sh's with_huggingface_overlay: dataset and
    model downloads performed by an evaluation land in the upper layer and are
    discarded, so the 563 GB shared cache is never mutated by scoring.
    """
    merged = work_dir / "merged_huggingface"
    upper = work_dir / "upper_huggingface"
    workdir = work_dir / "fuse_workdir"
    for d in (merged, upper, workdir):
        d.mkdir(parents=True, exist_ok=True)

    mounted = False
    if shutil.which("fuse-overlayfs"):
        opts = f"lowerdir={config.HF_HOME},upperdir={upper},workdir={workdir}"
        proc = subprocess.run(
            ["fuse-overlayfs", "-o", opts, str(merged)],
            capture_output=True,
            text=True,
        )
        mounted = proc.returncode == 0
        if not mounted:
            log(f"[hf] fuse-overlayfs failed ({proc.stderr.strip()}) -- binding the cache directly")
    else:
        log("[hf] fuse-overlayfs not found -- binding the cache directly")

    try:
        if mounted:
            _set_active("mount", merged)
        yield merged if mounted else config.HF_HOME
    finally:
        if mounted:
            _set_active("mount", None)
            subprocess.run(["fusermount", "-u", str(merged)], capture_output=True)
        for d in (merged, upper, workdir):
            shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------------- one attempt


# Use model_server.free_port(max_port=..., reserve_offset=...) to allocate
# HTTP ports while reserving space for the SGLang gRPC offset.


def _serve_mounts(model_path: Path) -> list[str]:
    """Host paths the serving container must see, each at its own path inside.

    Same path on both sides, so a model directory named on the command line means the
    same thing in both places. Resolved targets are included as well: vtb-env/ may be a
    symlink onto a bigger disk, and mounting only the link lands the container on a path
    that does not resolve in it.
    """
    mounts: list[str] = []
    seen: set[str] = set()
    for anchor in (
        model_path.parent,
        config.HF_HOME,
        config.HF_HOME.resolve(),
        config.VTB_ROOT,
        config.VTB_ROOT.resolve(),
    ):
        path = Path(anchor)
        if not path.exists() or str(path) in seen:
            continue
        seen.add(str(path))
        mounts += ["-v", f"{path}:{path}"]
    return mounts


# Where docker cp lands the model when the daemon refuses volume mounts.
CP_MODEL_DIR = "/vtb-serve-model"

_MOUNT_PROBE: dict = {}


def _bind_mounts_allowed() -> bool:
    """Whether this docker daemon accepts -v bind mounts, probed once.

    Some managed hosts' docker authZ plugin denies every volume mount (and --ipc host with it),
    so the serve container cannot be handed the model directory that way.
    Probed rather than configured: the same repo also runs on plain hosts where
    mounts work, and a per-host flag is one more thing to forget. The serve
    image is the probe image -- _require_serve_image() ran before this.
    """
    if "allowed" not in _MOUNT_PROBE:
        probe = subprocess.run(
            ["docker", "run", "--rm", "-v", "/etc/hostname:/vtb-mount-probe",
             "--entrypoint", "/bin/true", config.serve_image()],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_docker_env(),
            timeout=120,
        )
        _MOUNT_PROBE["allowed"] = probe.returncode == 0
    return _MOUNT_PROBE["allowed"]


def _serve_env_args(cp_mode: bool = False) -> list[str]:
    # In cp mode the host cache path does not exist inside the container; a
    # throwaway location keeps the server from trying to use a missing path.
    hf_home = "/tmp/vtb-hf" if cp_mode else str(config.HF_HOME)
    env_args = [
        "-e",
        f"HF_HOME={hf_home}",
        "-e",
        f"VTB_API_KEY={config.API_KEY}",
    ]
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            env_args += ["-e", f"{name}={os.environ[name]}"]
    for name in config.SERVE_PASSTHROUGH_ENV:
        value = os.environ.get(name)
        if value:
            env_args += ["-e", f"{name}={value}"]
    return env_args


def _docker_env() -> dict:
    """Environment for a docker command, with the daemon named explicitly.

    Not optional: with DOCKER_HOST unset the CLI follows the active docker *context*, so
    a host where a rootless context is selected would serve from the rootless daemon
    while looking like it had not. runner_paths.docker_host() is the one place that
    decides which daemon this repo uses.
    """
    env = os.environ.copy()
    host = runner_paths.docker_host()
    if host:
        env["DOCKER_HOST"] = host
    return env


def _serve_payload(model_path: Path, served_name: str, port: int) -> tuple[str, list[str]]:
    """Return (backend label, the server command)."""
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
        "--api-key",
        config.API_KEY,
        "--served-model-name",
        served_name,
        "--trust-remote-code",
    ]
    if config.SERVE_TOOL_CALL_PARSER:
        cmd += ["--tool-call-parser", config.SERVE_TOOL_CALL_PARSER]
    if config.SERVE_REASONING_PARSER:
        cmd += ["--reasoning-parser", config.SERVE_REASONING_PARSER]
    extra = list(config.SERVE_EXTRA_ARGS) + list(config.SGLANG_EXTRA_ARGS)
    if config.EVAL_GPU_COUNT > 1 and not model_server.has_flag(extra, "--tp-size", "--tp"):
        cmd += ["--tp-size", str(config.EVAL_GPU_COUNT)]
    cmd += extra
    return "sglang", cmd


def _serve_container_name(port: int) -> str:
    return f"vtb-grading-serve-{config.SERVE_BACKEND}-{port}".replace("_", "-")


def _remove_container(name: str) -> None:
    """Remove a serving container by name, whether or not it is still running.

    Tolerates already being gone, so it is safe on every exit path.
    """
    subprocess.run(
        ["docker", "rm", "--force", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_docker_env(),
    )


def serve_image_present() -> bool:
    """Whether the selected backend's serving image exists on the chosen daemon.

    Public because the service reports this at startup: a missing image is an
    operator's problem, and finding out at grading time costs a whole run.

    A missing docker CLI answers False rather than raising. This is called during
    startup, and a machine with no docker at all should get the same actionable warning
    as one whose image was never built -- not a service that refuses to boot.
    """
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", config.serve_image()],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_docker_env(),
        )
    except OSError:
        return False
    return probe.returncode == 0


def _require_serve_image() -> None:
    if serve_image_present():
        return
    raise RuntimeError(
        f"{config.SERVE_BACKEND} serving image {config.serve_image()} is not present. "
        f"Build it with `bash containers/build_image.sh {config.SERVE_BACKEND}`, point "
        f"VTB_SERVE_{config.SERVE_BACKEND.upper()}_IMAGE at one that exists, set "
        "VTB_SERVE_RUNTIME=host to serve from this host's own install, or pass "
        "--api-base/VTB_API_BASE for an already-running endpoint."
    )


def _serve_docker_command(model_path: Path, payload: list[str], name: str) -> list[str]:
    """The docker command that serves the delivered model.

    Host networking is intentional: the health poll and the endpoint both address the
    host's own ports, and a fixed host port must be exactly where backend task
    containers can reach it.
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        # Named so the container can be removed by name. The docker CLI is only a
        # client: killing it -- which is what an attempt timeout or a service SIGTERM
        # does -- leaves the server running and the evaluation card allocated, and every
        # later submission then waits behind the GPU free-memory gate for a reason that
        # looks unrelated.
        "--name",
        name,
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size",
        os.environ.get("VTB_SERVE_DOCKER_SHM_SIZE", "32g"),
    ]
    # Real isolation, and deliberately not paired with CUDA_VISIBLE_DEVICES: with
    # `device=7` the container sees that one card as its device 0, so passing the host's
    # index in as well would point it at a device that is not there. With multiple cards,
    # docker_gpus_arg quotes the device list in the form Docker's CSV parser requires.
    cmd += ["--gpus", runner_paths.docker_gpus_arg(config.EVAL_GPU_SPEC)]
    cmd += _serve_mounts(model_path)
    cmd += _serve_env_args()
    # Explicit, so the payload runs whatever ENTRYPOINT the image carries: a serving
    # image that sets one would otherwise read these as extra flags to a server it
    # starts itself.
    cmd += ["--entrypoint", payload[0]]
    cmd += [config.serve_image(), *payload[1:]]
    return cmd


def _serve_create_and_copy(model_path: Path, payload: list[str], name: str,
                           log: Logger) -> None:
    """Create the serving container and docker cp the model into it.

    The authZ-restricted fallback for `docker run -v`: the model lands in the
    container's writable layer, and the payload already names CP_MODEL_DIR as
    its --model-path. --ipc host is denied on the same hosts, so --shm-size
    alone carries the shared memory vllm/sglang need; --network host stays
    allowed there, which is what keeps 127.0.0.1:<port> a valid endpoint here.
    """
    create = [
        "docker", "create", "--name", name,
        "--network", "host",
        "--shm-size", os.environ.get("VTB_SERVE_DOCKER_SHM_SIZE", "32g"),
    ]
    gpu = config.EVAL_GPU.strip()
    # Same confinement, and for the same reason, as _serve_docker_command.
    create += ["--gpus", f"device={gpu}" if gpu else "all"]
    create += _serve_env_args(cp_mode=True)
    create += ["--entrypoint", payload[0], config.serve_image(), *payload[1:]]

    def checked(step: str, cmd: list[str]) -> None:
        res = subprocess.run(cmd, capture_output=True, text=True, env=_docker_env())
        if res.returncode != 0:
            raise RuntimeError(
                f"{step} failed (rc={res.returncode}): {res.stderr.strip()[:300]}"
            )

    checked("docker create", create)
    started = time.time()
    checked("docker cp", ["docker", "cp", str(model_path), f"{name}:{CP_MODEL_DIR}"])
    log(f"[serve] model copied into the container in {time.time() - started:.0f}s "
        f"({model_path} -> {CP_MODEL_DIR})")


def _tail_text(path: Path, max_bytes: int = 8192) -> str:
    """Return the tail of a UTF-8-ish log file for failure messages."""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace").strip()


@contextlib.contextmanager
def served_model(model_path: Path, log: Logger, work_dir: Path) -> Iterator[str]:
    """Serve the delivered model locally and yield its OpenAI-compatible endpoint.

    Every benchmark here holds harbor orchestration, not a model runtime. The model
    is served once on this host, with the runtime selected by VTB_SERVE_RUNTIME=docker|host. The endpoint is handed to evaluate.py
    as --api-base, while task isolation stays in the backend's per-task containers.
    """
    import urllib.error
    import urllib.request

    if config.EXTERNAL_API_BASE:
        base = config.EXTERNAL_API_BASE.rstrip("/")
        expected = model_path.name or "model"
        log(f"[serve] NOT serving the checkpoint: using the external endpoint {base} "
            f"(VTB_SWE_API_BASE). The score is attributed to {expected}, but this "
            "service did not load it -- do not publish official numbers from this run.")
        try:
            with urllib.request.urlopen(f"{base}/models", timeout=30) as resp:
                served = [m.get("id") for m in (json.loads(resp.read()).get("data") or [])]
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"VTB_SWE_API_BASE={base} is not answering /models: {exc}"
            ) from exc
        # A stale server from an earlier model is the failure this catches: it answers
        # perfectly, and every score would silently belong to the wrong checkpoint.
        if expected not in served:
            raise RuntimeError(
                f"VTB_SWE_API_BASE={base} serves {served}, not {expected!r}. Either point "
                f"it at the submitted checkpoint or start it with "
                f"--served-model-name {expected}"
            )
        log(f"[serve] external endpoint confirmed serving {expected!r}")
        yield base
        return

    if config.SERVE_PORT:
        port = config.SERVE_PORT
        if config.SERVE_BACKEND == "sglang":
            model_server.validate_sglang_port(port)
    elif config.SERVE_BACKEND == "sglang":
        port = model_server.free_port(
            max_port=SGLANG_MAX_HTTP_PORT, reserve_offset=SGLANG_GRPC_PORT_OFFSET
        )
    else:
        port = model_server.free_port()
    served_name = model_path.name or "model"
    backend, payload = _serve_payload(model_path, served_name, port)
    container_name: Optional[str] = None
    if config.SERVE_RUNTIME == "docker":
        _require_serve_image()
        container_name = _serve_container_name(port)
        if _bind_mounts_allowed():
            cmd = _serve_docker_command(model_path, payload, container_name)
            runtime_label = f"docker {config.serve_image()}"
        else:
            # authZ blocks -v: create the container, docker cp the model in,
            # then attach. `start -a` reports the container's own exit code,
            # so the poll() below still catches a server that dies at boot.
            payload = [CP_MODEL_DIR if a == str(model_path) else a for a in payload]
            _serve_create_and_copy(model_path, payload, container_name, log)
            cmd = ["docker", "start", "-a", container_name]
            runtime_label = f"docker {config.serve_image()} (model cp'd in; authZ blocks mounts)"
        env = _docker_env()
    else:
        # Whatever is installed here. The container path gets these through -e; a host
        # process needs them in its own environment, CUDA_VISIBLE_DEVICES included --
        # there is no --gpus to confine it to the evaluation card.
        cmd = list(payload)
        env = os.environ.copy()
        env["HF_HOME"] = str(config.HF_HOME)
        env["VTB_API_KEY"] = config.API_KEY
        if config.EVAL_GPU_SPEC:
            env["CUDA_VISIBLE_DEVICES"] = config.EVAL_GPU_SPEC
        runtime_label = f"host {payload[0]}"
    parser_note = f", tool parser {config.SERVE_TOOL_CALL_PARSER}"
    if config.SERVE_REASONING_PARSER:
        parser_note += f", reasoning parser {config.SERVE_REASONING_PARSER}"
    # Cards and tensor parallel belong in this line for the same reason the port does:
    # they are decided from the environment the service was started with, and an empty
    # VTB_EVAL_GPUS means tp=1 rather than "use the host's cards". A 9B model served from
    # one card against sixteen concurrent agents thrashes its KV cache instead of failing,
    # which is a slowdown nobody can see in a log that does not say the number.
    if config.EVAL_GPU_COUNT:
        gpu_note = f", cards {config.EVAL_GPU_SPEC} (tp={config.EVAL_GPU_COUNT})"
    else:
        gpu_note = ", cards inherited (tp=1, VTB_EVAL_GPUS unset)"
    log(
        f"[serve] starting {backend} endpoint on 0.0.0.0:{port} via "
        f"{runtime_label}{gpu_note}{parser_note}{' (fixed port)' if config.SERVE_PORT else ''}"
    )
    # The serve log lands inside work_dir so it is archived with the job; the
    # rc/tail below turn a serve-side boot failure into an actionable message
    # (the rc=125 authZ denial used to take a manual command reconstruction).
    serve_log_path = work_dir / f"{backend}_serve_{port}.log"
    serve_log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[serve] writing {backend} startup log to {serve_log_path}")
    serve_log = serve_log_path.open("ab")
    proc = subprocess.Popen(
        cmd,
        stdout=serve_log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    _set_active("serve", proc)
    # Recorded separately from the process: killing the client does not stop the
    # container, so the interruption handler needs the name as well as the pid.
    _set_active("serve_container", container_name)
    base = f"http://127.0.0.1:{port}/v1"
    try:
        deadline = time.time() + config.SERVE_TIMEOUT_SECONDS
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                serve_log.flush()
                tail = _tail_text(serve_log_path)
                detail = f"; see {serve_log_path}"
                if tail:
                    detail += f"; log tail:\n{tail}"
                raise RuntimeError(f"{backend} server exited while starting (rc={proc.returncode}){detail}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(3)
        if not ready:
            raise TimeoutError(
                f"{backend} endpoint not ready within {config.SERVE_TIMEOUT_SECONDS}s"
            )
        log(f"[serve] {backend} endpoint ready at {base}")
        yield base
    finally:
        _set_active("serve", None)
        _set_active("serve_container", None)
        if proc.poll() is None:
            _kill_tree(proc)
        serve_log.close()
        if container_name:
            _remove_container(container_name)


def _build_command(
    model_path: Path,
    limit: Optional[int],
    extra_args: list,
    out_json: Path,
    api_base: Optional[str] = None,
    sample_ids: Optional[list] = None,
) -> list:
    """The command that scores one attempt: the task's own evaluate.py, on this host.

    One command shape serves both Harbor-backed shims and benchmark-owned direct
    evaluators: each accepts the delivered model path, an OpenAI-compatible endpoint,
    a work directory, selection arguments, and the output contract. Harbor-backed task
    code remains isolated in task containers; direct evaluators run only their thin,
    repository-owned scoring orchestration on the host.
    """
    cmd = [
        sys.executable,
        "evaluate.py",
        "--model-path",
        str(model_path),
        "--templates-dir",
        TEMPLATES_REL,
        # Per-attempt, not the runner's shared default. Without this every attempt
        # writes to vtb-env/scratch/<benchmark>_run, so the retry ladder's later
        # attempts overwrite the trial tree and logs of the earlier ones -- and two
        # gradings of different models would land in the same place.
        #
        # Derived from out_json, which is already named metrics_p<phase>_a<attempt>,
        # so this needs no new arguments to stay unique.
        "--work-dir",
        str(out_json.parent / f"harness_{out_json.stem}"),
    ]
    if api_base:
        cmd += ["--api-base", api_base]
    # limit=None means "leave evaluate.py's own default subset alone". Passing
    # --limit 0 would NOT do that: evaluate.py forwards any value other than -1
    # straight through, so 0 would evaluate zero samples.
    if limit is not None:
        cmd += ["--limit", str(limit)]
    cmd += list(extra_args)
    # Named samples rather than a prefix. The service has already checked they are
    # inside the validation split; --limit is sent as -1 alongside, so the two
    # selections cannot combine into something neither side asked for.
    if sample_ids:
        cmd += ["--only", *[str(sid) for sid in sample_ids]]
    cmd += ["--json-output-file", str(out_json)]
    return cmd


def attempt_log_path(work_dir: Path, phase: int, attempt: int) -> Path:
    """Where one attempt's stdout lands. Shared so the service can tail it live."""
    return work_dir / f"eval_p{phase}_a{attempt}.log"


def inspect_log_dir(work_dir: Path) -> Path:
    """Where inspect-ai writes its own evaluation logs, per job.

    Without INSPECT_LOG_DIR, inspect defaults to `./logs` relative to the working
    directory -- which is the task directory inside the checkout, so evaluations
    were quietly depositing logs into the repo (2.3 GB accumulated across tasks).
    Runtime artifacts belong under VTB_ROOT, and per-job keeps concurrent
    evaluations from interleaving.
    """
    return work_dir / "inspect_logs"


def newest_inspect_log(work_dir: Path) -> Optional[Path]:
    logs = sorted(inspect_log_dir(work_dir).glob("*.json"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def dataset_total(work_dir: Path) -> Optional[int]:
    """Full sample count of the evaluated dataset, from inspect's own log.

    inspect records `eval.dataset.samples` as the size of the *whole* dataset
    even when --limit trimmed the run, so this is the denominator a final-mode
    evaluation needs in order to report a percentage. The log appears once the
    task starts, i.e. after the model server has booted.
    """
    log = newest_inspect_log(work_dir)
    if log is None:
        return None
    try:
        with open(log, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    total = ((data.get("eval") or {}).get("dataset") or {}).get("samples")
    return int(total) if isinstance(total, int) and total > 0 else None


def _run_once(
    model_path: Path,
    limit: Optional[int],
    extra_args: list,
    work_dir: Path,
    phase: int,
    attempt: int,
    log: Logger,
    eval_task: str,
    api_base: Optional[str] = None,
    sample_ids: Optional[list] = None,
    harness_snapshot: Optional[Path] = None,
    harness_import_path: Optional[str] = None,
) -> AttemptResult:
    out_json = work_dir / f"metrics_p{phase}_a{attempt}.json"
    log_path = attempt_log_path(work_dir, phase, attempt)
    out_json.unlink(missing_ok=True)

    started = time.time()
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    # Keep inspect's logs out of the checkout; see inspect_log_dir().
    log_dir = inspect_log_dir(work_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    env["INSPECT_LOG_DIR"] = str(log_dir)
    if harness_snapshot is not None or harness_import_path is not None:
        if harness_snapshot is None or harness_import_path is None:
            raise RuntimeError("editable harness snapshot and import path must be provided together")
        # The import path names the package and module, e.g. harness.mini_swe_agent:X
        # -> harness/mini_swe_agent.py must be in the snapshot.
        entry = harness_snapshot / (harness_import_path.split(":", 1)[0].split(".", 1)[1] + ".py")
        if not entry.is_file():
            raise RuntimeError(f"editable harness snapshot is incomplete: {harness_snapshot}")
        # The parent holds the `harness` package and is server-owned after submit.
        env["VTB_EDITABLE_HARNESS_ROOT"] = str(harness_snapshot.parent)
        env["VTB_EDITABLE_HARNESS_AGENT"] = harness_import_path
    if config.EVAL_GPU_SPEC:
        # Pin anything this attempt starts itself to the reserved card pool. The served
        # model gets the same cards through `--gpus device=`, which is stronger; this
        # covers the orchestration process.
        env["CUDA_VISIBLE_DEVICES"] = config.EVAL_GPU_SPEC

    # What the evaluation needs from this host:
    #
    # DOCKER_HOST: harbor starts each task's container itself. With this
    #   unset the client follows the active docker context, so which daemon a score
    #   comes from would depend on a setting outside this repo.
    #   runner_paths.docker_host() is the one place that decides.
    # PATH: the harbor installed for this checkout wins over anything else on PATH, so
    #   a score comes from the version this repo pins.
    # XDG_CACHE_HOME: harbor's registry metadata cache. It must match the prefetch
    #   script so a prepared dataset is reused during grading.
    env.setdefault("XDG_CACHE_HOME", str(config.VTB_ROOT / "cache"))
    docker_host = runner_paths.docker_host()
    if docker_host:
        env["DOCKER_HOST"] = docker_host
    tools_bin = config.VTB_ROOT / "tools" / "bin"
    if tools_bin.is_dir():
        env["PATH"] = f"{tools_bin}:{env.get('PATH', '')}"

    with huggingface_overlay(work_dir, log) as hf_source:
        # Scoring reads the shared cache and writes to a throwaway upper layer. Every
        # evaluation runs on the host, so the overlay has to be pointed at through the
        # environment here -- otherwise the mount is built and then ignored while
        # downloads land in the shared cache.
        env["HF_HOME"] = str(hf_source)
        cmd = _build_command(model_path, limit, extra_args, out_json, api_base, sample_ids)

        # Where evaluate.py runs. It is found relative to its own task directory, and a
        # subprocess given no cwd would look wherever the service was started from and
        # report the file missing.
        cwd = str(config.task_dir(eval_task))

        log(f"[eval] phase {phase} attempt {attempt}: {' '.join(cmd)}")
        error: Optional[str] = None
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"$ {' '.join(cmd)}\n\n")
            fh.flush()
            proc = subprocess.Popen(
                cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                start_new_session=True
            )
            # Registered so a signal handler can kill this tree; the new session
            # means it would otherwise survive the service and keep the GPU.
            _set_active("proc", proc)
            try:
                rc = proc.wait(timeout=config.EVAL_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                rc = -1
                error = f"evaluation exceeded {config.EVAL_TIMEOUT_SECONDS}s"
                _kill_tree(proc)
            finally:
                _set_active("proc", None)

    elapsed = time.time() - started

    metrics: Optional[dict] = None
    score: Optional[float] = None
    if out_json.exists():
        try:
            metrics = json.loads(out_json.read_text(encoding="utf-8"))
            raw = metrics.get("accuracy")
            if isinstance(raw, (int, float)):
                score = float(raw)
        except (OSError, ValueError) as exc:
            error = error or f"could not parse metrics: {exc}"

    if score is None and error is None:
        tail = _tail(log_path, 40)
        error = f"exit code {rc}, no accuracy in metrics output. log tail:\n{tail}"
    elif rc != 0 and error is None:
        # A non-zero exit is the harness saying it did not do the work, and it outranks
        # any number left in the metrics file. Without this an evaluation that selected
        # no samples at all -- harbor refuses a task filter that matches nothing --
        # returned accuracy 0.0 and was recorded as a finished evaluation, which reads
        # as "the model solved none of them" rather than "nothing was evaluated".
        tail = _tail(log_path, 40)
        error = (
            f"exit code {rc} with accuracy={score} in the metrics output; the harness "
            f"failed, so that number does not describe an evaluation. log tail:\n{tail}"
        )

    ok = score is not None and error is None
    log(f"[eval] phase {phase} attempt {attempt}: {'accuracy=%.6f' % score if ok else 'FAILED'} ({elapsed:.0f}s)")
    return AttemptResult(ok, score, metrics, phase, attempt, elapsed, None if ok else error, log_path, extra_args)


def _kill_tree(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            continue


def _tail(path: Path, lines: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(log unavailable)"
    return "\n".join(content[-lines:])


# -------------------------------------------------------------- retry ladder


def evaluate(
    ctx: RunContext,
    model_path: Path,
    limit: Optional[int],
    work_dir: Path,
    log: Logger,
    on_attempt: Optional[Callable[[int, int, Path], None]] = None,
    on_gpu_wait: Optional[WaitCallback] = None,
    eval_task: Optional[str] = None,
    job_id: Optional[str] = None,
    sample_ids: Optional[list] = None,
    harness_snapshot: Optional[Path] = None,
    harness_import_path: Optional[str] = None,
) -> dict:
    """Run the three-phase ladder, returning the first successful result.

    Phase 1 uses evaluate.py's own budget; phases 2 and 3 may lower a limit for the
    benchmarks that have one worth lowering (see config.TOKEN_LADDER). The retries are
    there for evaluations that fail for an infrastructure reason rather than because the
    model was wrong.

    on_attempt is called with (ladder phase, attempt within phase, log path)
    just before each attempt starts, so the service can report live progress by
    tailing the log of whichever attempt is currently running.

    eval_task selects which benchmark's evaluate.py runs, defaulting to the
    run's own. It differs only when validation is configured to use a separate
    benchmark, in which case the token budget ladder follows that benchmark too.

    job_id is what a cancellation names. Checked between attempts rather than only at
    the start, because the retries are the whole difficulty: a cancelled attempt looks
    like a failed one, and the ladder's answer to a failure is another attempt.
    """
    eval_task = eval_task or ctx.task
    work_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict] = []

    def cancelled() -> dict:
        log(f"[eval] stopped after {len(attempts)} attempt(s): cancelled by the run")
        return {
            "ok": False,
            "cancelled": True,
            "score": None,
            "metrics": None,
            "error": "cancelled by the run that submitted it",
            "attempt_log": attempts,
        }

    if not wait_for_gpu(log, on_gpu_wait=on_gpu_wait, job_id=job_id):
        if cancel_requested(job_id):
            return cancelled()
        return {
            "ok": False,
            "score": None,
            "metrics": None,
            "error": "evaluation GPU never became available",
            "attempt_log": attempts,
        }

    def run_ladder(api_base: Optional[str]) -> Optional[dict]:
        for phase, budget in enumerate(config.PHASE_ATTEMPTS, start=1):
            extra_args = config.token_args(eval_task, phase)
            for attempt in range(1, budget + 1):
                if cancel_requested(job_id):
                    return cancelled()
                time.sleep(config.ATTEMPT_SETTLE_SECONDS)
                if on_attempt is not None:
                    on_attempt(phase, attempt, attempt_log_path(work_dir, phase, attempt))
                result = _run_once(
                    model_path, limit, extra_args, work_dir, phase, attempt, log, eval_task,
                    api_base, sample_ids, harness_snapshot, harness_import_path,
                )
                attempts.append(result.to_json())
                if result.ok:
                    return {
                        "ok": True,
                        "score": result.score,
                        "metrics": result.metrics,
                        "error": None,
                        "attempt_log": attempts,
                        "winning_attempt": len(attempts),
                    }
                # The attempt that a cancellation kills reports an exit code and no
                # metrics, which is the same shape as a broken one. Only the flag
                # tells them apart, so it is read before retrying.
                if cancel_requested(job_id):
                    return cancelled()
        return None

    # Harbor-backed and direct evaluators drive the model over HTTP, so the worker
    # serves it once and hands evaluate.py the endpoint. A future self-served benchmark
    # may opt out through config.needs_served_model.
    if config.needs_served_model(eval_task):
        try:
            with served_model(model_path, log, work_dir) as api_base:
                success = run_ladder(api_base)
        except (RuntimeError, TimeoutError) as exc:
            # A cancellation kills the server mid-startup, which surfaces here as the
            # server having exited. Reported as what it was, not as a serving failure.
            if cancel_requested(job_id):
                return cancelled()
            return {
                "ok": False,
                "score": None,
                "metrics": None,
                "error": f"could not serve the model for {eval_task}: {exc}",
                "attempt_log": attempts,
            }
    else:
        success = run_ladder(None)

    if success is not None:
        return success

    return {
        "ok": False,
        "score": None,
        "metrics": None,
        "error": f"all {len(attempts)} evaluation attempts failed",
        "attempt_log": attempts,
    }
