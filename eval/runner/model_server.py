"""Serving the checkpoint under test, locally, for a whole suite.

Generic middleware: the model does not belong to the benchmark backend. Whatever runs the
task containers -- harbor -- the weights are loaded once on this host and reached
over an OpenAI-compatible HTTP endpoint. Serving is sglang.

The server runs in docker. `VTB_SERVE_RUNTIME=host` serves from this host's own
install instead, for a machine where one is already there.

Only used when no --api-base was given. Under the grading service the model is served
upstream by eval.grading.executor.
"""

from __future__ import annotations

import atexit
import os
import shlex
import sys
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

from . import runtime_paths

SGLANG_GRPC_PORT_OFFSET = 10000
SGLANG_MAX_HTTP_PORT = 65535 - SGLANG_GRPC_PORT_OFFSET


# Loading a large checkpoint and capturing CUDA graphs is minutes, and a suite waits
# on it before the first task. VTB_SERVE_TIMEOUT is read by the grading service's
# executor as well, so one endpoint has one deadline whichever side started it.
HEALTH_TIMEOUT = float(os.environ.get("VTB_SERVE_TIMEOUT") or 1800)


def api_key() -> str:
    """Endpoint API key."""
    return os.environ.get("VTB_API_KEY") or "inspectai"


def serve_backend(value: Optional[str] = None) -> str:
    backend = (value or os.environ.get("VTB_SERVE_BACKEND") or "sglang").strip().lower()
    if backend != "sglang":
        raise RuntimeError(f"serve backend must be 'sglang', got {backend!r}")
    return backend


def serve_runtime(backend: str) -> str:
    """How to run the server: in a container, or from whatever is on this host.

    docker is the default, and the only one with nothing to install.

    `host` is the escape hatch: a machine where sglang is already importable
    and running a container would only add a layer. It has to be asked for by name,
    so a machine never falls back to it silently.
    """
    runtime = (os.environ.get("VTB_SERVE_RUNTIME") or "docker").strip().lower()
    if runtime not in {"docker", "host"}:
        raise RuntimeError(f"serve runtime must be 'docker' or 'host', got {runtime!r}")
    return runtime


def csv_list(raw: str) -> list[str]:
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def eval_gpus() -> list[str]:
    return csv_list(os.environ.get("VTB_EVAL_GPUS", "").strip() or os.environ.get("VTB_EVAL_GPU", "").strip())


def eval_gpu_spec() -> str:
    return ",".join(eval_gpus()) or os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()


def has_flag(args: list[str], *flags: str) -> bool:
    for arg in args:
        for flag in flags:
            if arg == flag or arg.startswith(f"{flag}="):
                return True
    return False


def port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def free_port(*, max_port: int = 65535, reserve_offset: int = 0) -> int:
    if reserve_offset <= 0:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    import random

    low = 20000
    high = min(max_port, 65535 - reserve_offset)
    if high < low:
        raise RuntimeError(f"no valid port range leaves room for offset {reserve_offset}")

    for _ in range(1000):
        port = random.randint(low, high)
        if port_is_free(port) and port_is_free(port + reserve_offset):
            return port
    raise RuntimeError(
        f"could not find a free port pair in [{low}, {high}] with offset {reserve_offset}"
    )


def validate_sglang_port(port: int) -> None:
    grpc_port = port + SGLANG_GRPC_PORT_OFFSET
    if grpc_port > 65535:
        raise RuntimeError(
            f"SGLang HTTP port {port} would derive invalid gRPC port {grpc_port}; "
            f"set VTB_SERVE_PORT <= {SGLANG_MAX_HTTP_PORT} or leave it unset"
        )
    if not port_is_free(grpc_port):
        raise RuntimeError(
            f"SGLang derived gRPC port {grpc_port} is already in use; "
            "choose another VTB_SERVE_PORT or leave it unset"
        )


def _host_has_sglang() -> bool:
    proc = subprocess.run(
        [sys.executable, "-c", "import sglang"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


class ModelServer:
    """The delivered model, served once for the whole suite.

    Sampling parameters are deliberately not passed. They come from the
    generation_config.json shipped inside the delivered model directory, which is the
    rule every benchmark here follows -- overriding them would score a model under
    settings it did not ship.
    """

    def __init__(
        self,
        model_path: str,
        *,
        served_name: str,
        label: str = "eval",
        backend: Optional[str] = None,
        fixed_port: Optional[int] = None,
    ) -> None:
        self.model_path = model_path
        self.served_name = served_name
        self.label = label
        self.backend = serve_backend(backend)
        self.runtime = serve_runtime(self.backend)
        self.fixed_port = fixed_port
        self.port: Optional[int] = None
        self.process: Optional[subprocess.Popen] = None
        self.how: str = "unknown"

    def _tool_parser(self) -> str:
        return (
            os.environ.get("VTB_SERVE_TOOL_CALL_PARSER")
            or os.environ.get("VTB_TOOL_CALL_PARSER")
            or "qwen3_coder"
        )

    def _reasoning_parser(self) -> str:
        return os.environ.get("VTB_SERVE_REASONING_PARSER") or "qwen3"

    def _payload(self, port: int) -> list[str]:
        key = api_key()
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--api-key",
            key,
            "--served-model-name",
            self.served_name,
            "--trust-remote-code",
        ]
        if self._tool_parser():
            command += ["--tool-call-parser", self._tool_parser()]
        if self._reasoning_parser():
            command += ["--reasoning-parser", self._reasoning_parser()]
        extra = shlex.split(os.environ.get("VTB_SERVE_EXTRA_ARGS") or "")
        extra += shlex.split(os.environ.get("VTB_SGLANG_EXTRA_ARGS") or "")
        gpu_count = len(eval_gpus())
        if gpu_count > 1 and not has_flag(extra, "--tp-size", "--tp"):
            command += ["--tp-size", str(gpu_count)]
        command += extra
        return command

    def _image(self) -> str:
        """The serving image.

        VTB_SERVE_SGLANG_DOCKER_IMAGE is honoured ahead of the generic name because
        it is what existing run scripts and notes here already set.
        """
        return (
            os.environ.get("VTB_SERVE_SGLANG_DOCKER_IMAGE")
            or os.environ.get("VTB_SGLANG_DOCKER_IMAGE")
            or runtime_paths.SGLANG_IMAGE
        )

    def _mounts(self) -> list[str]:
        """Host paths the server has to see, each at its own path inside.

        Same path on both sides so a model directory named on the command line means
        the same thing in both places. Resolved targets are included as well:
        vtb-env/ inside the checkout may be a symlink to a shared cache, and mounting
        only the link would land the container on a path that does not resolve in it.
        """
        model_dir = Path(self.model_path).expanduser().resolve()
        anchor = model_dir if model_dir.is_dir() else model_dir.parent
        mounts: list[str] = []
        seen: set[str] = set()
        for candidate in (
            anchor,
            runtime_paths.VTB_ROOT,
            runtime_paths.VTB_ROOT.resolve(),
            runtime_paths.HF_HOME,
            runtime_paths.HF_HOME.resolve(),
        ):
            if candidate.is_dir() and str(candidate) not in seen:
                seen.add(str(candidate))
                mounts += ["-v", f"{candidate}:{candidate}"]
        return mounts

    def _docker_command(self, payload: list[str], port: int) -> list[str]:
        image = self._image()
        command = [
            "docker",
            "run",
            "--rm",
            # Named so stop() can remove the container by name. Killing the docker
            # client is not enough: it is only a client, and a SIGKILL to it leaves
            # the server running and the card allocated.
            "--name",
            self._container_name(port),
            # The endpoint is reached over the host's own address, and the health
            # poll below goes to 127.0.0.1, so the server has to be on the host's
            # network namespace rather than behind a published port.
            "--network",
            "host",
            "--ipc",
            "host",
            "--shm-size",
            os.environ.get("VTB_SERVE_DOCKER_SHM_SIZE", "32g"),
        ]
        # `device=N` is real isolation: only that card is present inside the container,
        # so nothing running in here can land on a card it was not given. Multi-device
        # specs need Docker's quoted device-list form.
        command += ["--gpus", runtime_paths.docker_gpus_arg(eval_gpu_spec())]
        command += self._mounts()
        command += ["-e", f"HF_HOME={runtime_paths.HF_HOME}"]
        command += ["-e", f"VTB_API_KEY={api_key()}"]
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if os.environ.get(name):
                command += ["-e", f"{name}={os.environ[name]}"]
        # Explicit, so the payload runs whatever the image's own ENTRYPOINT is: a
        # serving image that sets one would otherwise swallow these arguments as
        # extra flags to a server it starts itself.
        command += ["--entrypoint", payload[0]]
        self.how = f"docker {image}"
        return [*command, image, *payload[1:]]

    def _container_name(self, port: int) -> str:
        return f"vtb-serve-{self.backend}-{self.label}-{port}".replace("_", "-")

    def _require_image(self) -> None:
        image = self._image()
        probe = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self._docker_env(),
        )
        if probe.returncode == 0:
            return
        raise RuntimeError(
            f"no way to serve the model with {self.backend}: the image {image} is not "
            f"present. Build it with `bash containers/build_image.sh {self.backend}`, "
            f"point VTB_{self.backend.upper()}_IMAGE at one that exists, set "
            "VTB_SERVE_RUNTIME=host to use this host's own install, or pass --api-base "
            "to use an endpoint that is already running."
        )

    def _docker_env(self) -> dict:
        """Environment for a docker command, with the daemon named explicitly.

        Not optional: with DOCKER_HOST unset the CLI follows the active docker
        *context*, so a host where a rootless context is selected serves from the
        rootless daemon while looking like it did not. runtime_paths.docker_host()
        is the one place that decides which daemon this repo uses.
        """
        env = dict(os.environ)
        host = runtime_paths.docker_host()
        if host:
            env["DOCKER_HOST"] = host
        return env

    def _serve_command(self, port: int) -> list[str]:
        payload = self._payload(port)
        if self.runtime == "docker":
            self._require_image()
            return self._docker_command(payload, port)

        if _host_has_sglang():
            self.how = "sglang on selected host interpreter"
            return [sys.executable, *payload[1:]]
        raise RuntimeError(
            "VTB_SERVE_RUNTIME=host was asked for, but sglang is not available "
            "on this host. Unset it to serve in a container, or pass --api-base to use "
            "an endpoint that is already running."
        )

    def _serving_plan(self, port: int) -> str:
        """Describe the resolved serving port, GPU allocation, and tensor parallelism.

        Tensor parallelism follows VTB_EVAL_GPUS and defaults to one when unset.
        Logging the resolved plan helps diagnose insufficient KV-cache capacity and
        repeated prefilling under concurrent evaluation.
        """
        gpus = eval_gpus()
        if gpus:
            return f"port={port} cards={','.join(gpus)} tp={len(gpus)}"
        return (
            f"port={port} cards=all-visible tp=1 "
            "(VTB_EVAL_GPUS unset: one card serves, whatever the host has)"
        )

    def start(self) -> str:
        import urllib.error
        import urllib.request

        if self.fixed_port:
            port = self.fixed_port
            validate_sglang_port(port)
        else:
            # The backend is always sglang (config.py refuses anything else at import),
            # so every port comes from the range that keeps the gRPC offset free.
            port = free_port(max_port=SGLANG_MAX_HTTP_PORT, reserve_offset=SGLANG_GRPC_PORT_OFFSET)
        command = self._serve_command(port)
        print(f"[{self.label}] serving the model with {self.backend} via {self.how}", flush=True)
        print(f"[{self.label}] serve plan: {self._serving_plan(port)}", flush=True)
        env = self._docker_env()
        if self.runtime == "host" and eval_gpu_spec():
            env["CUDA_VISIBLE_DEVICES"] = eval_gpu_spec()
        self.process = subprocess.Popen(command, start_new_session=True, env=env)
        self.port = port
        atexit.register(self.stop)

        deadline = time.time() + HEALTH_TIMEOUT
        while time.time() < deadline:
            if self.process.poll() is not None:
                self._remove_container()
                raise RuntimeError(f"{self.backend} exited while starting (rc={self.process.returncode})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                    if response.status == 200:
                        return f"http://127.0.0.1:{port}/v1"
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        self.stop(force=True)
        raise TimeoutError(f"{self.backend} did not become ready within {HEALTH_TIMEOUT:.0f}s")

    def _remove_container(self) -> None:
        """Remove the serving container, if this run is the thing that started one.

        A docker client is not the server: it is a client of a daemon, and the
        container survives the client being killed -- holding the whole card, so the
        next submission then waits out the free-memory gate and fails for a reason that
        looks unrelated. Hence an explicit removal.
        """
        if self.runtime != "docker" or self.port is None:
            return
        subprocess.run(
            ["docker", "rm", "--force", self._container_name(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self._docker_env(),
        )

    def stop(self, force: bool = False) -> None:
        if self.process is None or self.process.poll() is not None:
            self._remove_container()
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL if force else signal.SIGTERM)
        except (ProcessLookupError, OSError):
            self._remove_container()
            return
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        self._remove_container()


