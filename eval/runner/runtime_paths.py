"""Where things live. One definition, derived from the checkout's own location.

The git checkout holds only source: code, and the container *definitions*
(containers/*.Dockerfile). Everything a run produces -- scratch, results, grading
state, the model cache, the backend's caches -- lands under <repo>/vtb-env/, which
.gitignore excludes and which can be deleted and repopulated from scratch.

Container images are the one thing that is neither. They are build products, but
they live in the docker daemon's own store and are named by tag, so there is no
path here that holds them, and nothing has to check whether an image file exists
before using it.

Nothing reaches above the checkout and nothing is a symlink: a clone works with no
configuration. Overriding is one variable -- VTB_ROOT moves all of it -- and the
specific variables below move one piece each.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# <repo>/eval/runner/runtime_paths.py -> parents[2] is the checkout.
REPO_ROOT = Path(os.environ.get("VTB_REPO") or Path(__file__).resolve().parents[2]).expanduser()

# Everything a run produces: disposable, gitignored, safe to delete.
VTB_ROOT = Path(os.environ.get("VTB_ROOT") or (REPO_ROOT / "vtb-env")).expanduser()

# The container definitions -- inputs to a build, tracked in git.
CONTAINER_DEFS_DIR = REPO_ROOT / "containers"

# The images built from them, by tag. containers/build_image.sh builds these names
# and containers/README.md says what each is for; one variable moves the whole set,
# and one VTB_*_IMAGE moves a single image to any reference a daemon can resolve.
IMAGE_NAMESPACE = os.environ.get("VTB_IMAGE_NAMESPACE") or "vibetrainbench"
IMAGE_TAG = os.environ.get("VTB_IMAGE_TAG") or "latest"


def _image(name: str, env: str) -> str:
    return os.environ.get(env) or f"{IMAGE_NAMESPACE}/{name}:{IMAGE_TAG}"


# The sandbox a researcher agent works in, and the model server.
TRAINER_IMAGE = _image("trainer", "VTB_TRAINER_IMAGE")
SGLANG_IMAGE = _image("sglang", "VTB_SGLANG_IMAGE")

# The two halves of what a run produces, and the distinction is worth keeping straight:
#
#   RESULTS_DIR  what is kept: the delivered final_model, metrics, audit manifests, the
#                agent's trace. Small, and the record of the run.
#   SCRATCH_DIR  what is worked in: job sandboxes, intermediate checkpoints, backend
#                trial trees. Large, and deletable once results have been collected.
#
# src/commit_utils/set_env_vars.sh sets the same two variables for the shell side, so a
# training run and the benchmark harness agree on both.
RESULTS_DIR = Path(os.environ.get("VTB_RESULTS_DIR") or (VTB_ROOT / "results"))
SCRATCH_DIR = Path(os.environ.get("VTB_SCRATCH") or (VTB_ROOT / "scratch"))
# Harbor and other registry-aware tools follow XDG_CACHE_HOME. Pin the default under
# vtb-env so a prefetch run and a later evaluation hit the same downloaded metadata.
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME") or (VTB_ROOT / "cache"))
GRADING_STATES_DIR = VTB_ROOT / "grading_states"
GRADING_STATE_DIR = Path(os.environ.get("VTB_GRADING_STATE") or (GRADING_STATES_DIR / "direct"))
HF_HOME = Path(os.environ.get("HF_HOME") or (VTB_ROOT / "resources" / "data" / "hf_cache"))

# Harbor dataset exports. Most benchmarks use these for inspection/image discovery;
# Tau3 consumes its validated export directly so evaluation never depends on an implicit
# per-user Harbor cache or a run-time patch.
HUB_TASK_SPECS_DIR = Path(
    os.environ.get("VTB_HUB_TASK_SPECS")
    or os.environ.get("VTB_HARBOR_DATASET_DIR")  # legacy prefetch alias
    or (VTB_ROOT / "hub_task_specs")
)
# Tau3 is evaluated from its validated local export, never from Harbor's implicit
# ~/.cache. This is the one canonical dataset root used by preparation and grading.
TAU3_TASK_SPECS_DIR = (
    HUB_TASK_SPECS_DIR / "sierra-research__tau3-bench" / "tau3-bench"
)

# Benchmark execution tools installed on the host by scripts/prepare/install_backend_tools.sh: uv, the
# standalone interpreter it downloads, and the harbor entry points. Under
# vtb-env so a wipe takes them too and a second checkout gets its own.
#
# task_backends.py resolves a tool by name from here, which is what keeps a run on the
# version this repo pins rather than one that happens to be on PATH.
TOOLS_DIR = VTB_ROOT / "tools"
TOOLS_BIN = TOOLS_DIR / "bin"

# Pinned third-party packages shipped into agent workspaces as references (today: the
# mini-swe-agent wheel an editable SWE harness is developed against). Downloaded once,
# verified against the index's published sha256, reused by every later preparation.
def packages_dir() -> Path:
    """Where the pinned reference packages live.

    A function rather than a constant so the value is read when it is used, like
    eval_concurrency below: workspace preparation can run in a process whose
    environment differs from import time.
    """
    override = os.environ.get("VTB_PACKAGES")
    if override:
        return Path(override).expanduser()
    return VTB_ROOT / "packages"

# Where a rootful daemon listens. Named because "use the system daemon" has to be said
# as an address, not as an absence: with no DOCKER_HOST the CLI falls back to the active
# *docker context*, and the rootless installer creates and selects a `rootless` context.
# Leaving DOCKER_HOST unset therefore keeps a run on the rootless daemon while looking
# like it switched -- which is exactly what happened here once.
SYSTEM_DOCKER_SOCKET = "unix:///var/run/docker.sock"


def docker_host() -> Optional[str]:
    """Which docker daemon a task container should be built and run by.

    The system (rootful) daemon is the default. Rootless needs no group membership and
    kept its images inside vtb-env, but it has two limits this benchmark ran into: its
    subuid range maps container UIDs only up to 65535, so swe-bench's images fail to
    unpack (UID 197609), and under cgroup v1 it can neither reach GPUs nor enforce the
    cpu/memory limits a task.toml asks for.

    Resolution, in order:
      VTB_DOCKER_HOST        this repo's explicit override, whatever it names.
      VTB_DOCKER_ROOTLESS=1  the per-user rootless socket, for a host with no rootful
                             daemon or no docker group membership.
      DOCKER_HOST            respected as inherited.
      system socket          the default.

    Every branch returns an address rather than None, because "use the system daemon"
    cannot be said by leaving DOCKER_HOST unset: the CLI then follows the active docker
    *context*, and the rootless installer creates and selects a `rootless` one. An unset
    variable therefore keeps a run on the rootless daemon while looking like it switched.
    """
    override = os.environ.get("VTB_DOCKER_HOST")
    if override:
        return override
    if (os.environ.get("VTB_DOCKER_ROOTLESS") or "").lower() in ("1", "true", "yes"):
        return f"unix:///run/user/{os.getuid()}/docker.sock"
    inherited = os.environ.get("DOCKER_HOST")
    if inherited:
        return inherited
    return SYSTEM_DOCKER_SOCKET


def docker_gpus_arg(gpu_spec: str) -> str:
    """Docker CLI --gpus value for all GPUs or a host device list."""
    gpu_spec = gpu_spec.strip()
    if not gpu_spec:
        return "all"
    if "," in gpu_spec:
        # The Docker CLI parses --gpus as CSV; multi-device lists must be quoted
        # inside the argument even when subprocess already bypasses the shell.
        return f'"device={gpu_spec}"'
    return f"device={gpu_spec}"


EVAL_CONCURRENCY_DEFAULT = 16


def eval_concurrency() -> int:
    """Trials the evaluation runs in parallel.

    Defined once because more than the runner depends on it: the grading service keys a
    run's timing history on it and derives an ETA from it, so a value read differently
    in two places would file a duration under one concurrency and read it back under
    another. scripts/prepare/configure_evaluation_environment.sh exports the same variable
    for the shell launchers.

    A function rather than a constant so the value is read when it is used; a
    nonsensical setting falls back rather than propagating a zero into a division.
    """
    try:
        value = int(os.environ.get("VTB_EVAL_CONCURRENT") or EVAL_CONCURRENCY_DEFAULT)
    except (TypeError, ValueError):
        return EVAL_CONCURRENCY_DEFAULT
    return value if value > 0 else EVAL_CONCURRENCY_DEFAULT


def ensure() -> None:
    """Create the directories a run writes to.

    Images are not here: `docker build` needs no prepared directory, which is one
    of the things that got simpler. HF_HOME is, because its default is inside
    vtb-env and a first run has no cache yet.
    """
    for path in (VTB_ROOT, RESULTS_DIR, SCRATCH_DIR, GRADING_STATE_DIR,
                 CACHE_DIR, HUB_TASK_SPECS_DIR, HF_HOME):
        path.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Every resolved path and image, for a preflight check or a run's own log."""
    rows = [
        ("VTB_REPO", REPO_ROOT),
        ("VTB_ROOT", VTB_ROOT),
        ("container defs", CONTAINER_DEFS_DIR),
        ("trainer image", TRAINER_IMAGE),
        ("sglang image", SGLANG_IMAGE),
        ("docker host", docker_host()),
        ("backend tools", TOOLS_BIN),
        ("results", RESULTS_DIR),
        ("registry cache", CACHE_DIR),
        ("hub task specs", HUB_TASK_SPECS_DIR),
        ("Tau2/Tau3 task specs", TAU3_TASK_SPECS_DIR),
        ("pinned packages", packages_dir()),
        ("scratch", SCRATCH_DIR),
        ("grading states root", GRADING_STATES_DIR),
        ("grading state", GRADING_STATE_DIR),
        ("HF_HOME", HF_HOME),
    ]
    width = max(len(name) for name, _ in rows)
    lines = []
    for name, value in rows:
        # An image reference and a daemon address are not paths, so the existence
        # note that is useful for a directory would be nonsense next to them.
        note = "" if isinstance(value, str) else ("" if Path(value).exists() else "   (missing)")
        lines.append(f"  {name:<{width}}  {value}{note}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
