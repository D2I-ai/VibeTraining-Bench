# Environment for a training run. Sourced by src/run_task.sh.
#
# Every default here points inside vtb-env/, the one gitignored directory that holds
# everything a run produces. This file was the last place still defaulting to bare
# relative paths -- `results`, `containers` -- which wrote a run's output into the
# checkout. That is fixed below; the shape now matches eval/runner/runtime_paths.py,
# which is where the python side resolves the same locations and the same images.
#
# Anything already set in the environment wins, so a caller can still redirect a single
# path without editing this.

# The checkout, found from this file rather than from the caller's cwd: run_task.sh may
# be invoked from anywhere, and a relative default would then resolve somewhere else.
if [ -z "${VTB_REPO:-}" ]; then
    VTB_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    export VTB_REPO
fi

# Helper: set a variable to a default if it is unset, empty, or the literal UNDEFINED
# (which a job scheduler substitutes for a parameter nobody filled in).
set_default() {
    local var_name="${1:-}"
    local default_value="${2:-}"
    local current_value
    eval "current_value=\"\${$var_name:-}\""

    if [ -z "$current_value" ] || [ "$current_value" = "UNDEFINED" ]; then
        export "$var_name"="$default_value"
    fi
}

# Everything a run produces. Disposable, gitignored, safe to delete wholesale.
set_default VTB_ROOT "${VTB_REPO}/vtb-env"

# The Harbor checkout that seeds editable-harness templates and provides adapter-local
# packages (Tau). Defaults to the sibling checkout next to this repo -- a relative
# derivation, so no machine-specific absolute path is baked in. README "Setup" covers
# the clone; point this at another checkout to use a fork that lives elsewhere.
# The default only applies when the checkout actually exists: a set-but-invalid path
# fails _harbor_repo_path() loudly for every benchmark's evaluation, while an unset
# one is a clean fallback (installed-package harness seeds; Tau refused with a clear
# message). A checkout is recognized the way task_backends does: by its adapters/ dir.
if [ -d "${VTB_REPO}/../harbor/adapters" ]; then
    set_default VTB_HARBOR_REPO "$(realpath -m "${VTB_REPO}/../harbor")"
fi

# Where a run's delivery and logs go. Absolute, so `exec 1>${EVAL_DIR}/output.log`
# cannot land in the checkout.
#
# This holds the *keeping* half of a run: the delivered final_model, the metrics, the
# audit manifests, and a copy of the agent's trace. Small enough to keep indefinitely.
set_default VTB_RESULTS_DIR "${VTB_ROOT}/results"

# Where the run's sandbox lives: the job directory the agent works in, its whole
# filesystem, every intermediate checkpoint, and the raw session store. This is the
# *working* half -- large, and disposable once a run's results have been collected.
#
# Defaulted under VTB_ROOT rather than /tmp, so a training run's scratch lands on the
# same big disk as the rest of its output and stays inside the one directory meant to
# hold everything a run produces. eval/runner/runtime_paths.py reads the same variable,
# so the benchmark harness and the training sandbox agree on where scratch is.
set_default VTB_SCRATCH "${VTB_ROOT}/scratch"

# The images, as docker references rather than paths: the daemon resolves a tag, so
# there is no image directory to configure and nothing that can half-resolve to an
# image which was never built.
#
# The namespace/tag pair is the same one eval/runner/runtime_paths.py builds these names
# from, so `VTB_IMAGE_TAG=v2 bash src/run_task.sh ...` moves the whole set on both
# sides at once, and a single VTB_*_IMAGE still moves one image anywhere a daemon can
# resolve.
set_default VTB_IMAGE_NAMESPACE "vibetrainbench"
set_default VTB_IMAGE_TAG "latest"

# The image the agent works inside: slime, Megatron-LM and the agent CLIs. See
# containers/trainer.Dockerfile, built by `bash containers/build_image.sh trainer`.
set_default VTB_TRAINER_IMAGE "${VTB_IMAGE_NAMESPACE}/trainer:${VTB_IMAGE_TAG}"

# Which docker daemon this run talks to, said as an address. Mirrors
# runtime_paths.docker_host(), and for the same reason: with DOCKER_HOST unset the CLI
# follows the active docker *context*, and the rootless installer creates and selects a
# `rootless` one -- so "use the system daemon" cannot be expressed by leaving this
# unset. Set VTB_DOCKER_ROOTLESS=1 on a host with no docker group membership.
if [ -n "${VTB_DOCKER_HOST:-}" ]; then
    export DOCKER_HOST="$VTB_DOCKER_HOST"
elif [ "${VTB_DOCKER_ROOTLESS:-}" = "1" ]; then
    export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
else
    export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"
fi

# Host-side Hugging Face cache. Keep the default under ${VTB_ROOT} so a checkout is
# self-contained; operators can still point HF_HOME at an external cache before sourcing
# this file.
set_default HF_HOME "${VTB_ROOT}/resources/data/hf_cache"
set_default VTB_MODEL_RESOURCE_ROOT "${VTB_ROOT}/resources/models/qwen3_5_9b_base"

# Which prompt template eval/agent_task/ renders. `workflow` is the default general
# task; `research` is the full-detail variant. `harness_connectivity`, `train_test`,
# `hf_train_test` and `mock_test` are the bounded modes. Documents that are not
# prompts live in eval/agent_task/workspace_docs/ and are copied into the workspace,
# never rendered.
set_default VTB_PROMPT "workflow"

# Appended to the results directory name, for telling two batches apart.
set_default VTB_EXPERIMENT_NAME ""

# The agent's home *inside* its container: the job directory is mounted here, so this is
# where its workspace, its scaffold state and its shell history live. A container path,
# so it is fixed regardless of where the host keeps the job directory.
#
# The generic "agent" account avoids embedding a developer identity in container paths.
# eval/agent_task/workspace_docs/grading.md and
# containers/other_home_data/.codex use the same path.
export VTB_AGENT_HOME="/home/agent"

# Default the agent to the overlay-mounted Hugging Face cache view. The lowerdir is the
# repo-local host HF_HOME above, and the overlay was meant to take per-run writes to
# scratch -- but in practice it does not: the container runs as root while fuse-overlayfs
# runs as this account, so creating any new path in the merged tree fails with EPERM.
# Treat the agent's view as read-only. run_task.sh points HF_DATASETS_CACHE at a real
# bind mount so the datasets library has somewhere to write.
# Override VTB_AGENT_HF_HOME=/tmp/hf_cache for a fully isolated empty per-job cache.
set_default VTB_AGENT_HF_HOME "${VTB_AGENT_HOME}/hf_cache"

# Refuse hub traffic from inside the agent container. Every dataset and checkpoint a run
# may train on is declared in its resource manifest and already in the cache above, so a
# hub call means either a mistake or a reach outside the manifest. Off by default it
# still failed, just later and less legibly: an unauthenticated request went out to
# huggingface.co and then died on EPERM creating a lock directory, which one run read as
# a download problem and spent several turns on. Local paths are unaffected. Set to 0
# only for a run that is deliberately allowed to fetch (an empty per-job cache, say).
set_default VTB_AGENT_HF_OFFLINE "1"

# Optional train-test resource selection. Recipes are named files under training/recipes/;
# direct resources are comma-separated ids under training/resources/ (for debugging only).
set_default VTB_TRAIN_RECIPE ""
set_default VTB_TRAIN_RESOURCES ""

# Keep the host account's ~/.local/lib off the container's import path. The agent's
# home is a bind mount, so without this a `pip install --user` on the host would be
# importable inside the container and a run would depend on this machine. Installing
# inside the trainer still works: it runs as root, and writes to the image's own
# site-packages.
export PYTHONNOUSERSITE=1
