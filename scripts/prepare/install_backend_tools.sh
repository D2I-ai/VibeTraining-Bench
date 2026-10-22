#!/usr/bin/env bash
# Install the benchmark execution backend tool on the host, at a pinned version.
#
# harbor runs every benchmark's tasks and starts the task containers itself.
#
# Why on the host rather than in a container: it is a pip package plus a container
# CLI, and it starts the task containers itself. Running it inside another
# container added a nesting layer with no benefit -- one more place for tmpdir and
# namespace settings to be passed through, and two layers to look through when a trial
# fails. The task containers are still containers; only the runner moved.
#
# Host interpreters may be too old for it. The .python-version contract selects
# a uv-managed interpreter for the runtime and Harbor, so
# nothing depends on what the system ships. Versions come from
# containers/requirements.txt, the single place that says what decides a score --
# harbor pinned to a fork commit there (a PEP 508 direct reference), so the grep below
# accepts both `harbor==X` and `harbor @ git+...` forms.
#
# Everything lands under vtb-env/ so it is disposable with the rest of the runtime
# state, and .gitignore already excludes it.
#
# Usage:
#   bash scripts/prepare/install_backend_tools.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VTB_ROOT="${VTB_ROOT:-$REPO_ROOT/vtb-env}"
TOOLS_DIR="$VTB_ROOT/tools"
PIN_HELPER="scripts/prepare/python/verify_harbor.py"
HARBOR_REPO="${VTB_HARBOR_REPO:-$REPO_ROOT/../harbor}"

PYTHON_VERSION="$(tr -d '\r\n' < .python-version)"
mkdir -p "$TOOLS_DIR"

# uv, and the tools it installs, both under vtb-env: this must not touch ~/.local or
# the system, so a second checkout gets its own and nothing here outlives a wipe.
export UV_INSTALL_DIR="$TOOLS_DIR/bin"
export UV_TOOL_BIN_DIR="$TOOLS_DIR/bin"
export XDG_DATA_HOME="$TOOLS_DIR/data"
export UV_CACHE_DIR="$TOOLS_DIR/cache"
export UV_PYTHON_INSTALL_DIR="$TOOLS_DIR/python"
mkdir -p "$UV_INSTALL_DIR" "$XDG_DATA_HOME" "$UV_CACHE_DIR"

if [ ! -x "$UV_INSTALL_DIR/uv" ]; then
    echo "[install] uv -> $UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

# Bootstrap without executing repository code under the system interpreter.
if [ ! -x "$TOOLS_DIR/runtime/bin/python3" ]; then
    uv venv --python "$PYTHON_VERSION" --managed-python "$TOOLS_DIR/runtime"
fi
export VTB_PYTHON="$TOOLS_DIR/runtime/bin/python3"
source src/commit_utils/configure_python.sh
HARBOR_SPEC="$("$VTB_PYTHON" "$PIN_HELPER" pin --field requirement)"
HARBOR_COMMIT="$("$VTB_PYTHON" "$PIN_HELPER" pin --field commit)"
echo "[install] $HARBOR_SPEC"
uv tool install --force --python "$PYTHON_VERSION" --managed-python "$HARBOR_SPEC"

# Verify provenance, not only the package's human-readable version (the pinned fork
# reports 0.22.0). Then prepare the source checkout used by adapters/harness templates
# and require it to be the same clean commit as the installed CLI.
"$VTB_PYTHON" "$PIN_HELPER" verify-install --harbor-bin "$UV_INSTALL_DIR/harbor"
"$VTB_PYTHON" "$PIN_HELPER" prepare-source --checkout "$HARBOR_REPO"
# harbor starts the task containers through docker, so the CLI is a hard
# requirement. The daemon is only warned about: it can be started later, and which
# daemon a run uses is resolved at run time by runtime_paths.docker_host(), not here.
command -v docker >/dev/null || { echo "FATAL: docker not on PATH" >&2; exit 1; }
DOCKER_STATE="reachable"
docker info >/dev/null 2>&1 || DOCKER_STATE="NOT reachable -- start it, or set VTB_DOCKER_HOST / VTB_DOCKER_ROOTLESS=1"

echo
echo "installed:"
echo "  harbor CLI $($UV_INSTALL_DIR/harbor --version 2>&1 | head -1)  commit=$HARBOR_COMMIT"
echo "  harbor src $HARBOR_REPO  commit=$HARBOR_COMMIT"
echo "  docker     $(docker --version 2>&1 | head -1)  (daemon: $DOCKER_STATE)"
echo
echo "put it on PATH for a run:"
echo "  source src/commit_utils/configure_python.sh"
echo "  export PATH=\"$UV_INSTALL_DIR:\$PATH\""
echo "  export VTB_HARBOR_REPO=\"$HARBOR_REPO\""
