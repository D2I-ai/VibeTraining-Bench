# Generic researcher selection shared by benchmark launchers (host-side only).
source "$(dirname "${BASH_SOURCE[0]}")/load_launch_env.sh" || return 1
AGENT_CLI="${AGENT_CLI:-claude}"
RESEARCHER="${RESEARCHER:-}"
RESEARCHER_SAFE=""
if [ -n "$RESEARCHER" ]; then
    RESEARCHER_SAFE="$("$VTB_PYTHON" "$REPO/agents/config.py" model-slug --researcher "$RESEARCHER")" || return 1
fi
# Freeze explicit selections across detached tmux shells; do not inherit stale server env.
export AGENT_CLI RESEARCHER
VTB_AGENT_ADAPTERS_DIR_Q="$(printf '%q' "${VTB_AGENT_ADAPTERS_DIR:-}")"
VTB_LOCAL_ENV_FILE_Q="$(printf '%q' "${VTB_LOCAL_ENV_FILE:-}")"
require_researcher() {
    "$VTB_PYTHON" "$REPO/agents/config.py" check "$AGENT_CLI" --researcher "$RESEARCHER"
}
remember_smoke_run() {
    "$VTB_PYTHON" "$REPO/agents/config.py" record-result "$AGENT_CLI" --researcher "$RESEARCHER" \
        --state "$1" --default-result-dir "$2"
}
smoke_result_path() {
    "$VTB_PYTHON" "$REPO/agents/config.py" result-path "$AGENT_CLI" --researcher "$RESEARCHER" \
        --state "$1" --default-result-dir "$2"
}
