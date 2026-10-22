# Source trusted host-side configuration; never bind this file into the agent home.
# VTB_LOCAL_ENV_FILE is opt-in executable shell configuration, not a data file.
_vtb_launch_repo="${REPO:-${VTB_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
_vtb_local_file="${VTB_LOCAL_ENV_FILE:-}"
_vtb_had_allexport=0
case "$-" in *a*) _vtb_had_allexport=1 ;; esac
set -a
if [ -f "$_vtb_launch_repo/api_key.env" ]; then
    source "$_vtb_launch_repo/api_key.env" || return 1
fi
if [ -n "$_vtb_local_file" ]; then
    case "$_vtb_local_file" in /*) ;; *) _vtb_local_file="$_vtb_launch_repo/$_vtb_local_file" ;; esac
    [ -f "$_vtb_local_file" ] || { printf 'FATAL: local configuration missing: %s\n' "$_vtb_local_file" >&2; return 1; }
    _vtb_local_file="$(realpath "$_vtb_local_file")"
    source "$_vtb_local_file" || return 1
fi
source "$(dirname "${BASH_SOURCE[0]}")/configure_python.sh" || return 1
export CODEX_API_KEY="${CODEX_API_KEY:-${OPENAI_API_KEY:-}}"
export VTB_LOCAL_ENV_FILE="$_vtb_local_file"
export VTB_AGENT_ADAPTERS_DIR="${VTB_AGENT_ADAPTERS_DIR:-}"
[ "$_vtb_had_allexport" -eq 1 ] || set +a
unset _vtb_launch_repo _vtb_local_file _vtb_had_allexport
