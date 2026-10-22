# Host interpreter contract. Source before executing repository Python code.
# Never resolve a shell alias/function named python, or silently use system Python 3.8.
_vtb_python_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_vtb_python_root="${VTB_ROOT:-$_vtb_python_repo/vtb-env}"
_vtb_python_version="$(tr -d '\r\n' < "$_vtb_python_repo/.python-version")"
_vtb_python="${VTB_PYTHON:-}"
if [ -z "$_vtb_python" ]; then
    for _vtb_candidate in "$_vtb_python_repo/.venv/bin/python3" "$_vtb_python_root/tools/runtime/bin/python3"; do
        if [ -x "$_vtb_candidate" ]; then _vtb_python="$_vtb_candidate"; break; fi
    done
    if [ -z "$_vtb_python" ] && [ -x "$_vtb_python_root/tools/bin/harbor" ]; then
        # Harbor's installed venv already uses the same canonical Python minor.
        _vtb_candidate="$(dirname "$(readlink -f "$_vtb_python_root/tools/bin/harbor")")/python3"
        [ ! -x "$_vtb_candidate" ] || _vtb_python="$_vtb_candidate"
    fi
fi
if [ -z "$_vtb_python" ]; then
    printf 'FATAL: Python %s environment missing. Run bash scripts/prepare/install_backend_tools.sh or set VTB_PYTHON to a compatible venv interpreter.\n' "$_vtb_python_version" >&2
    return 1
fi
case "$_vtb_python" in /*) ;; *) _vtb_python="$(type -P "$_vtb_python")" || return 1 ;; esac
[ -x "$_vtb_python" ] || { printf 'FATAL: VTB_PYTHON is not executable\n' >&2; return 1; }
# Keep the venv's executable path (resolving its symlink would lose its dependencies).
export VTB_PYTHON="$_vtb_python" PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
"$VTB_PYTHON" -c 'import sys; expected=tuple(map(int,sys.argv[1].split("."))); actual=sys.version_info[:2]; sys.exit(0 if actual==expected else "FATAL: Python %s required; selected interpreter is %s" % (sys.argv[1], ".".join(map(str,actual))))' "$_vtb_python_version" || return 1
unset _vtb_python_repo _vtb_python_root _vtb_python_version _vtb_python _vtb_candidate
