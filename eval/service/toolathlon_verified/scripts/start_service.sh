#!/bin/bash
# Start the Toolathlon-VERIFIED evaluation service (official held-out FINAL
# test set) on this host, port 8061.
#
#   VTB_TOOLATHLON_VERIFIED_REPO=/abs/path/to/official/toolathlon_verified \
#   VTB_TOOLATHLON_VERIFIED_SECRETS=/abs/path/to/per-host/secrets \
#       bash eval/service/toolathlon_verified/scripts/start_service.sh
#
# Self-contained island under eval/service/. It never edits the official
# checkout: paths.py stages `official tree + this package's overlay/ + per-host
# secrets` into a throwaway copy under vtb-env/, and project code is docker-cp'd
# into the per-task container from that staged root (the official flow). No
# postgres dependency; the task image is pulled once, never built.
#
# What it does, each step skippable when already satisfied:
#   1. docker reachable
#   2. stage official+overlay+secrets -> vtb-env/toolathlon_verified_staged
#   3. digest-pinned Toolathlon task image present (docker pull, no build)
#   4. tasks/finalpool present in the staged tree
#   5. exact service dependencies present in the selected Python environment
#   6. service started with nohup on 0.0.0.0:8061, then /health polled
#
# Tunables (export before running):
#   VTB_TOOLATHLON_VERIFIED_REPO        REQUIRED  abs path to official checkout
#   VTB_TOOLATHLON_VERIFIED_SECRETS     optional  abs path to per-host secrets
#   TOOLATHLON_VERIFIED_MAX_CONCURRENT  default 12  tasks of one job at once
#   TASK_TIMEOUT                        default 2700 seconds per task (45 min)
#   TOOLATHLON_TASK_RETRIES             default 1   crash retries per task
#   TOOLATHLON_VERIFIED_PORT            default 8061
#   TOOLATHLON_VERIFIED_IMAGE           default lockon0927/toolathlon-task-image:1016beta@sha256:4d04fe4e0a6fdb4946f51bb05120cb44a0eef980231c11252f93b62897afcb9f
#
# SECURITY: the service has NO authentication, and 0.0.0.0 is required so the
# task containers (bridge networks) can reach the model endpoint through the
# docker0 gateway. The port must be restricted to trusted sources. GET /tasks
# answers with a count only -- this is a held-out set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"                 # eval/service/toolathlon_verified
REPO_ROOT="$(cd "$PKG_DIR/../../.." && pwd)"       # vibetrainbench checkout root
cd "$REPO_ROOT"
source "$REPO_ROOT/src/commit_utils/configure_python.sh" || exit 1

PORT="${TOOLATHLON_VERIFIED_PORT:-8061}"
IMAGE="${TOOLATHLON_VERIFIED_IMAGE:-lockon0927/toolathlon-task-image:1016beta@sha256:4d04fe4e0a6fdb4946f51bb05120cb44a0eef980231c11252f93b62897afcb9f}"
RUNTIME_DIR="${VTB_ROOT:-$REPO_ROOT/vtb-env}"
mkdir -p "$RUNTIME_DIR"

# Default the per-host secrets dir to the conventional location under the
# runtime dir when the caller did not set it explicitly. Kept beside (not
# inside) toolathlon_verified_staged so the staging rebuild never wipes it,
# and exported so the `python3 -c ... build_staged_root()` subprocess below
# picks it up as the third staging layer.
if [ -z "${VTB_TOOLATHLON_VERIFIED_SECRETS:-}" ] && [ -d "$RUNTIME_DIR/toolathlon_verified_secrets" ]; then
    export VTB_TOOLATHLON_VERIFIED_SECRETS="$RUNTIME_DIR/toolathlon_verified_secrets"
fi
LOG="$RUNTIME_DIR/toolathlon_verified-$PORT.log"
PIDFILE="$RUNTIME_DIR/toolathlon_verified-$PORT.pid"

echo "=== [1/6] docker ==="
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon unreachable"; exit 1; }
echo "  ok"

echo "=== [2/6] stage official + overlay + secrets ==="
if [ -z "${VTB_TOOLATHLON_VERIFIED_REPO:-}" ]; then
    echo "FATAL: VTB_TOOLATHLON_VERIFIED_REPO is not set; point it at the official"
    echo "       toolathlon_verified checkout (the read-only upstream this drives)."
    exit 1
fi
if [ -z "${VTB_TOOLATHLON_VERIFIED_SECRETS:-}" ]; then
    echo "  [warn] VTB_TOOLATHLON_VERIFIED_SECRETS unset; credentialed tasks"
    echo "         (canvas/notion/snowflake/...) will fail without it."
fi
STAGED="$("$VTB_PYTHON" -c 'from eval.service.toolathlon_verified import paths; print(paths.build_staged_root())')"
echo "  staged -> $STAGED"

echo "=== [3/6] image $IMAGE ==="
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "  present"
else
    echo "  pulling (official image, never built locally) ..."
    docker pull "$IMAGE"
fi

echo "=== [4/6] tasks/finalpool ==="
N=$(find "$STAGED/tasks/finalpool" -maxdepth 1 -type d ! -name finalpool | wc -l)
if [ "$N" -lt 100 ]; then
    echo "FATAL: only $N task dirs under staged tasks/finalpool (expected ~108)"
    exit 1
fi
echo "  $N task dirs (k8s tasks excluded at runtime)"

echo "=== [5/6] python deps for the service (fastapi/uvicorn) ==="
if ! "$VTB_PYTHON" scripts/prepare/python/verify_python_environment.py --requirements "$PKG_DIR/requirements.txt"; then
    echo "Install the pinned service dependencies into the selected environment first:"
    printf '  %q pip install --python %q -r %q\n' "$RUNTIME_DIR/tools/bin/uv" "$VTB_PYTHON" "$PKG_DIR/requirements.txt"
    exit 1
fi
echo "  ok"

echo "=== [6/6] start service on 0.0.0.0:$PORT ==="
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "  already running (pid $(cat "$PIDFILE")); stop it first:"
    echo "    kill \$(cat $PIDFILE)"
    exit 1
fi
export TOOLATHLON_VERIFIED_MAX_CONCURRENT="${TOOLATHLON_VERIFIED_MAX_CONCURRENT:-12}"
export TASK_TIMEOUT="${TASK_TIMEOUT:-2700}"
export TOOLATHLON_TASK_RETRIES="${TOOLATHLON_TASK_RETRIES:-1}"
# Module path (run from repo root) so the package's relative imports resolve.
nohup "$VTB_PYTHON" -m uvicorn eval.service.toolathlon_verified.app:app \
    --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "  pid $(cat "$PIDFILE"), log $LOG"
echo "  concurrency=$TOOLATHLON_VERIFIED_MAX_CONCURRENT task_timeout=${TASK_TIMEOUT}s retries=$TOOLATHLON_TASK_RETRIES"

for _ in $(seq 1 30); do
    if OUT=$(curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null); then
        echo "  $OUT"
        if echo "$OUT" | grep -q '"ok":true'; then
            echo
            echo "READY. Single-host deployment (same box as grading + serve):"
            echo "  export VTB_TOOLATHLON_VERIFIED_URL=http://172.17.0.1:$PORT"
            echo "  export VTB_MODEL_ADVERTISE_HOST=172.17.0.1"
            exit 0
        fi
        echo "  service is up but prerequisites are missing -- fix the above"
        exit 1
    fi
    sleep 2
done
echo "FATAL: service did not answer within 60s; check $LOG"
tail -30 "$LOG"
exit 1
