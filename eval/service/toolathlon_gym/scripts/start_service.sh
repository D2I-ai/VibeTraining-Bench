#!/bin/bash
# Start the Toolathlon-GYM evaluation service, parallel mode (port 8060).
#
#   VTB_TOOLATHLON_GYM_REPO=/abs/path/to/official/toolathlon_gym \
#       bash eval/service/toolathlon_gym/scripts/start_service.sh
#
# This service is a self-contained island under eval/service/. It never edits
# the official checkout: paths.py stages `official tree + this package's
# overlay/` into a throwaway copy under vtb-env/, and the image build + per-task
# bind-mount both read from that staged root.
#
# What it does, in order, each step skippable when already satisfied:
#   1. docker reachable
#   2. stage official+overlay -> vtb-env/toolathlon_gym_staged
#   3. toolathlon-pack:latest image        (docker build from the staged root)
#   4. code delivery strategy              (bind mounts, or bake_code.sh fallback)
#   5. digest-pinned postgres image pulled
#   6. exact service dependencies present in the selected Python environment
#   7. service started with nohup on 0.0.0.0:8060, then /health polled
#
# Tunables (export before running):
#   VTB_TOOLATHLON_GYM_REPO     REQUIRED    abs path to the official gym checkout
#   TOOLATHLON_MAX_CONCURRENT   default 12   tasks of one job running at once
#   TASK_TIMEOUT                default 2100 seconds per task (35 min)
#   PG_READY_TIMEOUT            default 300  seconds for a per-task pg init.sql
#   TOOLATHLON_PORT             default 8060
#   RUN_MCP_SMOKE               default 0    set 1 to run scripts/test_mcp_smoke.sh
#
# SECURITY: the service has NO authentication. 0.0.0.0 is required so the
# benchmark adapter can reach it, so the port must be restricted to trusted
# sources (security group / firewall).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"                 # eval/service/toolathlon_gym
REPO_ROOT="$(cd "$PKG_DIR/../../.." && pwd)"       # vibetrainbench checkout root
cd "$REPO_ROOT"
source "$REPO_ROOT/src/commit_utils/configure_python.sh" || exit 1

PORT="${TOOLATHLON_PORT:-8060}"
IMAGE="${TOOLATHLON_IMAGE:-toolathlon-pack:latest}"
export POSTGRES_IMAGE="${POSTGRES_IMAGE:-postgres:15@sha256:9b1d34adbce1dd07ee6e94b4a2cf698884b89bd44a6c9c12f5da8f3acbfe4957}"
RUNTIME_DIR="${VTB_ROOT:-$REPO_ROOT/vtb-env}"
mkdir -p "$RUNTIME_DIR"
LOG="$RUNTIME_DIR/toolathlon_gym-$PORT.log"
PIDFILE="$RUNTIME_DIR/toolathlon_gym-$PORT.pid"

echo "=== [1/7] docker ==="
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon unreachable"; exit 1; }
echo "  ok"

echo "=== [2/7] stage official + overlay ==="
if [ -z "${VTB_TOOLATHLON_GYM_REPO:-}" ]; then
    echo "FATAL: VTB_TOOLATHLON_GYM_REPO is not set; point it at the official"
    echo "       toolathlon_gym checkout (the read-only upstream this drives)."
    exit 1
fi
STAGED="$("$VTB_PYTHON" -c 'from eval.service.toolathlon_gym import paths; print(paths.build_staged_root())')"
echo "  staged -> $STAGED"

echo "=== [3/7] image $IMAGE ==="
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "  exists, skipping build"
else
    echo "  building (first time: 10-20 min) ..."
    docker build -t "$IMAGE" "$STAGED"
fi

echo "=== [4/7] code delivery strategy ==="
if docker run --rm -v /etc/hostname:/vtb-mount-probe --entrypoint /bin/bash \
        "$IMAGE" -c true >/dev/null 2>&1; then
    echo "  bind mounts allowed: the runner mounts the staged tree at /workspace"
else
    echo "  bind mounts blocked: baking code into the image (cp fallback mode)"
    ( cd "$STAGED" && bash scripts/bake_code.sh )
fi

echo "=== [5/7] $POSTGRES_IMAGE ==="
if docker image inspect "$POSTGRES_IMAGE" >/dev/null 2>&1; then
    echo "  pulled already"
else
    docker pull "$POSTGRES_IMAGE"
fi

if [[ "${RUN_MCP_SMOKE:-0}" == "1" ]]; then
    echo "=== [5.5/7] MCP smoke test (RUN_MCP_SMOKE=1) ==="
    ( cd "$STAGED" && bash scripts/test_mcp_smoke.sh ) \
        || echo "  [warn] MCP smoke not fully green; service still starts"
fi

echo "=== [6/7] python deps for the service (fastapi/uvicorn) ==="
if ! "$VTB_PYTHON" scripts/prepare/python/verify_python_environment.py --requirements "$PKG_DIR/requirements.txt"; then
    echo "Install the pinned service dependencies into the selected environment first:"
    printf '  %q pip install --python %q -r %q\n' "$RUNTIME_DIR/tools/bin/uv" "$VTB_PYTHON" "$PKG_DIR/requirements.txt"
    exit 1
fi
echo "  ok"

echo "=== [7/7] start service on 0.0.0.0:$PORT ==="
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "  already running (pid $(cat "$PIDFILE")); stop it first:"
    echo "    kill \$(cat $PIDFILE)"
    exit 1
fi
export TOOLATHLON_MAX_CONCURRENT="${TOOLATHLON_MAX_CONCURRENT:-12}"
export TASK_TIMEOUT="${TASK_TIMEOUT:-2100}"
export PG_READY_TIMEOUT="${PG_READY_TIMEOUT:-300}"
# Module path (run from repo root) so the package's relative imports resolve.
nohup "$VTB_PYTHON" -m uvicorn eval.service.toolathlon_gym.app:app \
    --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "  pid $(cat "$PIDFILE"), log $LOG"
echo "  concurrency=$TOOLATHLON_MAX_CONCURRENT task_timeout=${TASK_TIMEOUT}s"

for _ in $(seq 1 30); do
    if OUT=$(curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null); then
        echo "  $OUT"
        if echo "$OUT" | grep -q '"ok":true'; then
            echo
            echo "READY. From the benchmark host, test with:"
            echo "  curl http://<this-host-ip>:$PORT/health"
            echo "  curl http://<this-host-ip>:$PORT/tasks"
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
