#!/usr/bin/env bash
# One ToolGym validation and one Toolathlon Verified final connectivity smoke.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
source "$REPO/src/commit_utils/launcher_agent.sh"
TARGET="eigent-ai__toolathlon_gym"; VTB_ROOT="${VTB_ROOT:-$REPO/vtb-env}"; RESULTS_DIR="${VTB_RESULTS_DIR:-$VTB_ROOT/results}"
export HF_HOME="${HF_HOME:-$VTB_ROOT/resources/data/hf_cache}"
SMOKE_ID="${SMOKE_ID:-toolathlon-connectivity}"; PORT="${GRADING_PORT:-5843}"
STATE="${VTB_GRADING_STATE:-$VTB_ROOT/grading_states/smoke/toolathlon_${SMOKE_ID}}"; STARTUP_LOG="$STATE/startup.log"
SERVICE_SESSION="${SERVICE_SESSION:-vtb_grading_toolathlon_smoke_${SMOKE_ID}}"; RUN_SESSION="${RUN_SESSION:-vtb_run_toolathlon_smoke_${SMOKE_ID}}"
RUN_ID="${RUN_ID:-toolathlon-smoke-${SMOKE_ID}}"; RESULT_DIR="$RESULTS_DIR/${AGENT_CLI}_${RESEARCHER_SAFE}_1h_toolathlon_smoke/${TARGET}_Qwen_Qwen3.5-9B-Base_${RUN_ID}"
RESULT_DIR="$(smoke_result_path "$STATE" "$RESULT_DIR")"
GYM="${VTB_TOOLATHLON_URL:?set VTB_TOOLATHLON_URL}"; VERIFIED="${VTB_TOOLATHLON_VERIFIED_URL:?set VTB_TOOLATHLON_VERIFIED_URL}"; HOST="${VTB_MODEL_ADVERTISE_HOST:?set VTB_MODEL_ADVERTISE_HOST}"
raw_health() { VTB_GRADING_PORT="$PORT" "$VTB_PYTHON" -m eval.grading.client health; }
health() { payload="$(raw_health)" || { printf '%s\n' "$payload" >&2; return 1; }; echo "$payload" | grep -q '"status": "ok"'; echo "TOOLATHLON_SMOKE_SERVICE_OK"; }
preflight() {
    [ "${1:-}" = runtime ] || require_researcher || return
    VTB_TOOLATHLON_URL="$GYM" VTB_TOOLATHLON_VERIFIED_URL="$VERIFIED" \
        VTB_MODEL_ADVERTISE_HOST="$HOST" bash scripts/prepare/prepare_toolathlon.sh preflight
    bash scripts/prepare/download_base_model.sh snapshot-preflight
    [ ! -e "$STATE" ] || { echo "FATAL: state exists: $STATE" >&2; exit 1; }; [ ! -e "$RESULT_DIR" ] || { echo "FATAL: result exists: $RESULT_DIR" >&2; exit 1; }
    ss -tln 2>/dev/null | grep -q ":${PORT} " && { echo "FATAL: port busy: $PORT" >&2; exit 1; }; echo "TOOLATHLON_SMOKE_PREFLIGHT_OK"
}
service() {
    preflight runtime; mkdir -p "$STATE"
    tmux new-session -d -s "$SERVICE_SESSION" "cd '$REPO' && export VTB_PYTHON='$VTB_PYTHON' VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q && source src/commit_utils/load_launch_env.sh && export VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_ROOT='$VTB_ROOT' HF_HOME='$HF_HOME' VTB_RESULTS_DIR='$RESULTS_DIR' && \
        export VTB_GRADING_STATE='$STATE' VTB_VAL_SOURCE='file:benchmarks/eigent-ai__toolathlon_gym/split.json' VTB_FINAL_LIMIT=1 && \
        export VTB_GRADING_PORT='$PORT' VTB_GPU_COORDINATION_MODE=shared VTB_EVAL_GPUS=0 && \
        export VTB_TOOLATHLON_URL='$GYM' VTB_TOOLATHLON_VERIFIED_URL='$VERIFIED' VTB_MODEL_ADVERTISE_HOST='$HOST' && \
        source scripts/prepare/configure_evaluation_environment.sh && '$VTB_PYTHON' -m eval.grading.server 2>&1 | tee '$STARTUP_LOG'"
    for _ in $(seq 1 30); do sleep 1; raw_health >/dev/null 2>&1 && { echo "Toolathlon smoke grading: $SERVICE_SESSION"; return; }; tmux has-session -t "$SERVICE_SESSION" 2>/dev/null || { cat "$STARTUP_LOG" >&2; exit 1; }; done; cat "$STARTUP_LOG" >&2; exit 1
}
run() {
    require_researcher || return
    health >/dev/null; remember_smoke_run "$STATE" "$RESULT_DIR" || return
    tmux new-session -d -s "$RUN_SESSION" "cd '$REPO' && export VTB_PYTHON='$VTB_PYTHON' VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q && source src/commit_utils/load_launch_env.sh && export VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_ROOT='$VTB_ROOT' HF_HOME='$HF_HOME' VTB_RESULTS_DIR='$RESULTS_DIR' && \
        VTB_GRADING_STATE='$STATE' VTB_ADMIN_TOKEN_FILE='$STATE/admin.token' VTB_PROMPT=toolathlon_smoke VTB_TRAIN_RECIPE= VTB_GRADING_PORT='$PORT' \
        VTB_AGENT_GPUS=0 VTB_EXPERIMENT_NAME=_toolathlon_smoke VTB_EDITABLE_HARNESS=off \
        bash src/run_task.sh '$TARGET' '$AGENT_CLI' Qwen/Qwen3.5-9B-Base '$RUN_ID' 1 '$RESEARCHER' 1"
    echo "Toolathlon smoke run: $RUN_SESSION result=$RESULT_DIR"
}
verify() { "$VTB_PYTHON" scripts/run_smoke/python/verify_smoke_toolathlon.py "$STATE" "$RESULT_DIR"; }
cleanup() { [ ! -s "$STATE/service.pid" ] || kill "$(cat "$STATE/service.pid")" 2>/dev/null || true; tmux kill-session -t "$SERVICE_SESSION" 2>/dev/null || true; }
case "${1:-}" in preflight) preflight ;; service) service ;; health) health ;; run) run ;; verify) verify ;; cleanup) cleanup ;; *) echo "usage: $0 <preflight|service|health|run|verify|cleanup>" >&2; exit 2 ;; esac
