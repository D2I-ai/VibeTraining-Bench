#!/usr/bin/env bash
# One-sample validation and final smoke for SWE-Bench Verified.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
source "$REPO/src/commit_utils/launcher_agent.sh"
BENCHMARK="swe-bench__swe-bench-verified"; VTB_ROOT="${VTB_ROOT:-$REPO/vtb-env}"; RESULTS_DIR="${VTB_RESULTS_DIR:-$VTB_ROOT/results}"
export HF_HOME="${HF_HOME:-$VTB_ROOT/resources/data/hf_cache}"
SMOKE_ID="${SMOKE_ID:-swe-connectivity}"; PORT="${GRADING_PORT:-5842}"
STATE="${VTB_GRADING_STATE:-$VTB_ROOT/grading_states/smoke/swe_${SMOKE_ID}}"; STARTUP_LOG="$STATE/startup.log"
SERVICE_SESSION="${SERVICE_SESSION:-vtb_grading_swe_smoke_${SMOKE_ID}}"; RUN_SESSION="${RUN_SESSION:-vtb_run_swe_smoke_${SMOKE_ID}}"
RUN_ID="${RUN_ID:-swe-smoke-${SMOKE_ID}}"; RESULT_DIR="$RESULTS_DIR/${AGENT_CLI}_${RESEARCHER_SAFE}_1h_swe_smoke/${BENCHMARK}_Qwen_Qwen3.5-9B-Base_${RUN_ID}"
RESULT_DIR="$(smoke_result_path "$STATE" "$RESULT_DIR")"
raw_health() { VTB_GRADING_PORT="$PORT" "$VTB_PYTHON" -m eval.grading.client health; }
health() { payload="$(raw_health)" || { printf '%s\n' "$payload" >&2; return 1; }; echo "$payload" | grep -q '"status": "ok"'; echo "SWE_SMOKE_SERVICE_OK split:1"; }
preflight() {
    [ "${1:-}" = runtime ] || require_researcher || return
    bash scripts/prepare/prepare_swe.sh preflight
    bash scripts/prepare/download_base_model.sh snapshot-preflight
    [ ! -e "$STATE" ] || { echo "FATAL: state exists: $STATE" >&2; exit 1; }; [ ! -e "$RESULT_DIR" ] || { echo "FATAL: result exists: $RESULT_DIR" >&2; exit 1; }
    ss -tln 2>/dev/null | grep -q ":${PORT} " && { echo "FATAL: port busy: $PORT" >&2; exit 1; }
    echo "SWE_SMOKE_PREFLIGHT_OK state=$STATE port=$PORT"
}
service() {
    preflight runtime; mkdir -p "$STATE"
    tmux new-session -d -s "$SERVICE_SESSION" "cd '$REPO' && export VTB_PYTHON='$VTB_PYTHON' VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q && source src/commit_utils/load_launch_env.sh && export VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_ROOT='$VTB_ROOT' HF_HOME='$HF_HOME' VTB_RESULTS_DIR='$RESULTS_DIR' && \
        export VTB_GRADING_STATE='$STATE' VTB_VAL_SOURCE=split:1 VTB_FINAL_LIMIT=1 VTB_GRADING_PORT='$PORT' && \
        export VTB_AGENT_DISCLOSURE=score,per_sample_outcomes,problems,outputs VTB_GPU_COORDINATION_MODE=shared VTB_EVAL_GPUS=0 && \
        source scripts/prepare/configure_evaluation_environment.sh && '$VTB_PYTHON' -m eval.grading.server 2>&1 | tee '$STARTUP_LOG'"
    for _ in $(seq 1 30); do sleep 1; raw_health >/dev/null 2>&1 && { echo "SWE smoke grading: $SERVICE_SESSION"; return; }; tmux has-session -t "$SERVICE_SESSION" 2>/dev/null || { cat "$STARTUP_LOG" >&2; exit 1; }; done
    cat "$STARTUP_LOG" >&2; exit 1
}
run() {
    require_researcher || return
    health >/dev/null; tmux has-session -t "$RUN_SESSION" 2>/dev/null && { echo "FATAL: session exists" >&2; exit 1; }
    remember_smoke_run "$STATE" "$RESULT_DIR" || return
    tmux new-session -d -s "$RUN_SESSION" "cd '$REPO' && export VTB_PYTHON='$VTB_PYTHON' VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q && source src/commit_utils/load_launch_env.sh && set -a && { [ ! -f api_key.env ] || source api_key.env; } && set +a && \
        export VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_ROOT='$VTB_ROOT' HF_HOME='$HF_HOME' VTB_RESULTS_DIR='$RESULTS_DIR' && VTB_GRADING_STATE='$STATE' VTB_ADMIN_TOKEN_FILE='$STATE/admin.token' \
        VTB_PROMPT=mock_test VTB_TRAIN_RECIPE= VTB_GRADING_PORT='$PORT' VTB_AGENT_GPUS=0 VTB_EXPERIMENT_NAME=_swe_smoke \
        VTB_EDITABLE_HARNESS=off bash src/run_task.sh '$BENCHMARK' '$AGENT_CLI' Qwen/Qwen3.5-9B-Base '$RUN_ID' 1 '$RESEARCHER' 1"
    echo "SWE smoke run: $RUN_SESSION result=$RESULT_DIR"
}
verify() { "$VTB_PYTHON" scripts/run_smoke/python/verify_smoke_swe.py "$STATE" "$RESULT_DIR"; }
cleanup() { [ ! -s "$STATE/service.pid" ] || kill "$(cat "$STATE/service.pid")" 2>/dev/null || true; tmux kill-session -t "$SERVICE_SESSION" 2>/dev/null || true; }
case "${1:-}" in preflight) preflight ;; service) service ;; health) health ;; run) run ;; verify) verify ;; cleanup) cleanup ;; *) echo "usage: $0 <preflight|service|health|run|verify|cleanup>" >&2; exit 2 ;; esac
