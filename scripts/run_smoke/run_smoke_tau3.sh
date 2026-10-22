#!/usr/bin/env bash
# Reproducible one-task connectivity test for the Tau2-backed Tau3 benchmark.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

load_env() {
    set -a
    [ ! -f api_key.env ] || source api_key.env
    [ ! -f benchmarks/sierra-research__tau3-bench/.env ] || \
        source benchmarks/sierra-research__tau3-bench/.env
    set +a
}
load_env
source "$REPO/src/commit_utils/launcher_agent.sh"

VTB_ROOT="${VTB_ROOT:-$REPO/vtb-env}"
export HF_HOME="${HF_HOME:-$VTB_ROOT/resources/data/hf_cache}"
RESULTS_DIR="${VTB_RESULTS_DIR:-$VTB_ROOT/results}"
HUB_TASK_SPECS="${VTB_HUB_TASK_SPECS:-${VTB_HARBOR_DATASET_DIR:-$VTB_ROOT/hub_task_specs}}"
TAU3_TASK_SPECS="$HUB_TASK_SPECS/sierra-research__tau3-bench/tau3-bench"
VTB_HARBOR_REPO="${VTB_HARBOR_REPO:-$REPO/../harbor}"

# Stable by default so a later `verify`/`cleanup` addresses the same run. Set a new
# SMOKE_ID explicitly for every repeated smoke; preflight refuses stale state/results.
SMOKE_ID="${SMOKE_ID:-open-source-cold}"
PORT="${GRADING_PORT:-5830}"
STATE="${VTB_GRADING_STATE:-$VTB_ROOT/grading_states/smoke/tau3_${SMOKE_ID}}"
STARTUP_LOG="$STATE/startup.log"
SERVICE_SESSION="${SERVICE_SESSION:-vtb_grading_tau2_smoke_${SMOKE_ID}}"
RUN_SESSION="${RUN_SESSION:-vtb_run_tau2_smoke_${SMOKE_ID}}"
RUN_ID="${RUN_ID:-tau2-smoke-${SMOKE_ID}}"
RESULT_DIR="$RESULTS_DIR/${AGENT_CLI}_${RESEARCHER_SAFE}_1h_tau2_smoke/sierra-research__tau3-bench_Qwen_Qwen3.5-9B-Base_${RUN_ID}"
RESULT_DIR="$(smoke_result_path "$STATE" "$RESULT_DIR")"

health() {
    VTB_GRADING_PORT="$PORT" "$VTB_PYTHON" -m eval.grading.client health
}

preflight() {
    [ "${1:-}" = runtime ] || require_researcher || return
    "$VTB_PYTHON" scripts/prepare/python/preflight_tau2_for_tau3.py --state "$STATE" --port "$PORT"
    [ ! -e "$RESULT_DIR" ] || { echo "FATAL: result already exists: $RESULT_DIR" >&2; exit 1; }
    tmux has-session -t "$SERVICE_SESSION" 2>/dev/null && {
        echo "FATAL: tmux session exists: $SERVICE_SESSION" >&2; exit 1;
    }
    tmux has-session -t "$RUN_SESSION" 2>/dev/null && {
        echo "FATAL: tmux session exists: $RUN_SESSION" >&2; exit 1;
    }
    echo "TAU3_SMOKE_PREFLIGHT_OK state=$STATE port=$PORT"
}

start_service() {
    preflight runtime
    mkdir -p "$STATE"
    tmux new-session -d -s "$SERVICE_SESSION" "set -o pipefail; cd '$REPO' && export VTB_PYTHON='$VTB_PYTHON' VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q && source src/commit_utils/load_launch_env.sh && set -a && \
        { [ ! -f benchmarks/sierra-research__tau3-bench/.env ] || source benchmarks/sierra-research__tau3-bench/.env; } && \
        { [ ! -f api_key.env ] || source api_key.env; } && \
        export VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_ROOT='$VTB_ROOT' HF_HOME='$HF_HOME' VTB_RESULTS_DIR='$RESULTS_DIR' &&
        export VTB_HUB_TASK_SPECS='$HUB_TASK_SPECS' VTB_HARBOR_REPO='$VTB_HARBOR_REPO' && \
        source scripts/prepare/configure_evaluation_environment.sh && set +a && \
        VTB_GRADING_STATE='$STATE' VTB_VAL_SOURCE=split:1 \
        VTB_AGENT_DISCLOSURE=score,per_sample_outcomes,problems,outputs \
        VTB_GPU_COORDINATION_MODE=shared VTB_EVAL_GPUS=0 VTB_FINAL_LIMIT=1 \
        VTB_GRADING_PORT='$PORT' '$VTB_PYTHON' -u -m eval.grading.server 2>&1 | tee '$STARTUP_LOG'"
    for _ in $(seq 1 30); do
        sleep 1
        if health >/dev/null 2>&1; then
            echo "Tau2 smoke grading: session=$SERVICE_SESSION state=$STATE port=$PORT"
            return 0
        fi
        if ! tmux has-session -t "$SERVICE_SESSION" 2>/dev/null; then
            echo "FATAL: Tau3 grading service exited during startup" >&2
            [ ! -f "$STARTUP_LOG" ] || cat "$STARTUP_LOG" >&2
            tmux capture-pane -t "$SERVICE_SESSION" -p -S -80 >&2 2>/dev/null || true
            return 1
        fi
    done
    echo "FATAL: Tau3 grading service did not become healthy: state=$STATE port=$PORT" >&2
    [ ! -f "$STARTUP_LOG" ] || cat "$STARTUP_LOG" >&2
    tmux capture-pane -t "$SERVICE_SESSION" -p -S -80 >&2 2>/dev/null || true
    return 1
}

start_run() {
    require_researcher || return
    health >/dev/null || { echo "FATAL: grading service is not healthy" >&2; exit 1; }
    [ ! -e "$RESULT_DIR" ] || { echo "FATAL: result already exists: $RESULT_DIR" >&2; exit 1; }
    tmux has-session -t "$RUN_SESSION" 2>/dev/null && {
        echo "FATAL: tmux session exists: $RUN_SESSION" >&2; exit 1;
    }
    remember_smoke_run "$STATE" "$RESULT_DIR" || return
    tmux new-session -d -s "$RUN_SESSION" "cd '$REPO' && export VTB_PYTHON='$VTB_PYTHON' VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q && source src/commit_utils/load_launch_env.sh && set -a && \
        { [ ! -f benchmarks/sierra-research__tau3-bench/.env ] || source benchmarks/sierra-research__tau3-bench/.env; } && \
        { [ ! -f api_key.env ] || source api_key.env; } && set +a && \
        export VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_ROOT='$VTB_ROOT' HF_HOME='$HF_HOME' VTB_RESULTS_DIR='$RESULTS_DIR' &&
        export VTB_HUB_TASK_SPECS='$HUB_TASK_SPECS' VTB_HARBOR_REPO='$VTB_HARBOR_REPO' && \
        VTB_GRADING_STATE='$STATE' VTB_ADMIN_TOKEN_FILE='$STATE/admin.token' \
        VTB_PROMPT=tau2_smoke VTB_TRAIN_RECIPE= VTB_GRADING_PORT='$PORT' \
        VTB_AGENT_GPUS=0 VTB_EXPERIMENT_NAME=_tau2_smoke VTB_EDITABLE_HARNESS=off \
        bash src/run_task.sh sierra-research__tau3-bench '$AGENT_CLI' \
        Qwen/Qwen3.5-9B-Base '$RUN_ID' 1 '$RESEARCHER' 1"
    echo "Tau2 smoke run: session=$RUN_SESSION result=$RESULT_DIR"
}

verify() {
    [ -f "$RESULT_DIR/metrics.json" ] || { echo "FATAL: smoke is not complete: $RESULT_DIR" >&2; exit 1; }
    "$VTB_PYTHON" scripts/run_smoke/python/verify_smoke_tau3.py "$STATE" "$RESULT_DIR"
    backend="$(find "$STATE/archive" -name backend_command.txt -type f | head -1)"
    [ -n "$backend" ] || { echo "FATAL: no archived backend command" >&2; exit 1; }
    grep -Fq -- "--path $TAU3_TASK_SPECS" "$backend" || {
        echo "FATAL: Harbor did not use the canonical local task export" >&2; exit 1;
    }
    ! grep -q -- "--dataset" "$backend" || {
        echo "FATAL: Harbor fell back to the registry dataset" >&2; exit 1;
    }
    ! grep -RqsE "Could not patch missing tau3-bench task asset|McpError: Timed out|NonZeroAgentExitCodeError" \
        "$STATE/service.log" "$STATE/archive" || {
        echo "FATAL: smoke contains a cold-cache patch/agent failure" >&2; exit 1;
    }
    state_file="$(find "$STATE/archive" -name tau3_runtime_state.json -type f | head -1)"
    [ -n "$state_file" ] || { echo "FATAL: no Tau runtime state was archived" >&2; exit 1; }
    "$VTB_PYTHON" - "$state_file" <<'PY'
import json, sys
state = json.load(open(sys.argv[1]))
if not state.get("bootstrap_complete"):
    raise SystemExit("Tau runtime did not complete bootstrap")
print("TAU2_COLD_CACHE_SMOKE_OK", "steps=", state.get("step_count"))
PY
}

cleanup() {
    if [ -s "$STATE/service.pid" ]; then kill "$(cat "$STATE/service.pid")" 2>/dev/null || true; fi
    tmux kill-session -t "$SERVICE_SESSION" 2>/dev/null || true
}

case "${1:-}" in
    preflight) preflight ;;
    service) start_service ;;
    run) start_run ;;
    health) health ;;
    verify) verify ;;
    cleanup) cleanup ;;
    *) echo "usage: $0 <preflight|service|run|health|verify|cleanup>" >&2; exit 2 ;;
esac
