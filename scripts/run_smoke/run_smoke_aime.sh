#!/usr/bin/env bash
# Bounded AIME smoke: one AIME 2025 val problem + one AIME 2026 final problem.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

source "$REPO/src/commit_utils/launcher_agent.sh"

BENCHMARK="aime__26"
VAL_BENCHMARK="aime__aime"
VTB_ROOT="${VTB_ROOT:-$REPO/vtb-env}"
export HF_HOME="${HF_HOME:-$VTB_ROOT/resources/data/hf_cache}"
RESULTS_DIR="${VTB_RESULTS_DIR:-$VTB_ROOT/results}"
HUB_TASK_SPECS="${VTB_HUB_TASK_SPECS:-${VTB_HARBOR_DATASET_DIR:-$VTB_ROOT/hub_task_specs}}"
SMOKE_ID="${SMOKE_ID:-aime-connectivity}"
PORT="${GRADING_PORT:-5841}"
STATE="${VTB_GRADING_STATE:-$VTB_ROOT/grading_states/smoke/aime_${SMOKE_ID}}"
STARTUP_LOG="$STATE/startup.log"
SERVICE_SESSION="${SERVICE_SESSION:-vtb_grading_aime_smoke_${SMOKE_ID}}"
RUN_SESSION="${RUN_SESSION:-vtb_run_aime_smoke_${SMOKE_ID}}"
RUN_ID="${RUN_ID:-aime-smoke-${SMOKE_ID}}"
RESULT_DIR="$RESULTS_DIR/${AGENT_CLI}_${RESEARCHER_SAFE}_1h_aime_smoke/${BENCHMARK}_Qwen_Qwen3.5-9B-Base_${RUN_ID}"
RESULT_DIR="$(smoke_result_path "$STATE" "$RESULT_DIR")"
VALIDATION_SOURCE="task:$VAL_BENCHMARK"

raw_health() {
    VTB_GRADING_PORT="$PORT" "$VTB_PYTHON" -m eval.grading.client health
}

validate_health() {
    payload="$(raw_health)" || {
        printf '%s\n' "$payload" >&2
        return 1
    }
    HEALTH_PAYLOAD="$payload" "$VTB_PYTHON" - "$VALIDATION_SOURCE" <<'PY'
import json, os, sys
payload = json.loads(os.environ["HEALTH_PAYLOAD"])
if payload.get("status") != "ok":
    raise SystemExit(payload.get("error") or f"grading service is not healthy: {payload}")
expected = sys.argv[1]
validation = payload.get("validation") or {}
if validation.get("raw") != expected:
    raise SystemExit(f"validation source is {validation.get('raw')!r}; expected {expected!r}")
item = (validation.get("resolved_counts") or {}).get("aime__26") or {}
if item.get("validation_task") != "aime__aime" or item.get("count") != 30:
    raise SystemExit(f"AIME validation route is wrong: {item}")
print("AIME_SMOKE_SERVICE_OK", item.get("validation_source"), "count=", item.get("count"))
PY
}

preflight() {
    [ "${1:-}" = runtime ] || require_researcher || return
    export PATH="$VTB_ROOT/tools/bin:$PATH"
    for command in docker tmux fuse-overlayfs; do
        command -v "$command" >/dev/null || {
            echo "FATAL: required command is missing: $command" >&2; exit 1;
        }
    done
    bash scripts/prepare/prepare_aime.sh preflight
    docker image inspect vibetrainbench/trainer:latest >/dev/null
    docker image inspect vibetrainbench/sglang:latest >/dev/null
    bash scripts/prepare/download_base_model.sh snapshot-preflight
    [ ! -e "$STATE" ] || { echo "FATAL: smoke state already exists: $STATE" >&2; exit 1; }
    [ ! -e "$RESULT_DIR" ] || { echo "FATAL: smoke result already exists: $RESULT_DIR" >&2; exit 1; }
    tmux has-session -t "$SERVICE_SESSION" 2>/dev/null && {
        echo "FATAL: tmux session exists: $SERVICE_SESSION" >&2; exit 1;
    }
    tmux has-session -t "$RUN_SESSION" 2>/dev/null && {
        echo "FATAL: tmux session exists: $RUN_SESSION" >&2; exit 1;
    }
    if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
        echo "FATAL: grading port is already listening: $PORT" >&2; exit 1
    fi
    echo "AIME_SMOKE_PREFLIGHT_OK state=$STATE port=$PORT"
}

start_service() {
    preflight runtime
    mkdir -p "$STATE"
    tmux new-session -d -s "$SERVICE_SESSION" "\
        cd '$REPO' && export VTB_PYTHON='$VTB_PYTHON' VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q && source src/commit_utils/load_launch_env.sh && \
        export VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_ROOT='$VTB_ROOT' HF_HOME='$HF_HOME' VTB_RESULTS_DIR='$RESULTS_DIR' &&
        export VTB_HUB_TASK_SPECS='$HUB_TASK_SPECS' && \
        export VTB_GRADING_STATE='$STATE' VTB_VAL_SOURCE='$VALIDATION_SOURCE' && \
        export VTB_AGENT_DISCLOSURE=score,per_sample_outcomes,problems,outputs && \
        export VTB_GPU_COORDINATION_MODE=shared VTB_EVAL_GPUS=0 && \
        export VTB_FINAL_LIMIT=1 VTB_GRADING_PORT='$PORT' && \
        export VTB_AIME_DIRECT_SAMPLES_PER_TASK=1 VTB_AIME_DIRECT_CONCURRENCY=1 && \
        source scripts/prepare/configure_evaluation_environment.sh && \
        '$VTB_PYTHON' -m eval.grading.server 2>&1 | tee '$STARTUP_LOG'"
    for _ in $(seq 1 30); do
        sleep 1
        if raw_health >/dev/null 2>&1; then
            echo "AIME smoke grading: session=$SERVICE_SESSION state=$STATE port=$PORT"
            return 0
        fi
        if ! tmux has-session -t "$SERVICE_SESSION" 2>/dev/null; then
            echo "FATAL: grading service exited during startup; log follows:" >&2
            [ ! -f "$STARTUP_LOG" ] || cat "$STARTUP_LOG" >&2
            exit 1
        fi
    done
    echo "FATAL: grading service did not become healthy; log follows:" >&2
    [ ! -f "$STARTUP_LOG" ] || cat "$STARTUP_LOG" >&2
    exit 1
}

start_run() {
    require_researcher || return
    validate_health >/dev/null
    [ ! -e "$RESULT_DIR" ] || { echo "FATAL: smoke result already exists: $RESULT_DIR" >&2; exit 1; }
    tmux has-session -t "$RUN_SESSION" 2>/dev/null && {
        echo "FATAL: tmux session exists: $RUN_SESSION" >&2; exit 1;
    }
    remember_smoke_run "$STATE" "$RESULT_DIR" || return
    tmux new-session -d -s "$RUN_SESSION" "cd '$REPO' && export VTB_PYTHON='$VTB_PYTHON' VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q && source src/commit_utils/load_launch_env.sh && set -a && \
        { [ ! -f api_key.env ] || source api_key.env; } && set +a && \
        export VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_ROOT='$VTB_ROOT' HF_HOME='$HF_HOME' VTB_RESULTS_DIR='$RESULTS_DIR' &&
        export VTB_HUB_TASK_SPECS='$HUB_TASK_SPECS' && \
        VTB_GRADING_STATE='$STATE' VTB_ADMIN_TOKEN_FILE='$STATE/admin.token' \
        VTB_PROMPT=aime_smoke VTB_TRAIN_RECIPE= VTB_GRADING_PORT='$PORT' \
        VTB_AGENT_GPUS=0 VTB_EXPERIMENT_NAME=_aime_smoke VTB_EDITABLE_HARNESS=off \
        bash src/run_task.sh '$BENCHMARK' '$AGENT_CLI' \
        Qwen/Qwen3.5-9B-Base '$RUN_ID' 1 '$RESEARCHER' 1"
    echo "AIME smoke run: session=$RUN_SESSION result=$RESULT_DIR"
}

verify() {
    [ -f "$RESULT_DIR/metrics.json" ] || {
        echo "FATAL: smoke is not complete: $RESULT_DIR" >&2; exit 1;
    }
    "$VTB_PYTHON" scripts/run_smoke/python/verify_smoke_aime.py "$STATE" "$RESULT_DIR"
}

cleanup() {
    if [ -s "$STATE/service.pid" ]; then kill "$(cat "$STATE/service.pid")" 2>/dev/null || true; fi
    tmux kill-session -t "$SERVICE_SESSION" 2>/dev/null || true
    tmux kill-session -t "$RUN_SESSION" 2>/dev/null || true
}

case "${1:-}" in
    preflight) preflight ;;
    service) start_service ;;
    health) validate_health ;;
    run) start_run ;;
    verify) verify ;;
    cleanup) cleanup ;;
    *) echo "usage: $0 <preflight|service|health|run|verify|cleanup>" >&2; exit 2 ;;
esac
