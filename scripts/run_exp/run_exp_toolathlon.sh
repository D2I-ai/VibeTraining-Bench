#!/usr/bin/env bash
# Published Toolathlon experiment: 8x A100, 16h, editable D/T and fixed harness.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
source "$REPO/src/commit_utils/launcher_agent.sh"
BENCHMARK="eigent-ai__toolathlon_gym"; MODEL="${MODEL:-Qwen/Qwen3.5-9B-Base}"
RECIPE="${RECIPE:-qwen3_5_9b/slime_sft_toolathlon}"; RUN_ID="${RUN_ID:-toolathlon-research}"
HOURS="${HOURS:-16}"; AGENT_CLI="${AGENT_CLI:-claude}"; RESEARCHER="${RESEARCHER:-}"
NUM_GPUS="${NUM_GPUS:-8}"; AGENT_GPUS="${AGENT_GPUS:-0,1,2,3,4,5,6,7}"; EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
export VTB_ROOT="${VTB_ROOT:-$REPO/vtb-env}"; export HF_HOME="${HF_HOME:-$VTB_ROOT/resources/data/hf_cache}"; export VTB_MODEL_RESOURCE_ROOT="${VTB_MODEL_RESOURCE_ROOT:-$VTB_ROOT/resources/models/qwen3_5_9b_base}"; RESULTS_DIR="${VTB_RESULTS_DIR:-$VTB_ROOT/results}"
GRADING_STATE="${VTB_GRADING_STATE:-$VTB_ROOT/grading_states/formal/toolathlon}"
GRADING_HOST="${GRADING_HOST:-${VTB_GRADING_HOST:-127.0.0.1}}"; GRADING_CONNECT_HOST="$GRADING_HOST"; [ "$GRADING_CONNECT_HOST" = "0.0.0.0" ] && GRADING_CONNECT_HOST="127.0.0.1"
GRADING_PORT="${GRADING_PORT:-${VTB_GRADING_PORT:-5800}}"; SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:-${VTB_SERVE_EXTRA_ARGS:-}}"; SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
SERVICE_SESSION="${SERVICE_SESSION:-vtb_grading_toolathlon}"; RUN_SESSION="${RUN_SESSION:-vtb_run_toolathlon}"
VAL_SOURCE="${VAL_SOURCE:-file:benchmarks/eigent-ai__toolathlon_gym/split.json}"; FINAL_LIMIT="${FINAL_LIMIT:--1}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-93600}"; DISCLOSURE="${DISCLOSURE:-score,per_sample_outcomes,problems,outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-_toolathlon_research}"
TOOLATHLON_URL="${VTB_TOOLATHLON_URL:?set VTB_TOOLATHLON_URL to the ToolGym service}"
VERIFIED_URL="${VTB_TOOLATHLON_VERIFIED_URL:?set VTB_TOOLATHLON_VERIFIED_URL to Toolathlon Verified}"
ADVERTISE_HOST="${VTB_MODEL_ADVERTISE_HOST:?set VTB_MODEL_ADVERTISE_HOST for service callbacks}"
TOOLATHLON_TIMEOUT="${VTB_TOOLATHLON_TIMEOUT:-10800}"; VERIFIED_TIMEOUT="${VTB_TOOLATHLON_VERIFIED_TIMEOUT:-13500}"

health() { VTB_GRADING_HOST="$GRADING_CONNECT_HOST" VTB_GRADING_PORT="$GRADING_PORT" "$VTB_PYTHON" -m eval.grading.client health; }
start_service() {
    VTB_TOOLATHLON_URL="$TOOLATHLON_URL" \
    VTB_TOOLATHLON_VERIFIED_URL="$VERIFIED_URL" \
    VTB_MODEL_ADVERTISE_HOST="$ADVERTISE_HOST" \
        bash scripts/prepare/prepare_toolathlon.sh preflight
    [ ! -e "$GRADING_STATE" ] || { echo "FATAL: use a fresh grading state: $GRADING_STATE" >&2; exit 1; }
    tmux has-session -t "$SERVICE_SESSION" 2>/dev/null && { echo "FATAL: tmux session exists: $SERVICE_SESSION" >&2; exit 1; }
    ss -tln 2>/dev/null | grep -q ":${GRADING_PORT} " && { echo "FATAL: port is busy: $GRADING_PORT" >&2; exit 1; }
    tmux new-session -d -s "$SERVICE_SESSION" "cd '$REPO' && export VTB_PYTHON='$VTB_PYTHON' VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q && source src/commit_utils/load_launch_env.sh && export VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_ROOT='$VTB_ROOT' HF_HOME='$HF_HOME' VTB_RESULTS_DIR='$RESULTS_DIR' VTB_MODEL_RESOURCE_ROOT='$VTB_MODEL_RESOURCE_ROOT' && \
        export VTB_GRADING_STATE='$GRADING_STATE' VTB_VAL_SOURCE='$VAL_SOURCE' VTB_FINAL_LIMIT='$FINAL_LIMIT' VTB_GRADING_HOST='$GRADING_HOST' && \
        export VTB_AGENT_DISCLOSURE='$DISCLOSURE' VTB_GPU_COORDINATION_MODE=shared VTB_EVAL_GPUS='$EVAL_GPUS' && \
        export VTB_GRADING_EVAL_TIMEOUT='$EVAL_TIMEOUT' VTB_GRADING_PORT='$GRADING_PORT' && \
        export VTB_TOOLATHLON_URL='$TOOLATHLON_URL' VTB_TOOLATHLON_VERIFIED_URL='$VERIFIED_URL' && \
        export VTB_MODEL_ADVERTISE_HOST='$ADVERTISE_HOST' VTB_TOOLATHLON_TIMEOUT='$TOOLATHLON_TIMEOUT' VTB_TOOLATHLON_VERIFIED_TIMEOUT='$VERIFIED_TIMEOUT' && \
        export VTB_SERVE_EXTRA_ARGS='$SERVE_EXTRA_ARGS' VTB_SGLANG_EXTRA_ARGS='$SGLANG_EXTRA_ARGS' && source scripts/prepare/configure_evaluation_environment.sh && \
        exec '$VTB_PYTHON' -m eval.grading.server"
    echo "Toolathlon grading service: session=$SERVICE_SESSION state=$GRADING_STATE port=$GRADING_PORT"
}
start_run() {
    require_researcher || return
    "$VTB_PYTHON" scripts/prepare/python/verify_recipe_cache.py "$RECIPE"
    health >/dev/null || { echo "FATAL: grading service is unavailable" >&2; exit 1; }
    tmux has-session -t "$RUN_SESSION" 2>/dev/null && { echo "FATAL: tmux session exists: $RUN_SESSION" >&2; exit 1; }
    model_safe="$(echo "$MODEL" | tr '/:[]' '____')"; gpu_suffix=""; [ "$NUM_GPUS" -le 1 ] || gpu_suffix="_${NUM_GPUS}gpu"
    result="$RESULTS_DIR/${AGENT_CLI}_${RESEARCHER_SAFE}_${HOURS}h${gpu_suffix}${EXPERIMENT_NAME}/${BENCHMARK}_${model_safe}_${RUN_ID}"
    [ ! -e "$result" ] || { echo "FATAL: result exists: $result" >&2; exit 1; }
    tmux new-session -d -s "$RUN_SESSION" "cd '$REPO' && export VTB_PYTHON='$VTB_PYTHON' VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q && source src/commit_utils/load_launch_env.sh && set -a && { [ ! -f api_key.env ] || source api_key.env; } && set +a && \
        export VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_ROOT='$VTB_ROOT' HF_HOME='$HF_HOME' VTB_RESULTS_DIR='$RESULTS_DIR' VTB_MODEL_RESOURCE_ROOT='$VTB_MODEL_RESOURCE_ROOT' && VTB_GRADING_STATE='$GRADING_STATE' \
        VTB_ADMIN_TOKEN_FILE='$GRADING_STATE/admin.token' VTB_PROMPT=workflow VTB_TRAIN_RECIPE='$RECIPE' \
        VTB_GRADING_HOST='$GRADING_CONNECT_HOST' VTB_GRADING_PORT='$GRADING_PORT' VTB_AGENT_GPUS='$AGENT_GPUS' VTB_EXPERIMENT_NAME='$EXPERIMENT_NAME' \
        VTB_EDITABLE_HARNESS=off bash src/run_task.sh '$BENCHMARK' '$AGENT_CLI' '$MODEL' '$RUN_ID' '$HOURS' '$RESEARCHER' '$NUM_GPUS'"
    echo "Toolathlon experiment: session=$RUN_SESSION result=$result harness=fixed"
}
case "${1:-}" in service) start_service ;; health) health ;; run) start_run ;; *) echo "usage: $0 <service|health|run>" >&2; exit 2 ;; esac
