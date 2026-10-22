#!/usr/bin/env bash
# Published Tau3 experiment: 8x A100, 16h, editable D/T and fixed harness.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAU3_BENCHMARK="sierra-research__tau3-bench"
TAU3_ENV="$REPO/benchmarks/$TAU3_BENCHMARK/.env"

if [ -f "$TAU3_ENV" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$TAU3_ENV"
    set +a
fi
if [ -f "$REPO/api_key.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$REPO/api_key.env"
    set +a
fi

source "$REPO/src/commit_utils/launcher_agent.sh"

export VTB_ROOT="${VTB_ROOT:-$REPO/vtb-env}"
export HF_HOME="${HF_HOME:-$VTB_ROOT/resources/data/hf_cache}"
export VTB_MODEL_RESOURCE_ROOT="${VTB_MODEL_RESOURCE_ROOT:-$VTB_ROOT/resources/models/qwen3_5_9b_base}"
RESULTS_DIR="${VTB_RESULTS_DIR:-$VTB_ROOT/results}"
HUB_TASK_SPECS="${VTB_HUB_TASK_SPECS:-${VTB_HARBOR_DATASET_DIR:-$VTB_ROOT/hub_task_specs}}"
VTB_HARBOR_REPO="${VTB_HARBOR_REPO:-$REPO/../harbor}"
export VTB_ROOT VTB_RESULTS_DIR="$RESULTS_DIR" VTB_HUB_TASK_SPECS="$HUB_TASK_SPECS" VTB_HARBOR_REPO
BENCHMARK="${BENCHMARK:-$TAU3_BENCHMARK}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B-Base}"
RECIPE="${RECIPE:-qwen3_5_9b/slime_sft_tau}"
RUN_ID="${RUN_ID:-tau3_research}"
HOURS="${HOURS:-16}"
AGENT_CLI="${AGENT_CLI:-claude}"
RESEARCHER="${RESEARCHER:-}"
NUM_GPUS="${NUM_GPUS:-8}"
AGENT_GPUS="${AGENT_GPUS:-0,1,2,3,4,5,6,7}"
EVAL_GPUS="${EVAL_GPUS:-$AGENT_GPUS}"
GRADING_HOST="${GRADING_HOST:-${VTB_GRADING_HOST:-127.0.0.1}}"
GRADING_CONNECT_HOST="$GRADING_HOST"; [ "$GRADING_CONNECT_HOST" = "0.0.0.0" ] && GRADING_CONNECT_HOST="127.0.0.1"
GRADING_PORT="${GRADING_PORT:-${VTB_GRADING_PORT:-5800}}"
SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:-${VTB_SERVE_EXTRA_ARGS:-}}"; SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
SERVICE_SESSION="${SERVICE_SESSION:-vtb_grading_tau3}"
RUN_SESSION="${RUN_SESSION:-vtb_run_tau3}"
GRADING_STATE="${VTB_GRADING_STATE:-$VTB_ROOT/grading_states/formal/tau3}"
FINAL_LIMIT="${FINAL_LIMIT:--1}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-93600}"
DISCLOSURE="${DISCLOSURE:-score,per_sample_outcomes,problems,outputs}"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_exp/run_exp_tau3.sh <service|health|run>

This launcher is restricted to sierra-research__tau3-bench. It loads only
benchmarks/sierra-research__tau3-bench/.env and optional repository api_key.env.
EOF
}

require_tau3() {
    [ "$BENCHMARK" = "$TAU3_BENCHMARK" ] || {
        echo "FATAL: run_exp_tau3.sh only supports $TAU3_BENCHMARK" >&2
        exit 2
    }
    [ -d "$VTB_HARBOR_REPO/adapters/tau3-bench" ] || {
        echo "FATAL: Harbor checkout lacks adapters/tau3-bench: $VTB_HARBOR_REPO" >&2
        exit 1
    }
    docker image inspect "${TAU2_IMAGE:-vibetrainbench/tau2-bench:043b1989}" >/dev/null 2>&1 || {
        echo "FATAL: Tau2 image is absent; run scripts/prepare/prepare_tau3.sh image" >&2
        exit 1
    }
}

health() {
    (cd "$REPO" && VTB_GRADING_HOST="$GRADING_CONNECT_HOST" VTB_GRADING_PORT="$GRADING_PORT" "$VTB_PYTHON" -m eval.grading.client health)
}

start_service() {
    require_tau3
    "$VTB_PYTHON" "$REPO/scripts/prepare/python/preflight_tau2_for_tau3.py" \
        --formal --state "$GRADING_STATE" --port "$GRADING_PORT"
    if tmux has-session -t "$SERVICE_SESSION" 2>/dev/null; then
        echo "FATAL: tmux session already exists: $SERVICE_SESSION" >&2
        exit 1
    fi
    tmux new-session -d -s "$SERVICE_SESSION" "cd '$REPO' && export VTB_PYTHON='$VTB_PYTHON' VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q && source src/commit_utils/load_launch_env.sh && set -a && \
        { [ ! -f benchmarks/sierra-research__tau3-bench/.env ] || source benchmarks/sierra-research__tau3-bench/.env; } && \
        { [ ! -f api_key.env ] || source api_key.env; } && \
        export VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_ROOT='$VTB_ROOT' HF_HOME='$HF_HOME' VTB_RESULTS_DIR='$RESULTS_DIR' VTB_MODEL_RESOURCE_ROOT='$VTB_MODEL_RESOURCE_ROOT' &&
        export VTB_HUB_TASK_SPECS='$HUB_TASK_SPECS' VTB_HARBOR_REPO='$VTB_HARBOR_REPO' && \
        source scripts/prepare/configure_evaluation_environment.sh && set +a && \
        VTB_GRADING_STATE='$GRADING_STATE' VTB_GRADING_HOST='$GRADING_HOST' \
        VTB_AGENT_DISCLOSURE='$DISCLOSURE' VTB_GPU_COORDINATION_MODE=shared \
        VTB_EVAL_GPUS='$EVAL_GPUS' VTB_FINAL_LIMIT='$FINAL_LIMIT' \
        VTB_GRADING_EVAL_TIMEOUT='$EVAL_TIMEOUT' VTB_SERVE_EXTRA_ARGS='$SERVE_EXTRA_ARGS' VTB_SGLANG_EXTRA_ARGS='$SGLANG_EXTRA_ARGS' \
        VTB_GRADING_PORT='$GRADING_PORT' '$VTB_PYTHON' -m eval.grading.server"
    echo "Tau3 grading service started in tmux: $SERVICE_SESSION"
}

start_run() {
    require_researcher || return
    "$VTB_PYTHON" "$REPO/scripts/prepare/python/verify_recipe_cache.py" "$RECIPE"
    require_tau3
    health >/dev/null 2>&1 || { echo "FATAL: grading service is unavailable" >&2; exit 1; }
    if tmux has-session -t "$RUN_SESSION" 2>/dev/null; then
        echo "FATAL: tmux session already exists: $RUN_SESSION" >&2
        exit 1
    fi
    model_safe="$(echo "$MODEL" | tr '/:[]' '____')"; gpu_suffix=""; [ "$NUM_GPUS" -le 1 ] || gpu_suffix="_${NUM_GPUS}gpu"
    result="$RESULTS_DIR/${AGENT_CLI}_${RESEARCHER_SAFE}_${HOURS}h${gpu_suffix}_tau3_research/${BENCHMARK}_${model_safe}_${RUN_ID}"
    [ ! -e "$result" ] || { echo "FATAL: result exists: $result" >&2; exit 1; }
    tmux new-session -d -s "$RUN_SESSION" "cd '$REPO' && export VTB_PYTHON='$VTB_PYTHON' VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q && source src/commit_utils/load_launch_env.sh && set -a && \
        { [ ! -f benchmarks/sierra-research__tau3-bench/.env ] || source benchmarks/sierra-research__tau3-bench/.env; } && \
        { [ ! -f api_key.env ] || source api_key.env; } && set +a && \
        export VTB_AGENT_ADAPTERS_DIR=$VTB_AGENT_ADAPTERS_DIR_Q VTB_LOCAL_ENV_FILE=$VTB_LOCAL_ENV_FILE_Q VTB_ROOT='$VTB_ROOT' HF_HOME='$HF_HOME' VTB_RESULTS_DIR='$RESULTS_DIR' VTB_MODEL_RESOURCE_ROOT='$VTB_MODEL_RESOURCE_ROOT' &&
        export VTB_HUB_TASK_SPECS='$HUB_TASK_SPECS' VTB_HARBOR_REPO='$VTB_HARBOR_REPO' && \
        VTB_GRADING_STATE='$GRADING_STATE' VTB_ADMIN_TOKEN_FILE='$GRADING_STATE/admin.token' \
        VTB_PROMPT=workflow VTB_TRAIN_RECIPE='$RECIPE' VTB_GRADING_HOST='$GRADING_CONNECT_HOST' VTB_GRADING_PORT='$GRADING_PORT' \
        VTB_AGENT_GPUS='$AGENT_GPUS' VTB_EXPERIMENT_NAME=_tau3_research \
        VTB_EDITABLE_HARNESS=off bash src/run_task.sh '$BENCHMARK' '$AGENT_CLI' '$MODEL' '$RUN_ID' '$HOURS' '$RESEARCHER' '$NUM_GPUS'"
    echo "Tau3 research run started in tmux: $RUN_SESSION result=$result"
}

case "${1:-}" in
    service) start_service ;;
    health) health ;;
    run) start_run ;;
    *) usage; exit 2 ;;
esac
