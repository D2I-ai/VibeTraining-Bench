#!/usr/bin/env bash
# Prepare SWE-Bench Verified task packages/images and validate local runtime prerequisites.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source "$(dirname "${BASH_SOURCE[0]}")/../../src/commit_utils/configure_python.sh" || exit 1
export VTB_ROOT="${VTB_ROOT:-$REPO_ROOT/vtb-env}"
BENCHMARK="swe-bench__swe-bench-verified"
TASK_EXPORT="${VTB_HUB_TASK_SPECS:-${VTB_HARBOR_DATASET_DIR:-$VTB_ROOT/hub_task_specs}}/$BENCHMARK"

usage() {
    cat <<'EOF'
Usage: bash scripts/prepare/prepare_swe.sh <datasets|tasks|preflight|all>

  datasets   download the base model and SWE training datasets into the HF cache
  tasks      export SWE-Bench Verified tasks and prefetch their Docker images
  preflight  validate the prepared local runtime without running an evaluation
  all        datasets -> tasks -> preflight

Training datasets and the base model are downloaded separately through Hugging Face
according to the selected recipe.
EOF
}

download_datasets() {
    bash scripts/prepare/download_base_model.sh
    "$VTB_PYTHON" scripts/prepare/python/verify_recipe_cache.py --download "${RECIPE:-qwen3_5_9b/slime_sft_swe}"
    echo "SWE_DATASETS_OK"
}

prepare_tasks() {
    bash scripts/prepare/prefetch_benchmark_images.sh --benchmarks "$BENCHMARK"
}

preflight() {
    export PATH="$VTB_ROOT/tools/bin:$PATH"
    local harbor_repo="${VTB_HARBOR_REPO:-$REPO_ROOT/../harbor}"
    local command_name image
    for command_name in docker harbor tmux; do
        command -v "$command_name" >/dev/null || {
            echo "FATAL: missing $command_name; complete the repository setup first" >&2
            exit 1
        }
    done
    docker info >/dev/null 2>&1 || { echo "FATAL: Docker daemon is unavailable" >&2; exit 1; }
    "$VTB_PYTHON" scripts/prepare/python/verify_harbor.py verify-install --harbor-bin "$(command -v harbor)"
    "$VTB_PYTHON" scripts/prepare/python/verify_harbor.py verify-source --checkout "$harbor_repo"
    [ -d "$harbor_repo/adapters/swebench" ] || {
        echo "FATAL: Harbor SWE adapter missing: $harbor_repo/adapters/swebench" >&2
        exit 1
    }
    [ -d "$TASK_EXPORT" ] || {
        echo "FATAL: SWE task export missing: $TASK_EXPORT" >&2
        echo "Run: bash scripts/prepare/prepare_swe.sh tasks" >&2
        exit 1
    }
    for image in vibetrainbench/trainer:latest vibetrainbench/sglang:latest; do
        docker image inspect "$image" >/dev/null 2>&1 || {
            echo "FATAL: required image missing: $image" >&2
            echo "Run: bash containers/build_image.sh all" >&2
            exit 1
        }
    done
    if [ "${1:-}" = "--formal" ]; then
        "$VTB_PYTHON" scripts/prepare/python/verify_recipe_cache.py "${RECIPE:-qwen3_5_9b/slime_sft_swe}"
    fi
    echo "SWE_PREPARE_OK tasks=$TASK_EXPORT"
}

case "${1:-}" in
    datasets) download_datasets ;;
    tasks) prepare_tasks ;;
    preflight) preflight "${2:-}" ;;
    all) download_datasets; prepare_tasks; preflight --formal ;;
    *) usage; exit 2 ;;
esac
