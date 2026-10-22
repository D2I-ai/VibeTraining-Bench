#!/usr/bin/env bash
# Validate Toolathlon's local adapters, external services, and shared runtime images.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source "$(dirname "${BASH_SOURCE[0]}")/../../src/commit_utils/configure_python.sh" || exit 1
export VTB_ROOT="${VTB_ROOT:-$REPO_ROOT/vtb-env}"

usage() {
    cat <<'EOF'
Usage: bash scripts/prepare/prepare_toolathlon.sh <datasets|preflight|all>

  datasets   download the base model and Toolathlon training pool into the HF cache
  preflight  validate local adapters, runtime images, and both external services
  all        datasets -> preflight; Toolathlon has no Harbor task-image preparation

Required environment:
  VTB_TOOLATHLON_URL           ToolGym validation service URL
  VTB_TOOLATHLON_VERIFIED_URL  Toolathlon Verified service URL
  VTB_MODEL_ADVERTISE_HOST     model-serving host reachable by both services

Training datasets and the base model are downloaded separately through Hugging Face
according to the selected recipe. This script validates services but does not start them.
EOF
}

download_datasets() {
    bash scripts/prepare/download_base_model.sh
    "$VTB_PYTHON" scripts/prepare/python/verify_recipe_cache.py --download "${RECIPE:-qwen3_5_9b/slime_sft_toolathlon}"
    echo "TOOLATHLON_DATASETS_OK"
}

preflight() {
    local gym_url="${VTB_TOOLATHLON_URL:-}"
    local verified_url="${VTB_TOOLATHLON_VERIFIED_URL:-}"
    local advertise_host="${VTB_MODEL_ADVERTISE_HOST:-}"
    local command_name image

    [ -n "$gym_url" ] || { echo "FATAL: set VTB_TOOLATHLON_URL" >&2; exit 1; }
    [ -n "$verified_url" ] || { echo "FATAL: set VTB_TOOLATHLON_VERIFIED_URL" >&2; exit 1; }
    [ -n "$advertise_host" ] || { echo "FATAL: set VTB_MODEL_ADVERTISE_HOST" >&2; exit 1; }
    for command_name in curl docker tmux; do
        command -v "$command_name" >/dev/null || {
            echo "FATAL: missing $command_name; complete the repository setup first" >&2
            exit 1
        }
    done
    docker info >/dev/null 2>&1 || { echo "FATAL: Docker daemon is unavailable" >&2; exit 1; }
    [ -f benchmarks/eigent-ai__toolathlon_gym/split.json ] || {
        echo "FATAL: ToolGym split is missing" >&2
        exit 1
    }
    [ -f benchmarks/hkust-nlp__toolathlon_verified/evaluate.py ] || {
        echo "FATAL: Toolathlon Verified adapter is missing" >&2
        exit 1
    }
    for image in vibetrainbench/trainer:latest vibetrainbench/sglang:latest; do
        docker image inspect "$image" >/dev/null 2>&1 || {
            echo "FATAL: required image missing: $image" >&2
            echo "Run: bash containers/build_image.sh all" >&2
            exit 1
        }
    done
    curl -sf -m 10 "${gym_url%/}/health" >/dev/null || {
        echo "FATAL: ToolGym service unavailable: $gym_url" >&2
        exit 1
    }
    curl -sf -m 10 "${verified_url%/}/health" >/dev/null || {
        echo "FATAL: Toolathlon Verified service unavailable: $verified_url" >&2
        exit 1
    }
    if [ "${1:-}" = "--formal" ]; then
        "$VTB_PYTHON" scripts/prepare/python/verify_recipe_cache.py "${RECIPE:-qwen3_5_9b/slime_sft_toolathlon}"
    fi
    echo "TOOLATHLON_PREPARE_OK gym=$gym_url verified=$verified_url advertise_host=$advertise_host"
}

case "${1:-}" in
    datasets) download_datasets ;;
    preflight) preflight "${2:-}" ;;
    all) download_datasets; preflight --formal ;;
    *) usage; exit 2 ;;
esac
