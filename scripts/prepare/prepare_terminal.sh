#!/usr/bin/env bash
# Validate Terminal-Bench TBLite context pack and download training data.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
source "$(dirname "${BASH_SOURCE[0]}")/../../src/commit_utils/configure_python.sh" || exit 1
export VTB_ROOT="${VTB_ROOT:-$REPO/vtb-env}"

case "${1:-}" in
    datasets)
        bash scripts/prepare/download_base_model.sh
        "$VTB_PYTHON" scripts/prepare/python/verify_recipe_cache.py --download "${RECIPE:-qwen3_5_9b/slime_sft_terminal}"
        echo "TERMINAL_DATASETS_OK"
        ;;
    preflight)
        export PATH="$VTB_ROOT/tools/bin:$PATH"
        command -v harbor >/dev/null || { echo "FATAL: Harbor is not installed" >&2; exit 1; }
        "$VTB_PYTHON" scripts/prepare/python/verify_harbor.py verify-install --harbor-bin "$(command -v harbor)"
        payload="$REPO/benchmarks/terminal-bench__terminal-bench-2-1/validation_context"
        [ -f "$payload/manifest.json" ] || {
            echo "FATAL: Terminal validation context missing: $payload" >&2
            exit 1
        }
        if [ "${2:-}" = "--formal" ]; then
            "$VTB_PYTHON" scripts/prepare/python/verify_recipe_cache.py "${RECIPE:-qwen3_5_9b/slime_sft_terminal}"
        fi
        echo "TERMINAL_PREPARE_OK context=$payload"
        ;;
    all)
        bash "$REPO/scripts/prepare/prepare_terminal.sh" datasets
        bash "$REPO/scripts/prepare/prepare_terminal.sh" preflight --formal
        ;;
    *) echo "usage: $0 <datasets|preflight|all>" >&2; exit 2 ;;
esac
