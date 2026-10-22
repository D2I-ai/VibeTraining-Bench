#!/usr/bin/env bash
# Register any Harbor Hub dataset (owner/name) as a visible local benchmark.
#
# Source declarations stay under benchmarks/<safe-alias>/; downloaded task packages are
# runtime data under vtb-env/hub_task_specs/. Experiment execution remains owned by the
# benchmark-specific smoke and formal launchers.
#
# Usage:
#   bash scripts/prepare/register_harbor_benchmark.sh aime/aime
#   VTB_HARBOR_REPO=/path/to/harbor bash scripts/prepare/register_harbor_benchmark.sh sierra-research/tau3-bench
set -euo pipefail

DATASET="${1:?usage: register_harbor_benchmark.sh <owner/name>}"

case "$DATASET" in
    */*) ;;
    *)
        echo "FATAL: '$DATASET' is not an owner/name Harbor Hub dataset." >&2
        exit 1
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source "$(dirname "${BASH_SOURCE[0]}")/../../src/commit_utils/configure_python.sh" || exit 1

VTB_ROOT="${VTB_ROOT:-$REPO_ROOT/vtb-env}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$VTB_ROOT/cache}"
export PATH="$VTB_ROOT/tools/bin:$PATH"

ALIAS="${VTB_HUB_ALIAS:-$("$VTB_PYTHON" - "$DATASET" <<'PY'
import re
import sys
name = sys.argv[1].strip().strip("/")
print(re.sub(r"[^A-Za-z0-9_.-]+", "__", name))
PY
)}"
BENCH_DIR="$REPO_ROOT/benchmarks/$ALIAS"
SPEC_DIR="$VTB_ROOT/hub_task_specs/$ALIAS"

command -v harbor >/dev/null || {
    echo "FATAL: harbor not found. Install it with: bash scripts/prepare/install_backend_tools.sh" >&2
    exit 1
}
"$VTB_PYTHON" scripts/prepare/python/verify_harbor.py verify-install --harbor-bin "$(command -v harbor)"

mkdir -p "$BENCH_DIR" "$SPEC_DIR" "$XDG_CACHE_HOME"

DATASET="$DATASET" ALIAS="$ALIAS" BENCH_DIR="$BENCH_DIR" "$VTB_PYTHON" eval/runner/register_hub_benchmark.py

echo "[hub] registered $DATASET as benchmark '$ALIAS' in benchmarks/$ALIAS/benchmark.json"

if [ "${VTB_HUB_DOWNLOAD:-1}" != "0" ]; then
    if [ ! -f "$SPEC_DIR/.downloaded" ]; then
        echo "[hub] exporting task specs to $SPEC_DIR"
        harbor download "$DATASET" --output-dir "$SPEC_DIR" --overwrite
        date -u +%FT%TZ > "$SPEC_DIR/.downloaded"
    else
        echo "[hub] task specs already exported at $SPEC_DIR"
    fi
fi

echo "[hub] preparation complete; use the matching smoke or formal experiment launcher to run it"
