#!/usr/bin/env bash
# Materialize and validate the pinned direct AIME evaluation datasets.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source "$(dirname "${BASH_SOURCE[0]}")/../../src/commit_utils/configure_python.sh" || exit 1
export VTB_ROOT="${VTB_ROOT:-$REPO/vtb-env}"
export HF_HOME="${HF_HOME:-$VTB_ROOT/resources/data/hf_cache}"
AIME_SFT_OUTPUT="$VTB_ROOT/resources/data/aime_sft"

usage() {
    echo "usage: $0 <datasets|training-data|preflight|all> [--formal (preflight)|--offline (training-data)]" >&2
}

materialize() {
    bash scripts/prepare/download_base_model.sh
    "$VTB_PYTHON" benchmarks/aime__aime/materialize_dataset.py --overwrite
    "$VTB_PYTHON" benchmarks/aime__26/materialize_dataset.py --overwrite
}

prepare_training_data() {
    local recipe="${RECIPE:-qwen3_5_9b/slime_sft_aime}"
    local output_parent
    if [ "${1:-}" = "--offline" ]; then
        "$VTB_PYTHON" scripts/prepare/python/verify_recipe_cache.py --check-sources "$recipe"
    else
        "$VTB_PYTHON" scripts/prepare/python/verify_recipe_cache.py --download "$recipe"
    fi
    mkdir -p "$(dirname "$AIME_SFT_OUTPUT")"
    output_parent="$(realpath -m "$(dirname "$AIME_SFT_OUTPUT")")"
    source scripts/prepare/configure_evaluation_environment.sh
    docker --host "$DOCKER_HOST" run --rm --network none --user "$(id -u):$(id -g)" \
        -e HOME=/tmp \
        -v "$REPO:/repo:ro" \
        -v "$(realpath -m "$HF_HOME"):/hf:ro" \
        -v "$output_parent:/materialized" \
        --entrypoint python3 "${VTB_TRAINER_IMAGE:-vibetrainbench/trainer:latest}" \
        /repo/scripts/prepare/python/aime_sft_pipeline/prepare.py \
        --repo /repo --hf-home /hf --output-dir /materialized/aime_sft
}

preflight() {
    "$VTB_PYTHON" - <<'PY'
import json
import os
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "benchmarks"))
from _direct_aime import load_tasks

vtb_root = Path(os.environ.get("VTB_ROOT") or repo / "vtb-env").expanduser()
hub = Path(
    os.environ.get("VTB_HUB_TASK_SPECS")
    or os.environ.get("VTB_HARBOR_DATASET_DIR")
    or vtb_root / "hub_task_specs"
).expanduser()
expected = {
    "aime__aime": ("opencompass/AIME2025", "a6ad95f611d72cf628a80b58bd0432ef6638f958"),
    "aime__26": ("MathArena/aime_2026", "d2de22f3c656b4f56cf8981212186377d1e23bc3"),
}
errors = []
for benchmark, (dataset, revision) in expected.items():
    root = (hub / benchmark).resolve()
    try:
        tasks, total = load_tasks(root, -1, None)
        source = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError) as exc:
        errors.append(f"{benchmark}: {exc}")
        continue
    if total != 30 or len(tasks) != 30:
        errors.append(f"{benchmark}: expected 30 tasks")
    if source.get("dataset") != dataset or source.get("revision") != revision:
        errors.append(f"{benchmark}: source pin mismatch")

if errors:
    print("AIME_PREFLIGHT_FAILED", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
print("AIME_PREFLIGHT_OK")
print(f"  - validation: {hub / 'aime__aime'} (30 tasks)")
print(f"  - final: {hub / 'aime__26'} (30 tasks)")
PY
}

case "${1:-}" in
    datasets) materialize ;;
    training-data)
        [ -z "${2:-}" ] || [ "$2" = "--offline" ] || { usage; exit 2; }
        prepare_training_data "${2:-}"
        ;;
    preflight)
        preflight
        if [ "${2:-}" = "--formal" ]; then
            "$VTB_PYTHON" scripts/prepare/python/verify_recipe_cache.py "${RECIPE:-qwen3_5_9b/slime_sft_aime}"
        fi
        ;;
    all) materialize; prepare_training_data; preflight; "$VTB_PYTHON" scripts/prepare/python/verify_recipe_cache.py "${RECIPE:-qwen3_5_9b/slime_sft_aime}" ;;
    *) usage; exit 2 ;;
esac
