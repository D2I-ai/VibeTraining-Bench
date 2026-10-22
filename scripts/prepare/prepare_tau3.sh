#!/usr/bin/env bash
# Prepare the Tau3 runtime environment. Tau3 uses Tau2 source and images internally.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source "$(dirname "${BASH_SOURCE[0]}")/../../src/commit_utils/configure_python.sh" || exit 1
export VTB_ROOT="${VTB_ROOT:-$REPO_ROOT/vtb-env}"

usage() {
    cat <<'EOF'
Usage: bash scripts/prepare/prepare_tau3.sh <datasets|image|tasks|preflight|all>

  datasets   download the base model and Tau3 training datasets into the HF cache
  image      fetch the pinned Tau2 source (PR ref when needed) and build its image
  tasks      export all 375 Tau3 Harbor tasks, patch atomically, and validate them
  preflight  perform local readiness checks without running an evaluation
  all        datasets -> image -> tasks -> preflight

This script prepares Tau3's runtime dependencies; it never starts training or evaluation.
Training datasets are downloaded separately through Hugging Face according to the selected
recipe. Published split JSON and baseline/context packs are supplied separately at their
declared runtime paths.
EOF
}

download_datasets() {
    bash scripts/prepare/download_base_model.sh
    "$VTB_PYTHON" scripts/prepare/python/verify_recipe_cache.py --download "${RECIPE:-qwen3_5_9b/slime_sft_tau}"
    echo "TAU3_DATASETS_OK"
}

build_image() (
    local tau2_repo="${TAU2_BENCH_REPO_DIR:-$REPO_ROOT/../tau2-bench}"
    local tau2_repo_url="${TAU2_BENCH_REPO_URL:-https://github.com/sierra-research/tau2-bench.git}"
    local pinned_commit pinned_ref
    pinned_commit="$("$VTB_PYTHON" -c 'from eval.runner.tau3_assets import TAU2_COMMIT; print(TAU2_COMMIT)')"
    pinned_ref="$("$VTB_PYTHON" -c 'from eval.runner.tau3_assets import TAU2_PR_REF; print(TAU2_PR_REF)')"
    local expected_commit="${TAU2_BENCH_COMMIT:-$pinned_commit}"
    local expected_ref="${TAU2_BENCH_REF:-$pinned_ref}"
    local image="${TAU2_IMAGE:-vibetrainbench/tau2-bench:${expected_commit:0:8}}"
    local dockerfile="$REPO_ROOT/containers/tau2-bench-base.Dockerfile"
    local domestic_network_mode effective_pip_index_url resolved commit build_context

    case "$(printf '%s' "${VTB_DOMESTIC_NETWORK_MODE:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true) domestic_network_mode=1 ;;
        *) domestic_network_mode=0 ;;
    esac
    if [ -n "${PIP_INDEX_URL:-}" ]; then
        effective_pip_index_url="$PIP_INDEX_URL"
    elif [ "$domestic_network_mode" = "1" ]; then
        effective_pip_index_url="https://repo.huaweicloud.com/repository/pypi/simple"
    else
        effective_pip_index_url="https://pypi.org/simple"
    fi

    [ -f "$dockerfile" ] || { echo "FATAL: missing $dockerfile" >&2; exit 1; }
    if [ ! -d "$tau2_repo/.git" ]; then
        echo "[tau2-source] cloning $tau2_repo_url -> $tau2_repo"
        git clone "$tau2_repo_url" "$tau2_repo"
    fi
    [ -z "$(git -C "$tau2_repo" status --porcelain)" ] || {
        echo "FATAL: Tau2 checkout has uncommitted changes: $tau2_repo" >&2
        exit 1
    }
    if ! git -C "$tau2_repo" cat-file -e "${expected_commit}^{commit}" 2>/dev/null; then
        echo "[tau2-source] fetching unmerged pin $expected_ref"
        git -C "$tau2_repo" fetch origin \
            "$expected_ref:refs/remotes/origin/vtb-tau3-pinned"
    fi
    resolved="$(git -C "$tau2_repo" rev-parse "${expected_commit}^{commit}")"
    [ "$resolved" = "$expected_commit" ] || {
        echo "FATAL: $expected_ref resolved to $resolved, expected $expected_commit" >&2
        exit 1
    }
    if [ "$(git -C "$tau2_repo" rev-parse HEAD)" != "$expected_commit" ]; then
        echo "[tau2-source] checking out pinned commit $expected_commit"
        git -C "$tau2_repo" checkout --detach "$expected_commit"
    fi
    commit="$(git -C "$tau2_repo" rev-parse HEAD)"
    [ "$commit" = "$expected_commit" ] || {
        echo "FATAL: expected Tau2 commit $expected_commit, found $commit" >&2
        exit 1
    }

    build_context="$(mktemp -d)"
    trap 'rm -rf -- "$build_context"' EXIT
    tar -C "$tau2_repo" -cf - pyproject.toml README.md src data | tar -C "$build_context" -xf -
    cp "$dockerfile" "$build_context/tau2-bench-base.Dockerfile"

    echo "[tau2-image] domestic_network_mode=$domestic_network_mode pip_index=$effective_pip_index_url"
    DOCKER_HOST="${VTB_DOCKER_HOST:-${DOCKER_HOST:-unix:///var/run/docker.sock}}" \
        docker build \
        --file "$build_context/tau2-bench-base.Dockerfile" \
        --tag "$image" \
        --build-arg "TAU2_BENCH_COMMIT=$commit" \
        --build-arg "VTB_DOMESTIC_NETWORK_MODE=$domestic_network_mode" \
        --build-arg "PIP_INDEX_URL=$effective_pip_index_url" \
        "$build_context"

    DOCKER_HOST="${VTB_DOCKER_HOST:-${DOCKER_HOST:-unix:///var/run/docker.sock}}" \
        docker run --rm -i --network none --entrypoint python3 "$image" - <<'PY'
import importlib.metadata as metadata
import tau2  # noqa: F401
expected = {"litellm": "1.83.0", "mcp": "1.25.0", "fastmcp": "3.0.0"}
for package, version in expected.items():
    found = metadata.version(package)
    if found != version:
        raise SystemExit(f"{package}: expected {version}, found {found}")
print("Tau2/Tau3 runtime dependency check passed")
PY
    echo "Tau2 base image ready: $image ($commit)"
)

prepare_tasks() {
    VTB_HARBOR_REPO="${VTB_HARBOR_REPO:-$REPO_ROOT/../harbor}" \
        bash scripts/prepare/prefetch_benchmark_images.sh \
        --benchmarks sierra-research__tau3-bench \
        --datasets-only
}

preflight() {
    set -a
    [ ! -f api_key.env ] || source api_key.env
    [ ! -f benchmarks/sierra-research__tau3-bench/.env ] || \
        source benchmarks/sierra-research__tau3-bench/.env
    set +a
    "$VTB_PYTHON" scripts/prepare/python/preflight_tau2_for_tau3.py "$@"
}

case "${1:-}" in
    datasets) download_datasets ;;
    image) build_image ;;
    tasks) prepare_tasks ;;
    preflight) shift; preflight "$@" ;;
    all) download_datasets; build_image; prepare_tasks; preflight --formal ;;
    *) usage; exit 2 ;;
esac
