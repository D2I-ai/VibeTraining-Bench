# Environment shared by smoke and formal grading services. Source it; do not execute it.
#
#     source scripts/prepare/configure_evaluation_environment.sh
#
# Explicit values already present in the environment always win.

if [ -z "${VTB_REPO:-}" ]; then
    VTB_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    export VTB_REPO
fi

export VTB_ROOT="${VTB_ROOT:-$VTB_REPO/vtb-env}"
source "$(dirname "${BASH_SOURCE[0]}")/../../src/commit_utils/configure_python.sh" || return 1

# Always name the Docker daemon. An unset DOCKER_HOST follows the active Docker context,
# which can silently select a rootless daemon that cannot run SWE-Bench images.
if [ -n "${VTB_DOCKER_HOST:-}" ]; then
    export DOCKER_HOST="$VTB_DOCKER_HOST"
else
    case "$(printf '%s' "${VTB_DOCKER_ROOTLESS:-}" | tr 'A-Z' 'a-z')" in
        1|true|yes) export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock" ;;
        *) export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}" ;;
    esac
fi

# Use the Harbor version installed for this checkout and keep all runtime caches under
# VTB_ROOT. Training datasets are resolved through Hugging Face into this HF cache.
case ":$PATH:" in
    *":$VTB_ROOT/tools/bin:"*) ;;
    *) export PATH="$VTB_ROOT/tools/bin:$PATH" ;;
esac
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$VTB_ROOT/cache}"
export HF_HOME="${HF_HOME:-$VTB_ROOT/resources/data/hf_cache}"

# Adapter-local harnesses and editable-harness seeds come from a Harbor checkout.
if [ -z "${VTB_HARBOR_REPO:-}" ] && [ -d "${VTB_REPO}/../harbor/adapters" ]; then
    export VTB_HARBOR_REPO="$(realpath -m "${VTB_REPO}/../harbor")"
fi

# A locally served endpoint still needs a non-empty matching API key for inspect-ai.
export VTB_API_KEY="${VTB_API_KEY:-inspectai}"
export VTB_EVAL_CONCURRENT="${VTB_EVAL_CONCURRENT:-16}"

# Translate the deprecated SWE-specific names once at the environment boundary.
if [ -n "${VTB_SWE_API_BASE:-}" ] && [ -z "${VTB_API_BASE:-}" ]; then
    export VTB_API_BASE="$VTB_SWE_API_BASE"
fi
if [ -n "${VTB_SWE_SERVE_TIMEOUT:-}" ] && [ -z "${VTB_SERVE_TIMEOUT:-}" ]; then
    export VTB_SERVE_TIMEOUT="$VTB_SWE_SERVE_TIMEOUT"
fi

# Evaluation GPU selection also determines tensor parallelism. Formal launchers provide an
# explicit value; the fallbacks keep manual smoke setup predictable.
if [ -z "${VTB_EVAL_GPUS:-}${VTB_EVAL_GPU:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        export VTB_EVAL_GPUS="$CUDA_VISIBLE_DEVICES"
        _vtb_gpus_from="CUDA_VISIBLE_DEVICES"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        VTB_EVAL_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)"
        export VTB_EVAL_GPUS
        _vtb_gpus_from="every visible card"
    else
        _vtb_gpus_from="unset (no nvidia-smi)"
    fi
else
    _vtb_gpus_from="already set"
fi

echo "[eval-env] cards=${VTB_EVAL_GPUS:-<none>} (${_vtb_gpus_from}) -> tp=$(printf '%s' "${VTB_EVAL_GPUS:-}" | awk -F, '{print NF}')"
echo "[eval-env] docker=${DOCKER_HOST}  hf=${HF_HOME}  concurrent=${VTB_EVAL_CONCURRENT}"
