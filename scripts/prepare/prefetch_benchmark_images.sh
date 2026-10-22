#!/usr/bin/env bash
# Pre-download Harbor task metadata and Docker task images for registered benchmarks.
#
# Intended for long unattended preparation runs: it is resumable, keeps Harbor's own
# registry cache and exported task metadata under vtb-env/, and skips Docker images
# that are already present in the rootless Docker store.
#
# Usage:
#   bash scripts/prepare/prefetch_benchmark_images.sh
#   bash scripts/prepare/prefetch_benchmark_images.sh --dry-run
#   bash scripts/prepare/prefetch_benchmark_images.sh --benchmarks terminal-bench__terminal-bench-2-1,swe-bench__swe-bench-verified
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source "$(dirname "${BASH_SOURCE[0]}")/../../src/commit_utils/configure_python.sh" || exit 1

VTB_ROOT="${VTB_ROOT:-$REPO_ROOT/vtb-env}"
TOOLS_DIR="$VTB_ROOT/tools"
TOOLS_BIN="$TOOLS_DIR/bin"
DATASET_DIR="${VTB_HUB_TASK_SPECS:-${VTB_HARBOR_DATASET_DIR:-$VTB_ROOT/hub_task_specs}}"
IMAGE_LIST="${VTB_IMAGE_LIST:-$VTB_ROOT/benchmark_images.txt}"
FAILED_LIST="${VTB_IMAGE_FAILED_LIST:-$VTB_ROOT/benchmark_images_failed.txt}"
LOG_DIR="${VTB_PREFETCH_LOG_DIR:-$VTB_ROOT/logs}"
LOG_FILE="$LOG_DIR/prefetch_benchmark_images.log"

# Registries rate-limit anonymous pulls -- public.ecr.aws answers `toomanyrequests:
# Rate exceeded` partway through a few hundred images. Each pull is retried with a
# growing wait, and a pull that still fails is recorded and skipped rather than ending
# the run: an unattended window should come back with a list of what is missing, not
# with 90 images left untried because the 91st was throttled.
PULL_RETRIES="${VTB_PULL_RETRIES:-5}"
PULL_BACKOFF_SECONDS="${VTB_PULL_BACKOFF:-30}"
PULL_DELAY_SECONDS="${VTB_PULL_DELAY:-2}"
MAX_PER_BENCHMARK="${VTB_MAX_IMAGES_PER_BENCHMARK:-0}"

# Keep Harbor's registry cache beside the other runtime state. The task backend and grading
# service use the same default, so prefetch and later evaluation reuse metadata and images.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$VTB_ROOT/cache}"
# Which daemon, decided the same way eval/runner/runtime_paths.py decides it: the system daemon
# unless told otherwise. The address is always named rather than left unset, because with
# nothing set the CLI follows the active docker *context* -- and the rootless installer
# creates and selects a `rootless` one, so an unset variable keeps the pull on the
# rootless daemon while appearing to have switched. Prefetching into the daemon that will
# not run the evaluation is the one way this script can waste a whole weekend.
if [ -n "${VTB_DOCKER_HOST:-}" ]; then
    export DOCKER_HOST="$VTB_DOCKER_HOST"
else
    case "$(printf '%s' "${VTB_DOCKER_ROOTLESS:-}" | tr 'A-Z' 'a-z')" in
        1|true|yes) export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock" ;;
        *) export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}" ;;
    esac
fi
export PATH="$TOOLS_BIN:$PATH"

DRY_RUN=0
DOWNLOAD_DATASETS=1
PULL_IMAGES=1
RETRY_FAILED=0
BENCHMARKS="terminal-bench__terminal-bench-2-1,swe-bench__swe-bench-verified"

usage() {
    cat <<'EOF'
Usage: bash scripts/prepare/prefetch_benchmark_images.sh [options]

Pre-downloads Harbor registry metadata/task exports and Docker task images for
registered benchmarks. Dataset-backed entries are read from benchmarks/registry.py.
Tau3 is special: its patched export becomes the evaluator's canonical task source, so
prefetch is a required setup transaction rather than an optional cache warm-up.

Options:
  --benchmarks a,b       comma-separated subset; default: terminal-bench__terminal-bench-2-1,swe-bench__swe-bench-verified
  --datasets-only        download/export Harbor registry datasets, do not pull images
  --images-only          pull images from already exported/local task trees
  --retry-failed         pull only the images recorded as failed by an earlier run
  --max-per-benchmark N  keep at most N images per benchmark (0 = all)
  --dry-run              print what would be done without network pulls/downloads
  -h, --help             show this help

Environment:
  VTB_ROOT                runtime root; default: ./vtb-env
  VTB_HUB_TASK_SPECS      registry task exports; default: $VTB_ROOT/hub_task_specs
  VTB_HARBOR_DATASET_DIR  legacy alias used when VTB_HUB_TASK_SPECS is unset
  VTB_IMAGE_LIST          generated image list; default: $VTB_ROOT/benchmark_images.txt
  VTB_IMAGE_FAILED_LIST   images that could not be pulled; default: $VTB_ROOT/benchmark_images_failed.txt
  DOCKER_HOST             explicit daemon socket; otherwise resolved as below
  VTB_DOCKER_ROOTLESS=1   use the per-user rootless daemon instead of the system one
                          (default is the system/rootful daemon)
  XDG_CACHE_HOME          Harbor registry cache root; default: $VTB_ROOT/cache
  VTB_PULL_RETRIES        attempts per image; default: 5
  VTB_PULL_BACKOFF        seconds added per retry; default: 30
  VTB_PULL_DELAY          seconds between images, to stay under rate limits; default: 2
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --benchmarks)
            BENCHMARKS="${2:?--benchmarks needs a comma-separated value}"
            shift 2
            ;;
        --datasets-only)
            PULL_IMAGES=0
            shift
            ;;
        --images-only)
            DOWNLOAD_DATASETS=0
            shift
            ;;
        --retry-failed)
            RETRY_FAILED=1
            DOWNLOAD_DATASETS=0
            shift
            ;;
        --max-per-benchmark)
            MAX_PER_BENCHMARK="${2:?--max-per-benchmark needs a number}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "FATAL: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

run_or_print() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

mkdir -p "$VTB_ROOT" "$DATASET_DIR" "$LOG_DIR" "$XDG_CACHE_HOME"
# Appended, not truncated: several preparation windows in a row are normal, and the
# earlier ones are the record of what was already pulled.
printf '\n===== %s =====\n' "$(date -Is)" >> "$LOG_FILE"

command -v harbor >/dev/null || {
    echo "FATAL: harbor not found. Install it with: bash scripts/prepare/install_backend_tools.sh" >&2
    exit 1
}
# patch_export_tau3 imports the pinned Harbor source to reuse its task patch. Run it
# with Harbor's own isolated Python so a clean host needs no duplicate Harbor install.
HARBOR_BIN="$(command -v harbor)"
HARBOR_PYTHON="$(dirname "$(readlink -f "$HARBOR_BIN")")/python"
if [ ! -x "$HARBOR_PYTHON" ]; then
    echo "FATAL: Harbor's isolated interpreter is missing; rerun install_backend_tools.sh" >&2
    exit 1
fi
"$VTB_PYTHON" scripts/prepare/python/verify_harbor.py verify-install --harbor-bin "$HARBOR_BIN"
command -v docker >/dev/null || {
    echo "FATAL: docker not found on PATH" >&2
    exit 1
}

# The planner runs in a substitution whose exit status this assignment keeps, so an
# unknown benchmark aborts here instead of prefetching nothing and exiting 0.
plan_text="$(
    VTB_PREFETCH_BENCHMARKS="$BENCHMARKS" \
    VTB_PREFETCH_DATASET_DIR="$DATASET_DIR" \
    "$VTB_PYTHON" - <<'PY'
from pathlib import Path
import os
import re
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / "benchmarks"))
import registry  # noqa: E402

requested = [name.strip() for name in os.environ["VTB_PREFETCH_BENCHMARKS"].split(",")]
if any(not name for name in requested):
    raise SystemExit(f"empty benchmark name in {os.environ['VTB_PREFETCH_BENCHMARKS']!r}")

dataset_dir = Path(os.environ["VTB_PREFETCH_DATASET_DIR"])
slug_pat = re.compile(r"[^A-Za-z0-9_.-]+")
known = sorted(registry.BENCHMARKS)

for name in requested:
    if name not in registry.BENCHMARKS:
        raise SystemExit(f"unknown benchmark {name!r}; known: {', '.join(known)}")
    spec = registry.BENCHMARKS[name]
    if spec.direct_eval:
        raise SystemExit(
            f"{name}: direct evaluator datasets are not Harbor exports; "
            "use bash scripts/prepare/prepare_aime.sh datasets"
        )
    dataset = spec.dataset
    if dataset:
        slug = slug_pat.sub("_", name)
        print(f"dataset\t{name}\t{dataset}\t{dataset_dir / slug}")
    elif spec.tasks_subdir:
        print(f"local\t{name}\t{repo / 'benchmarks' / name / spec.tasks_subdir}")
    else:
        raise SystemExit(f"{name}: no dataset or tasks_subdir in registry")
PY
)" || { echo "FATAL: could not build the prefetch plan (see above)" >&2; exit 1; }
mapfile -t plan <<< "$plan_text"

echo "[prefetch] benchmarks: $BENCHMARKS" | tee -a "$LOG_FILE"
echo "[prefetch] datasets:   $DATASET_DIR" | tee -a "$LOG_FILE"
echo "[prefetch] image list: $IMAGE_LIST" | tee -a "$LOG_FILE"
echo "[prefetch] cache:      $XDG_CACHE_HOME" | tee -a "$LOG_FILE"
echo "[prefetch] docker:     ${DOCKER_HOST:-<docker default: system daemon>}" | tee -a "$LOG_FILE"
echo "[prefetch] harbor:     $(harbor --version 2>&1 | head -1)" | tee -a "$LOG_FILE"

if docker info >/dev/null 2>&1; then
    docker_root="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || true)"
    rootless="$(docker info -f '{{.SecurityOptions}}' 2>/dev/null | grep -c rootless || true)"
    if [ -n "$docker_root" ]; then
        echo "[prefetch] docker root: $docker_root (rootless=$rootless)" | tee -a "$LOG_FILE"
        # Only a rootless daemon is expected to store images under vtb-env; the system
        # daemon's data-root is the admin's choice, so say where it is rather than warn.
        case "$docker_root" in
            "$VTB_ROOT"/*) ;;
            *)
                if [ "$rootless" = "0" ]; then
                    echo "[prefetch] note: system daemon stores images at $docker_root, outside vtb-env" | tee -a "$LOG_FILE"
                else
                    echo "[prefetch] WARN: rootless docker root is outside VTB_ROOT; large pulls may land outside vtb-env" | tee -a "$LOG_FILE"
                fi
                ;;
        esac
    fi
elif [ "$PULL_IMAGES" = "1" ] && [ "$DRY_RUN" != "1" ]; then
    echo "FATAL: docker daemon is not reachable at ${DOCKER_HOST:-the default socket}" >&2
    echo "       For the system daemon you must be in the docker group and your docker" >&2
    echo "       context must not be pinned to rootless: docker context use default." >&2
    echo "       To use the rootless daemon instead, set VTB_DOCKER_ROOTLESS=1." >&2
    exit 1
fi

roots=()
staged_exports=()
declare -A tau_publish_targets=()
cleanup_staged_exports() {
    for path in "${staged_exports[@]}"; do
        [ ! -e "$path" ] || rm -rf -- "$path"
    done
}
trap cleanup_staged_exports EXIT

if [ "$RETRY_FAILED" = "1" ]; then
    [ -s "$FAILED_LIST" ] || {
        echo "[prefetch] nothing to retry: $FAILED_LIST is empty or absent" | tee -a "$LOG_FILE"
        exit 0
    }
    cp "$FAILED_LIST" "$IMAGE_LIST"
    echo "[images] retrying $(grep -c . "$IMAGE_LIST") previously failed image(s)" | tee -a "$LOG_FILE"
elif [ "$DOWNLOAD_DATASETS" = "1" ]; then
    for entry in "${plan[@]}"; do
        IFS=$'\t' read -r kind name source out_dir <<< "$entry"
        if [ "$kind" = "dataset" ]; then
            echo "[dataset] $name -> $source" | tee -a "$LOG_FILE"
            if [ "$name" = "sierra-research__tau3-bench" ] && [ "$DRY_RUN" != "1" ]; then
                stage="$(mktemp -d "${out_dir}.download-XXXXXX")"
                staged_exports+=("$stage")
                tau_publish_targets["$stage"]="$out_dir"
                run_or_print harbor download --export --overwrite -o "$stage" "$source" 2>&1 | tee -a "$LOG_FILE"
                roots+=("$name=$stage")
            else
                run_or_print harbor download --export --overwrite -o "$out_dir" "$source" 2>&1 | tee -a "$LOG_FILE"
                roots+=("$name=$out_dir")
            fi
        else
            echo "[dataset] $name -> local $source" | tee -a "$LOG_FILE"
            roots+=("$name=$source")
        fi
    done
else
    for entry in "${plan[@]}"; do
        IFS=$'\t' read -r kind name source out_dir <<< "$entry"
        if [ "$kind" = "dataset" ]; then
            roots+=("$name=$out_dir")
        else
            roots+=("$name=$source")
        fi
    done
fi

if [ "$RETRY_FAILED" != "1" ]; then
    # Harbor applies this patch in its run path, but not to task trees exported by
    # `harbor download --export`. Patch only Tau3 exports; all other benchmarks keep
    # their current cache format and do not require a Harbor source checkout.
    if [ "$DRY_RUN" != "1" ]; then
        for root_index in "${!roots[@]}"; do
            root="${roots[$root_index]}"
            name="${root%%=*}"
            path="${root#*=}"
            if [ "$name" = "sierra-research__tau3-bench" ]; then
                harbor_repo="${VTB_HARBOR_REPO:-}"
                if [ -z "$harbor_repo" ] && [ -d "$REPO_ROOT/../harbor/adapters" ]; then
                    harbor_repo="$(realpath -m "$REPO_ROOT/../harbor")"
                fi
                if [ ! -d "$harbor_repo/adapters/tau3-bench" ]; then
                    echo "FATAL: Tau3 prefetch needs VTB_HARBOR_REPO with adapters/tau3-bench" >&2
                    exit 1
                fi
                "$VTB_PYTHON" scripts/prepare/python/verify_harbor.py verify-source --checkout "$harbor_repo"
                tau2_expected_revision="$("$VTB_PYTHON" -c 'from eval.runner.tau3_assets import TAU2_COMMIT; print(TAU2_COMMIT)')"
                tau2_default_image="$("$VTB_PYTHON" -c 'from eval.runner.tau3_assets import TAU2_BASE_IMAGE; print(TAU2_BASE_IMAGE)')"
                tau2_image="${TAU2_IMAGE:-$tau2_default_image}"
                docker image inspect "$tau2_image" >/dev/null 2>&1 || {
                    echo "FATAL: Tau2 base image is missing: $tau2_image" >&2
                    echo "  build it first: bash scripts/prepare/prepare_tau3.sh image" >&2
                    exit 1
                }
                tau2_revision="$(docker image inspect "$tau2_image" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
                tau2_litellm="$(docker image inspect "$tau2_image" --format '{{index .Config.Labels "org.vibetrainbench.litellm.version"}}')"
                if [ "$tau2_revision" != "$tau2_expected_revision" ] || [ "$tau2_litellm" != "1.83.0" ]; then
                    echo "FATAL: Tau2 image is stale or lacks pinned Tau runtime dependencies: $tau2_image" >&2
                    echo "  rebuild it: bash scripts/prepare/prepare_tau3.sh image" >&2
                    exit 1
                fi
                "$HARBOR_PYTHON" scripts/prepare/python/patch_export_tau3.py \
                    --export-dir "$path" --harbor-repo "$harbor_repo" 2>&1 | tee -a "$LOG_FILE"
                canonical="${tau_publish_targets[$path]:-$path}"
                if [ "$canonical" != "$path" ]; then
                    backup=""
                    if [ -e "$canonical" ]; then
                        backup="${canonical}.previous-$$"
                        mv "$canonical" "$backup"
                    fi
                    if ! mv "$path" "$canonical"; then
                        [ -z "$backup" ] || mv "$backup" "$canonical"
                        echo "FATAL: could not publish patched Tau3 export: $canonical" >&2
                        exit 1
                    fi
                    [ -z "$backup" ] || rm -rf -- "$backup"
                    roots[$root_index]="$name=$canonical"
                    path="$canonical"
                fi
                echo "[tau3] canonical task source: $path/tau3-bench" | tee -a "$LOG_FILE"
            fi
        done
    fi

    if [ "$DRY_RUN" = "1" ]; then
        any_root=0
        for root in "${roots[@]}"; do
            if [ -e "${root#*=}" ]; then
                any_root=1
                break
            fi
        done
        if [ "$any_root" = "0" ]; then
            echo "[dry-run] no exported/local task roots exist yet; skipping image list refresh" | tee -a "$LOG_FILE"
            exit 0
        fi
    fi

    # Both carriers of an image reference, task.toml and Dockerfile FROM -- see
    # eval/runner/collect_task_images.py for why reading only the first prefetched nothing
    # for swe-bench-verified.
    "$VTB_PYTHON" eval/runner/collect_task_images.py \
        --output "$IMAGE_LIST" \
        --max-per-root "$MAX_PER_BENCHMARK" \
        "${roots[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

image_count="$(grep -c . "$IMAGE_LIST" || true)"
echo "[images] images to consider: $image_count" | tee -a "$LOG_FILE"

if [ "$PULL_IMAGES" != "1" ]; then
    echo "[prefetch] dataset-only mode complete" | tee -a "$LOG_FILE"
    exit 0
fi

if [ "$image_count" = "0" ]; then
    echo "FATAL: no image references found. If this was not --dry-run, check dataset downloads." >&2
    exit 1
fi

pull_with_retries() {
    # A throttled registry is the expected failure, not an exceptional one, so the wait
    # grows with each attempt. `docker pull` is not piped into tee here: a pipeline's
    # exit status would be tee's, and every failure would look like a success.
    local image="$1"
    local attempt=1
    while :; do
        if docker pull "$image" >>"$LOG_FILE" 2>&1; then
            return 0
        fi
        if [ "$attempt" -ge "$PULL_RETRIES" ]; then
            return 1
        fi
        local wait=$((PULL_BACKOFF_SECONDS * attempt))
        echo "[image] attempt $attempt/$PULL_RETRIES failed, waiting ${wait}s: $image" | tee -a "$LOG_FILE"
        sleep "$wait"
        attempt=$((attempt + 1))
    done
}

: > "$FAILED_LIST"
pulled=0
present=0
failed=0
index=0
while IFS= read -r image; do
    [ -n "$image" ] || continue
    index=$((index + 1))
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] docker pull $image" | tee -a "$LOG_FILE"
        continue
    fi
    if docker image inspect "$image" >/dev/null 2>&1; then
        present=$((present + 1))
        continue
    fi
    echo "[image] ($index/$image_count) pull: $image" | tee -a "$LOG_FILE"
    if pull_with_retries "$image"; then
        pulled=$((pulled + 1))
    else
        failed=$((failed + 1))
        echo "$image" >> "$FAILED_LIST"
        echo "[image] GAVE UP after $PULL_RETRIES attempts: $image" | tee -a "$LOG_FILE"
    fi
    [ "$PULL_DELAY_SECONDS" = "0" ] || sleep "$PULL_DELAY_SECONDS"
done < "$IMAGE_LIST"

echo "[prefetch] pulled=$pulled already_present=$present failed=$failed" | tee -a "$LOG_FILE"
if [ "$failed" != "0" ]; then
    echo "[prefetch] failed images listed in $FAILED_LIST; retry with: bash scripts/prepare/prefetch_benchmark_images.sh --retry-failed" | tee -a "$LOG_FILE"
fi
echo "[prefetch] complete" | tee -a "$LOG_FILE"
echo "[prefetch] log: $LOG_FILE"
