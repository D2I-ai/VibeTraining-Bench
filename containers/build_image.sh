#!/bin/bash
# Build one of this repo's container images.
#
# Definitions live in containers/<name>.Dockerfile and are tracked; the images
# themselves live in the docker daemon's own store, so nothing multi-gigabyte lands
# in the checkout and every consumer refers to an image by tag rather than by path
# -- no directory to configure and no existence check to carry.
#
# Usage:
#   bash containers/build_image.sh trainer
#   bash containers/build_image.sh sglang
#   bash containers/build_image.sh all
#
# Extra arguments after the names go to `docker build`, so a one-off is:
#   bash containers/build_image.sh sglang -- --build-arg SGLANG_IMAGE=lmsysorg/sglang:v0.5.15.post1-cu129@sha256:c527318558f8c9d74a104c3a1f2e404b92da3e3c6a58c5aec2df16e05b56e63d
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The image namespace. Matches the VTB_*_IMAGE defaults in
# src/commit_utils/set_env_vars.sh, so a build and a run agree without either
# being told about the other.
NAMESPACE="${VTB_IMAGE_NAMESPACE:-vibetrainbench}"
TAG="${VTB_IMAGE_TAG:-latest}"

ALL=(trainer sglang)

names=()
extra=()
while [ $# -gt 0 ]; do
    case "$1" in
        --) shift; extra=("$@"); break ;;
        all) names+=("${ALL[@]}") ;;
        -*) extra+=("$1") ;;
        *) names+=("$1") ;;
    esac
    shift
done

if [ ${#names[@]} -eq 0 ]; then
    echo "usage: build_image.sh <name>... [-- docker build args]" >&2
    echo "known: ${ALL[*]} all" >&2
    exit 1
fi

for name in "${names[@]}"; do
    dockerfile="containers/${name}.Dockerfile"
    if [ ! -f "$dockerfile" ]; then
        echo "FATAL: no definition at $dockerfile" >&2
        echo "known: ${ALL[*]}" >&2
        exit 1
    fi
done

for name in "${names[@]}"; do
    image="${NAMESPACE}/${name}:${TAG}"
    echo "[build] containers/${name}.Dockerfile -> ${image}"
    # Build context is the repo root, not containers/: model-stack.txt and
    # check_stack.py are copied in by their repo-relative path so the same paths
    # read the same in the Dockerfile and in a grep.
    docker build \
        -f "containers/${name}.Dockerfile" \
        -t "$image" \
        ${extra[@]+"${extra[@]}"} \
        .
done

echo
echo "[build] done. Present now:"
for name in "${names[@]}"; do
    docker image inspect "${NAMESPACE}/${name}:${TAG}" \
        --format '  {{index .RepoTags 0}}  {{.Size}} bytes  created {{.Created}}' 2>/dev/null \
        || echo "  ${NAMESPACE}/${name}:${TAG}  MISSING"
done
