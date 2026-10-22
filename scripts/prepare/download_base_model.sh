#!/usr/bin/env bash
# Prepare the manifest-pinned base model. Download does not run GPU conversion.
# Existing resources with unknown provenance are never overwritten.
#
# Usage:
#   bash scripts/prepare/download_base_model.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source "$(dirname "${BASH_SOURCE[0]}")/../../src/commit_utils/configure_python.sh" || exit 1

export VTB_ROOT="${VTB_ROOT:-$REPO_ROOT/vtb-env}"
case "${1:-download}" in
    download|convert|preflight|snapshot-preflight)
        action="${1:-download}"
        [ "$#" -eq 0 ] || shift
        exec "$VTB_PYTHON" scripts/prepare/python/prepare_model.py "$action" "$@" ;;
    *) echo "usage: $0 <download|convert|preflight|snapshot-preflight> [--offline (download only)]" >&2; exit 2 ;;
esac
