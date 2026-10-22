#!/bin/bash
# 一次性脚本：把项目代码打包进 toolathlon-pack 镜像（authZ 禁挂载环境的配套方案）
# 用法：bash scripts/bake_code.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

IMAGE="toolathlon-pack:latest"
BAKE_TMP="bake-code-tmp"

docker rm -f "$BAKE_TMP" >/dev/null 2>&1 || true

echo "[bake] Creating temp container ..."
docker create --name "$BAKE_TMP" -w /workspace "$IMAGE" true >/dev/null

echo "[bake] Copying project code (~46MB) ..."
docker cp main.py "$BAKE_TMP:/workspace/main.py"
docker cp test_mcp_servers.py "$BAKE_TMP:/workspace/test_mcp_servers.py"
# The '/.' matters: `docker cp tasks c:/workspace/tasks` NESTS the source inside
# an existing target directory (giving /workspace/tasks/tasks), so a re-bake
# silently leaves the imported copy untouched at its original version. Copying
# the CONTENTS overwrites in place, which is what a re-bake has to do.
docker cp tasks/. "$BAKE_TMP:/workspace/tasks"
docker cp configs/. "$BAKE_TMP:/workspace/configs"
docker cp utils/. "$BAKE_TMP:/workspace/utils"
docker cp scripts/. "$BAKE_TMP:/workspace/scripts"
docker cp pyproject.toml "$BAKE_TMP:/workspace/pyproject.toml"

echo "[bake] Committing image (overwrites $IMAGE) ..."
docker commit "$BAKE_TMP" "$IMAGE" >/dev/null
docker rm -f "$BAKE_TMP" >/dev/null

echo "[bake] Done. Verify:"
echo "  docker run --rm $IMAGE ls /workspace"

