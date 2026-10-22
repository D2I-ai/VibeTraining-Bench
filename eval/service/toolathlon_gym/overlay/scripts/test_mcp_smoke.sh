#!/bin/bash
# MCP 全量冒烟测试：在 bake 好的镜像里逐个拉起 25 个 MCP server，
# 走 initialize -> tools/list -> 每个服务器一个只读工具调用。
#
# 前提：
#   - toolathlon-pack:latest 已构建且代码已烤入（bash scripts/bake_code.sh）
#   - postgres:15@sha256:9b1d34adbce1dd07ee6e94b4a2cf698884b89bd44a6c9c12f5da8f3acbfe4957 镜像已拉取
#
# 用法：
#   bash scripts/test_mcp_smoke.sh              # 测全部 25 个
#   bash scripts/test_mcp_smoke.sh 12306        # 只测指定服务器（可多个）
#   bash scripts/test_mcp_smoke.sh --list-tools # 只验证启动与握手，不调工具
#   KEEP=1 bash scripts/test_mcp_smoke.sh       # 跑完保留容器便于排查
#
# 说明：
#   - init.sql.gz 用 -v 只读挂载进 pg（可挂载环境；禁挂载时改回 docker cp）
#   - yaml 里 PG host 写死两种（postgres / toolathlon_pg），且
#     test_mcp_servers.py 的 yaml env 优先于进程环境，故给 pg 挂两个别名
#   - 测试完默认清理所有临时资源；退出码 = 测试结果（0 全过）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

IMAGE="${TOOLATHLON_IMAGE:-toolathlon-pack:latest}"
POSTGRES_IMAGE="${POSTGRES_IMAGE:-postgres:15@sha256:9b1d34adbce1dd07ee6e94b4a2cf698884b89bd44a6c9c12f5da8f3acbfe4957}"
NET="tge-net-test"
PG="tge-pg-test"
AG="tge-test-ag"

cleanup() {
    echo
    echo "Cleaning up test environment ..."
    docker rm -f "$AG" "$PG" >/dev/null 2>&1 || true
    docker network rm "$NET" >/dev/null 2>&1 || true
}

# 0) 前置检查：代码必须已烤进镜像
docker run --rm --entrypoint /bin/bash "$IMAGE" \
    -c "test -f /workspace/test_mcp_servers.py" 2>/dev/null \
    || { echo "FATAL: 代码未烤入镜像。先运行: bash scripts/bake_code.sh"; exit 1; }
docker image inspect "$POSTGRES_IMAGE" >/dev/null 2>&1 \
    || { echo "FATAL: 缺少 $POSTGRES_IMAGE。先运行: docker pull $POSTGRES_IMAGE"; exit 1; }

if [[ "${KEEP:-0}" != "1" ]]; then
    trap cleanup EXIT
fi
cleanup   # 清掉上一次运行可能留下的同名资源

echo "=== [1/4] 网络 + postgres（-v 只读挂载 init.sql.gz）==="
docker network create "$NET" >/dev/null
# 注意：创建时不带 --network，否则后面 network connect 挂别名会报
# "endpoint already exists"。启动前用一次 connect 同时挂两个别名。
docker create --name "$PG" \
    -e POSTGRES_DB=toolathlon_gym \
    -e POSTGRES_USER=eigent \
    -e POSTGRES_PASSWORD=camel \
    -v "$(pwd)/db/init.sql.gz:/docker-entrypoint-initdb.d/init.sql.gz:ro" \
    "$POSTGRES_IMAGE" >/dev/null

# 别名：12306/youtube 系 yaml 连 "postgres"，snowflake yaml 连 "toolathlon_pg"
docker network connect --alias postgres --alias toolathlon_pg "$NET" "$PG"
docker start "$PG" >/dev/null

echo "=== [2/4] 等待 pg 导入完成 ==="
for i in $(seq 1 60); do
    if docker exec "$PG" pg_isready -U eigent -d toolathlon_gym >/dev/null 2>&1 \
       && docker exec "$PG" psql -U eigent -d toolathlon_gym -tAc \
            "SELECT 1 FROM information_schema.schemata WHERE schema_name='canvas'" \
            2>/dev/null | grep -q 1; then
        echo "  pg ready（约 $((i * 3))s）"
        break
    fi
    if [[ $i == 60 ]]; then
        echo "FATAL: pg 3 分钟内未就绪"; docker logs --tail 20 "$PG"; exit 1
    fi
    sleep 3
done

echo "=== [3/4] 测试容器 ==="
# PG_USER/PG_PASSWORD/PG_DATABASE 是 tool_servers.py 的 PGHOST->PG_* 桥接的等价物：
# 12306/notion 等 server 只认 PG_* 名字，不补这三个，冒烟会拿着 yaml 里占位的
# postgres/postgres 去认证而失败（真实任务运行时由桥接逻辑兜底，不受影响）。
docker create --name "$AG" --network "$NET" \
    -e PGHOST="$PG" -e PG_HOST="$PG" -e PGPORT=5432 -e PG_PORT=5432 \
    -e PGUSER=eigent -e PG_USER=eigent \
    -e PGPASSWORD=camel -e PG_PASSWORD=camel \
    -e PGDATABASE=toolathlon_gym -e PG_DATABASE=toolathlon_gym \
    -e LOCAL_SERVERS_PATH=/opt/local_servers \
    -e PYTHON_BIN=/opt/venv/bin/python3 \
    -w /workspace "$IMAGE" sleep 7200 >/dev/null
docker start "$AG" >/dev/null

echo "=== [4/4] 运行 test_mcp_servers.py $* ==="
# 把任意参数（服务器名 / --list-tools / --verbose）原样透传
docker exec "$AG" /opt/venv/bin/python3 test_mcp_servers.py "$@"

