# Toolathlon-GYM evaluation service

A self-contained evaluation island under `eval/service/`. It exposes a small
FastAPI surface that a benchmark adapter POSTs to, and orchestrates one
throwaway Docker container (plus a per-task Postgres) per task. For the complete
Gym-validation/Verified-final workflow, start with the repository's
[Toolathlon guide](../README.md).

It is **decoupled from the official upstream**: the official gym checkout is an
external, read-only tree named by an environment variable, and this package
never writes to it.

## Layout

```
eval/service/toolathlon_gym/
  app.py            FastAPI surface (default :8060)
  runner.py         per-task container orchestration
  paths.py          resolves the external official checkout + stages the overlay
  tasks.json        the 12-task validation set this service serves
  overlay/          our customisations, layered onto the official tree
  requirements.txt  host-side deps only (fastapi/uvicorn/pydantic)
  scripts/
    start_service.sh
```

## Two-layer staging

`paths.py` builds a throwaway *staged* tree under `vtb-env/toolathlon_gym_staged`:

1. **official raw_code** — copied from `VTB_TOOLATHLON_GYM_REPO` (read-only).
2. **overlay/** — this package's files, copied on top (they win on collision).

The overlay is pinned to upstream commit
`45bbff81419fedbb6d5a2cb46029d2980a7c93c4`. `paths.py` rejects a checkout at
any other revision or with local changes. The image build and per-task
bind-mount both read from the staged root; a sidecar commit marker forces an
automatic rebuild if an older staged tree is present. The staged tree lives
under `vtb-env/` and is disposable.

## Environment

| Variable | Required | Meaning |
| --- | --- | --- |
| `VTB_TOOLATHLON_GYM_REPO` | yes | abs path to a clean official gym checkout at commit `45bbff81419fedbb6d5a2cb46029d2980a7c93c4` |
| `TOOLATHLON_PORT` | no (8060) | service port |
| `TOOLATHLON_MAX_CONCURRENT` | no (12) | tasks of one job running at once |
| `TOOLATHLON_TASK_RETRIES` | no (1) | retries for infrastructure failures; model failures are not retried |
| `TASK_TIMEOUT` | no (2100) | seconds per task |
| `PG_READY_TIMEOUT` | no (300) | seconds for a per-task pg to import init.sql |

## Official checkout

The upstream this service drives is not vendored here; clone it once and pin it
to the commit the overlay is maintained against (`paths.py` rejects any other
revision or a dirty tree):

```bash
git clone https://github.com/eigent-ai/toolathlon_gym.git /abs/path/to/official/toolathlon_gym
git -C /abs/path/to/official/toolathlon_gym checkout 45bbff81419fedbb6d5a2cb46029d2980a7c93c4
```

## Run

Prepare the dedicated Python 3.12 service environment and exact dependency pins first;
follow [service environment setup](../README.md#start-the-services). Startup validates
`requirements.txt`; it does not install packages into the host user site.

```bash
VTB_TOOLATHLON_GYM_REPO=/abs/path/to/official/toolathlon_gym \
    bash eval/service/toolathlon_gym/scripts/start_service.sh
```

The script builds the `toolathlon-pack:latest` image from the staged root
(first run: 10-20 min), pulls `postgres:15@sha256:9b1d34adbce1dd07ee6e94b4a2cf698884b89bd44a6c9c12f5da8f3acbfe4957`, then serves on `0.0.0.0:8060`
and polls `/health` until `{"ok": true}`.

Inspect the resolved paths without starting anything:

```bash
source src/commit_utils/configure_python.sh
"$VTB_PYTHON" -m eval.service.toolathlon_gym.paths
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/evaluate` | submit a job (list of task ids) |
| GET | `/jobs/{id}` | job status + per-task results |
| GET | `/jobs` | list jobs |
| GET | `/tasks` | the served task ids |
| GET | `/health` | prerequisite check (`{"ok": true}` when serveable) |

Each task result is three-state: `pass=true` / `pass=false` (the evaluator
ruled) / `pass=null` (infrastructure failure — ungraded, not counted as a
model failure).

## Security

The service has **no authentication**; `0.0.0.0` is required so the benchmark
host can reach it. Restrict the port to trusted sources (security group /
firewall). Anyone who can reach it can submit jobs and read results.
