# Toolathlon-VERIFIED evaluation service

A self-contained evaluation island under `eval/service/`. It exposes a small
FastAPI surface a benchmark adapter POSTs to, and orchestrates the official
**held-out FINAL test set** with the official phased flow per task
(`preprocess -> stash -> agent -> restore -> eval`). For the complete
Gym-validation/Verified-final workflow, start with the repository's
[Toolathlon guide](../README.md).

It is **decoupled from the official upstream**: the official verified checkout
is an external, read-only tree named by an environment variable, and this
package never writes to it. The task image is *pulled*, never built locally.

## Layout

```
eval/service/toolathlon_verified/
  app.py            FastAPI surface (default :8061)
  runner.py         per-task container orchestration (phased official flow)
  paths.py          resolves the external checkout + stages overlay + secrets
  overlay/          our customisations, layered onto the official tree
  requirements.txt  host-side deps only (fastapi/uvicorn/pydantic)
  scripts/
    start_service.sh
```

## Three-layer staging

`paths.py` builds a throwaway *staged* tree under
`vtb-env/toolathlon_verified_staged`, in three passes (later layers win):

1. **official raw_code** — copied from `VTB_TOOLATHLON_VERIFIED_REPO` (read-only).
2. **overlay/** — this package's files (our customisations).
3. **per-host secrets** — copied from `VTB_TOOLATHLON_VERIFIED_SECRETS`, a tree
   whose layout mirrors the repo root (e.g. `configs/token_key_session.py`,
   `configs/gcp-oauth.keys.json`, `configs/google_credentials.json`,
   `configs/snowflake_rsa_key*`, ...). Optional: a host running only
   non-credentialed tasks needs none.

The overlay is pinned to upstream commit
`9be8d8fe07a497b18ee61e3f2ae694e9797f39eb`. `paths.py` rejects a checkout at
any other revision or with local changes. Project code is `docker cp`'d into
the per-task container from the staged root (the official flow); a sidecar
commit marker forces an automatic rebuild if an older staged tree is present.
The overlay never edits upstream, and secrets are never committed here.

## Environment

| Variable | Required | Meaning |
| --- | --- | --- |
| `VTB_TOOLATHLON_VERIFIED_REPO` | yes | abs path to a clean official verified checkout at commit `9be8d8fe07a497b18ee61e3f2ae694e9797f39eb` |
| `VTB_TOOLATHLON_VERIFIED_SECRETS` | no | abs path to the per-host secrets tree |
| `TOOLATHLON_VERIFIED_PORT` | no (8061) | service port |
| `TOOLATHLON_VERIFIED_MAX_CONCURRENT` | no (12) | tasks of one job at once |
| `TASK_TIMEOUT` | no (2700) | seconds per task |
| `TOOLATHLON_TASK_RETRIES` | no (1) | crash retries per task |
| `TOOLATHLON_VERIFIED_IMAGE` | no | override the pulled task image |
| `VTB_TOOL_MODEL_BASE_URL` | optional | reader-selected OpenAI-compatible endpoint for the optional upstream provider adapter; exported service setting is forwarded into task containers |

Normal grading always uses the `unified` provider and the submitted checkpoint's request-supplied `MODEL_API_URL`; it does not require an external model-provider endpoint. If directly selecting an optional upstream provider, configure its endpoint and per-host credential explicitly. Host settings and local adapter files are not automatically available to the separately started service.

## Official checkout

The upstream this service drives is not vendored here; clone it once and pin it
to the commit the overlay is maintained against (`paths.py` rejects any other
revision or a dirty tree):

```bash
git clone https://github.com/hkust-nlp/Toolathlon.git /abs/path/to/official/toolathlon_verified
git -C /abs/path/to/official/toolathlon_verified checkout 9be8d8fe07a497b18ee61e3f2ae694e9797f39eb
```

## Run

Prepare the dedicated Python 3.12 service environment and exact dependency pins first;
follow [service environment setup](../README.md#start-the-services). Startup validates
`requirements.txt`; it does not install packages into the host user site.

```bash
VTB_TOOLATHLON_VERIFIED_REPO=/abs/path/to/official/toolathlon_verified \
VTB_TOOLATHLON_VERIFIED_SECRETS=/abs/path/to/per-host/secrets \
    bash eval/service/toolathlon_verified/scripts/start_service.sh
```

The script stages the tree, pulls `lockon0927/toolathlon-task-image:1016beta@sha256:4d04fe4e0a6fdb4946f51bb05120cb44a0eef980231c11252f93b62897afcb9f`,
checks `tasks/finalpool`, then serves on `0.0.0.0:8061` and polls `/health`.
For host preparation and the secrets layout, see [Environment](#environment) and [Security](#security) below, plus the parent [Toolathlon workflow](../README.md).

Inspect the resolved paths without starting anything:

```bash
source src/commit_utils/configure_python.sh
"$VTB_PYTHON" -m eval.service.toolathlon_verified.paths
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/evaluate` | submit a job (list of task ids) |
| GET | `/jobs/{id}` | job status + per-task results |
| GET | `/jobs` | list jobs |
| GET | `/tasks` | **count only** — this is a held-out set; ids live in the adapter's manifest |
| GET | `/health` | prerequisite check (`{"ok": true}` when serveable) |

Each task result is three-state: `pass=true` / `pass=false` (the evaluator
ruled) / `pass=null` (infrastructure failure — ungraded, not counted as a
model failure).

## Notes

- **k8s tasks**: a handful of tasks stand up a `kind` cluster; the runner names
  and tears down the cluster per task. `kubectl`/`kind` must be on PATH for
  those; other tasks need only Docker.
- **canvas conflict groups**: tasks that share mutable canvas state are
  serialised behind a per-group lock so parallel jobs cannot corrupt each other.
- **split-MCP**: when MCP servers run on a separate host, set
  `TOOLATHLON_MCP_HOST`; the runner rewrites the redirect yamls accordingly.

## Security

The service has **no authentication**; `0.0.0.0` is required so task containers
(bridge networks) can reach the model endpoint through the docker0 gateway.
Restrict the port to trusted sources. `GET /tasks` deliberately answers with a
count only.
