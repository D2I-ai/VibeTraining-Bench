# Toolathlon

This guide covers the complete Toolathlon workflow in VibeTraining-Bench. The service-specific READMEs remain the source of truth for implementation details. General grading policy is documented in [eval/grading/README.md](../grading/README.md).

## Architecture

Toolathlon validation and final evaluation use separate benchmark-owned HTTP services. This
keeps the final task set hidden from the outer agent.

| Stage | Benchmark | Tasks | Service | Purpose |
|---|---|---:|---|---|
| Validation | `eigent-ai__toolathlon_gym` | 12 | Toolathlon Gym, port 8060 | Agent-visible feedback during research. |
| Final | `hkust-nlp__toolathlon_verified` | 88 | Toolathlon Verified, port 8061 | Held-out score selected by the Gym benchmark's `split.json`. |

```text
outer agent -> grading service -> benchmark adapter -> Toolathlon service
                                                   -> per-task containers -> model endpoint
```

The grading worker serves a submitted checkpoint and invokes the benchmark adapter. The
adapter submits a job to the corresponding Toolathlon service. That service starts isolated
task containers, which call back to the model endpoint advertised by the grading host.

## Prerequisites

Complete the repository [quickstart](../../README.md#quickstart), [container setup](../../containers/README.md), and [runtime environment setup](../../scripts/prepare/README.md). Toolathlon additionally requires
clean external upstream checkouts at the exact revisions supported by the overlays:

```bash
# Gym validation service.
git clone https://github.com/eigent-ai/toolathlon_gym.git /abs/path/to/toolathlon_gym
git -C /abs/path/to/toolathlon_gym checkout 45bbff81419fedbb6d5a2cb46029d2980a7c93c4

# Verified final-evaluation service.
git clone https://github.com/hkust-nlp/Toolathlon.git /abs/path/to/toolathlon_verified
git -C /abs/path/to/toolathlon_verified checkout 9be8d8fe07a497b18ee61e3f2ae694e9797f39eb
```

The launchers reject a checkout at another revision or with local changes. They copy the
upstream source into disposable staged trees under `vtb-env/` and apply the repository's
overlays there; the upstream checkouts remain unchanged.

### Prepare Toolathlon data

Evaluation task definitions and the Gym database seed are included in the two pinned upstream
checkouts above; there is no separate evaluation-dataset download. The Verified task runtime
is supplied by its Docker image, which the service launcher pulls automatically.

The default Toolathlon training recipe uses the pinned `dive-mix8000.jsonl` pool from
`guoxinzju/vibetrainbench-toolathlon`. Download it before formal training. The dedicated
no-training smoke needs the base-model snapshot and evaluation services, not this pool.
Run from the repository root after building `vibetrainbench/trainer:latest`:

```bash
bash scripts/prepare/prepare_toolathlon.sh datasets
```

This populates the exact snapshot declared by
[`training/resources/data/toolathlon_dive_mix8000/resource_config.json`](../../training/resources/data/toolathlon_dive_mix8000/resource_config.json).
The base model is a separate prerequisite; follow the [recipe HF cache contract](../../training/recipes/README.md#hugging-face-dataset-contract).

Verified credentials belong in a separate per-host directory whose layout mirrors the
upstream repository. Never put real credentials in this checkout, an overlay, a Docker image,
or committed documentation. The required secret-file layout and service-specific dependencies are documented in [Toolathlon Verified service](toolathlon_verified/README.md).

## Start the services

First prepare a dedicated Python 3.12 host environment and install the exact service dependencies.
Both service manifests currently use the same pins:

```bash
export VTB_ROOT="${VTB_ROOT:-$PWD/vtb-env}"
"$VTB_ROOT/tools/bin/uv" venv --python 3.12 "$VTB_ROOT/tools/toolathlon-service"
export VTB_PYTHON="$VTB_ROOT/tools/toolathlon-service/bin/python3"
"$VTB_ROOT/tools/bin/uv" pip install --python "$VTB_PYTHON" \
  -r eval/service/toolathlon_gym/requirements.txt
source src/commit_utils/configure_python.sh
"$VTB_PYTHON" scripts/prepare/python/verify_python_environment.py \
  --requirements eval/service/toolathlon_verified/requirements.txt
```

The startup scripts verify these pins and fail if missing; they do not install unpinned user-site packages.
Service/tool credentials are managed independently in the service deployments. These scripts do not source
the outer researcher's `api_key.env` or `.local/runtime.env`.

Run both commands from the VibeTrainBench repository root. Each launcher backgrounds its
service and writes its PID and log under `vtb-env/`, so the commands can run sequentially.

```bash
VTB_TOOLATHLON_GYM_REPO=/abs/path/to/toolathlon_gym \
  bash eval/service/toolathlon_gym/scripts/start_service.sh

VTB_TOOLATHLON_VERIFIED_REPO=/abs/path/to/toolathlon_verified \
VTB_TOOLATHLON_VERIFIED_SECRETS=/abs/path/to/toolathlon_verified_secrets \
  bash eval/service/toolathlon_verified/scripts/start_service.sh
```

The Gym launcher builds `toolathlon-pack:latest` and pulls `postgres:15@sha256:9b1d34adbce1dd07ee6e94b4a2cf698884b89bd44a6c9c12f5da8f3acbfe4957`. The Verified
launcher pulls `lockon0927/toolathlon-task-image:1016beta@sha256:4d04fe4e0a6fdb4946f51bb05120cb44a0eef980231c11252f93b62897afcb9f`. Detailed staging, image, and API
behavior is documented in:

- [Toolathlon Gym service](toolathlon_gym/README.md)
- [Toolathlon Verified service](toolathlon_verified/README.md)

Confirm readiness before starting grading:

```bash
curl -fsS http://127.0.0.1:8060/health
curl -fsS http://127.0.0.1:8061/health
curl -fsS http://127.0.0.1:8060/tasks   # ordered Gym task ids
curl -fsS http://127.0.0.1:8061/tasks   # count only; final ids remain hidden
```

Both services listen on `0.0.0.0` and have no built-in authentication. Restrict ports 8060
and 8061 to trusted hosts with a firewall or security group.

## Connect grading to Toolathlon

Set the service URLs to addresses reachable from the grading process. Set
`VTB_MODEL_ADVERTISE_HOST` to the grading/model-serving host address reachable from the
per-task Docker containers. Do not advertise `127.0.0.1`: loopback inside a task container
points back to that container.

```bash
export VTB_TOOLATHLON_URL="http://<gym-service-host>:8060"
export VTB_TOOLATHLON_VERIFIED_URL="http://<verified-service-host>:8061"
export VTB_MODEL_ADVERTISE_HOST="<model-serving-host-ip>"
```

For a single-host deployment, `<model-serving-host-ip>` is normally the Docker bridge gateway
such as `172.17.0.1`. On a multi-host deployment, use the routable address of the host serving
the submitted model.

`run_exp_toolathlon.sh` passes these values explicitly into its tmux grading session. This matters
when a tmux server already exists, because a new session does not reliably inherit variables
recently exported by the calling shell.

## Validate before a full run

Use the bounded Toolathlon smoke to check the locally served checkpoint, tool calling,
service health, and one real validation/final job. The outer researcher calls its separately
configured provider API. Smoke supports `AGENT_CLI`/`RESEARCHER` selection.

Configure the selected [adapter credentials](../../agents/README.md#credentials) in the environment,
a trusted `VTB_LOCAL_ENV_FILE`, or optional ignored `api_key.env`. Export the service URLs/callback
host above and run the no-training smoke in phases. Service startup alone does not prove model API connectivity:

```bash
export AGENT_CLI=claude RESEARCHER="<your-research-model>"
SMOKE_ID=toolathlon-smoke-001 bash scripts/run_smoke/run_smoke_toolathlon.sh preflight
SMOKE_ID=toolathlon-smoke-001 bash scripts/run_smoke/run_smoke_toolathlon.sh service
SMOKE_ID=toolathlon-smoke-001 bash scripts/run_smoke/run_smoke_toolathlon.sh health
SMOKE_ID=toolathlon-smoke-001 bash scripts/run_smoke/run_smoke_toolathlon.sh run
```

The dedicated smoke materializes the unchanged base model as `final_model`, then exercises
one ToolGym validation task and one Toolathlon Verified final task without training.

## Run an experiment

Wait for the smoke run session to exit; verify and clean it up. Before formal startup,
prepare model conversion and run `prepare_toolathlon.sh preflight --formal`. Use a fresh formal
state without the smoke's one-sample final cap:

```bash
SMOKE_ID=toolathlon-smoke-001 bash scripts/run_smoke/run_smoke_toolathlon.sh verify &&
SMOKE_ID=toolathlon-smoke-001 bash scripts/run_smoke/run_smoke_toolathlon.sh cleanup

# Run the next phase only after successful verification and formal preparation.
export VTB_GRADING_STATE="$VTB_ROOT/grading_states/formal/toolathlon_run_001"
export RUN_ID=toolathlon-research-001
bash scripts/run_exp/run_exp_toolathlon.sh service
bash scripts/run_exp/run_exp_toolathlon.sh health
bash scripts/run_exp/run_exp_toolathlon.sh run
```

Use a unique `RUN_ID`; the launcher refuses to overwrite an existing result directory. See
[scripts/README.md](../../scripts/README.md) for all launcher options and tmux operations.

## Configuration reference

| Variable | Default | Used by |
|---|---:|---|
| `VTB_TOOLATHLON_GYM_REPO` | required | Gym service: clean pinned upstream checkout. |
| `VTB_TOOLATHLON_VERIFIED_REPO` | required | Verified service: clean pinned upstream checkout. |
| `VTB_TOOLATHLON_VERIFIED_SECRETS` | optional | Verified service: per-host secret overlay. |
| `VTB_TOOLATHLON_URL` | `http://127.0.0.1:8060` | Gym benchmark adapter. |
| `VTB_TOOLATHLON_VERIFIED_URL` | `http://127.0.0.1:8061` | Verified benchmark adapter. |
| `VTB_MODEL_ADVERTISE_HOST` | required by launchers | Host address advertised to task containers. |
| `TOOLATHLON_MAX_CONCURRENT` | `12` | Concurrent Gym tasks per job. |
| `TOOLATHLON_VERIFIED_MAX_CONCURRENT` | `12` | Concurrent Verified tasks per job. |
| `TASK_TIMEOUT` | Gym `2100s`; Verified `2700s` | Per-task service wall-clock limit; set separately in each service process. |
| `TOOLATHLON_TASK_RETRIES` | `1` | Infrastructure retries per task; model failures are not retried. |
| `VTB_TOOLATHLON_TIMEOUT` | `3300s` | Whole Gym adapter job timeout. |
| `VTB_TOOLATHLON_VERIFIED_TIMEOUT` | `13500s` | Whole Verified adapter job timeout. |
| `EVAL_TIMEOUT` | `93600s` | Per-attempt timeout passed to the grading service launcher. |

Keep `EVAL_TIMEOUT` above the applicable adapter timeout. A shorter grading timeout kills a
healthy Toolathlon job before it writes metrics and may cause the retry ladder to repeat the
entire evaluation.

## Results and shutdown

| Artifact | Default location |
|---|---|
| Run output, delivered model, and final grading result | `vtb-env/results/` |
| Grading queue state and per-attempt logs | `vtb-env/grading_states/` (one isolated directory per service) |
| Gym service log/PID | `vtb-env/toolathlon_gym-8060.log` / `.pid` |
| Verified service log/PID | `vtb-env/toolathlon_verified-8061.log` / `.pid` |
| Disposable staged upstream trees | `vtb-env/toolathlon_gym_staged/`, `vtb-env/toolathlon_verified_staged/` |

Stop only the processes whose PID files belong to this checkout:

```bash
kill "$(cat vtb-env/toolathlon_gym-8060.pid)"
kill "$(cat vtb-env/toolathlon_verified-8061.pid)"
```

`vtb-env/` is gitignored runtime state. Do not commit it or copy it between machines as a
substitute for rebuilding the staged trees.

## Troubleshooting

1. **Staging rejects the upstream checkout** — check that `HEAD` equals the documented pinned
   commit and that `git status --short` is empty in the upstream repository.
2. **`/health` is not ready** — inspect the corresponding `vtb-env/toolathlon_*.log`; verify
   Docker access, required images, and staged task files.
3. **Task containers cannot call the model** — ensure the model listens on a non-loopback
   interface and `VTB_MODEL_ADVERTISE_HOST` is reachable from Docker bridge networks.
4. **The endpoint works for small prompts but tasks fail** — verify tool-call support,
   concurrency limits, rate limits, and the context window. Toolathlon schemas make requests
   substantially larger than ordinary chat probes.
5. **Evaluation restarts after a fixed interval** — ensure `EVAL_TIMEOUT` exceeds
   `VTB_TOOLATHLON_TIMEOUT` or `VTB_TOOLATHLON_VERIFIED_TIMEOUT`. An outer timeout discards a
   healthy partial job and restarts it from the beginning.
6. **Docker reports address-pool exhaustion** — stop stale task containers and remove only
   networks confirmed to belong to completed Toolathlon jobs; this is a shared-host safety
   decision, not something the launcher performs automatically.
7. **Verified credentialed tasks fail** — verify the per-host secrets tree and external
   provider access, including both domain reachability and any egress-IP allowlist.
