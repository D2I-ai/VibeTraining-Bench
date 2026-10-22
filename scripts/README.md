# Operational scripts

Run every command from the repository root. Workflows are benchmark-specific and grouped by purpose:

```text
scripts/
├── prepare/       environment setup, assets, and local preflight
├── run_smoke/     bounded validation/final connectivity checks
└── run_exp/       paper-aligned formal experiments
```

Python helpers live only under the corresponding `python/` directory. Shared infrastructure remains generic; each benchmark has one discoverable shell entrypoint in all three workflow directories.

## Workflow contract

1. Prepare the host, shared images, model/data cache, and benchmark assets.
2. Run a bounded smoke with fresh smoke state.
3. Clean up the smoke service.
4. Start a formal grading service with a different fresh state.
5. Check health.
6. Start the formal experiment.

Scripts never combine service startup and experiment launch. This prevents a failed service from being mistaken for a running experiment and leaves the operator in control of GPU/API-consuming actions.

## Preparation

| Benchmark | Entrypoint | Actions | Responsibility |
|---|---|---|---|
| SWE | `prepare/prepare_swe.sh` | `datasets`, `tasks`, `preflight`, `all` | Manifest-pinned training data/model, task images, runtime/formal checks. |
| Terminal | `prepare/prepare_terminal.sh` | `datasets`, `preflight`, `all` | Three pinned Terminal SFT files plus the benchmark-local TBLite context. |
| Tau3 | `prepare/prepare_tau3.sh` | `datasets`, `image`, `tasks`, `preflight`, `all` | Three pinned training files, Tau2 image, patched tasks, explicit split. |
| Toolathlon | `prepare/prepare_toolathlon.sh` | `datasets`, `preflight`, `all` | Pinned training data/model, adapters, images, callback host, external services. |
| AIME | `prepare/prepare_aime.sh` | `datasets`, `training-data`, `preflight`, `all` | Evaluation data/model plus deterministic conversion of three pinned mathematics corpora. |

Examples:

```bash
# Examples of asset-only preparation; invoke the matching runtime preflight next.
bash scripts/prepare/prepare_swe.sh datasets
bash scripts/prepare/prepare_swe.sh tasks
bash scripts/prepare/prepare_terminal.sh datasets
bash scripts/prepare/prefetch_benchmark_images.sh \
  --benchmarks terminal-bench__terminal-bench-2-1,openthoughts__openthoughts-tblite-val50
bash scripts/prepare/prepare_tau3.sh datasets
bash scripts/prepare/prepare_tau3.sh image
bash scripts/prepare/prepare_tau3.sh tasks
bash scripts/prepare/prepare_toolathlon.sh datasets
bash scripts/prepare/prepare_aime.sh datasets
bash scripts/prepare/prepare_aime.sh training-data
```

Large downloads and builds occur only when explicitly invoked. Preparation reads revisions/file patterns directly from recipe resource declarations. `preflight` checks runtime prerequisites for smoke; `preflight --formal` and formal `run` also check training files, baseline payloads, and matching model/conversion provenance. `all` invokes downloads and formal checks, but never GPU model conversion; prepare `download_base_model.sh convert` separately before using `all` for a formal experiment. Dataset revisions and preparation requirements are documented in [training/recipes/README.md](../training/recipes/README.md#data-readiness).

Shared preparation utilities:

| Script | Purpose |
|---|---|
| `prepare/install_backend_tools.sh` | Install the pinned Harbor CLI under `VTB_ROOT`. |
| `prepare/configure_evaluation_environment.sh` | Sourceable evaluation/grading environment defaults. |
| `prepare/prefetch_benchmark_images.sh` | Export Harbor task metadata and pull task images. |
| `prepare/register_harbor_benchmark.sh` | Register a Harbor Hub `owner/name` dataset locally. |

See [prepare/README.md](prepare/README.md) for environment variables and paths.

## Bounded smoke tests

Every smoke supports:

```text
preflight → service → health → run → wait for run-session exit → verify → cleanup
```

Smoke and formal launchers support `AGENT_CLI`/`RESEARCHER`. They default to the `claude` adapter; explicitly select a researcher model before preflight/run. `service`, `health`, and `cleanup` do not require provider credentials. Custom integrations are selected through `VTB_LOCAL_ENV_FILE`/`VTB_AGENT_ADAPTERS_DIR`; local code and configuration are gitignored.

Use a unique `SMOKE_ID`; do not use the resulting state or score as a formal result.

```bash
export AGENT_CLI=claude RESEARCHER="<your-model>"
SMOKE_ID=swe-connect-001 bash scripts/run_smoke/run_smoke_swe.sh preflight
SMOKE_ID=swe-connect-001 bash scripts/run_smoke/run_smoke_swe.sh service
SMOKE_ID=swe-connect-001 bash scripts/run_smoke/run_smoke_swe.sh health
SMOKE_ID=swe-connect-001 bash scripts/run_smoke/run_smoke_swe.sh run
# Wait for the printed run session to exit; proceed only after each phase succeeds.
SMOKE_ID=swe-connect-001 bash scripts/run_smoke/run_smoke_swe.sh verify &&
SMOKE_ID=swe-connect-001 bash scripts/run_smoke/run_smoke_swe.sh cleanup
```

| Entrypoint | Validation | Final |
|---|---|---|
| `run_smoke_swe.sh` | 1 SWE validation task | 1 official-final connectivity task; held-out membership is not guaranteed without canonical order |
| `run_smoke_terminal.sh` | 1 TBLite task | 1 Terminal-Bench task |
| `run_smoke_tau3.sh` | 1 Tau3 validation task | 1 Tau3 final task |
| `run_smoke_toolathlon.sh` | 1 ToolGym task | 1 Toolathlon Verified task |
| `run_smoke_aime.sh` | 1 AIME 2025 problem/completion | 1 AIME 2026 problem/completion |

`run` returns after starting detached tmux; it does not wait for evaluation completion. Monitor the printed run session. `verify` succeeds only after both expected jobs and the official `metrics.json` are complete. Smoke scripts use isolated grading state names. `cleanup` stops grading and removes its tmux wrapper; Terminal/AIME additionally remove the run wrapper. It is not a safe active-run cancellation mechanism: use the interruption steps below first. State and result files are preserved.

Toolathlon additionally requires:

```text
VTB_TOOLATHLON_URL
VTB_TOOLATHLON_VERIFIED_URL
VTB_MODEL_ADVERTISE_HOST
```

Smoke evaluates the pinned local `Qwen/Qwen3.5-9B-Base` checkpoint while the selected researcher uses its own model API. Adapter/model overrides change researcher wiring and the result-directory prefix, not the benchmark's base checkpoint or scoring policy. Keep launch inputs consistent across phases; `smoke_run.json` records the non-secret result location for later verification. Provider/model examples are documented in [the main API section](../README.md#api-setup-and-model-selection) and [agents/README.md](../agents/README.md). Endpoint URLs are deployment choices.

## Formal experiments

Every formal launcher supports exactly:

```text
service → health → run
```

Example:

```bash
export VTB_ROOT="${VTB_ROOT:-$PWD/vtb-env}"
export VTB_GRADING_STATE="$VTB_ROOT/grading_states/formal/swe_run_001"
export AGENT_CLI=claude RESEARCHER="<your-research-model>" RUN_ID=swe-run-001
bash scripts/run_exp/run_exp_swe.sh service
bash scripts/run_exp/run_exp_swe.sh health
bash scripts/run_exp/run_exp_swe.sh run
```

| Entrypoint | Benchmark | GPUs | Budget | Recipe | Harness |
|---|---|---:|---:|---|---|
| `run_exp_swe.sh` | SWE-Bench Verified | 8×A100 | 24h | `slime_sft_swe` | editable |
| `run_exp_terminal.sh` | Terminal-Bench 2.1 | 8×A100 | 24h | `slime_sft_terminal` | editable |
| `run_exp_tau3.sh` | Tau3-Bench | 8×A100 | 16h | `slime_sft_tau` | fixed |
| `run_exp_toolathlon.sh` | Toolathlon | 8×A100 | 16h | `slime_sft_toolathlon` | fixed |
| `run_exp_aime.sh` | AIME | 8×A100 | 16h | `slime_sft_aime` | fixed |

The launchers expose launcher-input overrides for controlled local diagnostics, but paper comparisons use the defaults above. A service refuses an existing grading-state path; choose a new `VTB_GRADING_STATE` instead of reusing smoke or interrupted formal state.

Expected formal phase outcomes:

| Phase | Success gate |
|---|---|
| `service` | Prints grading session, state, and port; the detached service remains alive. |
| `health` | Returns JSON/status `ok` with the expected benchmark, validation, disclosure, and GPU policy. |
| `run` | Prints research session/result path; this means started, not completed. |
| completion | Research session exits normally; result path contains `final_model/`, `metrics.json`, logs, and manifests. Grading remains running until stopped. |

## Completion, interruption, and safe shutdown

### Monitor without stopping

Use the exact session names printed by launchers:

```bash
tmux ls
tmux attach -t <run-session>
```

Detach while leaving the job running with `Ctrl-b`, then `d`. Do not press `Ctrl-C` merely to leave the view.

### Successful completion

1. Wait until `tmux has-session -t <run-session>` reports no session.
2. Confirm the printed result directory contains a valid `metrics.json`, archived `final_model/`, resource manifest, and logs.
3. For smoke, run `verify`, then `cleanup`.
4. For formal runs, stop grading gracefully after no other run uses it:

```bash
tmux attach -t <grading-session>
# Press Ctrl-C once and wait for the shutdown message/session exit.
```

If attach is unavailable but the state belongs exclusively to this service, send SIGTERM through its PID file and wait for exit:

```bash
service_pid="$(cat <grading-state>/service.pid)"
kill -TERM "$service_pid"
while kill -0 "$service_pid" 2>/dev/null; do sleep 1; done
```

Preserve grading state with published results. Removing a tmux session is not a substitute for stopping the underlying service.

### Interrupt an active run

1. Stop the owned research command first:

```bash
tmux send-keys -t <run-session> C-c
```

2. Wait for its cleanup/traps. Attach and inspect logs if the session does not exit.
3. If an owned grading job remains queued/running, identify it and cancel it with the token that owns it:

```bash
source src/commit_utils/configure_python.sh
"$VTB_PYTHON" -m eval.grading.client --base http://127.0.0.1:<grading-port> queue \
  --token-file <owned-token-file>
"$VTB_PYTHON" -m eval.grading.client --base http://127.0.0.1:<grading-port> cancel \
  --token-file <owned-token-file> --job-id <job-id> \
  --reason "operator interrupted run"
```

A client timeout or terminal disconnect does not cancel a grading job. Wait until cancellation releases the GPU.

4. Stop grading with `Ctrl-C` or SIGTERM as above only after confirming it is not shared by another active run.
5. Use `tmux kill-session` only to remove a stale tmux wrapper after the owned process is already stopped. Never delete shared containers, networks, mounts, state, or results without confirming ownership.

## State and outputs

Default runtime roots:

```text
${VTB_ROOT}/results/                               experiment outputs
${VTB_ROOT}/grading_states/formal/<benchmark>/      formal grading state
${VTB_ROOT}/grading_states/smoke/<benchmark>_<id>/   smoke grading state
${VTB_ROOT}/grading_states/direct/                  direct-service default
${VTB_ROOT}/hub_task_specs/                          exported Harbor task packages
```

Keep `VTB_GRADING_STATE` consistent across service, run, verification, and cleanup phases. Do not move a live service's state; persisted paths may be absolute. For repeated formal runs, choose a fresh path such as `${VTB_ROOT}/grading_states/formal/swe_run_002`.

A formal result directory contains the exact prompt, outer-agent trace, workspace resource manifest, training logs, submitted model, and final metrics. See [training/README.md](../training/README.md) and [eval/grading/README.md](../eval/grading/README.md).

## Entrypoint conventions

- Shell entrypoints use `prepare_<benchmark>.sh`, `run_smoke_<benchmark>.sh`, or `run_exp_<benchmark>.sh`.
- Python helpers live under purpose-local `python/` directories.
- Smoke and formal state are always separate.
- Baseline/context packs and split files are supplied at declared paths; preparation validates them without regeneration.
- Start services and experiments separately; smoke and formal launchers do not provide an `all` action.
