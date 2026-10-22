# Shared grading service

The grading service is the trust boundary between research and evaluation. It registers runs, queues submissions, serves each submitted checkpoint through SGLang, invokes the benchmark evaluator, filters validation feedback, and persists official final results. Benchmark harnesses remain the scoring authority.

Normal users start it through `scripts/run_exp/run_exp_<benchmark>.sh`; `src/run_task.sh` registers the run and submits the final model automatically.

- Operational phases: [scripts/README.md](../../scripts/README.md)
- Runtime variables/paths: [scripts/prepare/README.md](../../scripts/prepare/README.md)
- Benchmark declarations: [benchmarks/README.md](../../benchmarks/README.md)
- Recipe validation contract: [training/recipes/README.md](../../training/recipes/README.md)

## Phased startup

```bash
bash scripts/run_exp/run_exp_swe.sh service
bash scripts/run_exp/run_exp_swe.sh health
AGENT_CLI=claude RESEARCHER="<your-research-model>" RUN_ID=swe-run-001 \
  bash scripts/run_exp/run_exp_swe.sh run
```

`service`, `health`, and `run` are deliberately separate. Never start `run` before health succeeds. Every formal service requires a fresh `VTB_GRADING_STATE`; smoke state and interrupted formal state are not interchangeable.

For an operator-managed service with GPUs disjoint from training:

```bash
export VTB_ROOT="${VTB_ROOT:-$PWD/vtb-env}"
export VTB_GPU_COORDINATION_MODE=isolated
export VTB_EVAL_GPUS=7
export VTB_GRADING_STATE="$VTB_ROOT/grading_states/direct_custom_001"
source scripts/prepare/configure_evaluation_environment.sh
"$VTB_PYTHON" -m eval.grading.server
```

Set overrides before sourcing the environment. Direct startup is an advanced path; benchmark launchers encode the experiment policy.

## Architecture

| Component | Responsibility |
|---|---|
| `server/` | HTTP routing, registration, queue ownership, disclosure, archival, and lifecycle. |
| `executor.py` | Serve checkpoint, invoke evaluator, apply retry ladder, and tear down resources. |
| `store.py` | Tokens, run/submission history, baselines, task statistics, and timing. |
| `valsplit.py` | Parse validation sources and reconstruct validation/held-out metrics. |
| `progress.py` | Live phase/count/elapsed/ETA information. |
| `client.py` | Standard-library CLI/HTTP client used by operators and agent containers. |
| `show_agent_grading.py` | Extract grading calls/replies from a completed research trace. |

One service has one evaluation worker. Multiple runs may queue concurrently, but only one grading job owns the configured evaluation GPU allocation at a time.

## Authentication and role isolation

Registration is operator-only and uses `${VTB_GRADING_STATE}/admin.token`. Do not pass the admin token on a command line or mount grading state into the research container.

Each run receives mode-`0600` scoped tokens:

| Role | Sandbox root | Capabilities |
|---|---|---|
| `agent` | writable task workspace | Submit models from its workspace for service-selected validation; inspect/cancel only its jobs. Cannot request final mode or write official metrics. |
| `official` | host evaluation directory | Submit archived delivery for validation/final and write official artifacts. Never mounted into the research container. |

Submitted model paths are resolved inside the token's registered sandbox root and must be complete readable model directories.

## Validation and final routing

Resolution order:

1. operator `VTB_VAL_SOURCE` override;
2. benchmark `validation_source` in `benchmark.json`;
3. fallback `split:30%`.

| Form | Validation set | Official final set |
|---|---|---|
| `split:<count>` | First count in canonical order | Ordered remainder only when complete canonical order is known; otherwise target evaluation with reconstruction when sufficient per-sample evidence exists. |
| `split:<percent>%` | First percentage; total must be known | Same canonical-order requirement/fallback as count splits. |
| `task:<benchmark-id>` | Entire named benchmark | Entire target benchmark. |
| `file:<path>` | IDs/file-owned routing | Separately declared final benchmark when present; otherwise the target benchmark. No remainder is inferred from the validation file. |
| `json:<absolute-path>` | Explicit `val` IDs | Explicit `test` IDs; currently Tau3-only. |
| `none` | Disabled | Target benchmark remains operator-controlled. |

Agent-role submissions always run validation. They may request a smaller positive limit but cannot choose samples outside the selected validation set.

For a true held-out `split:` final, the service needs complete canonical ordered sample IDs. A complete evaluation or reviewed baseline may teach that order; a subset run cannot. Without that order, the service may evaluate the target benchmark and reconstruct partitions only when the evaluator provides sufficient per-sample evidence. Connectivity smoke verifies validation and official-final execution, not disjoint membership. If task size/order changes, old and new scores are incomparable and a new state is required.

A recipe may declare the expected benchmark and validation source. Registration fails on mismatch so candidates and baselines cannot be compared across different splits.

### Benchmark policy

| Benchmark | Validation → final |
|---|---|
| SWE | fixed validation prefix → held-out SWE remainder |
| Terminal | TBLite benchmark → full Terminal-Bench target |
| Tau3 | JSON `val` → JSON `test` |
| Toolathlon | ToolGym file/list → Toolathlon Verified |
| AIME | AIME 2025 → AIME 2026 |

The Tau3 membership file is prepared under `VTB_ROOT`; API responses expose only count/digest, not hidden membership.

## Disclosure policy

`VTB_AGENT_DISCLOSURE` is a comma-separated set:

| Value | Agent-visible information |
|---|---|
| `score` | Aggregate validation score/metrics. |
| `per_sample_outcomes` | Sample IDs/status/correctness without hidden targets. |
| `problems` | Disclosed validation problem text. |
| `outputs` | Candidate outputs/trajectories allowed by the benchmark policy. |

Direct-service default is `score`. Formal launchers may enable all four for the benchmark's validation set. Every added field increases contamination risk. Reference answers, verifier targets, official final rows, and undisclosed service internals are always refused.

| Variable | Purpose |
|---|---|
| `VTB_AGENT_DISCLOSURE` | Allowed fields. |
| `VTB_AGENT_DISCLOSURE_MAX_SAMPLES` | Row cap; `0` means the submitting job's own limit. |
| `VTB_AGENT_DISCLOSURE_MAX_CHARS` | Per-text-field truncation; default `12000`. |

Agent and official tokens receive different views even for the same completed evaluation.

## Grading state

`VTB_GRADING_STATE` is service identity, policy state, and audit record—not a generic cache.

| Path | Purpose |
|---|---|
| `admin.token` | Operator credential, mode `0600`. |
| `service.pid` | Active service PID. |
| `service.log` | Registration, queue, attempts, disclosure, and result events. |
| `task_stats.json` | Per-benchmark `dataset_total`, canonical `ordered_ids`, and observed `val_ids`. Determines held-out scope. |
| `work/<job-id>/` | Active/failed evaluator attempts and logs. |
| `archive/` | Successful evaluation records used for audit/disclosure. |
| validation/baseline references | Resource-backed baseline/context linkage and compatibility state. |

Per-run artifacts live in the registered result/evaluation directory:

| File | Purpose |
|---|---|
| `submissions.json` | Ordered candidate submissions, scores, errors, attempts, and baseline comparisons. |
| `metrics.json` | Official final result written only by an official submission. |
| `final_eval.log` | Evaluator output for official final scoring. |

Rules:

- use new state for every smoke/formal service;
- do not edit `task_stats.json`, token files, or split files after startup;
- do not reuse a state after changing benchmark order, validation source, or external assets;
- preserve state with published results for audit;
- never seed formal state by copying smoke state.

## Baselines and validation context

A `validation_baseline` resource is keyed by benchmark, base model, and validation source. A `validation_context` resource is an agent-facing diagnostic pack. Both are read-only and separate from training data.

Baseline and context packs are supplied at declared paths. Preparation validates them and fails if a required pack is missing; it does not regenerate these artifacts.

## Editable harness policy

A benchmark may declare a constrained editable harness. Workspace preparation copies only allowlisted files; submission validation checks path confinement, size, syntax, import/base-class contract, and exact snapshot provenance.

Default harness policy:

| Benchmark | Harness |
|---|---|
| SWE | editable |
| Terminal | editable |
| Tau3 | fixed (`VTB_EDITABLE_HARNESS=off`) |
| Toolathlon | fixed (`VTB_EDITABLE_HARNESS=off`) |
| AIME | fixed/no inner harness |

Harness code never receives hidden final targets or verifier code. Changing the harness policy changes the experiment and must be reported.

## Queue, retry, and cancellation

`POST /submit` returns a job ID immediately. `submit --wait` polls until `done`, `failed`, or `cancelled`; running replies may include progress/ETA.

Infrastructure failures use the configured retry ladder. A model score is accepted only from successful evaluator output; failed attempts are not converted silently to zero. Tokens may cancel only their own jobs, and partial results are diagnostic rather than comparable scores.

## Client CLI

```bash
# On the host, select the repository environment first.
source src/commit_utils/configure_python.sh
"$VTB_PYTHON" -m eval.grading.client health
"$VTB_PYTHON" -m eval.grading.client queue --token-file /path/to/grading_token
"$VTB_PYTHON" -m eval.grading.client submit \
  --token-file /path/to/grading_token \
  --model-path final_model --limit 1 --wait
```

| Command | Purpose |
|---|---|
| `health` | Service, queue, GPU, validation, and disclosure status. |
| `register` / `unregister` | Operator run lifecycle. |
| `submit` | Queue a checkpoint. Agent role implies validation. |
| `queue` | Owned job details and aggregate other-run count. |
| `cancel` | Cancel an owned job. |
| `sample` | Read one sample subject to disclosure. |
| `baseline` | Read resource-backed baseline data; modifying baselines requires operator privileges. |

Prefer token files over shell variables. Only official tokens may request final mode/write metrics.

## HTTP API

The CLI is preferred. The underlying endpoints are:

| Endpoint | Authentication | Purpose |
|---|---|---|
| `GET /health` | none; admin reveals additional host details | Service status/policy. |
| `POST /register` | admin | Register scoped run token. |
| `POST /unregister` | run token | Invalidate token. |
| `POST /submit` | run token | Queue model/harness. |
| `GET /result` | run token + job | Owned result/progress. |
| `GET /queue` | run token | Owned queue view. |
| `POST /cancel` | run token + job | Cancel owned job. |
| `GET /sample` | run token + job + sample | Disclosed sample fields. |
| `GET /baseline` | run token | Read compatible baseline. |

Do not build research-agent behavior against server internals; use the CLI or documented JSON API.

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `VTB_GRADING_STATE` | `${VTB_ROOT}/grading_states/direct` direct; `grading_states/formal/<benchmark>` or `grading_states/smoke/<benchmark>_<id>` in launchers | Persistent service identity/state. |
| `VTB_GRADING_HOST` | `127.0.0.1` | Bind host. |
| `VTB_GRADING_PORT` | `5800` | HTTP port. |
| `VTB_GPU_COORDINATION_MODE` | `isolated` direct, `shared` launchers | Evaluation/training GPU coordination. |
| `VTB_EVAL_GPUS` | required/resolved by environment | Serving/evaluation devices. |
| `VTB_EVAL_CONCURRENT` | `16` | Concurrent benchmark trials. |
| `VTB_GRADING_EVAL_TIMEOUT` | `28800` direct | Per-attempt timeout. |
| `VTB_FINAL_LIMIT` | `-1` | Operator cap on final samples. |
| `VTB_VAL_SOURCE` | benchmark/fallback | Optional operator override. |
| `VTB_MAX_AGENT_CANCELS` | `5` | Agent cancellation quota; `0` unlimited. |
| `VTB_SERVE_RUNTIME` | `docker` | Serving runtime. |
| `VTB_SGLANG_IMAGE` | `vibetrainbench/sglang:latest` | Serving image. |
| `VTB_API_KEY` | `inspectai` | Local served-model API key. |
| `VTB_SERVE_TIMEOUT` | `1800` | Server startup timeout. |
| `VTB_SERVE_TOOL_CALL_PARSER` | `qwen3_coder` | Tool-call parser. |
| `VTB_SERVE_REASONING_PARSER` | `qwen3` | Reasoning parser. |
| `VTB_SERVE_EXTRA_ARGS`, `VTB_SGLANG_EXTRA_ARGS` | empty | Controlled server arguments. |
| `VTB_SERVE_DOCKER_SHM_SIZE` | `32g` | Serving-container shared memory. |

See [scripts/prepare/README.md](../../scripts/prepare/README.md) for storage/Docker/GPU variables.

## Failure diagnostics

| Symptom | Check |
|---|---|
| `health` unavailable | tmux session and `${VTB_GRADING_STATE}/service.log`; service may have exited before health. |
| validation-source mismatch | launcher `VAL_SOURCE`, recipe `validation`, benchmark declaration, and fresh state. |
| no held-out final IDs | `task_stats.json` has complete `ordered_ids`; do not infer order from a subset. |
| model repeatedly re-prefills | `VTB_EVAL_GPUS`/tensor parallelism and serving context/cache settings. |
| many task infrastructure failures | Docker daemon/network pool/task images and evaluator attempt logs. |
| stale behavior after edits | restart the long-running service with new state; services read policy/assets at startup. |
| missing baseline/context | resource declaration and external materialized path; do not substitute smoke data. |

## Verification

GPU-free service contract:

```bash
bash eval/grading/smoke_test.sh
```

This covers registration, path confinement, token isolation, validation policy, retry/cancellation, persistence, and shutdown. End-to-end connectivity must then use the matching bounded `scripts/run_smoke/run_smoke_<benchmark>.sh` phases; formal evaluation remains user-started.
