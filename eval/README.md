# Evaluation subsystem

Evaluation is isolated from research. A research agent may submit checkpoints and receive policy-filtered validation feedback; only the official role can score the hidden final set and write official metrics.

## Components

| Directory | Responsibility | Documentation |
|---|---|---|
| `grading/` | Token-gated service, validation/final policy, disclosure, queue, archive, and official result persistence. | [grading/README.md](grading/README.md) |
| `runner/` | Serve a checkpoint, resolve benchmark backend, execute evaluation, and normalize metrics. | Runtime paths are in [scripts/prepare/README.md](../scripts/prepare/README.md). |
| `agent_task/` | Outer-agent prompts and workspace-facing rules. | See prompt routing below. |
| `service/` | Benchmark-owned services and HTTP adapters, including Toolathlon Gym/Verified. | [service/README.md](service/README.md) |
| `templates/` | Shared model chat templates used by serving/evaluation. | Selected by runner/model-server configuration. |

Benchmark declarations and evaluator shims live under [benchmarks/](../benchmarks/README.md).

## End-to-end flow

1. A benchmark-specific launcher starts the grading service with fresh state.
2. `src/run_task.sh` prepares the recipe/resources and registers agent/official scopes.
3. The outer agent trains in its isolated workspace.
4. Agent-role submissions enter the validation queue.
5. The service starts SGLang for the submitted checkpoint.
6. `eval.runner.run` invokes the selected benchmark's `evaluate.py` through Harbor, a direct evaluator, or a custom service adapter.
7. The service filters validation feedback according to disclosure policy.
8. The archived final delivery is submitted with the official role.
9. Official final metrics and audit artifacts are persisted outside the research workspace.

See [grading/README.md](grading/README.md) for role and routing details.

## Prompt routing

`VTB_PROMPT` selects a Markdown prompt under `eval/agent_task/`; `get_prompt.py` resolves it and injects benchmark/model/budget context.

| Prompt class | Intended use |
|---|---|
| `workflow` / research prompts | Formal research/training workflow. |
| `mock_test` | Generic no-training connectivity; must remain benchmark-neutral. |
| benchmark-specific smoke prompts | Minimal model materialization/submission behavior for bounded smoke. |
| workspace docs | Stable grading/agent rules copied into the job. |

Benchmark-specific instructions must be gated by the exact benchmark identity and kept out of shared prompts. Prompts never include hidden final IDs, answers, verifier secrets, or official tokens.

## Backends

| Backend pattern | Execution |
|---|---|
| Harbor | Harbor runs an inner agent in task containers and returns verifier outcomes. |
| Direct evaluator | Evaluator calls the served checkpoint directly; AIME is the reference. |
| Custom service | Evaluator submits to a benchmark-owned service; Toolathlon is the reference. |

All patterns must emit the shared metrics/per-sample contract so validation disclosure and final archival remain consistent.

## Environment

Training and evaluation resolve environment from separate sourceable files:

| Side | Environment file |
|---|---|
| Research/training | `src/commit_utils/set_env_vars.sh` |
| Grading/evaluation | `scripts/prepare/configure_evaluation_environment.sh` |

Launchers load them automatically. The authoritative variable/path reference is [scripts/prepare/README.md](../scripts/prepare/README.md).

## Templates

Jinja templates under `templates/` provide model-family chat formatting when a base model has no compatible template. The model server/evaluator selects templates; benchmark task code should not rewrite them implicitly. A template change can alter tool calls and scores and must be treated as an evaluation-contract change.

## Extend evaluation

- Add/modify benchmark declarations through [benchmarks/README.md](../benchmarks/README.md).
- Add grading behavior only when it is benchmark-independent and preserves role isolation.
- Put benchmark-specific external service deployment under `eval/service/<name>/`.
- Add prompt content under `agent_task/` only when it belongs in the research workspace.
- Keep final targets and official credentials server-side.

## Verification

```bash
# GPU-free service contract
bash eval/grading/smoke_test.sh

# Then run the selected benchmark's bounded smoke phases.
bash scripts/run_smoke/run_smoke_swe.sh preflight
```

Do not start service/GPU/API-consuming phases automatically; follow [scripts/README.md](../scripts/README.md).
