# Benchmarks and registration

Benchmarks are declarative registry entries plus an evaluator contract. The grading service selects a benchmark by its canonical `name`; the evaluator remains the sole scoring authority.

- Runtime paths: [scripts/prepare/README.md](../scripts/prepare/README.md)
- Operational entrypoints: [scripts/README.md](../scripts/README.md)
- Validation/final routing: [eval/grading/README.md](../eval/grading/README.md)
- Benchmark-owned services: [eval/service/README.md](../eval/service/README.md)

## Registered benchmark layout

```text
benchmarks/<safe-name>/
├── benchmark.json        registry declaration
├── evaluate.py           evaluator CLI adapter/shim
├── benchmark.txt         optional agent-facing benchmark summary
├── split.json            optional ordered validation membership
├── baseline/             optional validation diagnostics
├── validation_context/   optional benchmark-owned context pack
├── tasks/                optional in-repository task tree
└── task_context/         evaluation protocol, harness, and data notes
```

Large Harbor task exports live under `${VTB_ROOT}/hub_task_specs`, not in Git.

## Fastest Harbor Hub registration

Register any Harbor Hub `owner/name` dataset:

```bash
bash scripts/prepare/register_harbor_benchmark.sh owner/name
```

The script:

1. derives a filesystem-safe `owner__name` alias (overridable with `VTB_HUB_ALIAS`);
2. writes `benchmarks/<alias>/benchmark.json` and its evaluator shim;
3. exports task specifications under `${VTB_ROOT}/hub_task_specs/<alias>` unless `VTB_HUB_DOWNLOAD=0`;
4. leaves execution to the benchmark-specific prepare/smoke/formal workflow.

Review and commit the generated declaration only after verifying licensing, task count/order, agent choice, and evaluation semantics. Runtime task packages remain gitignored.

## `benchmark.json` schema

Minimal Harbor-backed example:

```json
{
  "name": "owner__benchmark",
  "backend": "harbor",
  "dataset": "owner/benchmark",
  "dataset_total": 100,
  "default_limit": -1,
  "agent": "codex",
  "agentic": true,
  "environment": "docker",
  "validation_source": "split:20",
  "required_env": [],
  "agent_kwargs": {},
  "notes": "Evaluation contract and provenance."
}
```

| Field | Meaning |
|---|---|
| `name` | Canonical registry ID and launcher target. |
| `backend` | `harbor` or a benchmark-owned custom/direct adapter. |
| `dataset` | Harbor Hub/source dataset identifier when applicable. |
| `tasks_subdir` | In-repository task directory when no registry dataset is used. |
| `dataset_total` | Expected canonical task count. |
| `task_ids` | Optional explicit ordered IDs; order is part of split reproducibility. |
| `default_limit` | Evaluator default (`-1` means all). |
| `agent` | Inner agent implementation; `mini-swe-agent` and `codex` are built-in shorthands. |
| `agentic` | Whether task execution uses an inner agent trajectory. |
| `environment` | Execution isolation, normally `docker`. |
| `validation_source` | Default validation routing contract. |
| `extra_args` | Evaluator/backend CLI arguments. |
| `agent_kwargs` | Request sampling, agent version/config, retry, and step controls. |
| `required_env` | Environment variables that must exist before evaluation. |
| `direct_eval` | Direct model-call settings for a no-agent evaluator. |
| `editable_harness` | Harness template/import metadata and allowed editable files. |
| `adapter` | Source checkout/path/version metadata for adapter-backed benchmarks. |
| `notes` | Human-readable scope and provenance. |

Registry validation rejects unsafe task IDs, invalid limits, malformed editable-harness declarations, and incompatible source combinations.

## Backend patterns

### Harbor-backed benchmark

Use `backend: harbor` and either `dataset` or `tasks_subdir`. Harbor owns task containers, agent execution, and verifier outcomes. Prefetch with `scripts/prepare/prefetch_benchmark_images.sh`.

### Direct evaluator

Use a benchmark-owned `evaluate.py` and `direct_eval` settings when scoring is a deterministic model-call protocol without task containers or an inner agent. AIME is the reference implementation. The evaluator must emit the shared metrics/per-sample contract and preserve hidden answers in grading state.

### Custom service adapter

Use a custom adapter when another service owns task orchestration. Toolathlon is the reference implementation: its evaluator submits a job to the service, polls it, and translates results to the grading contract. Document service deployment under `eval/service/` and declare every required environment variable.

## Validation and final routing

Declare one source:

| Source | Meaning |
|---|---|
| `split:N` | Deterministic first `N` ordered target tasks validate; ordered remainder is final only after complete canonical order is known. |
| `task:<benchmark>` | Separate benchmark validates; target benchmark is final. |
| `json:<path>` | A JSON file explicitly declares validation/final IDs. |
| `file:<path>` | Local validation IDs; a separately declared final benchmark may be used, otherwise final targets the benchmark. No remainder is inferred. |

Task order is load-bearing for `split:N` and limit prefixes. Persisted `task_stats.json` records `ordered_ids` and `val_ids`; a subset run cannot teach complete order. Without order, split final may use target evaluation/reconstruction fallback rather than a guaranteed disjoint task. Changing the declaration/order requires fresh state.

See [eval/grading/README.md](../eval/grading/README.md#validation-and-final-routing).

## Editable harness declaration

SWE and Terminal allow harness edits. A declaration identifies:

- the installed/source template;
- the import path and base class;
- the exact allowed files;
- an agent package version when the template depends on an installed loop.

A formal launcher may disable editing with `VTB_EDITABLE_HARNESS=off`; Tau3, Toolathlon, and AIME disable editing by default. Never expose verifier, hidden test, or grading-service code as editable harness material.

## Add a benchmark

1. Choose a stable canonical ID (`owner__name`).
2. Select Harbor, direct, or custom backend pattern.
3. Add `benchmark.json` and `evaluate.py`.
4. Define canonical task order/count and validation source.
5. Add `task_context/eval_protocol.md`; add harness/data docs only when applicable.
6. Add complete required environment and external-asset contracts.
7. Add matching `scripts/prepare/prepare_<benchmark>.sh` behavior.
8. Add bounded `scripts/run_smoke/run_smoke_<benchmark>.sh` validation/final verification.
9. Add `scripts/run_exp/run_exp_<benchmark>.sh` with the benchmark's hardware, budget, recipe, and harness policy.
10. Add a compatible recipe/resource set and cross-link it.
11. Verify smoke state is isolated from formal state and final IDs are never disclosed.

## Benchmark index

| Benchmark | Backend | Validation → final | Protocol and special requirements |
|---|---|---|---|
| SWE-Bench Verified | Harbor | validation split → held-out remainder | [protocol](swe-bench__swe-bench-verified/task_context/eval_protocol.md), [editable harness](swe-bench__swe-bench-verified/task_context/harness.md), task-image prefetch |
| Terminal-Bench 2.1 | Harbor | TBLite → Terminal-Bench 2.1 | [protocol](terminal-bench__terminal-bench-2-1/task_context/eval_protocol.md), [editable harness](terminal-bench__terminal-bench-2-1/task_context/harness.md), benchmark-local TBLite context pack |
| Tau3-Bench | Harbor/local patched export | explicit split → hidden test | [protocol](sierra-research__tau3-bench/task_context/eval_protocol.md), pinned Tau2 runtime image, validated 375-task export, benchmark-local split and baseline context |
| Toolathlon | custom services | ToolGym → Toolathlon Verified | [protocol](eigent-ai__toolathlon_gym/task_context/eval_protocol.md), [training-pool notes](eigent-ai__toolathlon_gym/task_context/data_pool.md), two external services and callback host |
| AIME | direct evaluator | AIME 2025 → AIME 2026 | [2025 protocol](aime__aime/task_context/eval_protocol.md), [2026 protocol](aime__26/task_context/eval_protocol.md), pinned evaluation materialization |

Benchmark protocols describe scoring and disclosure-sensitive behavior. They do not replace the central grading policy.
