# Recipe schema and experiment assembly

A recipe is the reproducible contract between a formal launcher, workspace preparation, training resources, and the grading service. Recipe names are paths relative to `training/recipes/`, without `.json`.

For the training lifecycle, see [training/README.md](../README.md). For validation/final semantics, see [eval/grading/README.md](../../eval/grading/README.md).

## Recipe schema

```json
{
  "name": "qwen3_5_9b/example",
  "description": "Human-readable purpose.",
  "compatible_benchmarks": ["owner__benchmark"],
  "compatible_model_families": ["Qwen/Qwen3.5-9B-Base"],
  "resources": [
    "scripts/slime_sft",
    "models/qwen3_5_9b_base",
    "data/example_dataset"
  ],
  "entrypoint_hint": "train_resources/scripts/slime_sft/run_train.sh",
  "validation": {
    "benchmark": "owner__benchmark",
    "val_source": "split:50",
    "description": "Validation contract for this recipe."
  },
  "env_defaults": {
    "SLIME_MODEL_CONFIG": "qwen3.5-9B.sh",
    "LOSS_MASK_TYPE": "qwen3_5",
    "NUM_GPUS": "8",
    "TP_SIZE": "4",
    "PP_SIZE": "1",
    "CP_SIZE": "1",
    "MAX_TOKENS_PER_GPU": "16384",
    "MAX_PROMPT_LEN": "16384",
    "ROLLOUT_BATCH_SIZE": "64",
    "GLOBAL_BATCH_SIZE": "64",
    "NUM_EPOCH": "1",
    "SAVE_INTERVAL": "50",
    "SAVE_OPTIM": "0",
    "LR": "1e-5",
    "MIN_LR": "1e-6",
    "LR_WARMUP_FRACTION": "0.05",
    "TRAIN_DATA": "train.jsonl"
  }
}
```

### Top-level fields

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Canonical path-like recipe identifier. Must match its path under `training/recipes/`. |
| `description` | yes | Scope, data mixture, and intended use. State whether it is formal or debug. |
| `compatible_benchmarks` | yes | Benchmark IDs accepted by workspace preparation. |
| `compatible_model_families` | yes | Allowed model ID prefixes. |
| `resources` | yes | Ordered resource IDs under `training/resources/`; must be non-empty. |
| `entrypoint_hint` | yes for trainable recipes | Intended script inside the prepared workspace. It is documentation, not automatic execution. |
| `validation` | when validation feedback is used | Benchmark and `val_source` that must match service registration. |
| `env_defaults` | yes for the shared Slime entrypoint | Uppercase scalar trainer defaults recorded in the locked manifest; must include every setting required by the selected trainer. |

Explicit values chosen inside the prepared workspace may override `env_defaults`, subject to launcher GPU visibility and benchmark policy.

## Resource declarations

Each resource lives under `training/resources/<resource-id>/`. `resource_config.json` records type, provenance, and resolution.

### Hugging Face dataset contract

Formal training data should use `type: hf_cache_dataset` with immutable metadata:

```json
{
  "id": "data/example_dataset",
  "type": "hf_cache_dataset",
  "description": "Purpose and schema of the corpus.",
  "repo_id": "organization/dataset",
  "revision": "full-immutable-commit-sha",
  "cache_path": "${VTB_ROOT}/resources/data/hf_cache/hub/datasets--organization--dataset",
  "snapshot_path": "${HF_HOME}/hub/datasets--organization--dataset/snapshots/full-immutable-commit-sha",
  "subsets": ["default"],
  "data_files": {
    "default": "data/*.parquet"
  },
  "license": "declared dataset license",
  "schema_check": "messages-compatible transformation contract",
  "row_count_check": 12345
}
```

Required resource metadata:

- exact `repo_id`;
- immutable full revision, not `main`;
- subset/config and file globs;
- license and provenance;
- expected row count and schema validation;
- deterministic transformation instructions when raw rows are not OpenAI `messages` JSONL;
- an offline loading hint using `snapshot_path`, never the Hub repo ID.

The cache is not copied into the workspace. Only the declaration is copied; agents read the shared snapshot and write derived files into job-local storage.

### Other resource types

| Type | Use |
|---|---|
| `scripts` | Trainer/runtime scripts copied into the workspace. |
| `cached_model` / `local_model` | Declared HF checkpoint and optional Megatron checkpoint. |
| `hf_dataset` | A dataset with a separately materialized payload. |
| `local` | Small committed/debug data only; use dataset resource types for large corpora. |
| `validation_baseline` | Read-only baseline outcomes matched to benchmark/model/validation source. |
| `validation_context` | Agent-facing diagnostic pack, never training data. |

## Validation coupling

The recipe, launcher, benchmark registry, and grading service must agree:

```text
launcher BENCHMARK
  = recipe compatible_benchmarks entry
  = validation.benchmark
  = grading registration target

launcher VAL_SOURCE
  = recipe validation.val_source
  = grading-state task_stats/validation routing
```

Supported sources are documented in [eval/grading/README.md](../../eval/grading/README.md#validation-and-final-routing):

- `split:N`: first deterministic `N` ordered IDs validate; the ordered remainder is final only when complete canonical order is known. Otherwise target evaluation/reconstruction fallback applies.
- `task:<benchmark>`: another registered benchmark validates; the target benchmark is final.
- `json:<path>`: a JSON file declares explicit validation/final IDs.
- `file:<path>`: a local file supplies validation IDs and may name a separate final benchmark; no remainder is inferred from the validation list.

Changing a validation source requires a new grading state. Never reuse state created with a different task order or source.

## Trainer defaults

The shared Slime entrypoint refuses to start when required settings are missing. A complete recipe specifies at least the fields below; `TRAIN_DATA` must resolve to a concrete `messages` JSONL or be deterministically produced before training:

| Group | Fields |
|---|---|
| Model/loss | `SLIME_MODEL_CONFIG`, `LOSS_MASK_TYPE` |
| Topology | `NUM_GPUS`, `TP_SIZE`, `PP_SIZE`, `CP_SIZE` |
| Sequence/batch | `MAX_TOKENS_PER_GPU`, `MAX_PROMPT_LEN`, `ROLLOUT_BATCH_SIZE`, `GLOBAL_BATCH_SIZE` |
| Schedule | `NUM_EPOCH`, `LR`, `MIN_LR`, `LR_WARMUP_FRACTION` |
| Checkpoint | `SAVE_INTERVAL`, `SAVE_OPTIM` |
| Data | `TRAIN_DATA` or recipe-specific mixture controls |

Topology products and visible GPUs must be compatible. Formal launchers default to eight GPUs; debug overrides must not be reported as paper results.

## Formal recipes

| Recipe | Role |
|---|---|
| `slime_sft_swe` | SWE default; pinned HF resources. |
| `slime_sft_terminal` | Terminal default; three pinned HF training files. |
| `slime_sft_tau` | Tau3 default; AReaL, Nemotron, and Simia pinned. |
| `slime_sft_toolathlon` | Toolathlon default; pinned HF pool. |
| `slime_sft_aime` | AIME default; three pinned sources and deterministic materialization. |

Before launching a recipe, validate all referenced model/data resources and evaluation assets against their declarations.

## Data readiness

Prepare the declared resources before running formal preflight.

### Complete pinned HF resources

| Benchmark | Logical resource | HF repository | Revision |
|---|---|---|---|
| SWE | Orchard | `microsoft/Orchard` | `70c05ec1f20f823ae6adc60374922e9271bb74e2` |
| SWE | SWE-Dev-train | `zai-org/SWE-Dev-train` | `5888e5864886590ead78a69b6e60490a0d89a594` |
| SWE | SWE-rebench OpenHands trajectories | `nebius/SWE-rebench-openhands-trajectories` | `35455389ab51bf5e2306bfd436ef72d0f98bf882` |
| SWE | Open-SWE-Traces | `nvidia/Open-SWE-Traces` | `985223d41b0fbe8b913ae13c4b833e1007c60854` |
| Toolathlon | VibeTrainBench Toolathlon pool | `guoxinzju/vibetrainbench-toolathlon` | `22acd607a1e5be5a8dcac4fd0298c6e746b4b6aa` |
| Tau3 | AReaL, Nemotron interactive-agent, Simia | `Shuaiqi/Tau_Data` | `ec268e4c203bb9e015ce759813f57fae599c88c4` |
| Terminal | RTS reward-1, Terminal trajectories, TMAX RTS-style Qwen3.5 | `prt66/terminal_data` | `dd833a625e032842b2d84bd2b993b7c40e301f3c` |
| AIME | OpenThoughts3-1.2M | `open-thoughts/OpenThoughts3-1.2M` | `61bcf9d4eb38b30295efc2021227a63cc5bb34c8` |
| AIME | Nemotron-SFT-Math-v4 | `nvidia/Nemotron-SFT-Math-v4` | `84d42ad0cb960f07f951b9baa9ed2b46a5a18c66` |
| AIME | OpenR1-Math-220k default | `open-r1/OpenR1-Math-220k` | `e4e141ec9dea9f8326f4d347be56105859b2bd68` |

AIME preparation filters/converts the three pinned sources and concatenates them in the table order into `${VTB_ROOT}/resources/data/aime_sft/train.jsonl`; its manifest records 1,229,249 rows and the output SHA-256. Terminal's pinned upstream repository does not declare a license. Tau preparation downloads only the three declared training JSONL files, never the evaluation split.

### Evaluation assets—not training data

| Asset | Purpose | Location or preparation |
|---|---|---|
| TBLite context pack | Terminal validation diagnostics | benchmark-local `validation_context/` payload |
| Tau3 split JSON | explicit validation/final membership | benchmark-local split JSON |
| Baseline trajectory/context packs | agent-visible validation diagnostics | benchmark-local `baseline/` payloads, matched through short-name declarations |
| AIME 2025/2026 problems | direct validation/final evaluation | declared revisions/materialization via `prepare_aime.sh`; AIME 2026 `/rows` revision is not independently enforced |
| Harbor task exports | benchmark execution metadata | prepared by benchmark scripts |

These assets must never be added to a training mixture.

## Assemble a custom experiment

1. **Choose a benchmark and model.** Confirm the benchmark ID in `benchmarks/registry.py`/`benchmark.json` and model family used by the launcher.
2. **Declare HF resources.** Add pinned `resource_config.json` files with complete resource metadata.
3. **Define deterministic materialization.** Document how snapshot files become `messages` rows; keep derived output job-local.
4. **Create the recipe.** Add compatibility, resources, trainer defaults, and validation contract.
5. **Validate grading alignment.** Ensure recipe `val_source` equals the service source and benchmark ordering is stable.
6. **Prepare runtime assets.** Use the benchmark's `prepare_<benchmark>.sh` and populate HF snapshots.
7. **Run bounded smoke.** Verify one validation and one final path with fresh smoke state.
8. **Start formal phases.** Start `service`, check `health`, then start `run` with a new formal state.
9. **Audit outputs.** Preserve the locked resource manifest, prompt, trace, logs, final model, and metrics.

If a new benchmark or recipe changes the editable surface, document it in [benchmarks/README.md](../../benchmarks/README.md) and [eval/grading/README.md](../../eval/grading/README.md).
