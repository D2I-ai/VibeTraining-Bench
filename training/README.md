# Training subsystem

Training recipes define the initial model, data resources, trainer defaults, and validation contract presented to a research agent. They are declarative inputs: workspace preparation copies or references resources, but does not start training.

- Recipe schema and custom experiment assembly: [recipes/README.md](recipes/README.md)
- Runtime variables and paths: [scripts/prepare/README.md](../scripts/prepare/README.md)
- Validation/final routing: [eval/grading/README.md](../eval/grading/README.md)
- Formal launchers: [scripts/README.md](../scripts/README.md)

## Layout

```text
training/
├── recipes/                         experiment declarations
│   └── qwen3_5_9b/
├── resources/
│   ├── scripts/                     trainer entrypoints copied into workspaces
│   ├── models/                      model resource declarations/checkpoints
│   ├── data/                        HF dataset and local data declarations
│   └── validation/                  baseline and context declarations
└── README.md
```

Large payloads do not belong in Git. Model and dataset snapshots live under `VTB_ROOT`; resource declarations record immutable provenance and resolved locations.

## Workspace lifecycle

A formal launcher sets `VTB_TRAIN_RECIPE`, then calls `src/run_task.sh`:

1. resolve the recipe under `training/recipes/`;
2. verify benchmark/model compatibility;
3. resolve every resource under `training/resources/`;
4. validate required model/data/context paths;
5. create the isolated task workspace;
6. write `train_resources/`, `train_resource_manifest.lock.json`, and `environment.md`;
7. start the selected research-agent adapter;
8. collect `task/final_model/` for validation/final grading.

The research agent can modify only the surfaces allowed by the benchmark and launcher. Workspace preparation never executes the recipe's trainer automatically.

## Selecting a recipe

Use the benchmark-specific formal launcher:

```bash
RECIPE=qwen3_5_9b/slime_sft_swe \
RUN_ID=swe-example \
AGENT_CLI=claude RESEARCHER="<your-research-model>" \
  bash scripts/run_exp/run_exp_swe.sh run
```

Start the matching grading service and check `health` first. `RECIPE` overrides the launcher's default recipe for controlled experiments; paper comparisons use the documented default.

`VTB_TRAIN_RESOURCES` is a lower-level alternative for explicit comma-separated resource IDs. It is mutually exclusive with `VTB_TRAIN_RECIPE` and bypasses recipe compatibility, validation, and `env_defaults`; prefer a recipe for formal experiments.

## Default recipes

| Benchmark | Default recipe | Training data |
|---|---|---|
| SWE-Bench Verified | `slime_sft_swe` | Four pinned HF datasets. |
| Terminal-Bench 2.1 | `slime_sft_terminal` | Three files pinned from `prt66/terminal_data`. |
| Tau3-Bench | `slime_sft_tau` | All three training corpora pinned; JSON split downloaded separately from training files. |
| Toolathlon | `slime_sft_toolathlon` | Pinned HF pool. |
| AIME | `slime_sft_aime` | Three pinned mathematics sources materialized into one 1,229,249-row JSONL. |

Dataset revisions, preparation requirements, and resource metadata are documented in [recipes/README.md#data-readiness](recipes/README.md#data-readiness).

## Resource behavior

| Resource type | Workspace behavior |
|---|---|
| `scripts` | Copied under `task/train_resources/scripts/`. |
| `cached_model` / `local_model` | Validate the declared HF checkpoint (and optional `torch_dist`) and copy it into the workspace. |
| `hf_cache_dataset` | Copy only the declaration; expose the shared pinned snapshot read-only. |
| `hf_dataset` | Validate a separately materialized payload and copy it. |
| `local` | Copy committed/debug data; use dataset resource types for large training corpora. |
| `validation_baseline` | Resolve a benchmark/model/validation-source baseline and mount it read-only. |
| `validation_context` | Validate an external context pack and copy it to its declared workspace path. |

Training data, validation baselines, validation context, and evaluation questions are distinct resource classes. Evaluation data must never be mixed into the training corpus.

## Offline Hugging Face contract

Research containers run offline by default (`VTB_AGENT_HF_OFFLINE=1`). Before launch:

- download every model/dataset at the revision declared by the recipe;
- keep the shared cache under `HF_HOME`;
- load datasets from the declaration's expanded `snapshot_path` and `data_files` globs;
- never infer dataset structure from content-addressed `blobs/` names;
- write derived JSONL/Arrow files only into per-run workspace/scratch;
- record repo ID, immutable revision, subsets/files, license, schema, and row-count checks.

See [containers/README.md](../containers/README.md) for cache mounting and [scripts/prepare/README.md](../scripts/prepare/README.md) for path resolution.

## Shared Slime SFT entrypoint

`resources/scripts/slime_sft/run_train.sh` provides the common Qwen3.5 Slime/Megatron workflow:

1. apply recipe `env_defaults`;
2. resolve selected training JSONL and model resource;
3. convert the HF checkpoint to Megatron `torch_dist` when needed;
4. run SFT and record framework state/logs;
5. verify and convert the final checkpoint back to Hugging Face format;
6. write `task/final_model/`.

Training rows must contain a `messages` field. Important settings include:

| Group | Variables |
|---|---|
| Model/loss | `SLIME_MODEL_CONFIG`, `LOSS_MASK_TYPE` |
| Parallelism | `NUM_GPUS`, `TP_SIZE`, `PP_SIZE`, `CP_SIZE` |
| Batching | `MAX_TOKENS_PER_GPU`, `ROLLOUT_BATCH_SIZE`, `GLOBAL_BATCH_SIZE` |
| Schedule | `NUM_EPOCH`, `LR`, `MIN_LR`, `LR_WARMUP_FRACTION` |
| Checkpoints | `SAVE_INTERVAL`, `SAVE_OPTIM`, `SAVE_DIR`, `OUTPUT_DIR` |
| Data | `TRAIN_DATA`, `MAX_PROMPT_LEN` |

Avoid `SAVE_INTERVAL=1` in formal runs unless every intermediate checkpoint is intentionally retained; optimizer and `torch_dist` state can consume terabytes.

## Outputs

Default persistent outputs live under `${VTB_ROOT}/results/`:

| Artifact | Meaning |
|---|---|
| `prompt.md` | Exact outer-agent instructions. |
| `solve_out.txt` | Timestamped outer-agent trace. |
| `task/train_resource_manifest.lock.json` | Resolved recipe, resources, and defaults. |
| `task/logs/` | Training and framework-state logs. |
| `task/final_model/` | Checkpoint submitted for grading. |
| `final_model/` | Archived official submission. |
| `metrics.json` | Official final score after grading. |

Intermediate checkpoints and task trial trees belong in scratch and may be removed only after the persistent result has been audited.

## Adding an experiment

1. Add or reuse resource declarations.
2. Add a recipe under `training/recipes/`.
3. Pin all HF resources and document licenses/provenance.
4. Match recipe benchmark/model compatibility to the intended launcher.
5. Match recipe validation to the grading service's `val_source`.
6. Run workspace/resource validation with a debug-sized setup.
7. Use the matching prepare, smoke, and formal entrypoints.
8. Preserve manifests, logs, final model, and metrics.

The complete field-level procedure is in [recipes/README.md](recipes/README.md).
