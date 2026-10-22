# Runtime environment and paths

This is the authoritative reference for host/runtime variables and generated paths. For commands, see [scripts/README.md](../README.md); for container storage, see [containers/README.md](../../containers/README.md).

## Environment boundaries

Two sourceable files configure different sides of a run:

| File | Loaded by | Responsibility |
|---|---|---|
| `src/commit_utils/set_env_vars.sh` | `src/run_task.sh` | Research-agent container, trainer image, prompt, job home, HF cache, results, and scratch. |
| `scripts/prepare/configure_evaluation_environment.sh` | smoke/formal service launchers | Docker daemon, local Harbor tools, grading/model-serving environment, evaluation GPUs, and caches. |

Benchmark launchers source them automatically. Source the evaluation file manually only when deliberately starting `"$VTB_PYTHON" -m eval.grading.server` without a launcher.

### Python runtime

`.python-version` selects Python 3.12, matching `pyproject.toml`, isolated Harbor, and the checked trainer/SGLang image minor. `install_backend_tools.sh` bootstraps uv and `${VTB_ROOT}/tools/runtime/` without running project code under the system interpreter. Host helpers source `src/commit_utils/configure_python.sh`: explicit `VTB_PYTHON` wins, then checkout `.venv`, then the managed runtime, then an existing compatible Harbor environment. Missing/wrong versions fail before launch. No Python alias or user-site packages are used; inherited `PYTHONHOME`/`PYTHONPATH` are cleared. Python subprocesses use `sys.executable`; container commands use image-local `python3` and do not receive the host interpreter path.

```bash
source src/commit_utils/configure_python.sh
"$VTB_PYTHON" scripts/prepare/python/verify_python_environment.py
```

Toolathlon service startup verifies its exact `requirements.txt`; it never silently installs unpinned packages with `--user`. Install dependencies into a dedicated 3.12 environment using `uv pip install --python "$VTB_PYTHON" -r eval/service/toolathlon_gym/requirements.txt` (or the Verified manifest). Its MCP/task environments stay controlled by their upstream uv projects and pinned task images.

Direct-service variables use normal “explicit value, then default” resolution. Formal launchers first resolve their own input names and then export downstream `VTB_*` values into tmux; therefore an inherited downstream variable may be overwritten by a launcher default. Use the launcher inputs documented below rather than assuming every ambient `VTB_*` value wins.

## Harbor CLI and source checkout

`containers/requirements.txt` is the single authoritative Harbor pin. `install_backend_tools.sh` installs that exact git commit, verifies the installed distribution's `direct_url.json`, and prepares a clean source checkout at the same commit for adapters/editable harnesses:

```bash
# Optional: choose a non-default source location before installation.
export VTB_HARBOR_REPO=/absolute/path/to/harbor
bash scripts/prepare/install_backend_tools.sh
```

Without the override, source defaults to sibling `../harbor`. Existing dirty checkouts, a different origin, or a different commit fail rather than being overwritten. Expected result: `HARBOR_INSTALL_OK` and `HARBOR_SOURCE_OK` report the same 40-character commit; `${VTB_HARBOR_REPO}/adapters/` exists. Benchmark preparation rechecks installed provenance, and SWE/Tau3 also recheck source provenance.

## Runtime root

```text
VTB_REPO=<checkout>
VTB_ROOT=${VTB_REPO}/vtb-env
```

`vtb-env/` is gitignored runtime state. It can be populated from a clean clone using the preparation scripts and declared resources. To place it on a larger disk, either set `VTB_ROOT` consistently for every command or keep the default path as a symlink to the large disk. A stable symlink minimizes accidental split state between tools.

Print Python-side resolution with:

```bash
source src/commit_utils/configure_python.sh
"$VTB_PYTHON" -m eval.runner.runtime_paths
```

## Path map

| Variable/path | Default | Contents and retention |
|---|---|---|
| `VTB_PYTHON` | selected 3.12 virtual environment | Host interpreter, frozen across detached phases; never forwarded into the researcher container. |
| `VTB_RESULTS_DIR` | `${VTB_ROOT}/results` | Final model, metrics, prompts, manifests, and audit traces. Preserve. |
| `VTB_SCRATCH` | `${VTB_ROOT}/scratch` | Workspaces, intermediate checkpoints, and trial trees. Large and disposable after audit. |
| `VTB_GRADING_STATE` | `${VTB_ROOT}/grading_states/formal/<benchmark>` or `grading_states/smoke/<benchmark>_<SMOKE_ID>` | Tokens, queue, validation split state, timing, logs, and archives. Direct-service default: `grading_states/direct`. Never share between smoke/formal runs. |
| `HF_HOME` | `${VTB_ROOT}/resources/data/hf_cache` | Pinned model and dataset snapshots used offline. Shared read-only lower layer. |
| `XDG_CACHE_HOME` | `${VTB_ROOT}/cache` | Harbor/registry metadata cache. |
| `VTB_HUB_TASK_SPECS` | `${VTB_ROOT}/hub_task_specs` | Exported Harbor task packages; Tau3 consumes its validated export directly. |
| `VTB_PACKAGES` | `${VTB_ROOT}/packages` | Pinned reference packages used for editable harnesses. |
| `${VTB_ROOT}/tools` | fixed under `VTB_ROOT` | `uv`, Python 3.12, Harbor, and their isolated data/cache. |
| `VTB_MODEL_RESOURCE_ROOT` | `${VTB_ROOT}/resources/models/qwen3_5_9b_base` | Normalized HF checkpoint and matching Megatron conversion copied into workspaces. Can select a fresh directory without replacing existing resources. |
| `${VTB_ROOT}/logs` | fixed under `VTB_ROOT` | Preparation/prefetch logs. |

Docker images do not live under `VTB_ROOT`; they live in the selected daemon's `data-root`.

## Base-model resources

Smoke and formal training consume different materializations of the same base model:

| Resource | Expected location | Consumer |
|---|---|---|
| Shared Hub snapshot | `${HF_HOME}/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/<revision>` | no-training smoke and offline snapshot resolution |
| Recipe HF checkpoint | `${VTB_MODEL_RESOURCE_ROOT}/hf_checkpoint` | workspace model resource and HF→Megatron conversion inputs |
| Megatron checkpoint | `${VTB_MODEL_RESOURCE_ROOT}/torch_dist` | recipe workspace preparation/training |

The model declaration pins revision `68c46c4b3498877f3ef123c856ecfde50c39f404`. Download preparation respects `HF_HOME`, resolves that exact snapshot, copies it to the declared HF checkpoint, and normalizes Qwen chat EOS to `<|im_end|>` (ID `248046`) without modifying the shared snapshot.

```bash
# Explicit download/materialization; no GPU conversion is started.
bash scripts/prepare/download_base_model.sh download

# Separately user-started GPU conversion, never called by datasets/all/preflight.
bash scripts/prepare/download_base_model.sh convert

# Local checks only: weights, tokenizer, revision, and conversion provenance.
bash scripts/prepare/download_base_model.sh preflight
```

All five formal recipes require the same model resource. Both the HF checkpoint and its Megatron conversion carry matching `vtb_source.json` provenance. Existing resources with unknown/different provenance are refused before a download; use a fresh `VTB_MODEL_RESOURCE_ROOT` rather than overwriting existing checkpoints. Keep that override exported for preparation, formal preflight, and every launcher phase; launchers explicitly forward it into tmux. Never assign provenance to an unverified old `torch_dist` checkpoint.

```bash
# Operator-started recovery from a complete pinned cache; no network or GPU use.
export VTB_ROOT="${VTB_ROOT:-$PWD/vtb-env}"
export VTB_MODEL_RESOURCE_ROOT="$VTB_ROOT/resources/models/qwen3_5_9b_base_prepared_v2"
bash scripts/prepare/download_base_model.sh download --offline

# Separately operator-started GPU conversion of this newly prepared checkpoint.
bash scripts/prepare/download_base_model.sh convert
bash scripts/prepare/download_base_model.sh preflight
```

Offline materialization prints `BASE_MODEL_DOWNLOAD_OK`; formal model preflight prints `MODEL_RESOURCE_OK` only after conversion. It validates chat template and EOS consistency as well as provenance. Smoke uses `snapshot-preflight` instead: the exact pinned snapshot, tokenizer/config, and indexed weights must exist, even if other cached revisions are complete.

Training downloads are manifest-driven: `prepare_<benchmark>.sh datasets` selects the recipe's immutable revisions and only its `data_files` patterns. Runtime `preflight` supports smoke; `preflight --formal` additionally validates training/model/baseline resources. `all` includes that formal check but never launches model conversion. AIME `datasets` prepares evaluation data and the base model; `training-data` downloads three pinned mathematics sources and runs the CPU-only conversion pipeline under `scripts/prepare/python/aime_sft_pipeline/`. Terminal downloads the three JSONL files declared by its pinned HF resource.

AIME `training-data --offline` skips downloads, checks all cached raw inputs, and runs its converter with Docker networking disabled. AIME checks require all 120/12/10 declared Parquet shards, matching preparation-code digests, and a streamed SHA-256/row-count/messages-schema validation of the output. These checks can take time on the full corpus; they do not load it all into memory.

```bash
# Use only after the three pinned training sources have been downloaded.
bash scripts/prepare/prepare_aime.sh training-data --offline
bash scripts/prepare/prepare_aime.sh preflight --formal
```

## Docker selection

Resolution order:

1. `VTB_DOCKER_HOST`;
2. `VTB_DOCKER_ROOTLESS=1`;
3. inherited `DOCKER_HOST`;
4. `unix:///var/run/docker.sock`.

Rootful Docker is the default. SWE images may contain UIDs outside a rootless subuid range, and rootless/cgroup combinations may not expose GPUs or enforce task limits correctly.

| Variable | Purpose |
|---|---|
| `VTB_DOCKER_HOST` | Explicit daemon socket used by VibeTraining-Bench. |
| `VTB_DOCKER_ROOTLESS` | Opt in to the per-user rootless socket. Diagnostic only for unsupported tasks. |
| `VTB_TRAINER_IMAGE` | Override the trainer image reference. |
| `VTB_SGLANG_IMAGE` | Override the serving image reference. |
| `VTB_IMAGE_NAMESPACE`, `VTB_IMAGE_TAG` | Change names produced by container builds. |

## Launcher inputs and downstream variables

Formal launchers resolve these inputs and then export downstream variables into tmux:

| Launcher input | Downstream value | Meaning |
|---|---|---|
| `AGENT_GPUS` | `VTB_AGENT_GPUS` | GPUs visible to research/training. |
| `EVAL_GPUS` | `VTB_EVAL_GPUS` | Serving/evaluation GPUs; count determines tensor parallelism. |
| `GRADING_PORT` | `VTB_GRADING_PORT` | Grading API port. |
| `FINAL_LIMIT` | `VTB_FINAL_LIMIT` | Operator final-sample cap. |
| `DISCLOSURE` | `VTB_AGENT_DISCLOSURE` | Agent validation-feedback fields. |
| `RECIPE` | `VTB_TRAIN_RECIPE` | Recipe under `training/recipes/`. |
| `VAL_SOURCE` | `VTB_VAL_SOURCE` | Validation routing override. |
| `SERVE_EXTRA_ARGS` / `SGLANG_EXTRA_ARGS` | corresponding `VTB_*` values | Controlled serving arguments. |

Set the launcher-side name when using `run_exp_<benchmark>.sh`. Formal launchers force shared GPU coordination and benchmark harness policy; direct-service variables do not override those policies accidentally.

Direct-service/runtime variables:

| Variable | Meaning |
|---|---|
| `VTB_GPU_COORDINATION_MODE` | `shared` for coordinated one-host launchers; `isolated` for operator-managed disjoint GPUs. |
| `VTB_EVAL_CONCURRENT` | Parallel evaluation trials; default `16`. |
| `VTB_GRADING_HOST` | Grading bind/connect host. |
| `VTB_MODEL_PORT` | Local served-model port; default `8110`. |
| `VTB_SERVE_TIMEOUT` | Submitted-checkpoint startup deadline. |
| `VTB_TRAIN_RESOURCES` | Lower-level resources; mutually exclusive with a recipe. |
| `VTB_PROMPT` | Prompt under `eval/agent_task/`. |
| `VTB_RESULTS_DIR` | Persistent output root. |

Keep agent/evaluation GPU sets explicit when workloads share a host. A fresh service also requires a free port and nonexistent grading-state path.

Recipe defaults are documented in [training/recipes/README.md](../../training/recipes/README.md). Disclosure and validation routing are documented in [eval/grading/README.md](../../eval/grading/README.md).

## Benchmark-specific assets

| Benchmark | Runtime requirement |
|---|---|
| SWE | Harbor task export and task images under the shared cache/Docker store. |
| Terminal | TBLite context at `${VTB_REPO}/benchmarks/terminal-bench__terminal-bench-2-1/validation_context`. |
| Tau3 | Validated task export plus split JSON at `${VTB_REPO}/benchmarks/sierra-research__tau3-bench/tau3_val_test_split.json`; all three training corpora are downloaded without the split file. |
| Toolathlon | `VTB_TOOLATHLON_URL`, `VTB_TOOLATHLON_VERIFIED_URL`, and `VTB_MODEL_ADVERTISE_HOST`. |
| AIME | Pinned AIME 2025/2026 evaluation data under `${VTB_ROOT}/hub_task_specs`. |

These are evaluation/runtime assets, not training corpora. Baseline payloads live in `benchmarks/<id>/baseline/`; `training/resources/validation/` contains declarations only. Mount lookup matches benchmark, base model, and expanded validation source, so short logical IDs do not depend on host paths. Baselines are mounted read-only at `$HOME/grading_baseline`; Terminal context is copied to `$HOME/task/tblite_validation/`.

## Credentials and secrets

Copy `api_key.env.example` to `api_key.env` once; do not overwrite an existing file. The credential file is gitignored. `src/run_task.sh` forwards only variable names listed in the selected adapter's `adapter_config.json`; grading/admin/official authority is rejected. `VTB_LOCAL_ENV_FILE` optionally selects trusted executable local settings and `VTB_AGENT_ADAPTERS_DIR` opts into an ignored adapter root. Neither local configuration nor the host credential file is bound into the researcher workspace.

Rules:

- never commit provider keys, HF tokens, service tokens, or Toolathlon secrets;
- do not store an HF token inside the shared cache;
- do not export secrets globally when a scoped gitignored file is sufficient;
- grading admin/official tokens are generated inside fresh grading state and are role-separated;
- configure provider gateways and URLs through the selected adapter's supported environment variables.

See [agents/README.md](../../agents/README.md) for the provider-adapter contract.

## Compatibility aliases

`VTB_SWE_API_BASE` and `VTB_SWE_SERVE_TIMEOUT` are translated to the generic `VTB_API_BASE` and `VTB_SERVE_TIMEOUT` when the generic names are unset. They exist for compatibility and must not be used in new documentation or integrations.

`VTB_HARBOR_DATASET_DIR` is a legacy alias for `VTB_HUB_TASK_SPECS`; use the latter.

## State safety

Default state directories live under `${VTB_ROOT}/grading_states/`. Set `VTB_GRADING_STATE` consistently across every phase, including verification and cleanup. Do not move live state: paths recorded in registrations and archives may be absolute.

- Use a new `VTB_GRADING_STATE` for every smoke and formal service.
- Do not edit `task_stats.json` or token files after service startup.
- Do not point a formal service at smoke state.
- An aborted run may leave tmux sessions, containers, FUSE mounts, or registrations; inspect ownership before cleanup.
- Keep persistent results separate from scratch so intermediate checkpoints can be audited and then removed safely.
