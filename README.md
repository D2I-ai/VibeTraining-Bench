<div align="center">

# VibeTraining-Bench

### Data–Trainer–Harness Co-Design for Iterative System Improvement

Ruotian Peng, Yuanyi Ren, Shuaiqi Liu, Xin Guo, Fang Zhu, Yangyi Huang, Yue Wu, Zheng Wang, Yandong Wen, Zhihang Fu<br>
Zhejiang University · Westlake University · Alibaba Token Hub · South China University of Technology · The Chinese University of Hong Kong

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2610.26342-blue)](https://arxiv.org/abs/2610.26342)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub.io-purple)](https://d2i-ai.github.io/VibeTraining-Bench/index.html)
[![Survey](https://img.shields.io/badge/Survey-Preprints-green)](https://www.preprints.org/frontend/manuscript/3c41418aa2e782b90ee5da995565ecac/download_pub)
[![Code](https://img.shields.io/badge/Code-Repository-brightgreen)](#quickstart)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-5-blueviolet)](benchmarks/README.md)

**English** | [中文版](README.zh-CN.md)

A benchmark for evaluating whether research agents can jointly improve **training data**, the **trainer**, and the **inference harness**—the cross-component co-design capability at the heart of recursive self-improvement.

</div>

---

## Overview

Recursive self-improvement requires agents to diagnose system bottlenecks and coordinate improvements across training and inference. VibeTraining-Bench evaluates this capability on software engineering, terminal operation, multi-turn tool use, customer-service agents, and mathematical reasoning.

Each research agent starts from the same base model and resource budget. It may use validation feedback to refine the allowed components, then submits one final checkpoint. A separate grading service owns validation and final evaluation; hidden final data never enters the research workspace.

![VibeTraining-Bench overview](assets/images/main_figure.png)

*Research agents iteratively co-design data, trainer, and harness under a fixed budget, then submit the final system for isolated grading.*

## Relation to Our Prior Survey

VibeTraining-Bench builds on the unified research framework introduced in our prior survey, [*The Path to Recursive Self-Improving Agents: Foundation, Framework, and Future Directions*](https://www.preprints.org/manuscript/202608.0051) (Liu et al., 2026). This framework models the foundation model, agent harness, agent data system, agent trainer, and improvement mechanism as a coupled evolving system to characterize the scope, dependencies, and dynamics of self-improvement. Within this framework, VibeTraining-Bench provides a controlled evaluation of cross-component co-design, assessing whether research agents can jointly improve training data, the trainer, and the inference harness under a fixed resource budget.

## Highlights

- **Cross-component evaluation:** measure coordinated data, trainer, and harness improvement rather than isolated tuning.
- **Controlled comparison:** every research agent receives the same initial system, budget, and benchmark contract.
- **Isolated grading:** research and grading run separately, with final data and official credentials withheld from the agent.
- **Five task domains:** SWE-Bench Verified, Terminal-Bench 2.1, Tau3-Bench, Toolathlon, and AIME.
- **Reproducible operations:** each benchmark has matching `prepare`, bounded `run_smoke`, and formal `run_exp` entrypoints.

## API setup and model selection

There are several independent model-call roles. Configure each role's model ID, credentials, and endpoint for your deployment. Launchers default to `AGENT_CLI=claude` and require an explicit `RESEARCHER` for preflight/run. Custom adapters and deployment settings can be loaded through explicitly selected, gitignored local configuration.

| Role | Model selection | API requirement |
|---|---|---|
| Outer research agent | Smoke/formal launcher `AGENT_CLI` + `RESEARCHER` | Selected adapter's provider key; needed even for no-training smoke. |
| Model being trained/evaluated | Formal launcher `MODEL` + compatible `RECIPE` | Local checkpoint served by grading through an OpenAI-compatible endpoint; no external model-provider key for SWE, Terminal, or AIME inference. |
| Tau3 simulated user | `TAU2_USER_MODEL`, `TAU2_USER_BASE_URL`, `TAU2_USER_API_KEY` | Separate evaluator-side model API; independent of `RESEARCHER`. |
| Tau3 natural-language assertions | `TAU2_NL_ASSERTIONS_MODEL`, `TAU2_NL_ASSERTIONS_BASE_URL` | Evaluator-side model API configured with the Tau runtime's authentication settings. |
| Toolathlon | `VTB_TOOLATHLON_URL`, `VTB_TOOLATHLON_VERIFIED_URL`, `VTB_MODEL_ADVERTISE_HOST` | Two separately deployed [task services](eval/service/README.md) and a reachable checkpoint endpoint; task/tool credentials belong to those services. |
| Grading HTTP API | Launcher-selected host/port | Generated agent/admin/official tokens, not provider API keys. |

### Provider credentials and endpoint contracts

```bash
# Create once; edit the gitignored file without committing real credentials.
test -f api_key.env || cp api_key.env.example api_key.env
# Set only the key required by your selected adapter.
```

| `AGENT_CLI` | Credential in `api_key.env` | Model ID and interface |
|---|---|---|
| `claude` | `ANTHROPIC_API_KEY` | `RESEARCHER` is passed to Claude CLI `--model`; optional `ANTHROPIC_BASE_URL` selects an Anthropic-compatible messages endpoint. |
| `codex` | `OPENAI_API_KEY` or `CODEX_API_KEY` | Mapped to `CODEX_API_KEY`; `RESEARCHER` is passed to Codex CLI `--model`. Optional `OPENAI_BASE_URL` must support that CLI's API contract. |
| `claude_openrouter` | `OPENROUTER_API_KEY` | `RESEARCHER` selects a provider-supported model; `VTB_OPENROUTER_BASE_URL` is the adapter's endpoint override. |

Matching the protocol matters: an OpenAI-compatible URL cannot simply replace an Anthropic-compatible URL. Only environment variables declared in the selected adapter's `adapter_config.json` are forwarded into the research container. Configure endpoints through the adapter's supported settings; see [adapter contracts](agents/README.md) before introducing another provider or relay.

`api_key.env` is optional and sourced when present; credentials may instead be exported or supplied through a trusted `VTB_LOCAL_ENV_FILE`. Credentials are forwarded only for the selected adapter, by environment-variable name rather than credential-valued command arguments. Never put HF tokens in the shared cache or commit service secrets. Tau3 additionally loads its benchmark-local gitignored `.env`; configure its two evaluator model roles separately, including `TAU2_USER_REASONING_EFFORT` (an explicitly empty value is valid). Toolathlon task secrets belong in the separate service deployment described in [service setup](eval/service/README.md).

### Switching researcher and base models

A label such as `researcher1` is a run name, not a separate configuration variable. Each run has one outer researcher; select it with `RESEARCHER`. To compare models, launch separate runs with distinct `RUN_ID`/tmux sessions and fresh grading state, using the same benchmark/base model/recipe/budget. For example, after starting and health-checking the corresponding formal service:

```bash
AGENT_CLI=claude RESEARCHER="<research-model-A>" RUN_ID=researcher1 \
  bash scripts/run_exp/run_exp_swe.sh run

# Alternative for a separate run/service; do not overlap the same run session.
AGENT_CLI=codex RESEARCHER="<research-model-B>" RUN_ID=researcher2 \
  bash scripts/run_exp/run_exp_swe.sh run
```

Subagent model routing belongs to the selected adapter. The `claude` adapter can forward `CLAUDE_CODE_SUBAGENT_MODEL` and compatible fast-model settings; custom adapters must declare their own controls. Changing `MODEL` changes the checkpoint family, not the researcher API. It requires compatible resources, recipe validation, and serving support; the included recipes target `Qwen/Qwen3.5-9B-Base`. The lower-level invocation is `bash src/run_task.sh <benchmark> <adapter> <base-model> <run-id> <hours> <researcher-model> <gpu-count>`; the sixth argument becomes `AGENT_CONFIG` inside the adapter. Prefer the benchmark launchers.

## Quickstart

All commands run from the repository root. Commands that start services, consume GPUs, call model APIs, or launch experiments are intentionally separate and must be started by the user.

### 1. Prepare the container runtime

```bash
# Verify the rootful Docker daemon first.
docker info

# Install and verify the pinned Harbor CLI, then clone/verify its matching source.
# Set VTB_HARBOR_REPO first only when the default sibling ../harbor is unsuitable.
bash scripts/prepare/install_backend_tools.sh

# Build the trainer and SGLang images.
bash containers/build_image.sh all
```

Expected results: `docker info` shows a reachable rootful daemon; backend installation prints `HARBOR_INSTALL_OK` and `HARBOR_SOURCE_OK` with the same exact commit; image builds finish with `vibetrainbench/trainer:latest` and `vibetrainbench/sglang:latest` inspectable by Docker.

Host commands use Python **3.12**, declared by `.python-version` and `pyproject.toml`. Backend installation bootstraps a uv-managed environment under `vtb-env/tools/runtime/`; launchers select it automatically. An explicit `VTB_PYTHON=/absolute/path/to/venv/bin/python3` can select another 3.12 environment. Shell aliases, user-site packages, `PYTHONHOME`, and `PYTHONPATH` do not select the runtime. The trainer/SGLang images verify the same Python minor at build time; benchmark task images retain their own pinned environments.

For direct Python commands and tests, use the same interpreter:

```bash
source src/commit_utils/configure_python.sh
"$VTB_PYTHON" --version
"$VTB_PYTHON" -m unittest discover -s tests -q
```

Prepare the selected benchmark's assets before running its readiness check. Run each listed action separately from the repository root:

| Benchmark | Asset preparation actions | Runtime check |
|---|---|---|
| SWE | `bash scripts/prepare/prepare_swe.sh datasets`, then `tasks` | `bash scripts/prepare/prepare_swe.sh preflight` |
| Terminal | `bash scripts/prepare/prepare_terminal.sh datasets`, then prefetch below | `bash scripts/prepare/prepare_terminal.sh preflight` |
| Tau3 | `bash scripts/prepare/prepare_tau3.sh datasets`, then `image`, then `tasks` | `bash scripts/prepare/prepare_tau3.sh preflight` |
| Toolathlon | `bash scripts/prepare/prepare_toolathlon.sh datasets`; deploy both task services separately | `bash scripts/prepare/prepare_toolathlon.sh preflight` |
| AIME | `bash scripts/prepare/prepare_aime.sh datasets`, then `training-data` | `bash scripts/prepare/prepare_aime.sh preflight` |

For Terminal validation/final task packages and images:

```bash
bash scripts/prepare/prefetch_benchmark_images.sh \
  --benchmarks terminal-bench__terminal-bench-2-1,openthoughts__openthoughts-tblite-val50
```

`datasets` prepares the model and recipe-declared inputs; AIME's `training-data` performs CPU-only mathematics conversion. A no-training smoke needs the model snapshot and evaluation assets, not the formal training mixture or Megatron conversion. Runtime checks print the benchmark-specific success marker (`SWE_PREPARE_OK`, `TERMINAL_PREPARE_OK`, `TAU2_PREFLIGHT_OK`, `TOOLATHLON_PREPARE_OK`, or `AIME_PREFLIGHT_OK`). A nonzero exit means preparation is incomplete.

For **formal training**, also prepare the normalized HF checkpoint's matching Megatron conversion. This is a separate GPU operation—start it only when the cards are available:

```bash
bash scripts/prepare/download_base_model.sh convert
bash scripts/prepare/download_base_model.sh preflight
# Replace swe with the selected benchmark; this checks training/baseline resources too.
bash scripts/prepare/prepare_swe.sh preflight --formal
```

The `all` preparation actions include downloads and formal checks, but never GPU model conversion; they require that conversion to be prepared beforehand.

See [container setup](containers/README.md), [environment and paths](scripts/prepare/README.md), and [recipe data readiness](training/recipes/README.md#data-readiness). If existing model checkpoints lack provenance, select a fresh `VTB_MODEL_RESOURCE_ROOT` and use `download_base_model.sh download --offline` to normalize the exact cached snapshot without overwriting old weights. GPU conversion remains a separate operator action. Export that model-root override through all formal phases; it does not change the workspace's model-resource path.

### 2. Run a bounded connectivity smoke

Smoke runs prove validation and official-final connectivity with explicit small limits; they do not prove held-out membership or formal training-data readiness, and their scores are not benchmark results. All five smoke launchers accept `AGENT_CLI` and `RESEARCHER`; their base checkpoint remains pinned to `Qwen/Qwen3.5-9B-Base` (one hour, one GPU, one validation and one final sample). Configure the selected adapter's credential and compatible endpoint before starting. The researcher calls that API; evaluation calls the locally served checkpoint. Service/health/cleanup do not require a researcher API key.

Use a unique `SMOKE_ID` and run each phase separately. `run` starts detached tmux and returns immediately; wait for the printed run session to exit before `verify`:

```bash
export SMOKE_ID=swe-connect-001
export AGENT_CLI=claude RESEARCHER="<your-research-model>"
# Configure ANTHROPIC_API_KEY and, if needed, ANTHROPIC_BASE_URL in api_key.env.
# Base checkpoint: Qwen3.5-9B-Base; model API selection is independent of this checkpoint.
bash scripts/run_smoke/run_smoke_swe.sh preflight
bash scripts/run_smoke/run_smoke_swe.sh service
bash scripts/run_smoke/run_smoke_swe.sh health
bash scripts/run_smoke/run_smoke_swe.sh run
# Wait for the printed run session to exit before these commands.
bash scripts/run_smoke/run_smoke_swe.sh verify
bash scripts/run_smoke/run_smoke_swe.sh cleanup
```

Expected results:

| Phase | Success gate |
|---|---|
| `preflight` | Prints a benchmark-specific `*_SMOKE_PREFLIGHT_OK`. |
| `service` | Prints the grading tmux session/state/port and leaves the service running. |
| `health` | Reports service status `ok` and the intended validation policy. |
| `run` | Prints a detached run session and result path; this means **started**, not completed. |
| `verify` | Run only after the run session exits; prints `*_EXPERIMENT_SMOKE_OK` after both bounded jobs and `metrics.json` are complete. |
| `cleanup` | Run only after verification or stopping the owned run. Stops smoke grading and removes its tmux wrapper; Terminal/AIME also remove the run wrapper. State and results are preserved. |

Use `tmux attach -t <printed-run-session>` to monitor; detach without stopping it with `Ctrl-b`, then `d`. Replace `swe` with `terminal`, `tau3`, `toolathlon`, or `aime`. Never reuse smoke grading state for a formal run: validation membership, limits, and registrations belong to that service. See [safe completion and interruption](scripts/README.md#completion-interruption-and-safe-shutdown).

### 3. Start a formal experiment

Formal launchers also keep service startup, health checking, and experiment launch separate. Use a fresh benchmark-specific grading state:

```bash
export VTB_ROOT="${VTB_ROOT:-$PWD/vtb-env}"
export VTB_GRADING_STATE="$VTB_ROOT/grading_states/formal/swe_run_001"
export AGENT_CLI=claude RESEARCHER="<your-research-model>" RUN_ID=swe-run-001
export MODEL=Qwen/Qwen3.5-9B-Base RECIPE=qwen3_5_9b/slime_sft_swe
# Requires ANTHROPIC_API_KEY for the researcher; grading serves MODEL locally.
bash scripts/run_exp/run_exp_swe.sh service
bash scripts/run_exp/run_exp_swe.sh health
bash scripts/run_exp/run_exp_swe.sh run
```

Experiment defaults:

| Benchmark | Hardware | Budget | Default recipe | Harness |
|---|---:|---:|---|---|
| SWE-Bench Verified | 8×A100 | 24h | `slime_sft_swe` | editable |
| Terminal-Bench 2.1 | 8×A100 | 24h | `slime_sft_terminal` | editable |
| Tau3-Bench | 8×A100 | 16h | `slime_sft_tau` | fixed |
| Toolathlon | 8×A100 | 16h | `slime_sft_toolathlon` | fixed |
| AIME | 8×A100 | 16h | `slime_sft_aime` | fixed |

Expected results: `service` prints a grading tmux session and fresh state path; `health` returns status `ok`; `run` prints a detached research session and result path. The `run` command returning means **started**, not completed. Successful completion requires the run session to exit normally and the printed result directory to contain valid `metrics.json` plus the archived final model.

Monitor with `tmux attach -t <printed-run-session>` and detach with `Ctrl-b`, then `d`. After completion, the grading service remains running: attach to its printed session and press `Ctrl-C` for graceful shutdown. To interrupt early, stop the owned research session first, cancel any owned grading job, wait for GPU release, and only then stop grading; do not use `tmux kill-session` as the first shutdown step. See the exact [completion/interruption runbook](scripts/README.md#completion-interruption-and-safe-shutdown).

The launchers accept documented launcher-input overrides for controlled local runs; the settings above are the reference for paper comparisons. Changing `RESEARCHER` alone does not change the adapter/protocol. For a diagnostic final cap, set `FINAL_LIMIT=1` before `service` (not a paper result). Tau3/Toolathlon still need their independent evaluator/service API settings from the table above.

## Optional local adapters

Custom adapter code can live under gitignored `.local/agents/<adapter>/`, with `solve.sh`, `adapter_config.json`, and an optional trace parser. Set `VTB_AGENT_ADAPTERS_DIR` explicitly to opt in; there is no automatic local-code discovery or fallback. An ignored `.local/runtime.env` can contain adapter/model/endpoint settings, loaded only when `VTB_LOCAL_ENV_FILE` names it. This is executable shell configuration: use only files you trust.

```bash
# Load trusted local deployment settings.
export VTB_LOCAL_ENV_FILE="$PWD/.local/runtime.env"
# Keep this selection for every service/health/run/verify/cleanup phase.
bash scripts/run_smoke/run_smoke_swe.sh preflight
```

`api_key.env` and `.local/` configuration are gitignored. Only the selected adapter's declared variables reach the researcher container. Smoke records its non-secret adapter/model/result path in `smoke_run.json`, allowing later verification with the same state even when the API key is absent. Runs without that record require explicit adapter/model inputs for verification.

## Grading-state layout

```text
vtb-env/grading_states/
├── formal/<benchmark>/             formal launcher defaults
├── smoke/<benchmark>_<SMOKE_ID>/    isolated smoke services
└── direct/                         direct-service default
```

All ten launchers and Python defaults use this root. `VTB_GRADING_STATE` overrides the whole service directory; keep the same value across `service`, `health`, `run`, `verify`, and `cleanup` as applicable. Repeated formal experiments need a new directory, as in the example above. `VTB_ROOT` still relocates all runtime state.

Do not move live state or reuse it for a different service: registrations and archives may contain absolute paths. Preserve each run's state with its results, and use the same `VTB_GRADING_STATE` when verifying or stopping that run.

## Documentation index

| Topic | Documentation |
|---|---|
| Containers, Docker storage, FUSE, image contracts | [containers/README.md](containers/README.md) |
| Runtime environment, variables, paths, secrets | [scripts/prepare/README.md](scripts/prepare/README.md) |
| Prepare, smoke, and formal entrypoints | [scripts/README.md](scripts/README.md) |
| Training workflow and resources | [training/README.md](training/README.md) |
| Recipe schema and custom experiment assembly | [training/recipes/README.md](training/recipes/README.md) |
| Benchmark registry and adding a benchmark | [benchmarks/README.md](benchmarks/README.md) |
| Evaluation subsystem | [eval/README.md](eval/README.md) |
| Grading service, disclosure, validation/final routing | [eval/grading/README.md](eval/grading/README.md) |
| Benchmark-owned services | [eval/service/README.md](eval/service/README.md) |
| Research-agent/provider adapters | [agents/README.md](agents/README.md) |

## License

This repository is released under the [MIT License](LICENSE). Third-party benchmark corpora, task images, and upstream services keep their own licenses.

## Citation

If you find this work useful, please cite our paper.

```bibtex
@misc{peng2026vibetrainingbench,
  title         = {VibeTraining-Bench: Data-Trainer-Harness Co-Design for Iterative System Improvement},
  author        = {Ruotian Peng and Yuanyi Ren and Shuaiqi Liu and Xin Guo and Fang Zhu and Yangyi Huang and Yue Wu and Zheng Wang and Yandong Wen and Zhihang Fu},
  year          = {2026},
  eprint        = {2610.26342},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

```bibtex
@article{202608.0051,
  doi       = {10.20944/preprints202608.0051.v1},
  url       = {https://www.preprints.org/manuscript/202608.0051},
  year      = {2026},
  month     = {August},
  publisher = {Preprints},
  author    = {Shuaiqi Liu and Zhengkai Lin and Yuxiang Zhang and Yuanyi Ren and Yue Wu and Yongbin Li and Zheng Wang and Zhihang Fu and Jieping Ye},
  title     = {The Path to Recursive Self-Improving Agents: Foundation, Framework, and Future Directions},
  journal   = {Preprints}
}
```
