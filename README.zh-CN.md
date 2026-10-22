<div align="center">

# VibeTraining-Bench

### 面向迭代式系统改进的数据–训练器–推理框架协同设计

Ruotian Peng, Yuanyi Ren, Shuaiqi Liu, Xin Guo, Fang Zhu, Yangyi Huang, Yue Wu, Zheng Wang, Yandong Wen, Zhihang Fu<br>
浙江大学 · 西湖大学 · 阿里巴巴 Token Hub · 华南理工大学 · 香港中文大学

[![论文](https://img.shields.io/badge/Paper-arXiv%3A2610.26342-blue)](https://arxiv.org/abs/2610.26342)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub.io-purple)](https://d2i-ai.github.io/VibeTraining-Bench/index.zh.html)
[![Survey](https://img.shields.io/badge/Survey-Preprints-green)](https://www.preprints.org/frontend/manuscript/3c41418aa2e782b90ee5da995565ecac/download_pub)
[![代码](https://img.shields.io/badge/Code-Repository-brightgreen)](#快速开始)
[![基准](https://img.shields.io/badge/Benchmarks-5-blueviolet)](benchmarks/README.md)

[English](README.md) | **中文版**

用于评估研究智能体能否联合改进**训练数据**、**训练器**和**推理框架**，即递归自我改进所需的跨组件协同设计能力。

</div>

---

## 概览

递归自我改进要求智能体诊断系统瓶颈，并协调训练与推理侧的改进。VibeTraining-Bench 在软件工程、终端操作、多轮工具使用、客户服务智能体和数学推理任务上评估这种能力。

每个研究智能体从相同的基础模型和资源预算出发，可以利用验证反馈改进允许编辑的组件，最终提交一个检查点。独立的评分服务负责验证与最终评测；隐藏测试数据不会进入研究工作区。

![VibeTraining-Bench 概览](assets/images/main_figure.png)

*研究智能体在固定预算下迭代协同设计数据、训练器和推理框架，随后提交最终系统，由隔离的评分环境完成评测。*

## 与我们先前综述的关系

VibeTraining-Bench 以我们先前综述 [*The Path to Recursive Self-Improving Agents: Foundation, Framework, and Future Directions*](https://www.preprints.org/manuscript/202608.0051)（Liu 等，2026）提出的统一研究框架为基础。该框架将基础模型、agent harness、agent 数据系统、agent 训练器与改进机制建模为一个耦合演化系统，以刻画自改进的范围与依赖关系。在此框架下，VibeTraining-Bench 对跨组件协同设计进行受控评测，考察研究 agent 能否在固定资源预算下协同改进训练数据、训练器与推理 harness。

## 主要特点

- **跨组件评测：**关注数据、训练器和推理框架的协同改进，而非单一组件调优。
- **受控比较：**所有研究智能体使用相同初始系统、资源预算和基准约定。
- **隔离评分：**研究与评分环境分离，最终测试数据和正式凭据不会提供给研究智能体。
- **五类任务：**SWE-Bench Verified、Terminal-Bench 2.1、Tau3-Bench、Toolathlon 和 AIME。
- **可复现操作：**每个基准都有对应的 `prepare`、有界 `run_smoke` 和正式 `run_exp` 入口。

## 快速开始

所有命令都从仓库根目录执行。启动服务、占用 GPU、调用模型 API 或运行实验的命令均保持分阶段，并由用户明确启动。

### 1. 准备容器运行环境

```bash
# 首先确认 rootful Docker daemon 可用。
docker info

# 安装并验证固定 commit 的 Harbor CLI，随后自动克隆/验证同 commit 源码。
# 仅当默认 sibling 路径 ../harbor 不适用时，提前设置 VTB_HARBOR_REPO。
bash scripts/prepare/install_backend_tools.sh

# 构建 trainer 和 SGLang 镜像。
bash containers/build_image.sh all
```

预期结果：`docker info` 显示可访问的 rootful daemon；后端安装打印 commit 完全相同的 `HARBOR_INSTALL_OK` 和 `HARBOR_SOURCE_OK`；镜像构建结束后，Docker 可以 inspect `vibetrainbench/trainer:latest` 和 `vibetrainbench/sglang:latest`。

Prepare benchmark assets before runtime checks. Each action below is invoked separately; `preflight` alone does not download training data:

| Benchmark | Asset actions | Runtime check |
|---|---|---|
| SWE | `bash scripts/prepare/prepare_swe.sh datasets`, then `tasks` | `bash scripts/prepare/prepare_swe.sh preflight` |
| Terminal | `bash scripts/prepare/prepare_terminal.sh datasets`; prefetch Terminal/TBLite tasks below | `bash scripts/prepare/prepare_terminal.sh preflight` |
| Tau3 | `bash scripts/prepare/prepare_tau3.sh datasets`, then `image`, then `tasks` | `bash scripts/prepare/prepare_tau3.sh preflight` |
| Toolathlon | `bash scripts/prepare/prepare_toolathlon.sh datasets`; deploy both task services separately | `bash scripts/prepare/prepare_toolathlon.sh preflight` |
| AIME | `bash scripts/prepare/prepare_aime.sh datasets`, then `training-data` | `bash scripts/prepare/prepare_aime.sh preflight` |

```bash
bash scripts/prepare/prefetch_benchmark_images.sh \
  --benchmarks terminal-bench__terminal-bench-2-1,openthoughts__openthoughts-tblite-val50
```

Formal training requires a separate GPU model conversion followed by `preflight --formal`; `all` never performs that conversion. A no-training smoke needs only the model snapshot and evaluation assets, not the formal training mixture. See the [formal preparation steps](README.md#1-prepare-the-container-runtime).

预期结果：所选命令以对应的成功标记结束（`SWE_PREPARE_OK`、`TERMINAL_PREPARE_OK`、`TAU2_PREFLIGHT_OK`、`TOOLATHLON_PREPARE_OK` 或 `AIME_PREFLIGHT_OK`）。非零退出码表示准备尚未完成。

Host Python is pinned to the 3.12 minor by `.python-version` and `pyproject.toml`. Backend installation provisions an isolated environment; launchers automatically use it. For direct commands, source `src/commit_utils/configure_python.sh` and invoke `"$VTB_PYTHON"`. An explicit 3.12 virtual-environment interpreter can be selected with `VTB_PYTHON`; host aliases and user-site packages are not used. Trainer/SGLang builds verify the same minor. See [Python runtime](scripts/prepare/README.md#python-runtime).

详细说明见[容器环境](containers/README.md)、[环境变量与路径](scripts/prepare/README.md)和[训练数据就绪状态](training/recipes/README.md#data-readiness)。

### 2. 运行有界连通性 Smoke

Smoke 通过明确的小样本上限验证 validation 与 official-final 的连通性；它不证明 held-out 成员关系，其分数也不是正式基准结果。

Smoke and formal launchers support `AGENT_CLI`/`RESEARCHER`; the default adapter is `claude`, with an explicitly selected researcher API model. The smoke checkpoint remains pinned to `Qwen/Qwen3.5-9B-Base`. Only the selected adapter's declared credentials are forwarded. Custom adapters and configuration can be loaded through explicitly selected, gitignored local files. See [API roles and model selection](README.md#api-setup-and-model-selection).

使用唯一的 `SMOKE_ID` 并逐阶段执行。`run` 只会启动 detached tmux 后立即返回；必须等待打印出的 run session 结束后再执行 `verify`：

```bash
export AGENT_CLI=claude RESEARCHER="<your-research-model>"
SMOKE_ID=swe-connect-001 bash scripts/run_smoke/run_smoke_swe.sh preflight
SMOKE_ID=swe-connect-001 bash scripts/run_smoke/run_smoke_swe.sh service
SMOKE_ID=swe-connect-001 bash scripts/run_smoke/run_smoke_swe.sh health
SMOKE_ID=swe-connect-001 bash scripts/run_smoke/run_smoke_swe.sh run
# Wait for the run session to exit before verification; clean up only after success.
SMOKE_ID=swe-connect-001 bash scripts/run_smoke/run_smoke_swe.sh verify &&
SMOKE_ID=swe-connect-001 bash scripts/run_smoke/run_smoke_swe.sh cleanup
```

各阶段的成功条件：

| 阶段 | 成功条件 |
|---|---|
| `preflight` | 打印基准对应的 `*_SMOKE_PREFLIGHT_OK`。 |
| `service` | 打印 grading tmux session、state 和端口，并保持服务运行。 |
| `health` | 返回服务状态 `ok` 和预期 validation policy。 |
| `run` | 打印 detached run session 和结果路径；这里只表示“已启动”，不表示“已完成”。 |
| `verify` | 仅在 run session 退出后执行；两个有界任务和 `metrics.json` 完整时打印 `*_EXPERIMENT_SMOKE_OK`。 |
| `cleanup` | Run only after verification or stopping the owned run. Stops grading and removes its tmux wrapper; Terminal/AIME also remove the run wrapper. State/results are preserved. |

使用 `tmux attach -t <打印出的-run-session>` 观察；按 `Ctrl-b` 再按 `d` 可在不中止任务的情况下退出。可将 `swe` 替换为 `terminal`、`tau3`、`toolathlon` 或 `aime`。不得将 smoke grading state 用于正式实验。参见[完成、中断与安全退出](scripts/README.md#completion-interruption-and-safe-shutdown)。

### 3. 启动正式实验

正式实验同样将评分服务启动、健康检查和实验启动拆分。每次正式运行必须使用新的基准专属 grading state：

```bash
export VTB_ROOT="${VTB_ROOT:-$PWD/vtb-env}"
export VTB_GRADING_STATE="$VTB_ROOT/grading_states/formal/swe_run_001"
export AGENT_CLI=claude RESEARCHER="<your-research-model>" RUN_ID=swe-run-001
export MODEL=Qwen/Qwen3.5-9B-Base RECIPE=qwen3_5_9b/slime_sft_swe
# Configure ANTHROPIC_API_KEY in api_key.env for the outer researcher.
bash scripts/run_exp/run_exp_swe.sh service
bash scripts/run_exp/run_exp_swe.sh health
bash scripts/run_exp/run_exp_swe.sh run
```

论文默认设置：

| 基准 | 硬件 | 预算 | 默认 Recipe | Harness |
|---|---:|---:|---|---|
| SWE-Bench Verified | 8×A100 | 24h | `slime_sft_swe` | 可编辑 |
| Terminal-Bench 2.1 | 8×A100 | 24h | `slime_sft_terminal` | 可编辑 |
| Tau3-Bench | 8×A100 | 16h | `slime_sft_tau` | 固定 |
| Toolathlon | 8×A100 | 16h | `slime_sft_toolathlon` | 固定 |
| AIME | 8×A100 | 16h | `slime_sft_aime` | 固定 |

预期结果：`service` 打印 grading tmux session 和新的 state 路径；`health` 返回状态 `ok`；`run` 打印 detached research session 和结果路径。`run` 命令返回只表示“已启动”。成功完成要求 run session 正常退出，并且打印出的结果目录包含有效 `metrics.json` 和归档后的最终模型。

使用 `tmux attach -t <打印出的-run-session>` 观察，按 `Ctrl-b` 再按 `d` 可退出观察而不中止任务。实验完成后 grading service 仍会继续运行：attach 到打印出的 grading session 并按 `Ctrl-C`，触发优雅退出。若需提前中断，应先停止属于本次运行的 research session，再取消其 grading job、等待 GPU 释放，最后停止 grading；不要首先使用 `tmux kill-session`。精确步骤见[完成、中断与安全退出](scripts/README.md#completion-interruption-and-safe-shutdown)。

The launchers accept documented input overrides for controlled local runs; the settings above are the reference for paper comparisons.

## Grading-state layout

Default state paths are `vtb-env/grading_states/formal/<benchmark>/`, `vtb-env/grading_states/smoke/<benchmark>_<SMOKE_ID>/`, and `vtb-env/grading_states/direct/`. `VTB_GRADING_STATE` overrides the whole directory; keep the same value across launcher phases. Do not move live state or reuse smoke state for a formal service.

`RESEARCHER` selects the outer API model; `MODEL` selects the local checkpoint and must match `RECIPE`. Labels such as `researcher1` belong in `RUN_ID`, not a separate model-selection variable. Tau3's user/assertion models and Toolathlon service credentials are independent of the researcher. The [main API section](README.md#api-setup-and-model-selection) provides model-switching examples without prescribing deployment URLs.

## 文档索引

| 主题 | 文档 |
|---|---|
| 容器、Docker 存储、FUSE 和镜像约定 | [containers/README.md](containers/README.md) |
| 运行环境、变量、路径和密钥 | [scripts/prepare/README.md](scripts/prepare/README.md) |
| Prepare、smoke 和正式实验入口 | [scripts/README.md](scripts/README.md) |
| 训练流程和资源 | [training/README.md](training/README.md) |
| Recipe 字段和自定义实验装配 | [training/recipes/README.md](training/recipes/README.md) |
| Benchmark 注册和扩展 | [benchmarks/README.md](benchmarks/README.md) |
| Evaluation 子系统 | [eval/README.md](eval/README.md) |
| Grading service、披露等级和验证/最终评测路由 | [eval/grading/README.md](eval/grading/README.md) |
| 基准专属服务 | [eval/service/README.md](eval/service/README.md) |
| 研究智能体和 Provider 适配器 | [agents/README.md](agents/README.md) |

## 许可证

本仓库以 [MIT License](LICENSE) 发布。第三方基准语料、任务镜像和上游服务仍遵循其各自的许可证。

## 引用

如果这项工作对你有帮助，请引用我们的论文。

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
