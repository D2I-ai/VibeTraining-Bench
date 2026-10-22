"""Grading service configuration.

Shared, long-lived grading service for VibeTrainBench.

Design constraints (see eval/grading/README.md):
  * stdlib only -- the host pythons (3.8 system / 3.14 conda) have no flask,
    and delivery_audit.py already set the precedent of a stdlib-only tool.
  * The service NEVER evaluates in-process. Every evaluation is a subprocess
    running the untouched per-task evaluate.py, so that file remains the only
    scoring authority.
  * Isolation is where it belongs rather than everywhere. The delivered model is
    served in a container and reached over HTTP; each task's own code runs in the
    per-task container harbor starts. The thin orchestration between them
    runs on the host, because the tools it needs (harbor) are installed on
    the host and wrapping them would only mean handing a container the docker
    socket.
  * The service is shared across concurrent runs, so task / base_model /
    sandbox root are supplied per submission (via a run token), never fixed
    at startup.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Optional

# One place decides which image is which and which docker daemon this repo talks to;
# the benchmark runner reads the same module, so the grading service and a hand-run
# benchmark cannot disagree about either.
from ..runner import runtime_paths


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default).expanduser()


def _csv_list(raw: str) -> list[str]:
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


# ---------------------------------------------------------------- locations
# Repo root and the runtime storage root have exactly one definition, in
# eval/runner/runtime_paths.py (its module docstring covers why nothing reaches
# above the checkout and VTB_ROOT moves everything). This module used to re-derive
# both, and the two would disagree the day only one was edited.
REPO_ROOT = runtime_paths.REPO_ROOT
VTB_ROOT = runtime_paths.VTB_ROOT

# Optional Harbor checkout used by adapter-local Harbor agents. Most benchmarks do
# not need this, but Tau's registered agent is an import path under adapters/ and
# therefore requires the Harbor repo on PYTHONPATH during evaluation. Keep this as
# configuration/status rather than a hard service startup requirement so unrelated
# benchmarks still run when the Tau adapter branch is not installed.
HARBOR_REPO_RAW = (os.environ.get("VTB_HARBOR_REPO") or "").strip()
HARBOR_REPO = Path(HARBOR_REPO_RAW).expanduser().resolve() if HARBOR_REPO_RAW else None
HARBOR_REPO_HAS_ADAPTERS = bool(HARBOR_REPO and (HARBOR_REPO / "adapters").is_dir())

# Runtime storage: the disposable output of a run -- scratch, results, service state,
# model cache. Inside the checkout at vtb-env/, which .gitignore excludes, and safe to
# delete and repopulate. Container images do NOT live here: they live in the docker
# daemon's own store and are named by tag, built from the Dockerfiles under containers/.
# Where runtime storage lives stays a single decision: every default below hangs off
# VTB_ROOT, so a bigger disk means setting one variable.

# Service state (run registry, queue journal, baseline cache).
STATE_DIR = runtime_paths.GRADING_STATE_DIR

# There is no separate "evaluation container". All eight benchmarks in benchmarks/ are
# three-line shims into eval.runner.run, which drives harbor on the host and
# talks to an HTTP endpoint; wrapping that in a container of ours would supply nothing
# it uses and hide the tools it needs. What does run in a container is the model server
# (below) and each task's own environment, started by harbor.

# ---------------------------------------------------------------- model serving
# Backend-managed/agentic benchmarks drive an OpenAI-compatible HTTP endpoint. sglang is
# the only serving stack -- it carries the validated Qwen3.5 agentic parser pair (below) --
# and the variable exists so a mis-set environment fails loudly here rather than silently
# serving with another stack. VTB_MODEL_SERVER_BACKEND is accepted as a descriptive alias.
SERVE_BACKEND = (
    os.environ.get("VTB_SERVE_BACKEND")
    or os.environ.get("VTB_MODEL_SERVER_BACKEND")
    or "sglang"
).strip().lower()
if SERVE_BACKEND != "sglang":
    raise ValueError("VTB_SERVE_BACKEND must be 'sglang'")
# How to run the server: in a container, or from whatever is installed on this host.
#
# `host` is the escape hatch, for a machine where sglang is already importable
# and a container would only add a layer.
SERVE_RUNTIME = (os.environ.get("VTB_SERVE_RUNTIME") or "docker").strip().lower()
if SERVE_RUNTIME not in {"docker", "host"}:
    raise ValueError("VTB_SERVE_RUNTIME must be 'docker' or 'host'")

# The endpoint's API key.
API_KEY = os.environ.get("VTB_API_KEY") or "inspectai"

# The serving images, as tags rather than paths: the daemon resolves a tag, so there is
# no image file for this repo to locate or check for.
#
# VTB_SERVE_SGLANG_DOCKER_IMAGE is honoured first because existing run scripts and
# reports here already set it.
SERVE_SGLANG_IMAGE = (
    os.environ.get("VTB_SERVE_SGLANG_DOCKER_IMAGE")
    or os.environ.get("VTB_SERVE_SGLANG_IMAGE")
    or runtime_paths.SGLANG_IMAGE
)

# Tool parsing is a serving concern. The defaults are the Qwen3.5-compatible parser
# pair used by the validated agentic runs. Override these when a delivered model's chat
# template differs.
SERVE_TOOL_CALL_PARSER = (
    os.environ.get("VTB_SERVE_TOOL_CALL_PARSER")
    or os.environ.get("VTB_TOOL_CALL_PARSER")
    or "qwen3_coder"
)
SERVE_REASONING_PARSER = os.environ.get("VTB_SERVE_REASONING_PARSER") or "qwen3"
SERVE_PORT = int(os.environ.get("VTB_SERVE_PORT") or 0)

# An already-running OpenAI-compatible endpoint to evaluate against, instead of
# serving the delivered checkpoint here.
#
# This exists because serving locally is not always possible: the serving images are
# built from containers/*.def by apptainer, and on a host without apptainer (or whose
# accelerator the pinned runtime does not support) there is no way to produce one. The
# error raised in that situation already told operators to "use --api-base/
# VTB_SWE_API_BASE for an already-running endpoint" -- this is that path.
#
# It weakens a guarantee, so it is opt-in and loud: normally the score provably comes
# from the submitted model_path, because this service is what loaded it. Pointed at an
# endpoint somebody else started, the score is only as trustworthy as that endpoint.
# served_model() therefore compares the endpoint's advertised model name against the
# checkpoint directory's name and refuses on a mismatch, which catches the mistake that
# actually happens -- a stale server left running from an earlier model.
#
# Use it for wiring up and debugging a benchmark. Official numbers should come from a
# locally served checkpoint.
EXTERNAL_API_BASE = (os.environ.get("VTB_SWE_API_BASE") or "").strip()
SERVE_EXTRA_ARGS = shlex.split(os.environ.get("VTB_SERVE_EXTRA_ARGS") or "")
# Host paths the serving container must additionally see, bound at the same location.
#
# The serving image is built from a stock CUDA base, which is enough when the
# accelerator is an NVIDIA card: apptainer --nv injects the driver libraries and
# device nodes. It is NOT enough for an accelerator whose runtime lives in a host
# SDK directory -- a CUDA-compatible stack such as PPU keeps its libraries under
# /usr/local/PPU_SDK, which --nv knows nothing about, so vLLM inside the container
# would find no device at all.
#
# Kept as a configured list rather than hardcoded paths: naming a vendor here would
# tie the harness to one kind of hardware, and the set of paths differs per SDK.
#
#   VTB_SERVE_EXTRA_BINDS=/usr/local/PPU_SDK,/opt/accl-p
SERVE_EXTRA_BINDS = [
    p.strip() for p in (os.environ.get("VTB_SERVE_EXTRA_BINDS") or "").split(",") if p.strip()
]

# Host environment variables to forward into the serving container.
#
# Binding an SDK is only half of it: its loader has to be told where to look, which
# lives in variables like LD_LIBRARY_PATH and CUDA_HOME. Forwarding is opt-in and
# explicit because a blanket copy of the environment would drag the host's PATH and
# python settings in with it.
#
#   VTB_SERVE_PASSTHROUGH_ENV=LD_LIBRARY_PATH,CUDA_HOME,PPU_SDK,PPU_HOME
SERVE_PASSTHROUGH_ENV = [
    v.strip() for v in (os.environ.get("VTB_SERVE_PASSTHROUGH_ENV") or "").split(",") if v.strip()
]
SGLANG_EXTRA_ARGS = shlex.split(os.environ.get("VTB_SGLANG_EXTRA_ARGS") or "")


def serve_image() -> str:
    return SERVE_SGLANG_IMAGE

# Every installed benchmark's evaluate.py drives an already-served model over HTTP.
# Most are shims into eval.runner.run and Harbor; benchmark-owned direct evaluators such
# as AIME consume the same --api-base contract without task containers. The worker serves
# the delivered model in either case, so there is no exception list to keep in sync.
#
# VTB_SELF_SERVED_BENCHMARKS is the way back if a benchmark is ever added that loads the
# model itself: name it here and the worker will not serve one for it. `or` is not usable
# for this default -- an empty value is a meaningful setting, and an empty string is
# falsy, so unset and "nothing" have to be told apart explicitly.
_SELF_SERVED_ENV = os.environ.get("VTB_SELF_SERVED_BENCHMARKS")
SELF_SERVED_BENCHMARKS = frozenset(
    entry.strip()
    for entry in (
        _SELF_SERVED_ENV if _SELF_SERVED_ENV is not None else ""
    ).split(",")
    if entry.strip()
)


# Trials the evaluation runs in parallel. Defined on the runner side, where the value
# is actually consumed, and re-exported here because the grading side needs it for
# three things -- how long an evaluation should take, the key its history is filed
# under, and the estimate derived from the base model's timing -- and those must not
# be able to disagree with the runner about it.
eval_concurrency = runtime_paths.eval_concurrency


def editable_harness(task: str) -> dict[str, Any]:
    """Editable-harness metadata declared by one benchmark, or an empty object."""
    path = REPO_ROOT / "benchmarks" / task / "benchmark.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    value = payload.get("editable_harness") or {}
    return dict(value) if isinstance(value, dict) and value.get("enabled") is True else {}


def needs_served_model(task: str) -> bool:
    """Whether the worker must serve the model before running evaluate.py.

    True for everything installed here. False only for a task named in
    VTB_SELF_SERVED_BENCHMARKS, which is empty by default.
    """
    return task not in SELF_SERVED_BENCHMARKS


# Which tool-call parser to serve a delivered model with. Kept under the historical
# name for callers that imported it directly; new code should use
# SERVE_TOOL_CALL_PARSER and SERVE_REASONING_PARSER.
#
# Backend agents drive the model through tool calls, so the endpoint must be started
# with parser support. The right parser depends on the model's chat template, not its
# name: `hermes` for templates emitting <tool_call> tags, `qwen3_coder`/`qwen3_xml`
# for Qwen3-Coder-style <function=...> shapes.
TOOL_CALL_PARSER = SERVE_TOOL_CALL_PARSER


# Serving a model loads the whole checkpoint and captures CUDA graphs before the first
# task runs, so this is far longer than one attempt's own startup. VTB_SERVE_TIMEOUT is
# the same name eval/runner/model_server.py reads, and deliberately so: one endpoint,
# one deadline, whichever side started it. VTB_SWE_SERVE_TIMEOUT is the old name, kept
# working because it appears in operators' shell history; nothing here is SWE-specific.
SERVE_TIMEOUT_SECONDS = int(
    os.environ.get("VTB_SERVE_TIMEOUT")
    or os.environ.get("VTB_SWE_SERVE_TIMEOUT")
    or 1800
)

# Host-side HuggingFace cache. Evaluations get a throwaway fuse-overlayfs upper
# layer on top of it, so dataset/model downloads never pollute the shared cache.
HF_HOME = _env_path("HF_HOME", str(VTB_ROOT / "resources" / "data" / "hf_cache"))

# Baseline scores are keyed by (task, base_model) and shared across runs.
BASELINE_CACHE_DIR = STATE_DIR / "baselines"

# The base model's own validation transcripts, kept once and read by every run.
#
# Sharing is the point here for the same reason it is for the score: the base model's
# behaviour on the split does not depend on which run is looking, and one copy of ~600MB
# beats one per run. Runs mount this read-only rather than receiving a copy.
BASELINE_DISCLOSURE_DIR = STATE_DIR / "baseline_disclosure"

# ---------------------------------------------------------------- networking
# Port must stay clear of the ephemeral model servers an evaluate.py may spawn
# (they take --port). Keep this below
# the 8000+ range those default to.
PORT = int(os.environ.get("VTB_GRADING_PORT") or 5800)
BIND_HOST = os.environ.get("VTB_GRADING_HOST") or "127.0.0.1"

# ---------------------------------------------------------------- gpu policy
# How grading coordinates with training GPU use.
#
# isolated: the grading service has a disjoint card/pool, or runs on another machine.
# Agents should see ordinary queueing/running states and no GPU-release advice.
#
# shared: training and grading may use the same physical GPUs. The service still owns
# serialisation, but a validation submission can wait behind the agent's own GPU work, so
# agent-facing status may tell it to release GPU processes and do CPU-only work.
GPU_COORDINATION_MODE = (os.environ.get("VTB_GPU_COORDINATION_MODE") or "isolated").strip().lower()
if GPU_COORDINATION_MODE not in {"isolated", "shared"}:
    raise ValueError("VTB_GPU_COORDINATION_MODE must be 'isolated' or 'shared'")

# The GPU(s) reserved for evaluation. VTB_EVAL_GPUS is the plural form for
# multi-GPU serving; VTB_EVAL_GPU remains the backward-compatible single-card spelling.
# Empty means "inherit", which is only appropriate for smoke tests or already-isolated
# two-machine deployments.
EVAL_GPU = os.environ.get("VTB_EVAL_GPU", "").strip()
EVAL_GPUS = _csv_list(os.environ.get("VTB_EVAL_GPUS", "").strip() or EVAL_GPU)
EVAL_GPU_SPEC = ",".join(EVAL_GPUS)
EVAL_GPU_COUNT = len(EVAL_GPUS)

# Evaluation needs the card essentially to itself (evaluate.py serves the candidate at a
# fixed share of *total* memory, and the agent harness spikes above it).
# If a stray training process is on the eval card we wait rather than OOM.
GPU_FREE_MIB_REQUIRED = int(os.environ.get("VTB_GRADING_GPU_FREE_MIB") or 40000)
GPU_WAIT_SECONDS = int(os.environ.get("VTB_GRADING_GPU_WAIT") or 1800)
GPU_POLL_SECONDS = int(os.environ.get("VTB_GRADING_GPU_POLL") or 30)

# How long a cancellation waits for the cards to actually report free memory before
# answering anyway. Short, because the caller is blocked on it: a serving container
# that has been asked to stop takes seconds, and a caller told "released" when the
# memory is still held would meet it as an unexplained out-of-memory failure.
GPU_RELEASE_WAIT_SECONDS = int(os.environ.get("VTB_GRADING_GPU_RELEASE_WAIT") or 180)

# How long a cancellation waits for the worker to finish winding the job up, so the
# response can carry the partial result. Only the partial depends on it -- the cards
# are already free by then -- so it is short, and /result has the partial either way.
CANCEL_WINDUP_SECONDS = int(os.environ.get("VTB_GRADING_CANCEL_WINDUP") or 60)

# How many times an agent may cancel its own validation in one run. A cancellation is
# a legitimate way to take the shared cards back; submitting, watching the first few
# samples and cancelling repeatedly is a cheap way to shop for a flattering prefix,
# and the difference between the two is the count. 0 disables the cap.
MAX_AGENT_CANCELS = int(os.environ.get("VTB_MAX_AGENT_CANCELS") or 5)

# Optional operator-side cap for official final submissions. Empty or -1 keeps the normal
# full benchmark. Positive values are for smoke/train-test connectivity checks where burning
# the entire final set would be wasteful.
FINAL_LIMIT = int(os.environ.get("VTB_FINAL_LIMIT") or -1)

# ---------------------------------------------------------------- retry ladder
# Three phases of attempts (4 -> 3 -> 2). Phase 1 uses each evaluate.py's own default
# budget; phases 2 and 3 may lower a limit for the benchmarks that have one.
#
# No benchmark here lowers a step count any more. The three shell-loop ones used to retry
# at 40 then 25 steps, which never reached an agent (see task_backends.step_limit_args) and
# would have been wrong if it had: a working delivery already spends a median of 72 steps
# and up to 372, and the tasks that need the most steps are the hard ones. Cutting the
# budget does not shorten a task, it decides the answer -- and it decides it against
# exactly the samples that separate a good model from a mediocre one, while returning a
# number that looks like a score. A run that burns hours does so by repeating itself, and
# that is detectable directly (trial_results.no_progress_repeats) without capping anyone.
TOKEN_LADDER: dict[str, list[list[str]]] = {
    # task            phase 2   phase 3
    # An empty pair means "retry unchanged", declared rather than left out. The behaviour
    # is the same either way, but leaving a benchmark out of this table is also how a
    # benchmark that *should* have a ladder goes unnoticed -- so absence is reported at
    # startup (tasks_without_ladder) and this is how a benchmark says the repetition is
    # intended.
    #
    # The shell-loop benchmarks: a retry exists for a verifier that could not reach a
    # verdict or an attempt that died on infrastructure, and it repeats the same
    # measurement. When an attempt is too slow the answer is the timeout and the
    # degeneracy report, not a shorter leash.
    "swe-bench__swe-bench-verified": [[], []],
    "terminal-bench__terminal-bench-2-1": [[], []],
    # Same shell-loop shape as terminal-bench: a retry exists for a verifier that could
    # not reach a verdict, and it repeats the same measurement. An over-slow attempt
    # answers with the timeout and the degeneracy report, not a shorter leash.
    "openthoughts__openthoughts-tblite": [[], []],
    "openthoughts__openthoughts-tblite-val50": [[], []],
    # tau3-bench is a dialogue simulation rather than a shell loop, so its own agent
    # decides how many turns a task takes. There is no step knob to lower.
    "sierra-research__tau3-bench": [[], []],
    # toolathlon's CAMEL agent takes max_steps through the same --step-limit flag
    # (the adapter forwards it as max_steps). The default is toolathlon's own 100, and
    # the ladder lowers from there in the same spirit as the SWE suites above -- these
    # are agentic too: they run out of steps, not tokens.
    #
    # The cut is gentler than the SWE ones because measurement shows these tasks USE
    # their steps: a task was observed consuming all 50 of an earlier, lower limit
    # while still working. Halving the budget would convert "could not reach a verdict"
    # into "cut off earlier", which is not what a retry is for.
    "toolathlon_gym": [["--step-limit", "70"], ["--step-limit", "50"]],
    # The test twin runs the same adapter (toolathlon_gym's evaluate.py executed
    # against this directory's manifest), so its step knob is the same flag and
    # gets the same ladder.
    "toolathlon_gym_test": [["--step-limit", "70"], ["--step-limit", "50"]],
    # toolathlon_verified runs the official verified harness (OpenAI Agents SDK,
    # not camel) through its own service, but max_steps travels through the same
    # --step-limit flag. The official formal_run_v0 default is 200 -- twice
    # gym's -- and the verified tasks are longer-horizon, so the ladder lowers
    # from 200 with the same 70%/50% proportions gym uses, not gym's absolute
    # values (which would be a 65% cut before phase 2 even starts).
    "toolathlon_verified": [["--step-limit", "140"], ["--step-limit", "100"]],
    # gym60 / verified72 run the SAME adapters as their parents (same service,
    # same max_steps semantics), so their step knobs are the same flags and get
    # the same ladders -- gym60 from the 100 default, verified72 from 200.
    "toolathlon_gym60": [["--step-limit", "70"], ["--step-limit", "50"]],
    "toolathlon_verified72": [["--step-limit", "140"], ["--step-limit", "100"]],
    # aime, bfcl_parity and gpqa-diamond are single-turn: one question, one answer,
    # scored by an exact match. There is no budget to lower, and a retry exists only
    # for an attempt that failed for an infrastructure reason.
    "aime__aime": [[], []],
    "aime__26": [[], []],
    "gorilla__bfcl_parity": [[], []],
    "gpqa-diamond__gpqa-diamond": [[], []],
    # humanevalfix is agentic, but through openhands. --step-limit reaches an agent as
    # harbor's `--agent-kwarg step_limit=`, and whether the openhands adapter honours
    # that kwarg has not been verified here -- so this declares an unchanged retry
    # rather than a lowered limit that might be accepted and ignored.
    "bigcode__humanevalfix": [[], []],
}

# Attempts per phase. Phase 1 has no token override.
PHASE_ATTEMPTS = (4, 3, 2)

# Per-attempt wall clock ceiling: 8 hours.
EVAL_TIMEOUT_SECONDS = int(os.environ.get("VTB_GRADING_EVAL_TIMEOUT") or 28800)

# Short settle time before each attempt, giving a freed GPU a moment to report itself as
# free. Lowered in tests so the failure ladder can be exercised without waiting on real
# work.
ATTEMPT_SETTLE_SECONDS = float(os.environ.get("VTB_GRADING_SETTLE") or 5)

# ------------------------------------------------------------- eval protocol
# Kept as raw strings here and parsed by valsplit at service startup, so a
# malformed value stops the service with one clear message instead of raising
# from an import in whichever module happened to touch it first.
#
# VAL_SOURCE decides which samples an agent submission is scored on. An agent may
# request a smaller positive --limit, but the service caps it at this configured
# validation boundary. AGENT_DISCLOSURE decides how much of the outcome comes
# back. Both default to the most restrictive useful setting.
#
# MAX_SAMPLES is a ceiling on what may be disclosed, not a page size. It defaulted
# to 64, which silently returned 64 of a 150-sample split's per-sample outcomes and
# left the agent diagnosing a prefix of an evaluation it had paid for all of. 0 means
# the only bound is the job's own sample limit -- which for an agent is already capped
# at the validation split, so the held-out samples stay unreachable either way.
VAL_SOURCE_RAW = os.environ.get("VTB_VAL_SOURCE", "").strip()
AGENT_DISCLOSURE_RAW = os.environ.get("VTB_AGENT_DISCLOSURE", "").strip()
AGENT_DISCLOSURE_MAX_SAMPLES = int(os.environ.get("VTB_AGENT_DISCLOSURE_MAX_SAMPLES") or 0)
AGENT_DISCLOSURE_MAX_CHARS = int(os.environ.get("VTB_AGENT_DISCLOSURE_MAX_CHARS") or 12000)


def token_args(task: str, phase: int) -> list[str]:
    """Token-budget override for a 1-indexed phase. Phase 1 -> no override.

    Nothing overrides a step budget here any more, for the reason spelled out above
    TOKEN_LADDER: the samples that need the most steps are the hard ones, so a lowered
    ceiling answers them rather than measuring them. The table stays because a
    token-budget override is still the right shape for a benchmark that has one.
    """
    if phase <= 1:
        return []
    ladder = TOKEN_LADDER.get(task)
    if not ladder:
        return []
    idx = phase - 2
    return list(ladder[idx]) if 0 <= idx < len(ladder) else []


# Benchmarks live in <repo>/benchmarks/, beside the harness in <repo>/eval/ rather than
# inside it. One definition, so a move cannot leave two places disagreeing -- which
# it did: task_dir was updated and known_tasks was not, and the service reported an
# empty task list while every task directory was in place.
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"


def task_dir(task: str) -> Path:
    return BENCHMARKS_ROOT / task


# Two shapes, because a benchmark declares its subset in one of two ways. The
# inspect-ai ones spell out an argparse default; the SWE ones are a shim over the
# registry in benchmarks/registry.py and declare default_limit there. Both are read from
# the file that actually runs, which is the point -- a table here would drift.
_DEFAULT_LIMIT_RE = re.compile(
    r"""["']--limit["'][^)]*?default\s*=\s*(-?\d+)|default_limit\s*=\s*(-?\d+)""", re.S
)
_default_limit_cache: dict[str, Optional[int]] = {}
_dataset_total_cache: dict[str, Optional[int]] = {}
_validation_source_cache: dict[str, Optional[str]] = {}


def _registry_spec_attr(task: str, attr: str):
    """Read one benchmark registry attribute, returning None on lookup failure."""
    import importlib.util
    import sys

    spec_path = BENCHMARKS_ROOT / "registry.py"
    if not spec_path.is_file():
        return None
    try:
        module_name = "_vtb_task_registry"
        module_spec = importlib.util.spec_from_file_location(module_name, spec_path)
        if module_spec is None or module_spec.loader is None:
            return None
        module = importlib.util.module_from_spec(module_spec)
        # Registered before exec because registry.py defines a @dataclass, and
        # dataclass resolves its annotations by looking the module up in sys.modules;
        # without this the load fails with a bare
        # "'NoneType' object has no attribute '__dict__'".
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
            entry = getattr(module, "BENCHMARKS", {}).get(task)
            return getattr(entry, attr, None) if entry is not None else None
        finally:
            sys.modules.pop(module_name, None)
    except (OSError, ValueError, AttributeError):
        return None


def _registry_spec_value(task: str, attr: str) -> Optional[int]:
    """Read an integer benchmark registry attribute, if one is declared."""
    value = _registry_spec_attr(task, attr)
    return value if isinstance(value, int) and value > 0 else None


def _registry_default_limit(task: str) -> Optional[int]:
    """The limit a benchmark declares in benchmarks/registry.py, if it is declared there.

    Backend-managed benchmarks keep their subset size in the registry rather than in
    their evaluate.py, which is a three-line shim. Read live rather than copied into a
    table here, for the same reason the regex below reads the file that runs: two
    places to state one number is how they end up disagreeing -- and this one did,
    silently returning None for every task after the registry was introduced, which
    cost the progress reporter its percentages.
    """
    return _registry_spec_value(task, "default_limit")


def dataset_total(task: str) -> Optional[int]:
    """Declared full dataset size for a benchmark, if known before evaluation.

    Observed totals from evaluation logs remain the strongest evidence and are stored
    by the grading service. This declarative value exists for backend-managed datasets
    whose size is known from the benchmark registration but not discoverable from a
    local task manifest before the first run.
    """
    if task in _dataset_total_cache:
        return _dataset_total_cache[task]
    value = _registry_spec_value(task, "dataset_total")
    _dataset_total_cache[task] = value
    return value


def validation_source(task: str) -> Optional[str]:
    """Benchmark-local validation source declared in benchmark.json, if any.

    A benchmark may route its validation elsewhere (e.g. a curated fixed subset of a
    sibling dataset) by declaring validation_source in benchmark.json. ``${VTB_ROOT}``
    and ``${VTB_REPO}`` are expanded so private, runtime-owned split files need not be
    tied to one host path. This is only the benchmark's own declaration: the grading
    service's explicit VTB_VAL_SOURCE still overrides it (see
    GradingService._validation_source), and the split:<...> default applies when neither
    is set.
    """
    if task in _validation_source_cache:
        return _validation_source_cache[task]
    value = _registry_spec_attr(task, "validation_source")
    parsed = value.strip() if isinstance(value, str) and value.strip() else None
    if parsed:
        parsed = os.path.expandvars(
            parsed.replace("${VTB_ROOT}", str(VTB_ROOT)).replace("$VTB_ROOT", str(VTB_ROOT))
            .replace("${VTB_REPO}", str(REPO_ROOT)).replace("$VTB_REPO", str(REPO_ROOT))
        )
    _validation_source_cache[task] = parsed
    return parsed


def default_limit(task: str) -> Optional[int]:
    """The --limit default a task evaluates with, or None.

    A dev submission that passes no limit inherits this, so it is the sample total
    needed to turn a completion count into a percentage.

    Two sources, both read live so neither can drift from what actually runs: the
    registry (where a backend-managed benchmark declares its subset) and the task's own
    evaluate.py (where a self-contained one bakes an argparse default). None simply
    means no percentage is reported.
    """
    if task in _default_limit_cache:
        return _default_limit_cache[task]

    value = _registry_default_limit(task)
    if value is None:
        script = task_dir(task) / "evaluate.py"
        try:
            match = _DEFAULT_LIMIT_RE.search(script.read_text(encoding="utf-8"))
            if match:
                # Whichever of the two alternatives matched.
                parsed = int(match.group(1) or match.group(2))
                value = parsed if parsed > 0 else None
        except OSError:
            value = None
    _default_limit_cache[task] = value
    return value


def known_tasks() -> list[str]:
    root = BENCHMARKS_ROOT
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "evaluate.py").is_file())


def _benchmark_json(task: str) -> dict:
    try:
        return json.loads((task_dir(task) / "benchmark.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _agent_string(spec: dict) -> str:
    agent = spec.get("agent")
    if isinstance(agent, dict):
        return str(agent.get("default") or "")
    return str(agent or "")


def _requires_harbor_repo(agent: str) -> bool:
    return agent.startswith("adapters.") or ".adapters." in agent


def tasks_requiring_harbor_repo(tasks: Optional[list] = None) -> list[str]:
    """Installed benchmarks whose selected agent is imported from Harbor adapters/."""
    return sorted(
        task
        for task in (tasks if tasks is not None else known_tasks())
        if _requires_harbor_repo(_agent_string(_benchmark_json(task)))
    )


def harbor_repo_status(tasks: Optional[list] = None) -> dict:
    """Operator-facing status for adapter-local Harbor benchmark support."""
    required_by = tasks_requiring_harbor_repo(tasks)
    status = {
        "repo_env": "VTB_HARBOR_REPO",
        "required_by": required_by,
        "configured": HARBOR_REPO is not None,
        "has_adapters": HARBOR_REPO_HAS_ADAPTERS,
        "ready": not required_by or HARBOR_REPO_HAS_ADAPTERS,
    }
    if HARBOR_REPO is not None:
        status["path"] = str(HARBOR_REPO)
    return status


def tasks_without_ladder(tasks: Optional[list] = None) -> list:
    """Installed benchmarks that have no entry in TOKEN_LADDER.

    This used to mean something behavioural: without an entry, phases 2 and 3 repeated
    phase 1 and all nine attempts became the same attempt. Since no entry lowers a step
    budget any more (see above TOKEN_LADDER), that is now true of every benchmark, and
    the entry is bookkeeping -- it records that someone considered what a retry should
    change, and found nothing worth changing. Still reported at startup for that reason:
    a benchmark added later may have a token budget worth lowering, and this is the only
    place that would ever ask.
    """
    return sorted(t for t in (tasks if tasks is not None else known_tasks()) if t not in TOKEN_LADDER)
