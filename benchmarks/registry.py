"""Benchmark registry loader.

All benchmark-specific registration lives next to the benchmark itself as
`benchmarks/<name>/benchmark.json`. This module intentionally contains only schema,
normalisation and lookup logic; it should not grow per-benchmark entries.

A benchmark spec chooses one task source:

  dataset=      a dataset identifier resolved by Harbor or a benchmark-owned direct evaluator.
  tasks_subdir= a task tree carried in this repo (`-p <dir>`), kept for hand-written
                local task trees only.

Backend-level caches and downloaded hub task exports are runtime data under
`vtb-env/` (for example `vtb-env/cache` and `vtb-env/hub_task_specs`). They do not
belong in `benchmarks/`, which holds only lightweight registrations, shims, and
curated protocol files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Task-container backends. See eval/runner/task_backends.py for what each becomes.
#
# docker is the default and the only one wired up: it is harbor's own default, it builds
# a task's environment/Dockerfile when task.toml names no image, and it honours a task's
# no-network declaration.
DOCKER = "docker"


@dataclass(frozen=True)
class BenchmarkSpec:
    """How to run one benchmark.

    Exactly one of `dataset` and `tasks_subdir` is set: a registry name is resolved by
    the task backend, a subdirectory is a task tree in this repo.
    """

    name: str
    backend: str  # "harbor" or "customed" for a benchmark-owned adapter
    agent: str
    default_limit: int
    dataset_total: Optional[int] = None

    dataset: Optional[str] = None
    tasks_subdir: Optional[str] = None
    environment: str = DOCKER

    # A curated registry-backed subset. The IDs are passed to the backend as explicit
    # include filters, preserving their declared order and avoiding backend sampling.
    task_ids: tuple = ()
    # Optional benchmark-specific validation policy (e.g. "task:other__benchmark" --
    # validate on another benchmark entirely). The grading service's explicit
    # VTB_VAL_SOURCE setting still takes precedence over this value.
    validation_source: Optional[str] = None

    # The benchmark's nature, declared rather than chosen at run time. Agentic tasks
    # use a tool-loop harness; non-agentic direct-answer datasets still run through
    # whatever existing Harbor agent the benchmark registration selects. The flag is
    # declarative metadata -- records, reports and readers use it; the command line gets
    # whatever `agent` names, which must already match the nature.
    agentic: bool = True

    # Appended verbatim, for a benchmark that needs something specific.
    extra_args: tuple = ()
    # Backend agent-kwargs, passed as k=v.
    agent_kwargs: dict = field(default_factory=dict)
    # Environment variables the benchmark requires before any model or backend starts.
    required_env: tuple = ()
    # Benchmark-owned direct evaluator settings. These benchmarks still declare their
    # public dataset identifier, but evaluate.py bypasses the Harbor command builder.
    direct_eval: dict = field(default_factory=dict)
    # Free-form, for the runner to record and for a shim to read.
    notes: str = ""
    # Optional joint model+harness experiment metadata. The standard agent field remains
    # authoritative unless this is enabled by the grading service for one submission.
    editable_harness: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.dataset) == bool(self.tasks_subdir):
            raise ValueError(
                f"{self.name}: set exactly one of dataset= (registry) or "
                "tasks_subdir= (task tree in this repo)"
            )
        if self.backend not in {"harbor", "customed"}:
            raise ValueError(
                f"{self.name}: backend must be harbor or customed, not {self.backend!r}"
            )
        if self.task_ids and not self.dataset:
            raise ValueError(f"{self.name}: task_ids requires a registry dataset")
        if self.direct_eval and self.agent != "none":
            raise ValueError(f"{self.name}: direct_eval requires agent=none")
        if any(not task_id or task_id.strip() != task_id for task_id in self.task_ids):
            raise ValueError(f"{self.name}: task_ids entries must be non-empty and trimmed")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError(f"{self.name}: task_ids contains duplicate entries")
        if self.editable_harness:
            if not isinstance(self.editable_harness, dict):
                raise ValueError(f"{self.name}: editable_harness must be an object")
            if self.editable_harness.get("enabled") is not True:
                raise ValueError(f"{self.name}: editable_harness.enabled must be true")
            for key in ("import_path", "template_source", "allowed_files"):
                if key not in self.editable_harness:
                    raise ValueError(f"{self.name}: editable_harness requires {key}")
            if not isinstance(self.editable_harness["import_path"], str) or ":" not in self.editable_harness["import_path"]:
                raise ValueError(f"{self.name}: editable_harness.import_path must be module:Class")
            if not isinstance(self.editable_harness["allowed_files"], list) or not all(
                isinstance(item, str) and item and not item.startswith("/") and ".." not in Path(item).parts
                for item in self.editable_harness["allowed_files"]
            ):
                raise ValueError(f"{self.name}: editable_harness.allowed_files must be safe relative paths")


# Built-in agent shorthands accepted in benchmark.json. Longer import paths may be
# written directly for custom Harbor agents.
HARBOR_MINI_SWE = "mini-swe-agent"
HARBOR_DEFAULT_AGENT = "codex"


# Directory-local registrations written by humans or scripts/prepare/register_harbor_benchmark.sh live
# in benchmarks/<name>/benchmark.json. They keep generated hub benchmarks visible next
# to hand-written ones while heavy downloaded task packages stay under vtb-env/.
REGISTRY_DIR = Path(__file__).resolve().parent


def _normalise_agent(value) -> str:
    if isinstance(value, dict):
        env = value.get("env")
        default = value.get("default") or HARBOR_DEFAULT_AGENT
        if env and os.environ.get(str(env)):
            value = os.environ[str(env)]
        else:
            value = default
    value = str(value)
    if value in ("default", "harbor-default"):
        return HARBOR_DEFAULT_AGENT
    if value in ("mini-swe", "mini-swe-agent"):
        return HARBOR_MINI_SWE
    return value


def _spec_from_json(path: Path) -> BenchmarkSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    name = str(data.get("name") or path.parent.name)
    extra_args = tuple(data.get("extra_args") or ())
    agent_kwargs = dict(data.get("agent_kwargs") or {})
    required_env = tuple(str(v) for v in (data.get("required_env") or []))
    raw_editable_harness = data.get("editable_harness", {})
    if raw_editable_harness is None:
        raw_editable_harness = {}
    raw_task_ids = data.get("task_ids") or []
    if not isinstance(raw_task_ids, list):
        raise ValueError(f"{name}: task_ids must be a JSON array")
    task_ids = tuple(str(task_id) for task_id in raw_task_ids)
    return BenchmarkSpec(
        name=name,
        backend=str(data.get("backend") or data.get("framework") or "harbor"),
        agent=_normalise_agent(data.get("agent") or HARBOR_DEFAULT_AGENT),
        default_limit=int(data.get("default_limit", -1)),
        dataset_total=(int(data["dataset_total"]) if isinstance(data.get("dataset_total"), int) and data["dataset_total"] > 0 else None),
        dataset=data.get("dataset"),
        tasks_subdir=data.get("tasks_subdir"),
        environment=str(data.get("environment") or DOCKER),
        task_ids=task_ids,
        validation_source=(
            str(data["validation_source"]).strip() if data.get("validation_source") else None
        ),
        agentic=bool(data.get("agentic", False)),
        extra_args=extra_args,
        agent_kwargs=agent_kwargs,
        required_env=required_env,
        direct_eval=dict(data.get("direct_eval") or {}),
        notes=str(data.get("notes") or f"generated from {path}"),
        editable_harness=dict(raw_editable_harness),
    )


def _load_directory_specs() -> dict[str, BenchmarkSpec]:
    specs: dict[str, BenchmarkSpec] = {}
    for path in sorted(REGISTRY_DIR.glob("*/benchmark.json")):
        spec = _spec_from_json(path)
        specs[spec.name] = spec
    return specs


BENCHMARKS: dict[str, BenchmarkSpec] = _load_directory_specs()


def get(name: str) -> BenchmarkSpec:
    if name in BENCHMARKS:
        return BENCHMARKS[name]
    raise SystemExit(
        f"unknown benchmark {name!r}; known: {', '.join(sorted(BENCHMARKS))}. "
        "For a Harbor Hub dataset owner/name, run scripts/prepare/register_harbor_benchmark.sh so it "
        "creates benchmarks/<alias>/benchmark.json first."
    )


def task_source(spec: BenchmarkSpec, task_dir: Path) -> Optional[Path]:
    """The in-repo task tree for a spec, or None when it comes from the registry."""
    return (task_dir / spec.tasks_subdir) if spec.tasks_subdir else None
