#!/usr/bin/env python3
"""Resolve a run's declared validation source and baseline resource for run_task.sh.

A training recipe declares its validation source inline, in its own
`validation.val_source` field -- there is no separate profile file. This module
resolves that declaration and checks it against the grading service's effective
source for the task, and finds the baseline declaration by benchmark, model,
and expanded validation source. The short logical ID and materialized payload
path are read from the declaration rather than computed from directory names.

These used to be inline heredocs in run_task.sh. They are logic, not glue, and
a shell heredoc cannot be imported or unit tested; this module can.

Subcommands:
  val-source-for-recipe REPO RECIPE_NAME [TASK]
      Print the recipe's validation.val_source (empty line when undeclared).
      When TASK is given, refuse (exit 3, reason on stderr) a recipe whose
      validation block declares a different benchmark -- a recipe carried over
      to another benchmark must fail the run at launch, not validate against
      the wrong split.

  baseline-for-source REPO TASK BASE_MODEL VAL_SOURCE HEALTH_JSON
      Refuse (exit 3, reason on stderr) when the declared source disagrees with
      the service's effective source for the task. Otherwise print the baseline
      resource's payload directory (empty when no baseline resource exists).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional


def _expand_val_source(value: str, repo: Path) -> str:
    """Expand runtime-root placeholders before comparing a recipe with the service."""
    root = os.environ.get("VTB_ROOT") or str(repo / "vtb-env")
    return os.path.expandvars(
        value.replace("${VTB_ROOT}", root).replace("$VTB_ROOT", root).replace("${VTB_REPO}", str(repo))
    )


def _val_source_for_recipe(repo: Path, recipe_name: str, task: Optional[str] = None) -> None:
    repo = Path(repo)
    name = recipe_name.strip().strip("/")
    path = repo / "training" / "recipes" / (name if name.endswith(".json") else name + ".json")
    try:
        recipe = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # No recipe is not an error here: run_task.sh falls back to the legacy
        # baseline lookup. A malformed recipe fails loudly later, at
        # prepare_agent_workspace.py, which owns recipe validation.
        return
    validation = recipe.get("validation") or {}
    if not isinstance(validation, dict):
        print(f"recipe {name}: the validation block must be a JSON object", file=sys.stderr)
        sys.exit(3)
    declared_benchmark = str(validation.get("benchmark") or "")
    if task and declared_benchmark and declared_benchmark != task:
        print(
            f"recipe {name} declares a validation block for benchmark "
            f"{declared_benchmark!r}, not {task!r}",
            file=sys.stderr,
        )
        sys.exit(3)
    val_source = _expand_val_source(str(validation.get("val_source") or "").strip(), repo)
    if val_source:
        print(val_source)


def _baseline_for_source(repo: Path, task: str, base_model: str, val_source: str, health_path: Path) -> None:
    repo = Path(repo)
    sys.path.insert(0, str(repo))
    from eval.grading import valsplit

    val_source = valsplit.expand_val_source(repo, val_source)
    # Grammar check now, so a typo'd source fails the run at launch rather than
    # after the baseline was mounted against a different split.
    try:
        valsplit.parse_val_source(val_source)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(3)

    health = json.loads(Path(health_path).read_text(encoding="utf-8"))
    validation = health.get("validation") or {}
    # Prefer the per-task effective source (a benchmark's own validation_source
    # routing) over the service-wide default the health root reports.
    per_task = (validation.get("resolved_counts") or {}).get(task, {}) or {}
    service_source = str(per_task.get("validation_source") or validation.get("raw") or "")
    if valsplit.expand_val_source(repo, service_source) != valsplit.expand_val_source(repo, val_source):
        print(
            f"recipe declares val_source={val_source!r} but the grading service runs "
            f"val_source={service_source!r} for {task}; the baseline would be measured "
            "on a different split than this run's candidates",
            file=sys.stderr,
        )
        sys.exit(3)

    cfg = valsplit.find_baseline_resource(repo, task, base_model, val_source)
    if cfg is None:
        return  # no compatible baseline declared
    resource_id = cfg["id"]
    resource = cfg["_config_path"]
    # From here the baseline is declared, so a missing payload is fatal rather than a
    # silent skip. The recipe (or an operator import) named this resource precisely to
    # give the agent a reference point for "did my training change anything"; running
    # without it hands back an unexplained absence -- indistinguishable from a base model
    # that was never imported -- against hours of GPU. resolve_resource_path only expands
    # the path, it does not check the files exist, which is why the directory is verified
    # here rather than left to run_task's `[ -d ]` to swallow.
    payload = valsplit.resolve_resource_path(repo, cfg.get("materialized_path"))
    if payload is None:
        print(
            f"baseline resource {resource_id} is declared ({resource}) but its "
            "resource_config.json has no usable materialized_path",
            file=sys.stderr,
        )
        sys.exit(3)
    if not payload.is_dir():
        print(
            f"baseline resource {resource_id} is declared but its payload directory is "
            f"missing: {payload}. Restore the benchmark's published baseline payload; "
            "the run will not start without the "
            "reference validation it promised the agent",
            file=sys.stderr,
        )
        sys.exit(3)
    print(payload)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    command, args = argv[1], argv[2:]
    if command == "val-source-for-recipe" and len(args) in (2, 3):
        _val_source_for_recipe(Path(args[0]), args[1], args[2] if len(args) > 2 else None)
        return 0
    if command == "baseline-for-source" and len(args) == 5:
        _baseline_for_source(Path(args[0]), args[1], args[2], args[3], Path(args[4]))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
