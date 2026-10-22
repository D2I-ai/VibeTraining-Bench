#!/usr/bin/env python3
"""Atomically patch a Harbor Tau3 export into VTB's canonical local task tree.

The registry packages are immutable upstream artifacts and lag the pinned Harbor adapter.
Evaluation never patches them at run time: setup exports all 375 tasks, applies Harbor's
own task patch, rewrites both Dockerfiles to VTB's fixed offline base, writes manifests,
and only then replaces the unpatched tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from eval.runner.tau3_assets import (  # noqa: E402
    DATASET_SUBDIR,
    HARBOR_COMMIT,
    PATCH_MANIFEST,
    TAU2_COMMIT,
    TAU3_PATCH_SCHEMA,
    TAU2_BASE_IMAGE,
    TAU3_TASK_COUNT,
    TAU3_TASK_PREFIX,
    TASK_MANIFEST,
    TRACKED_TASK_FILES,
    sha256_file,
    validate_export,
)

STATIC_ASSETS = (
    Path("environment/docker-compose.yaml"),
    Path("environment/runtime-server/server.py"),
    Path("tests/evaluate.py"),
)

REQUIRED_TAU_ENV_NAMES = (
    "TAU2_USER_API_KEY",
    "TAU2_USER_BASE_URL",
    "TAU2_USER_MODEL",
    "TAU2_USER_REASONING_EFFORT",
    "TAU2_USER_TEMPERATURE",
    "TAU2_USER_LLM_ARGS_JSON",
    "TAU2_NL_ASSERTIONS_BASE_URL",
    "TAU2_NL_ASSERTIONS_MODEL",
    "TAU2_NL_ASSERTIONS_LLM_ARGS_JSON",
)

MAIN_DOCKERFILE = f'''ARG TAU2_BASE_IMAGE="{TAU2_BASE_IMAGE}"
FROM ${{TAU2_BASE_IMAGE}}

WORKDIR /app
'''

RUNTIME_DOCKERFILE = f'''ARG TAU2_BASE_IMAGE="{TAU2_BASE_IMAGE}"
FROM ${{TAU2_BASE_IMAGE}}

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY server.py ./
COPY task_config.json ./

EXPOSE 8000
CMD ["python3", "server.py"]
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--harbor-repo", required=True, type=Path)
    parser.add_argument(
        "--expected-tasks", type=int, default=TAU3_TASK_COUNT,
        help="test-only override; production exports must retain the 375-task default",
    )
    return parser.parse_args()


def _dataset_root(export_dir: Path) -> Path:
    nested = export_dir / DATASET_SUBDIR
    if nested.is_dir():
        return nested
    if any(export_dir.glob(f"{TAU3_TASK_PREFIX}*/task.toml")):
        return export_dir
    raise SystemExit(
        f"no Tau3 dataset tree below {export_dir}; expected {nested} or direct task directories"
    )


def _load_harbor_task_patcher(harbor_repo: Path) -> Callable[[list[Path]], None]:
    """Wrap the pinned Harbor patch instead of maintaining a second copy in VTB."""
    sys.path.insert(0, str(harbor_repo / "src"))
    from harbor.job import Job
    from harbor.models.task.id import PackageTaskId
    from harbor.tasks.client import TaskDownloadResult

    def patch(task_dirs: list[Path]) -> None:
        results = {
            PackageTaskId(org="sierra-research", name=task.name, ref="latest"):
            TaskDownloadResult(path=task, download_time_sec=0.0, cached=True)
            for task in task_dirs
        }
        Job._patch_tau3_bench_cached_tasks(results)

    return patch


def _ordered_task_dirs(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and (path / "task.toml").is_file()),
        key=lambda path: path.name,
    )


def _task_digests(task_dirs: list[Path]) -> dict[str, dict[str, str]]:
    return {
        task.name: {relative: sha256_file(task / relative) for relative in TRACKED_TASK_FILES}
        for task in task_dirs
    }


def _assert_no_runtime_downloads(task_dirs: list[Path]) -> None:
    forbidden = ("apt-get", "git clone", "pip install", "uv pip", "curl ", "wget ")
    for task in task_dirs:
        for relative in ("environment/Dockerfile", "environment/runtime-server/Dockerfile"):
            text = (task / relative).read_text(encoding="utf-8").lower()
            found = [token for token in forbidden if token in text]
            if found:
                raise ValueError(f"{task / relative}: runtime download commands remain: {found}")
            if TAU2_BASE_IMAGE.lower() not in text:
                raise ValueError(f"{task / relative}: fixed Tau3 task base is missing")


def _patch_tree(
    root: Path,
    template: Path,
    expected_tasks: int,
    harbor_commit: str,
    patch_cached_tasks: Callable[[list[Path]], None],
) -> None:
    task_dirs = _ordered_task_dirs(root)
    if len(task_dirs) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} Tau3 tasks, found {len(task_dirs)} under {root}")
    if len({task.name for task in task_dirs}) != len(task_dirs) or any(
        not task.name.startswith(TAU3_TASK_PREFIX) for task in task_dirs
    ):
        raise ValueError("Tau3 export contains duplicate or non-canonical task directories")

    # Harbor owns task.toml and adapter static-asset semantics. VTB only adds the
    # reproducible local-image layer and export transaction around that implementation.
    patch_cached_tasks(task_dirs)
    for task in task_dirs:
        task_toml = (task / "task.toml").read_text(encoding="utf-8")
        missing_env = [name for name in REQUIRED_TAU_ENV_NAMES if name not in task_toml]
        if missing_env:
            raise ValueError(f"{task}: Harbor Tau3 patch is missing env entries: {missing_env}")
        for relative in STATIC_ASSETS:
            source = template / relative
            destination = task / relative
            if not source.is_file():
                raise ValueError(f"pinned Harbor template is incomplete: {source}")
            if not destination.is_file() or sha256_file(destination) != sha256_file(source):
                raise ValueError(f"Harbor Tau3 patch left a missing/stale asset: {destination}")
        (task / "environment/Dockerfile").write_text(MAIN_DOCKERFILE, encoding="utf-8")
        (task / "environment/runtime-server/Dockerfile").write_text(
            RUNTIME_DOCKERFILE, encoding="utf-8"
        )

    _assert_no_runtime_downloads(task_dirs)
    ids = [task.name for task in task_dirs]
    (root / TASK_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "sierra-research/tau3-bench",
                "tasks": [{"task_id": task_id} for task_id in ids],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    task_ids_sha256 = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    patch = {
        "schema_version": TAU3_PATCH_SCHEMA,
        "harbor_commit": harbor_commit,
        "tau2_commit": TAU2_COMMIT,
        "task_base_image": TAU2_BASE_IMAGE,
        "task_count": len(ids),
        "task_ids_sha256": task_ids_sha256,
        "template_sha256": {str(path): sha256_file(template / path) for path in STATIC_ASSETS},
        "tasks": _task_digests(task_dirs),
    }
    (root / PATCH_MANIFEST).write_text(
        json.dumps(patch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    export_root = args.export_dir.expanduser().resolve()
    source_root = _dataset_root(export_root)
    harbor_repo = args.harbor_repo.expanduser().resolve()
    template = harbor_repo / "adapters/tau3-bench/src/tau3_bench/task-template"
    if not template.is_dir():
        raise SystemExit(f"Harbor Tau3 template is missing: {template}")
    harbor_commit = subprocess.check_output(
        ["git", "-C", str(harbor_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if harbor_commit != HARBOR_COMMIT:
        raise SystemExit(f"expected Harbor {HARBOR_COMMIT}, found {harbor_commit}")

    patch_cached_tasks = _load_harbor_task_patcher(harbor_repo)
    parent = source_root.parent
    stage = Path(tempfile.mkdtemp(prefix=f".{source_root.name}.patching-", dir=parent))
    backup = parent / f".{source_root.name}.unpatched-{os.getpid()}"
    try:
        shutil.rmtree(stage)
        shutil.copytree(source_root, stage, symlinks=False)
        _patch_tree(
            stage, template, args.expected_tasks, harbor_commit, patch_cached_tasks
        )
        if args.expected_tasks == TAU3_TASK_COUNT:
            validate_export(stage)
        os.replace(source_root, backup)
        try:
            os.replace(stage, source_root)
        except BaseException:
            os.replace(backup, source_root)
            raise
        shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(
        f"[tau3] patched {args.expected_tasks} tasks atomically; Harbor {harbor_commit}; "
        f"source={source_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
