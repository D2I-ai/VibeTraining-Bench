"""Contract for the prefetched, patched Tau3 task tree.

Tau3 is registry-backed upstream, but VTB deliberately evaluates a local export: the
registry packages lag the pinned Harbor adapter and Harbor's cold-cache patch runs before
all package files exist.  A local tree makes download -> patch -> validate an explicit
setup transaction and keeps evaluation independent of ~/.cache/harbor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from eval.runner.backend_pins import HARBOR_COMMIT

TAU3_BENCHMARK = "sierra-research__tau3-bench"
TAU3_DATASET = "sierra-research/tau3-bench"
TAU3_TASK_PREFIX = "tau3-bench__"
TAU3_TASK_COUNT = 375
TAU3_PATCH_SCHEMA = "vtb.tau3-task-export.v1"

TAU2_COMMIT = "043b1989e5869ecd231a753d439287fddbf64dcf"
TAU2_PR_REF = "refs/pull/469/head"
TAU2_BASE_IMAGE = "vibetrainbench/tau2-bench:043b1989"

DATASET_SUBDIR = "tau3-bench"
PATCH_MANIFEST = ".vibetrainbench_tau3_patch.json"
TASK_MANIFEST = "manifest.json"

TRACKED_TASK_FILES = (
    "task.toml",
    "environment/Dockerfile",
    "environment/runtime-server/Dockerfile",
    "environment/docker-compose.yaml",
    "environment/runtime-server/server.py",
    "tests/evaluate.py",
)


class Tau3AssetError(ValueError):
    """The local Tau3 export is absent, stale, or internally inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_task_ids(root: Path) -> list[str]:
    manifest = root / TASK_MANIFEST
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Tau3AssetError(f"Tau3 prefetch manifest is missing: {manifest}") from exc
    except ValueError as exc:
        raise Tau3AssetError(f"Tau3 prefetch manifest is invalid JSON: {manifest}: {exc}") from exc
    values = payload.get("tasks") if isinstance(payload, dict) else None
    ids = [str(item.get("task_id") or "") for item in values or [] if isinstance(item, dict)]
    if len(ids) != TAU3_TASK_COUNT or len(set(ids)) != TAU3_TASK_COUNT or any(
        not item.startswith(TAU3_TASK_PREFIX) for item in ids
    ):
        raise Tau3AssetError(
            f"Tau3 manifest must list {TAU3_TASK_COUNT} unique canonical task ids; found {len(ids)}"
        )
    return ids


def _load_patch_manifest(root: Path) -> dict[str, Any]:
    path = root / PATCH_MANIFEST
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Tau3AssetError(f"Tau3 patch manifest is missing: {path}") from exc
    except ValueError as exc:
        raise Tau3AssetError(f"Tau3 patch manifest is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Tau3AssetError(f"Tau3 patch manifest must be a JSON object: {path}")
    return payload


def validate_export(root: Path, *, verify_files: bool = True) -> dict[str, Any]:
    """Validate a canonical Tau3 export and return its patch manifest.

    Full hashing is intentional in preflight and at evaluator launch.  The files are
    small task definitions; hashing them is cheap compared with discovering stale task
    semantics after a multi-hour evaluation.
    """
    root = root.expanduser().resolve()
    ids = ordered_task_ids(root)
    patch = _load_patch_manifest(root)
    expected = {
        "schema_version": TAU3_PATCH_SCHEMA,
        "harbor_commit": HARBOR_COMMIT,
        "tau2_commit": TAU2_COMMIT,
        "task_base_image": TAU2_BASE_IMAGE,
        "task_count": TAU3_TASK_COUNT,
    }
    for key, value in expected.items():
        if patch.get(key) != value:
            raise Tau3AssetError(
                f"Tau3 patch manifest {key}={patch.get(key)!r}; expected {value!r}. "
                "Re-run the Tau3 prefetch."
            )
    if patch.get("task_ids_sha256") != hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest():
        raise Tau3AssetError("Tau3 task order digest does not match manifest.json")

    recorded = patch.get("tasks") or {}
    if set(recorded) != set(ids):
        raise Tau3AssetError("Tau3 patch manifest task set does not match manifest.json")
    if verify_files:
        for task_id in ids:
            task_dir = root / task_id
            files = recorded.get(task_id) or {}
            for relative in TRACKED_TASK_FILES:
                path = task_dir / relative
                expected_digest = files.get(relative)
                if not path.is_file():
                    raise Tau3AssetError(f"Tau3 task is incomplete: {path}")
                if not expected_digest or sha256_file(path) != expected_digest:
                    raise Tau3AssetError(f"Tau3 task file digest mismatch: {path}")
    return patch
