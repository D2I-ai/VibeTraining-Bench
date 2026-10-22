"""Shared filesystem helpers for AIME dataset materializers."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable


class MaterializeError(RuntimeError):
    pass


def output_default(repo: Path, benchmark: str) -> Path:
    vtb_root = Path(os.environ.get("VTB_ROOT") or (repo / "vtb-env")).expanduser()
    hub_root = Path(
        os.environ.get("VTB_HUB_TASK_SPECS")
        or os.environ.get("VTB_HARBOR_DATASET_DIR")
        or (vtb_root / "hub_task_specs")
    ).expanduser()
    return hub_root / benchmark


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MaterializeError(f"cannot read {path}: {exc}") from exc


def materialize_atomically(
    output: Path,
    overwrite: bool,
    populate: Callable[[Path], dict],
) -> dict:
    """Populate a sibling staging tree and publish it only after validation succeeds."""
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise MaterializeError(f"{output} already exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.materializing-", dir=output.parent))
    backup = output.parent / f".{output.name}.previous-{os.getpid()}"
    try:
        result = populate(stage)
        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(stage, output)
        except BaseException:
            if backup.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)
