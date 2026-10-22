"""Editable-harness submission helpers.

A benchmark may let the agent deliver a custom agent harness next to
final_model; these functions resolve that declaration, snapshot the harness
into the official sandbox, and decide scratch cleanup. Used by the service
core; split out so the editable-harness surface is one readable file.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Optional
from .. import config
from ..store import RunContext


def _editable_harness_submission(ctx: RunContext, model_path: Path) -> Optional[dict[str, Any]]:
    """Validate the harness paired with an editable benchmark's model directory.

    The API deliberately accepts no harness path. A candidate is always
    ``<candidate-root>/final_model`` plus ``<candidate-root>/harness`` so an agent
    cannot point grading at another run's code after the model path was confined by
    the token sandbox.
    """
    # A run may opt out of its benchmark's editable harness at registration (a
    # model-only ablation); its submissions are then plain model deliveries.
    if getattr(ctx, "editable_harness", None) == "off":
        return None
    metadata = config.editable_harness(ctx.task)
    if not metadata:
        return None
    candidate_root = model_path.parent
    harness = candidate_root / "harness"
    allowed = set(metadata.get("allowed_files") or [])
    import_path = metadata.get("import_path")
    if not isinstance(import_path, str) or not import_path.startswith("harness.") or ":" not in import_path:
        raise ValueError(f"editable harness has an unsupported import path: {import_path!r}")
    if not allowed or not harness.is_dir():
        raise ValueError("editable harness is missing beside the submitted model")

    files: list[Path] = []
    for path in harness.rglob("*"):
        if path.is_symlink():
            raise ValueError("editable harness may not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("editable harness contains a non-regular file")
        relative = path.relative_to(harness).as_posix()
        if relative not in allowed:
            raise ValueError(f"editable harness contains undeclared file: {relative}")
        files.append(path)
    missing = sorted(name for name in allowed if not (harness / name).is_file())
    if missing:
        raise ValueError(f"editable harness is missing required files: {', '.join(missing)}")
    if len(files) > 32 or sum(path.stat().st_size for path in files) > 8 * 1024 * 1024:
        raise ValueError("editable harness exceeds the file-count or size limit")
    module_name, class_name = import_path.split(":", 1)
    entry = harness / (module_name.split(".", 1)[1] + ".py")
    try:
        tree = ast.parse(entry.read_text(encoding="utf-8"), filename=str(entry))
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"editable harness entry is not valid Python: {exc}") from exc
    agent_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    base_name = str(metadata.get("base_class") or "BaseAgent")
    if agent_class is None or not any(
        (isinstance(base, ast.Name) and base.id == base_name)
        or (isinstance(base, ast.Attribute) and base.attr == base_name)
        for base in agent_class.bases
    ):
        raise ValueError(f"editable harness entry must define {class_name}({base_name})")

    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(harness).as_posix()):
        digest.update(path.relative_to(harness).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    metadata_name = "final_candidate.json" if ctx.role == "official" else "candidate.json"
    metadata_path = candidate_root / metadata_name
    if not metadata_path.is_file():
        raise ValueError(f"editable harness submission requires {metadata_name} beside the candidate")
    try:
        candidate_id = str((json.loads(metadata_path.read_text(encoding="utf-8")) or {}).get("candidate_id") or "")
    except (OSError, ValueError) as exc:
        raise ValueError(f"{metadata_name} is not valid JSON") from exc
    if not candidate_id:
        raise ValueError(f"{metadata_name} must contain candidate_id")
    return {
        "candidate_root": candidate_root,
        "source": harness,
        "import_path": import_path,
        "fingerprint": digest.hexdigest(),
        "candidate_id": candidate_id,
    }


def _snapshot_harness(submission: dict[str, Any], destination: Path) -> Path:
    target = destination / "harness"
    shutil.copytree(submission["source"], target)
    for path in target.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    target.chmod(0o555)
    return target


def _discard_success_scratch(work_dir: Path, harness_snapshot: Optional[Path]) -> None:
    """Remove large evaluation outputs but retain the immutable submitted harness."""
    preserved = harness_snapshot.parent if harness_snapshot is not None else None
    for child in work_dir.iterdir():
        if child == preserved:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
