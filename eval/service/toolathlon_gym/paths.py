"""Path resolution for the Toolathlon-GYM service.

The service drives an *external, read-only* official gym checkout, named by
VTB_TOOLATHLON_GYM_REPO. This mirrors how eval/runner/task_backends.py resolves
VTB_HARBOR_REPO: the backend owns its own repo resolution, so the generic
eval/runner/runtime_paths.py never has to learn about it.

The official checkout is never written to. Our customisations live in this
package's overlay/ and are layered on top of a throwaway *staged* copy
(official tree + overlay files), which is what both the image build and the
per-task bind-mount use. The checkout is required to match
``PINNED_UPSTREAM_COMMIT`` exactly so an upstream update cannot silently change
the source underneath the overlay.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# eval/service/toolathlon_gym/paths.py -> parents[3] is the vibetrainbench checkout.
_THIS = Path(__file__).resolve()
REPO_ROOT = Path(os.environ.get("VTB_REPO") or _THIS.parents[3]).expanduser()

# Disposable working area, gitignored like everything else under vtb-env/.
VTB_ROOT = Path(os.environ.get("VTB_ROOT") or (REPO_ROOT / "vtb-env")).expanduser()

OVERLAY_DIR = _THIS.parent / "overlay"

_ENV_REPO = "VTB_TOOLATHLON_GYM_REPO"

# The overlay is maintained against this exact upstream revision. A different
# checkout can silently change APIs or file layouts underneath the overlay.
PINNED_UPSTREAM_COMMIT = "45bbff81419fedbb6d5a2cb46029d2980a7c93c4"

# Junk that must never ride into the staged tree (build artefacts, VCS, caches).
_STAGE_IGNORE = shutil.ignore_patterns(
    ".git", ".venv", "venv", "__pycache__", "*.pyc", "*.pyo", ".cache",
    "node_modules", "dumps", "eval_output", "*.log", "*.pid",
)


def _looks_like_gym_checkout(path: Path) -> bool:
    """A gym checkout has main.py and a utils/ package at its root."""
    return (path / "main.py").is_file() and (path / "utils").is_dir()


def _git(path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(
            f"{_ENV_REPO}={path} must be a git checkout at pinned commit "
            f"{PINNED_UPSTREAM_COMMIT}: {detail.strip()}"
        ) from exc
    return result.stdout.strip()


def _validate_upstream_revision(path: Path) -> None:
    actual = _git(path, "rev-parse", "HEAD")
    if actual != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError(
            f"{_ENV_REPO}={path} is at {actual}, but this overlay requires "
            f"{PINNED_UPSTREAM_COMMIT}"
        )
    dirty = _git(path, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            f"{_ENV_REPO}={path} has local changes; the pinned official checkout "
            "must be clean before applying the overlay"
        )


def official_repo() -> Path:
    """Return the clean official checkout at the overlay's pinned revision."""
    raw = os.environ.get(_ENV_REPO)
    if not raw:
        raise RuntimeError(
            f"{_ENV_REPO} is not set; point it at the official toolathlon_gym "
            "checkout (the read-only upstream this service drives)."
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{_ENV_REPO}={path} is not a directory")
    if not _looks_like_gym_checkout(path):
        raise RuntimeError(
            f"{_ENV_REPO}={path} does not look like a gym checkout "
            "(expected main.py and utils/ at its root)."
        )
    _validate_upstream_revision(path)
    return path


def overlay_dir() -> Path:
    return OVERLAY_DIR


def staged_root() -> Path:
    """Where the staged (official + overlay) tree lives. Location only."""
    return VTB_ROOT / "toolathlon_gym_staged"


def build_staged_root(force: bool = False) -> Path:
    """Materialise the pinned official tree plus overlay into a staged root."""
    official = official_repo()
    staged = staged_root()
    revision_file = staged.with_name(f"{staged.name}.upstream_commit")
    staged_revision = ""
    if revision_file.is_file():
        staged_revision = revision_file.read_text(encoding="utf-8").strip()
    if force or not staged.exists() or staged_revision != PINNED_UPSTREAM_COMMIT:
        if staged.exists():
            shutil.rmtree(staged)
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(official, staged, ignore=_STAGE_IGNORE)
        revision_file.write_text(PINNED_UPSTREAM_COMMIT + "\n", encoding="utf-8")
    if OVERLAY_DIR.is_dir():
        _apply_overlay(OVERLAY_DIR, staged)
    return staged


def _apply_overlay(src: Path, dst: Path) -> None:
    """Copy overlay files onto the staged tree, overwriting in place."""
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def describe() -> str:
    lines = [
        f"{'pinned_upstream_commit':28} {PINNED_UPSTREAM_COMMIT}",
        f"{_ENV_REPO:28} {os.environ.get(_ENV_REPO) or '(unset)'}",
    ]
    try:
        lines.append(f"{'official_repo':28} {official_repo()}")
    except RuntimeError as exc:
        lines.append(f"{'official_repo':28} ERROR: {exc}")
    lines.append(f"{'overlay_dir':28} {OVERLAY_DIR} (exists={OVERLAY_DIR.is_dir()})")
    lines.append(f"{'staged_root':28} {staged_root()}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
