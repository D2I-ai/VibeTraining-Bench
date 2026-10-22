"""Path resolution for the Toolathlon-VERIFIED service.

The service drives an *external, read-only* official verified checkout, named
by VTB_TOOLATHLON_VERIFIED_REPO. This mirrors how eval/runner/task_backends.py
resolves VTB_HARBOR_REPO: the backend owns its own repo resolution, so the
generic eval/runner/runtime_paths.py never has to learn about it.

The official checkout is never written to. The staged tree that the per-task
containers actually run is layered in three passes (plan's three-layer model):

    1. official raw_code   copied from VTB_TOOLATHLON_VERIFIED_REPO (read-only)
    2. vibetrainbench overlay   this package's overlay/ (our customisations)
    3. per-host secrets    VTB_TOOLATHLON_VERIFIED_SECRETS (gitignored configs:
                           token_key_session.py, gcp-oauth.keys.json,
                           google_credentials.json, snowflake_rsa_key*, ...)

The checkout must match ``PINNED_UPSTREAM_COMMIT`` exactly so an upstream update
cannot silently change the source underneath the overlay; per-host secrets are
still applied last.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# eval/service/toolathlon_verified/paths.py -> parents[3] is the vibetrainbench checkout.
_THIS = Path(__file__).resolve()
REPO_ROOT = Path(os.environ.get("VTB_REPO") or _THIS.parents[3]).expanduser()

# Disposable working area, gitignored like everything else under vtb-env/.
VTB_ROOT = Path(os.environ.get("VTB_ROOT") or (REPO_ROOT / "vtb-env")).expanduser()

OVERLAY_DIR = _THIS.parent / "overlay"

_ENV_REPO = "VTB_TOOLATHLON_VERIFIED_REPO"
_ENV_SECRETS = "VTB_TOOLATHLON_VERIFIED_SECRETS"

# The overlay is maintained against this exact upstream revision. A different
# checkout can silently change APIs or file layouts underneath the overlay.
PINNED_UPSTREAM_COMMIT = "9be8d8fe07a497b18ee61e3f2ae694e9797f39eb"

# Junk that must never ride into the staged tree (build artefacts, VCS, caches).
_STAGE_IGNORE = shutil.ignore_patterns(
    ".git", ".venv", "venv", "__pycache__", "*.pyc", "*.pyo", ".cache",
    "node_modules", "dumps", "eval_output", "*.log", "*.pid",
)


def _looks_like_verified_checkout(path: Path) -> bool:
    """A verified checkout has main.py, scripts/ and tasks/finalpool/."""
    return (
        (path / "main.py").is_file()
        and (path / "scripts").is_dir()
        and (path / "tasks" / "finalpool").is_dir()
    )


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
            f"{_ENV_REPO} is not set; point it at the official Toolathlon_verified "
            "checkout (the read-only upstream this service drives)."
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{_ENV_REPO}={path} is not a directory")
    if not _looks_like_verified_checkout(path):
        raise RuntimeError(
            f"{_ENV_REPO}={path} does not look like a verified checkout "
            "(expected main.py, scripts/ and tasks/finalpool/ at its root)."
        )
    _validate_upstream_revision(path)
    return path


def overlay_dir() -> Path:
    return OVERLAY_DIR


def secrets_dir() -> Path | None:
    """Per-host secrets tree overlaid last, or None when unset.

    Optional on purpose: a host running only non-credentialed tasks needs no
    secrets. When set it must be a directory whose layout mirrors the repo root
    (e.g. configs/token_key_session.py), so it can be copied straight on top.
    """
    raw = os.environ.get(_ENV_SECRETS)
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{_ENV_SECRETS}={path} is not a directory")
    return path


def staged_root() -> Path:
    """Where the staged (official + overlay + secrets) tree lives. Location only."""
    return VTB_ROOT / "toolathlon_verified_staged"


def build_staged_root(force: bool = False) -> Path:
    """Materialise the pinned official tree plus overlay and secrets."""
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
        _apply_layer(OVERLAY_DIR, staged)
    secrets = secrets_dir()
    if secrets is not None:
        _apply_layer(secrets, staged)
    return staged


def _apply_layer(src: Path, dst: Path) -> None:
    """Copy a layer's files onto the staged tree, overwriting in place."""
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def describe() -> str:
    lines = [
        f"{'pinned_upstream_commit':32} {PINNED_UPSTREAM_COMMIT}",
        f"{_ENV_REPO:32} {os.environ.get(_ENV_REPO) or '(unset)'}",
    ]
    try:
        lines.append(f"{'official_repo':32} {official_repo()}")
    except RuntimeError as exc:
        lines.append(f"{'official_repo':32} ERROR: {exc}")
    lines.append(f"{'overlay_dir':32} {OVERLAY_DIR} (exists={OVERLAY_DIR.is_dir()})")
    lines.append(f"{_ENV_SECRETS:32} {os.environ.get(_ENV_SECRETS) or '(unset)'}")
    lines.append(f"{'staged_root':32} {staged_root()}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
