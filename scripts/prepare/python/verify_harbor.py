#!/usr/bin/env python3
"""Prepare and verify the pinned Harbor CLI and matching source checkout."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from eval.runner.backend_pins import HARBOR_COMMIT, HARBOR_PIN, HARBOR_REPO_URL  # noqa: E402


def _run(*args: str, cwd: Optional[Path] = None) -> str:
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _normalized_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if value.startswith("git+"):
        value = value[4:]
    parsed = urlsplit(value)
    path = parsed.path[:-4] if parsed.path.endswith(".git") else parsed.path
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path.rstrip("/"), "", ""))


def verify_install_local() -> None:
    try:
        dist = metadata.distribution("harbor")
    except metadata.PackageNotFoundError as exc:
        raise SystemExit("FATAL: Harbor is not installed in this Python environment") from exc
    raw = dist.read_text("direct_url.json")
    if not raw:
        raise SystemExit("FATAL: installed Harbor has no direct_url.json; expected the pinned git install")
    try:
        direct = json.loads(raw)
        vcs = direct["vcs_info"]
        installed_url = direct["url"]
        installed_commit = str(vcs["commit_id"]).lower()
        requested = str(vcs.get("requested_revision") or "").lower()
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"FATAL: invalid Harbor direct_url.json: {exc}") from exc
    if vcs.get("vcs") != "git":
        raise SystemExit(f"FATAL: installed Harbor VCS is {vcs.get('vcs')!r}, expected 'git'")
    if _normalized_url(installed_url) != _normalized_url(HARBOR_REPO_URL):
        raise SystemExit(
            f"FATAL: installed Harbor source is {installed_url}, expected {HARBOR_REPO_URL}"
        )
    if installed_commit != HARBOR_COMMIT or requested != HARBOR_COMMIT:
        raise SystemExit(
            "FATAL: installed Harbor commit/requested revision "
            f"is {installed_commit}/{requested}, expected {HARBOR_COMMIT}"
        )
    print(f"HARBOR_INSTALL_OK version={dist.version} commit={installed_commit} source={installed_url}")


def verify_install(harbor_bin: Path) -> None:
    try:
        first_line = harbor_bin.resolve().read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError) as exc:
        raise SystemExit(f"FATAL: cannot read Harbor launcher {harbor_bin}: {exc}") from exc
    if not first_line.startswith("#!"):
        raise SystemExit(f"FATAL: Harbor launcher has no Python shebang: {harbor_bin}")
    interpreter = Path(first_line[2:].strip())
    if not interpreter.is_file():
        raise SystemExit(f"FATAL: Harbor launcher Python is missing: {interpreter}")
    subprocess.run(
        [str(interpreter), str(Path(__file__).resolve()), "verify-install-local"],
        check=True,
    )


def _origin(checkout: Path) -> str:
    try:
        return _run("git", "remote", "get-url", "origin", cwd=checkout)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"FATAL: Harbor checkout has no readable origin: {checkout}: {exc}") from exc


def verify_source(checkout: Path) -> None:
    checkout = checkout.expanduser().resolve()
    if not (checkout / ".git").exists():
        raise SystemExit(f"FATAL: Harbor source checkout is missing: {checkout}")
    origin = _origin(checkout)
    if _normalized_url(origin) != _normalized_url(HARBOR_REPO_URL):
        raise SystemExit(f"FATAL: Harbor origin is {origin}, expected {HARBOR_REPO_URL}")
    status = _run("git", "status", "--porcelain", cwd=checkout)
    if status:
        raise SystemExit(f"FATAL: Harbor checkout has uncommitted changes: {checkout}")
    head = _run("git", "rev-parse", "HEAD", cwd=checkout).lower()
    if head != HARBOR_COMMIT:
        raise SystemExit(f"FATAL: Harbor checkout is {head}, expected {HARBOR_COMMIT}")
    if not (checkout / "adapters").is_dir():
        raise SystemExit(f"FATAL: Harbor checkout has no adapters directory: {checkout}")
    print(f"HARBOR_SOURCE_OK checkout={checkout} commit={head} source={origin}")


def prepare_source(checkout: Path) -> None:
    checkout = checkout.expanduser().resolve()
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", HARBOR_REPO_URL, str(checkout)], check=True)
    if not (checkout / ".git").exists():
        raise SystemExit(f"FATAL: Harbor source path exists but is not a git checkout: {checkout}")
    origin = _origin(checkout)
    if _normalized_url(origin) != _normalized_url(HARBOR_REPO_URL):
        raise SystemExit(f"FATAL: Harbor origin is {origin}, expected {HARBOR_REPO_URL}")
    if _run("git", "status", "--porcelain", cwd=checkout):
        raise SystemExit(f"FATAL: Harbor checkout has uncommitted changes: {checkout}")
    try:
        _run("git", "cat-file", "-e", f"{HARBOR_COMMIT}^{{commit}}", cwd=checkout)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "fetch", "origin", HARBOR_COMMIT], cwd=checkout, check=True)
    if _run("git", "rev-parse", "HEAD", cwd=checkout).lower() != HARBOR_COMMIT:
        subprocess.run(["git", "checkout", "--detach", HARBOR_COMMIT], cwd=checkout, check=True)
    verify_source(checkout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pin_parser = subparsers.add_parser("pin")
    pin_parser.add_argument("--field", choices=("url", "commit", "requirement"), required=True)

    install_parser = subparsers.add_parser("verify-install")
    install_parser.add_argument("--harbor-bin", type=Path, required=True)
    subparsers.add_parser("verify-install-local")

    source_parser = subparsers.add_parser("verify-source")
    source_parser.add_argument("--checkout", type=Path, required=True)
    prepare_parser = subparsers.add_parser("prepare-source")
    prepare_parser.add_argument("--checkout", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "pin":
        print({"url": HARBOR_PIN.url, "commit": HARBOR_PIN.commit, "requirement": HARBOR_PIN.requirement}[args.field])
    elif args.command == "verify-install":
        verify_install(args.harbor_bin)
    elif args.command == "verify-install-local":
        verify_install_local()
    elif args.command == "verify-source":
        verify_source(args.checkout)
    elif args.command == "prepare-source":
        prepare_source(args.checkout)


if __name__ == "__main__":
    main()
