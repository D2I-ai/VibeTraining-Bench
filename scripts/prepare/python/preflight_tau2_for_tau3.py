#!/usr/bin/env python3
"""Local readiness checks for the Tau2-backed Tau3 benchmark; performs no downloads."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from eval.runner import runtime_paths  # noqa: E402
from eval.runner.tau3_assets import (  # noqa: E402
    HARBOR_COMMIT,
    TAU2_BASE_IMAGE,
    TAU2_COMMIT,
    TAU3_TASK_COUNT,
    Tau3AssetError,
    validate_export,
)

REQUIRED_PROVIDER_VARS = (
    "TAU2_USER_API_KEY",
    "TAU2_USER_BASE_URL",
    "TAU2_USER_MODEL",
    "TAU2_NL_ASSERTIONS_BASE_URL",
    "TAU2_NL_ASSERTIONS_MODEL",
)

def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def _check_port(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true", help="also require the published Tau3 split")
    parser.add_argument("--state", type=Path, help="state path that must not already exist")
    parser.add_argument("--port", type=int, default=5830, help="grading port that must be free")
    args = parser.parse_args()

    errors: list[str] = []
    notes: list[str] = []

    harbor_bin = runtime_paths.TOOLS_BIN / "harbor"
    pin_helper = REPO / "scripts/prepare/python/verify_harbor.py"
    if not harbor_bin.is_file():
        errors.append(f"pinned Harbor CLI is missing: {harbor_bin}")
    else:
        try:
            notes.append(_run(sys.executable, str(pin_helper), "verify-install", "--harbor-bin", str(harbor_bin)))
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"installed Harbor provenance check failed: {exc}")

    harbor = Path(os.environ.get("VTB_HARBOR_REPO") or REPO.parent / "harbor")
    tau2 = Path(os.environ.get("TAU2_BENCH_REPO_DIR") or REPO.parent / "tau2-bench")
    for name, path, expected in (("Harbor", harbor, HARBOR_COMMIT), ("Tau2", tau2, TAU2_COMMIT)):
        try:
            found = _run("git", "-C", str(path), "rev-parse", "HEAD")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"{name} checkout unavailable at {path}: {exc}")
        else:
            if found != expected:
                errors.append(f"{name} checkout is {found}; expected {expected}")

    if shutil.which("docker") is None:
        errors.append("docker is not on PATH")
    else:
        try:
            raw = _run(
                "docker", "--host", runtime_paths.docker_host(),
                "image", "inspect", TAU2_BASE_IMAGE,
            )
            image = json.loads(raw)[0]
            labels = (image.get("Config") or {}).get("Labels") or {}
            expected_labels = {
                "org.opencontainers.image.revision": TAU2_COMMIT,
                "org.vibetrainbench.litellm.version": "1.83.0",
                "org.vibetrainbench.mcp.version": "1.25.0",
                "org.vibetrainbench.fastmcp.version": "3.0.0",
            }
            for key, value in expected_labels.items():
                if labels.get(key) != value:
                    errors.append(f"{TAU2_BASE_IMAGE} label {key}={labels.get(key)!r}; expected {value!r}")
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as exc:
            errors.append(f"Tau2 base image is missing/stale: {exc}")

    try:
        patch = validate_export(runtime_paths.TAU3_TASK_SPECS_DIR)
        notes.append(f"validated {patch['task_count']} patched tasks")
    except Tau3AssetError as exc:
        errors.append(str(exc))

    for key in REQUIRED_PROVIDER_VARS:
        if not os.environ.get(key, "").strip():
            errors.append(f"required provider variable is missing: {key}")
    # Empty is an intentional override that disables Tau2's incompatible default "low".
    if "TAU2_USER_REASONING_EFFORT" not in os.environ:
        errors.append("TAU2_USER_REASONING_EFFORT must be set (empty is valid for custom models)")

    from eval.runner import resource_assets as assets
    _, model_cfg = assets.read_resource(REPO, "models/qwen3_5_9b_base")
    errors.extend(assets.snapshot_errors(REPO, model_cfg))

    if args.state and args.state.expanduser().exists():
        errors.append(f"grading state already exists; use a fresh path: {args.state}")
    if not _check_port(args.port):
        errors.append(f"grading port is already listening: {args.port}")

    if args.formal:
        from eval.grading import config, valsplit
        from eval.runner.resource_assets import recipe_errors

        source = config.VAL_SOURCE_RAW or config.validation_source("sierra-research__tau3-bench")
        try:
            parsed = valsplit.parse_val_source(valsplit.expand_val_source(REPO, source or ""))
            if parsed.kind != "json":
                errors.append("formal Tau3 requires an explicit JSON validation/test split")
        except ValueError as exc:
            errors.append(f"formal Tau3 split is invalid: {exc}")
        errors.extend(recipe_errors(REPO, os.environ.get("RECIPE") or "qwen3_5_9b/slime_sft_tau"))
    else:
        notes.append("the published formal split was not required (smoke mode)")

    if errors:
        print("TAU2_PREFLIGHT_FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("TAU2_PREFLIGHT_OK")
    for note in notes:
        print(f"  - {note}")
    print(f"  - task source: {runtime_paths.TAU3_TASK_SPECS_DIR}")
    print(f"  - grading port {args.port} is free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
