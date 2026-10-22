#!/usr/bin/env python3
"""Verify that every HF dataset declared by a recipe is present in the local cache."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from eval.runner import resource_assets as assets, runtime_paths  # noqa: E402


def download_recipe(recipe_path: str) -> list[str]:
    """Explicit operator action: fetch only training files at manifest revisions."""
    recipe = json.loads(assets.recipe_file(REPO, recipe_path).read_text())
    cache = assets.hf_home(REPO)
    cache.mkdir(parents=True, exist_ok=True)
    errors = []
    for resource_id in recipe["resources"]:
        _, cfg = assets.read_resource(REPO, resource_id)
        if cfg.get("pending"):
            errors.append(f"{resource_id}: pending resource: {cfg['pending']}")
            continue
        if cfg.get("type") != "hf_cache_dataset":
            continue
        patterns = [alternative for pattern in assets.dataset_patterns(cfg)
                    for alternative in assets.expand_file_pattern(pattern)]
        command = ["docker", "--host", runtime_paths.docker_host(), "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
                   "-v", f"{cache}:/hf", "-e", "HF_HOME=/hf", "-e", "HOME=/tmp",
                   "--entrypoint", "hf", os.environ.get("VTB_TRAINER_IMAGE", "vibetrainbench/trainer:latest"),
                   "download", "--repo-type", "dataset", cfg["repo_id"],
                   "--revision", cfg["revision"]]
        for pattern in patterns:
            command.extend(["--include", pattern])
        print(f"[dataset] {resource_id} revision={cfg['revision']} cache={cache}", flush=True)
        subprocess.run(command, check=True)
        errors.extend(assets.dataset_errors(REPO, cfg))
    return errors


def check_sources(recipe_path: str) -> list[str]:
    """Offline raw-input gate; does not require models or derived payloads yet."""
    recipe = json.loads(assets.recipe_file(REPO, recipe_path).read_text())
    errors = []
    for resource_id in recipe["resources"]:
        _, cfg = assets.read_resource(REPO, resource_id)
        if cfg.get("type") == "hf_cache_dataset":
            errors.extend(assets.resource_errors(REPO, resource_id))
    return errors


def check_recipe(recipe_path: str) -> list[str]:
    return assets.recipe_errors(REPO, recipe_path)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--download", action="store_true", help="explicitly download declared training files")
    modes.add_argument("--check-sources", action="store_true", help="check cached raw HF datasets without downloading or requiring conversion")
    parser.add_argument("recipes", nargs="+")
    args = parser.parse_args()
    all_errors: list[str] = []
    for recipe_path in args.recipes:
        errors = download_recipe(recipe_path) if args.download else (check_sources(recipe_path) if args.check_sources else check_recipe(recipe_path))
        all_errors.extend(errors)
    if all_errors:
        print("RECIPE_CACHE_CHECK_FAILED", file=sys.stderr)
        for error in all_errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)
    marker = "TRAINING_DATA_DOWNLOAD_OK" if args.download else ("TRAINING_SOURCES_OK" if args.check_sources else "RECIPE_CACHE_OK")
    print(f"{marker} recipes={','.join(args.recipes)}")


if __name__ == "__main__":
    main()
