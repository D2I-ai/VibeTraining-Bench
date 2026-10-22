#!/usr/bin/env python3
"""Explicit model preparation: pinned HF materialization or user-started conversion."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from eval.runner import resource_assets as assets, runtime_paths  # noqa: E402

RESOURCE_ID = "models/qwen3_5_9b_base"


def normalize_checkpoint(checkpoint: Path, cfg: dict) -> None:
    path = checkpoint / "tokenizer_config.json"
    tokenizer = json.loads(path.read_text(encoding="utf-8"))
    token_ids = [int(key) for key, value in tokenizer.get("added_tokens_decoder", {}).items()
                 if value.get("content") == cfg["eos_token"]]
    if token_ids != [cfg["eos_token_id"]]:
        raise ValueError("declared EOS token ID does not match the pinned tokenizer")
    tokenizer["eos_token"] = cfg["eos_token"]
    path.write_text(json.dumps(tokenizer, indent=2, ensure_ascii=False) + "\n")
    template = tokenizer.get("chat_template")
    if isinstance(template, str):
        (checkpoint / "chat_template.jinja").write_text(template, encoding="utf-8")
    for name in ("config.json", "generation_config.json"):
        target = checkpoint / name
        if not target.is_file():
            continue
        payload = json.loads(target.read_text())
        payload["eos_token_id"] = cfg["eos_token_id"]
        if isinstance(payload.get("text_config"), dict):
            payload["text_config"]["eos_token_id"] = cfg["eos_token_id"]
        target.write_text(json.dumps(payload, indent=2) + "\n")
    (checkpoint / assets.MODEL_SOURCE_FILE).write_text(json.dumps(assets.model_signature(cfg), indent=2) + "\n")


def materialize_model(cfg: dict, *, offline: bool = False) -> None:
    destination = assets.resolve_path(REPO, cfg["hf_checkpoint_path"])
    if destination.exists():
        errors = assets.model_errors(REPO, cfg, require_conversion=False)
        if errors:
            raise ValueError(f"existing model resource has unknown/different provenance: {destination}; choose a fresh VTB_MODEL_RESOURCE_ROOT, do not overwrite it")
        print(f"BASE_MODEL_DOWNLOAD_OK already_prepared={destination}")
        return
    cache = assets.hf_home(REPO)
    if not offline:
        cache.mkdir(parents=True, exist_ok=True)
        image = os.environ.get("VTB_TRAINER_IMAGE", "vibetrainbench/trainer:latest")
        subprocess.run(["docker", "--host", runtime_paths.docker_host(), "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
                        "-v", f"{cache}:/hf", "-e", "HF_HOME=/hf", "-e", "HOME=/tmp",
                        "--entrypoint", "hf", image, "download", cfg["repo_id"],
                        "--revision", cfg["revision"]], check=True)
    snapshot = assets.resolve_path(REPO, cfg["snapshot_path"])
    if snapshot.name != cfg["revision"]:
        raise ValueError("model snapshot path does not match the declared immutable revision")
    errors = assets.snapshot_errors(REPO, cfg)
    if errors:
        raise ValueError("; ".join(errors))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "checkpoint"
        shutil.copytree(snapshot, staged, symlinks=False)
        normalize_checkpoint(staged, cfg)
        errors = assets.model_errors(REPO, dict(cfg, hf_checkpoint_path=str(staged)), require_conversion=False)
        if errors:
            raise ValueError("; ".join(errors))
        staged.rename(destination)
    print(f"BASE_MODEL_DOWNLOAD_OK revision={cfg['revision']} cache={cache} checkpoint={destination}")
    if cfg.get("torch_dist_path"):
        print("Megatron conversion is separate: bash scripts/prepare/download_base_model.sh convert")


def convert_model(cfg: dict) -> None:
    errors = assets.model_errors(REPO, cfg, require_conversion=False)
    if errors:
        raise ValueError("; ".join(errors))
    destination = assets.resolve_path(REPO, cfg["torch_dist_path"])
    if destination.exists():
        errors = assets.model_errors(REPO, cfg)
        if errors:
            raise ValueError(f"existing conversion has unknown/different provenance: {destination}; use a fresh model resource")
        print(f"MODEL_CONVERSION_OK already_prepared={destination}")
        return
    checkpoint = assets.resolve_path(REPO, cfg["hf_checkpoint_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="conversion-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "torch_dist"
        command = 'set -euo pipefail; cd /root/slime; source "scripts/models/$1"; PYTHONPATH=/root/Megatron-LM python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint /model --save /output/torch_dist'
        subprocess.run(["docker", "--host", runtime_paths.docker_host(), "run", "--rm", "--gpus", os.environ.get("VTB_CONVERSION_GPUS", "all"),
                        "-v", f"{checkpoint}:/model:ro", "-v", f"{tmp}:/output",
                        "--entrypoint", "bash", os.environ.get("VTB_TRAINER_IMAGE", "vibetrainbench/trainer:latest"),
                        "-c", command, "convert", cfg["slime_model_config"]], check=True)
        if not list(staged.glob("**/*.distcp")):
            raise ValueError("converter produced no Megatron shards")
        # Conversion runs as root; make its output readable/writable by the invoking owner.
        subprocess.run(["docker", "--host", runtime_paths.docker_host(), "run", "--rm", "-v", f"{tmp}:/output", "--entrypoint", "chown",
                        os.environ.get("VTB_TRAINER_IMAGE", "vibetrainbench/trainer:latest"),
                        "-R", f"{os.getuid()}:{os.getgid()}", "/output"], check=True)
        (staged / assets.MODEL_SOURCE_FILE).write_text(json.dumps(assets.model_signature(cfg), indent=2) + "\n")
        staged.rename(destination)
    print(f"MODEL_CONVERSION_OK checkpoint={destination} revision={cfg['revision']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "convert", "preflight", "snapshot-preflight"))
    parser.add_argument("--offline", action="store_true", help="materialize only from the pinned local snapshot; never download")
    args = parser.parse_args()
    if args.offline and args.action != "download":
        parser.error("--offline is only supported for download")
    _, cfg = assets.read_resource(REPO, RESOURCE_ID)
    if args.action == "download":
        materialize_model(cfg, offline=args.offline)
    elif args.action == "convert":
        convert_model(cfg)
    else:
        errors = assets.snapshot_errors(REPO, cfg) if args.action == "snapshot-preflight" else assets.model_errors(REPO, cfg)
        if errors:
            parser.error("; ".join(errors))
        print("MODEL_SNAPSHOT_OK" if args.action == "snapshot-preflight" else "MODEL_RESOURCE_OK")


if __name__ == "__main__":
    main()
