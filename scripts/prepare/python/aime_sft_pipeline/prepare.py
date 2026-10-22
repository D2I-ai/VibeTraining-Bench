#!/usr/bin/env python3
"""Materialize the pinned AIME mathematics sources into one slime SFT JSONL."""

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

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parents[3]))
from eval.runner import resource_assets as assets

SOURCES = (
    {
        "resource": "data/aime_open_thoughts3",
        "input_subdir": "data",
        "converter": "convert_math_to_slime_sft.py",
        "component": "open_thoughts3_math_sft.jsonl",
        "rows": 850_000,
    },
    {
        "resource": "data/aime_nemotron_math",
        "input_subdir": "data",
        "converter": "convert_cot_to_slime_sft.py",
        "component": "nemotron_cot_sft.jsonl",
        "rows": 285_516,
    },
    {
        "resource": "data/aime_openr1_math",
        "input_subdir": "data",
        "converter": "convert_default_to_slime_sft.py",
        "component": "openr1_math_220k_default_sft.jsonl",
        "rows": 93_733,
    },
)
TOTAL_ROWS = sum(int(item["rows"]) for item in SOURCES)


def load_resource(repo: Path, resource_id: str, hf_home: Path) -> tuple[dict, Path]:
    config_path = repo / "training/resources" / resource_id / "resource_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("id") != resource_id or config.get("type") != "hf_cache_dataset":
        raise ValueError(f"invalid AIME source declaration: {config_path}")
    raw_snapshot = str(config.get("snapshot_path") or "")
    snapshot = Path(raw_snapshot.replace("${HF_HOME}", str(hf_home))).resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"pinned source snapshot is missing: {snapshot}")
    errors = assets.dataset_errors(repo, dict(config, snapshot_path=str(snapshot)))
    if errors:
        raise ValueError("; ".join(errors))
    return config, snapshot


def source_signatures(repo: Path, hf_home: Path) -> list[dict]:
    signatures = []
    for item in SOURCES:
        config, _ = load_resource(repo, str(item["resource"]), hf_home)
        signatures.append(
            {
                "resource": item["resource"],
                "repo_id": config["repo_id"],
                "revision": config["revision"],
                "data_files": config["data_files"],
                "rows": item["rows"],
                "converter": item["converter"],
                "converter_sha256": assets.sha256_file(PIPELINE / str(item["converter"])),
            }
        )
    return signatures


def validate_existing(output_dir: Path, signatures: list[dict], *, code_signature: dict | None = None) -> None:
    train = output_dir / "train.jsonl"
    manifest_path = output_dir / "source_manifest.json"
    if not train.is_file() or train.stat().st_size == 0 or not manifest_path.is_file():
        raise ValueError(f"incomplete AIME materialization: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sources") != signatures or manifest.get("rows") != TOTAL_ROWS:
        raise ValueError(f"AIME materialization provenance differs from current declarations: {output_dir}")
    if manifest.get("bytes") != train.stat().st_size:
        raise ValueError(f"AIME materialization size differs from provenance: {output_dir}")
    if code_signature is not None and manifest.get("preparation_sha256") != code_signature:
        raise ValueError(f"AIME preparation code differs from provenance: {output_dir}")
    errors = assets.jsonl_errors(train, rows=TOTAL_ROWS, sha256=manifest.get("sha256", ""))
    if errors:
        raise ValueError("; ".join(errors))


def count_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def materialize(repo: Path, hf_home: Path, output_dir: Path) -> None:
    signatures = source_signatures(repo, hf_home)
    _, output_cfg = assets.read_resource(repo, "data/aime")
    code_signature = assets.preparation_signature(repo, output_cfg)
    if output_dir.exists():
        validate_existing(output_dir, signatures, code_signature=code_signature)
        print(f"AIME_SFT_ALREADY_READY rows={TOTAL_ROWS} output={output_dir}")
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".aime_sft_", dir=output_dir.parent))
    components: list[Path] = []
    try:
        for item in SOURCES:
            _, snapshot = load_resource(repo, str(item["resource"]), hf_home)
            component = staging / str(item["component"])
            subprocess.run(
                [
                    sys.executable,
                    str(PIPELINE / str(item["converter"])),
                    "--input-dir",
                    str(snapshot / str(item["input_subdir"])),
                    "--output",
                    str(component),
                ],
                check=True,
            )
            found = count_rows(component)
            if found != item["rows"]:
                raise ValueError(
                    f"{item['resource']} produced {found} rows; expected {item['rows']}"
                )
            components.append(component)

        train = staging / "train.jsonl"
        digest = hashlib.sha256()
        with train.open("wb") as destination:
            for component in components:
                with component.open("rb") as source:
                    for chunk in iter(lambda: source.read(8 << 20), b""):
                        destination.write(chunk)
                        digest.update(chunk)
                component.unlink()
        manifest = {
            "format": "slime_messages_jsonl",
            "materialized_file": "train.jsonl",
            "rows": TOTAL_ROWS,
            "bytes": train.stat().st_size,
            "sha256": digest.hexdigest(),
            "sources": signatures,
            "preparation_sha256": code_signature,
        }
        errors = assets.jsonl_errors(train, rows=TOTAL_ROWS, sha256=manifest["sha256"])
        if errors:
            raise ValueError("; ".join(errors))
        (staging / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"AIME_SFT_PREPARE_OK rows={TOTAL_ROWS} output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    hf_home = args.hf_home.resolve()
    output = args.output_dir.resolve()
    signatures = source_signatures(repo, hf_home)
    if args.check:
        _, output_cfg = assets.read_resource(repo, "data/aime")
        validate_existing(output, signatures, code_signature=assets.preparation_signature(repo, output_cfg))
        print(f"AIME_SFT_PREPARE_OK rows={TOTAL_ROWS} output={output}")
        return
    materialize(repo, hf_home, output)


if __name__ == "__main__":
    main()
