"""Shared, read-only resource checks for preparation and workspace assembly."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from pathlib import Path

from eval.grading import valsplit

MODEL_SOURCE_FILE = "vtb_source.json"


def hf_home(repo: Path) -> Path:
    root = Path(os.environ.get("VTB_ROOT") or repo / "vtb-env").expanduser()
    return Path(os.environ.get("HF_HOME") or root / "resources/data/hf_cache").expanduser().resolve()


def resolve_path(repo: Path, raw: str) -> Path:
    raw = raw.replace("${HF_HOME}", str(hf_home(repo)))
    result = valsplit.resolve_resource_path(repo, raw)
    if result is None:
        raise ValueError("resource path must not be empty")
    return result.resolve()


def read_resource(repo: Path, resource_id: str) -> tuple[Path, dict]:
    relative = Path(resource_id)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
        raise ValueError(f"unsafe resource ID: {resource_id}")
    root = repo / "training/resources" / relative
    if not root.is_dir():
        raise ValueError(f"resource missing: {root}")
    path = root / "resource_config.json"
    if path.is_file():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("id") != resource_id:
            raise ValueError(f"resource ID mismatch: {path}")
        return root, cfg
    if resource_id.startswith("scripts/"):
        return root, {"id": resource_id, "type": "local"}
    raise ValueError(f"resource declaration missing: {path}")


def dataset_patterns(cfg: dict) -> list[str]:
    patterns = []
    for value in (cfg.get("data_files") or {}).values():
        patterns.extend(value if isinstance(value, list) else [value])
    if not patterns or not all(isinstance(p, str) for p in patterns):
        raise ValueError(f"{cfg['id']}: data_files must declare file patterns")
    for pattern in patterns:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError(f"unsafe dataset file pattern: {pattern}")
    return patterns


def expand_file_pattern(pattern: str) -> list[str]:
    """Expand the small brace alternatives used in existing SWE manifests."""
    match = re.search(r"\{([^{}]+)\}", pattern)
    if match is None:
        return [pattern]
    return [expanded for option in match.group(1).split(",")
            for expanded in expand_file_pattern(pattern[:match.start()] + option + pattern[match.end():])]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preparation_signature(repo: Path, cfg: dict) -> dict[str, str]:
    result = {}
    for name in cfg.get("preparation_scripts", []):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe preparation script: {name}")
        result[name] = sha256_file(repo / relative)
    return result


def jsonl_errors(path: Path, *, rows: int, sha256: str) -> list[str]:
    """Stream integrity and messages-schema checks without loading a corpus into RAM."""
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                return [f"blank JSONL row at {path}:{count + 1}"]
            record = json.loads(line)
            messages = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(messages, list) or not messages:
                return [f"missing messages at {path}:{count + 1}"]
            for message in messages:
                if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
                    return [f"invalid message at {path}:{count + 1}"]
                if not isinstance(message.get("content"), str):
                    return [f"non-text message content at {path}:{count + 1}"]
                if "reasoning_content" in message and message["reasoning_content"] is not None and not isinstance(message["reasoning_content"], str):
                    return [f"invalid reasoning_content at {path}:{count + 1}"]
            count += 1
    errors = []
    if count != rows:
        errors.append(f"JSONL row count mismatch: {path} (expected {rows}, found {count})")
    if digest.hexdigest() != sha256:
        errors.append(f"JSONL SHA-256 mismatch: {path}")
    return errors


def dataset_errors(repo: Path, cfg: dict) -> list[str]:
    if not re.fullmatch(r"[0-9a-f]{40}", str(cfg.get("revision", ""))):
        return [f"{cfg['id']}: immutable 40-character revision is required"]
    snapshot = resolve_path(repo, cfg.get("snapshot_path", ""))
    errors = []
    for pattern in dataset_patterns(cfg):
        for alternative in expand_file_pattern(pattern):
            files = [Path(p) for p in glob.glob(str(snapshot / alternative), recursive=True)]
            if not files or any(not p.is_file() or p.stat().st_size == 0 for p in files):
                errors.append(f"{cfg['id']}: missing/empty dataset files: {snapshot / alternative}")
            expected = (cfg.get("file_count_check") or {}).get(pattern)
            if expected is not None and len(files) != expected:
                errors.append(f"{cfg['id']}: dataset file count mismatch: {snapshot / alternative} (expected {expected}, found {len(files)})")
    return errors


def materialized_dataset_errors(repo: Path, cfg: dict) -> list[str]:
    root = resolve_path(repo, cfg.get("materialized_path", ""))
    materialized_file = cfg.get("materialized_file")
    if not isinstance(materialized_file, str) or not materialized_file.strip():
        return [f"{cfg['id']}: materialized_file is required"]
    manifest_name = str(cfg.get("source_manifest") or "source_manifest.json")
    for name in (materialized_file, manifest_name):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            return [f"{cfg['id']}: unsafe materialized payload path: {name}"]
    payload = root / materialized_file
    manifest_path = root / manifest_name
    errors = []
    if not payload.is_file() or payload.stat().st_size == 0:
        errors.append(f"{cfg['id']}: materialized training data missing/empty: {payload}")
    if not manifest_path.is_file():
        errors.append(f"{cfg['id']}: materialization provenance missing: {manifest_path}")
        return errors
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("materialized_file") != materialized_file:
        errors.append(f"{cfg['id']}: materialized file differs from provenance: {manifest_path}")
    if manifest.get("rows") != cfg.get("num_rows"):
        errors.append(f"{cfg['id']}: materialized row count differs from declaration: {manifest_path}")
    if payload.is_file() and manifest.get("bytes") != payload.stat().st_size:
        errors.append(f"{cfg['id']}: materialized byte size differs from provenance: {manifest_path}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("sha256", ""))):
        errors.append(f"{cfg['id']}: materialized SHA-256 is missing/invalid: {manifest_path}")
    declared_sources = cfg.get("source_resources") or []
    manifest_sources = manifest.get("sources") or []
    if not isinstance(manifest_sources, list) or not all(isinstance(item, dict) for item in manifest_sources):
        errors.append(f"{cfg['id']}: malformed source provenance: {manifest_path}")
        return errors
    if [item.get("resource") for item in manifest_sources] != declared_sources:
        errors.append(f"{cfg['id']}: source order differs from declaration: {manifest_path}")
    else:
        for item in manifest_sources:
            _, source = read_resource(repo, item["resource"])
            if any(item.get(key) != source.get(key) for key in ("repo_id", "revision", "data_files")):
                errors.append(f"{cfg['id']}: source provenance is stale for {item['resource']}")
    if cfg.get("preparation_scripts") and manifest.get("preparation_sha256") != preparation_signature(repo, cfg):
        errors.append(f"{cfg['id']}: preparation code differs from materialization provenance")
    if not errors:
        errors.extend(jsonl_errors(payload, rows=int(cfg["num_rows"]), sha256=manifest["sha256"]))
    return errors


def weights_errors(path: Path) -> list[str]:
    if not path.is_dir():
        return [f"HF checkpoint directory missing: {path}"]
    index = path / "model.safetensors.index.json"
    files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        names = set((data.get("weight_map") or {}).values())
        if not names or any(Path(n).name != n for n in names):
            return [f"invalid weight index: {index}"]
        files = [path / n for n in names]
    if not files or any(not p.is_file() or p.stat().st_size == 0 for p in files):
        return [f"HF checkpoint weights missing/incomplete: {path}"]
    return []


def snapshot_errors(repo: Path, cfg: dict) -> list[str]:
    snapshot = resolve_path(repo, cfg.get("snapshot_path", ""))
    errors = weights_errors(snapshot)
    if not re.fullmatch(r"[0-9a-f]{40}", str(cfg.get("revision", ""))) or snapshot.name != cfg["revision"]:
        errors.append(f"model snapshot path/revision mismatch: {snapshot}")
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if not (snapshot / name).is_file():
            errors.append(f"model snapshot file missing: {snapshot / name}")
    return errors


def model_signature(cfg: dict) -> dict:
    return {key: cfg.get(key) for key in ("repo_id", "revision", "eos_token", "eos_token_id")}


def model_errors(repo: Path, cfg: dict, *, require_conversion: bool = True) -> list[str]:
    checkpoint = resolve_path(repo, cfg.get("hf_checkpoint_path", ""))
    errors = weights_errors(checkpoint)
    if cfg.get("revision"):
        manifest = checkpoint / MODEL_SOURCE_FILE
        if not manifest.is_file():
            errors.append(f"model provenance missing: {manifest}; run download_base_model.sh")
        else:
            source = json.loads(manifest.read_text(encoding="utf-8"))
            if source != model_signature(cfg):
                errors.append(f"model provenance differs from declared revision/normalization: {checkpoint}")
        tokenizer = checkpoint / "tokenizer_config.json"
        if not tokenizer.is_file():
            errors.append(f"tokenizer missing: {tokenizer}")
        else:
            tokenizer_cfg = json.loads(tokenizer.read_text())
            if cfg.get("eos_token") and tokenizer_cfg.get("eos_token") != cfg["eos_token"]:
                errors.append(f"tokenizer EOS normalization mismatch: {tokenizer}")
            template = checkpoint / "chat_template.jinja"
            if not template.is_file() or cfg["eos_token"] not in template.read_text():
                errors.append(f"chat template missing/incompatible: {template}")
            model_config = checkpoint / "config.json"
            if not model_config.is_file():
                errors.append(f"model config missing: {model_config}")
            else:
                model_cfg = json.loads(model_config.read_text())
                if model_cfg.get("eos_token_id") != cfg["eos_token_id"] or (isinstance(model_cfg.get("text_config"), dict) and model_cfg["text_config"].get("eos_token_id") != cfg["eos_token_id"]):
                    errors.append(f"model EOS normalization mismatch: {model_config}")
    if require_conversion and cfg.get("torch_dist_path"):
        converted = resolve_path(repo, cfg["torch_dist_path"])
        shards = list(converted.glob("**/*.distcp"))
        if not shards or any(p.stat().st_size == 0 for p in shards):
            errors.append(f"Megatron checkpoint missing/incomplete: {converted}; run download_base_model.sh convert")
        if cfg.get("revision"):
            provenance = converted / MODEL_SOURCE_FILE
            expected = model_signature(cfg)
            if not provenance.is_file() or json.loads(provenance.read_text()) != expected:
                errors.append(f"matching Megatron conversion provenance missing: {provenance}")
    return errors


def resource_errors(repo: Path, resource_id: str) -> list[str]:
    try:
        root, cfg = read_resource(repo, resource_id)
        if cfg.get("pending"):
            return [f"{resource_id}: pending resource: {cfg['pending']}"]
        kind = cfg.get("type", "local")
        if kind == "hf_cache_dataset":
            return dataset_errors(repo, cfg)
        if kind == "hf_dataset":
            return materialized_dataset_errors(repo, cfg)
        if kind in {"cached_model", "local_model"}:
            return model_errors(repo, cfg)
        if kind == "local" and resource_id.startswith("data/"):
            names = [cfg["materialized_file"]] if cfg.get("materialized_file") else [f["path"] for f in cfg.get("available_files", [])]
            if not names or any(not (root / p).is_file() or (root / p).stat().st_size == 0 for p in names):
                return [f"{resource_id}: local training payload missing"]
        return []
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"{resource_id}: {exc}"]


def recipe_file(repo: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe recipe path: {name}")
    return repo / "training/recipes" / relative.with_suffix(".json")


def recipe_errors(repo: Path, name: str) -> list[str]:
    try:
        recipe = json.loads(recipe_file(repo, name).read_text())
        errors = [error for item in recipe["resources"] for error in resource_errors(repo, item)]
        validation = recipe.get("validation", {})
        if validation:
            source = valsplit.expand_val_source(repo, validation["val_source"])
            valsplit.parse_val_source(source)
            # All formal recipes start from this declared base-model family.
            cfg = valsplit.find_baseline_resource(repo, validation["benchmark"], "Qwen/Qwen3.5-9B-Base", source)
            if cfg is not None:
                payload = resolve_path(repo, cfg["materialized_path"])
                sample_root = payload / "samples"
                expected = int(cfg.get("graded") or cfg.get("split_members") or 0)
                samples = [p for p in sample_root.iterdir() if p.is_dir()] if sample_root.is_dir() else []
                if not samples or (expected and len(samples) != expected):
                    errors.append(f"baseline samples missing/incomplete: {payload} (expected {expected}, found {len(samples)})")
                summary_path = payload / "summary.json"
                if not summary_path.is_file():
                    errors.append(f"baseline summary missing: {summary_path}")
                else:
                    summary = json.loads(summary_path.read_text())
                    if summary.get("task") != validation["benchmark"] or valsplit.expand_val_source(repo, summary.get("val_source", "")) != source:
                        errors.append(f"baseline identity/source mismatch: {summary_path}")
        benchmark_path = repo / "benchmarks" / validation.get("benchmark", "") / "benchmark.json"
        if validation and benchmark_path.is_file():
            benchmark = json.loads(benchmark_path.read_text())
            if benchmark.get("validation_context"):
                _, context = read_resource(repo, benchmark["validation_context"])
                context_root = resolve_path(repo, context["materialized_path"])
                if not (context_root / "manifest.json").is_file():
                    errors.append(f"validation context missing: {context_root}")
        return errors
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"{name}: {exc}"]
