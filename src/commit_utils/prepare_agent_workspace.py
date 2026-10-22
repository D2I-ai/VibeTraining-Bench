#!/usr/bin/env python3
"""Prepare the agent-visible workspace for a vibetrainbench run.

This script owns files that are copied into ${JOB_DIR}/task before the agent starts:
common evaluation instructions, benchmark context, and optional training resources. Runtime
secrets such as grading_token are intentionally written later by run_task.sh.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional

RESERVED_TOP_LEVEL = {
    "grading_token",
    "grading.md",
    "environment.md",
    "agent_rules.md",
    "timer.sh",
    "final_model",
    "templates",
    "train_resource_manifest.lock.json",
    "harness",
    "harness_reference",
    "candidates",
    "final_candidate.json",
}

# Where the editable framework copies live in the agent's workspace, named here so the
# workspace doc can state them without the agent having to go looking. The container
# bootstrap (src/commit_utils/bootstrap_mutable_framework.sh, sourced before the agent
# starts) copies the trainer image's /root/slime and /root/Megatron-LM here and exports
# SLIME_DIR/MEGATRON_DIR pointing at the copies, so the training entrypoint runs from
# them and edits there survive the run. If the bootstrap's layout moves, these move with
# it.
WORKSPACE_SLIME_DIR = "framework_src/slime"
WORKSPACE_MEGATRON_DIR = "framework_src/Megatron-LM"


def fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    raise SystemExit(2)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        fail(f"missing file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"expected JSON object in {path}")
    return data


def copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        fail(f"missing copy source: {src}")
    if dst.exists():
        fail(f"destination already exists, refusing to overwrite: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    else:
        shutil.copy2(src, dst)


def copy_dir_contents(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in sorted(src.iterdir()):
        copy_path(child, dst / child.name)


# Editable harness templates a benchmark may declare in benchmark.json. A tree source
# is copied and pruned to the allowlist; a single-module source becomes the package's
# entry module.
EDITABLE_HARNESS_SOURCES = ("installed-terminus-2", "installed-mini-swe-agent")


def editable_harness_config(bench_dir: Path) -> dict[str, Any]:
    config = load_json(bench_dir / "benchmark.json").get("editable_harness") or {}
    if not config:
        return {}
    if not isinstance(config, dict) or config.get("enabled") is not True:
        fail(f"{bench_dir}/benchmark.json has invalid editable_harness metadata")
    if config.get("template_source") not in EDITABLE_HARNESS_SOURCES:
        fail(
            f"unsupported editable harness template: {config.get('template_source')!r}; "
            f"known: {sorted(EDITABLE_HARNESS_SOURCES)}"
        )
    import_path = config.get("import_path")
    if not isinstance(import_path, str) or not import_path.startswith("harness.") or ":" not in import_path:
        fail(f"editable harness import_path must look like harness.<module>:<Class>, got {import_path!r}")
    if not config.get("allowed_files"):
        fail("editable harness allowed_files must not be empty")
    return config


def installed_harness_source(repo: Path, template_source: str) -> Path:
    """Locate the installed source an editable harness is seeded from.

    Both editable harnesses ship inside harbor; a template differs only in which
    subtree of harbor/agents/ holds it. The backend-installed package is authoritative:
    baseline jobs resolve registered agents from that package, so the editable SWE
    workspace must start from those exact bytes or preparation fails. Directory-based
    development harnesses may still fall back to a checkout.
    """
    subpath = {
        "installed-terminus-2": "terminus_2",
        "installed-mini-swe-agent": "installed/mini_swe_agent.py",
    }.get(template_source)
    if subpath is None:
        fail(f"unsupported editable harness template: {template_source!r}")
    # A directory source is complete only with its entry module and templates; a file
    # source just has to exist.
    valid = (
        (lambda p: (p / "terminus_2.py").is_file() and (p / "templates").is_dir())
        if template_source == "installed-terminus-2"
        else (lambda p: p.is_file())
    )
    tools = Path(os.environ.get("VTB_ROOT") or repo / "vtb-env") / "tools"
    roots = list(
        sorted(tools.glob("data/uv/tools/harbor/lib/python*/site-packages/harbor/agents"))
    )
    if template_source != "installed-mini-swe-agent":
        roots.extend(sorted(tools.glob("cache/**/harbor/agents")))
        harbor_repo = os.environ.get("VTB_HARBOR_REPO")
        if harbor_repo:
            roots.append(Path(harbor_repo) / "src" / "harbor" / "agents")
    for root_dir in roots:
        candidate = root_dir / subpath
        if valid(candidate):
            return candidate
    fail(
        f"installed harness source for {template_source} not found; "
        "run scripts/prepare/install_backend_tools.sh so workspace and baseline use "
        "the same Harbor package"
    )
    raise AssertionError("unreachable")


def copy_editable_harness(repo: Path, task_dir: Path, config: dict[str, Any]) -> None:
    allowed = set(config.get("allowed_files") or [])
    if not allowed:
        fail("editable_harness.allowed_files must not be empty")
    destination = task_dir / "harness"
    source = installed_harness_source(repo, str(config.get("template_source") or ""))
    if source.is_file():
        # A single-module source gets a package constructed around it, so the declared
        # import path (harness.<module>:<Class>) resolves.
        module_name = str(config["import_path"]).split(":", 1)[0].split(".", 1)[1]
        destination.mkdir(parents=True)
        (destination / "__init__.py").write_text(
            '"""Editable harness package seeded from the declared template source."""\n',
            encoding="utf-8",
        )
        copied_module = destination / f"{module_name}.py"
        copy_path(source, copied_module)
        if _sha256_of(copied_module) != _sha256_of(source):
            fail(f"editable harness copy does not match its baseline adapter source: {source}")
    else:
        copy_path(source, destination)
        for path in sorted(destination.rglob("*"), reverse=True):
            relative = path.relative_to(destination).as_posix()
            if path.is_dir():
                # A directory survives iff a declared editable file lives under it. The
                # existence of those files is verified by the `missing` check below; testing
                # (path / name) here would double the prefix and rmtree every non-empty
                # directory, which is exactly how the templates/ subtree was once deleted.
                if not any(name.startswith(relative + "/") for name in allowed):
                    shutil.rmtree(path, ignore_errors=True)
            elif relative not in allowed:
                path.unlink()
    missing = sorted(name for name in allowed if not (destination / name).is_file())
    if missing:
        fail(f"editable harness source is missing declared editable files: {missing}")


# The mini-swe-agent editable harness is harbor's shim; the agent loop it launches in
# the task container is the mini-swe-agent pip package, installed at evaluation time at
# the benchmark's pinned version. The shim alone is not enough to write L2 loop code
# against, so preparation ships that package's source as a read-only reference.
# Templates whose harness ships the whole package tree (terminus) need no reference.
HARNESS_PACKAGE_TEMPLATES = {"installed-mini-swe-agent": "mini-swe-agent"}


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_package_wheel(project: str, version: str, store: Path) -> tuple[Path, str]:
    """The version-pinned wheel, from the local package store, downloaded once if absent.

    A prepared run must not depend on a live package index: the first preparation on a
    host downloads the wheel against the index's published sha256, and every later run
    reuses that file. On a host without network access, seed the store by hand.
    """
    store.mkdir(parents=True, exist_ok=True)
    cached = sorted(store.glob(f"{project.replace('-', '_')}-{version}-*.whl"))
    if cached:
        return cached[0], _sha256_of(cached[0])
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{project}/{version}/json", timeout=60) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
        entry = next(u for u in meta["urls"] if str(u.get("filename", "")).endswith(".whl"))
        expected = entry["digests"]["sha256"]
        with urllib.request.urlopen(entry["url"], timeout=300) as resp:
            data = resp.read()
    except Exception as exc:
        fail(
            f"{project}=={version} is not in the package store ({store}) and could not "
            f"be downloaded: {exc}. On an offline host, place the wheel there by hand."
        )
    if hashlib.sha256(data).hexdigest() != expected:
        fail(f"downloaded {project}=={version} failed the index's sha256 check")
    dest = store / entry["filename"]
    dest.write_bytes(data)
    return dest, expected


def provision_harness_package_source(
    repo: Path, task_dir: Path, bench_dir: Path, harness_config: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Ship the pinned harness package's source into the workspace, read-only.

    The pin is the benchmark's `agent_kwargs.version` -- the same value the runner passes
    to harbor at evaluation time -- so the source the agent reads and the package it is
    scored on cannot drift apart. Only called when the run ships an editable harness.
    """
    project = HARNESS_PACKAGE_TEMPLATES.get(str(harness_config.get("template_source") or ""))
    if project is None:
        return None
    bench_meta = load_json(bench_dir / "benchmark.json")
    version = str((bench_meta.get("agent_kwargs") or {}).get("version") or "").strip()
    if not version:
        fail(
            f"{bench_dir}/benchmark.json declares an editable mini-swe-agent harness but "
            "no agent_kwargs.version pin for the loop package"
        )
    # runtime_paths owns every vtb-env path; imported here because this script is
    # invoked by path and sys.path[0] is its own directory, not the repo.
    sys.path.insert(0, str(repo))
    from eval.runner import runtime_paths

    store = runtime_paths.packages_dir()
    wheel, sha256 = _pinned_package_wheel(project, version, store)
    # Two views of the same pinned bytes: the unpacked tree for reading, the wheel for a
    # scratch-venv install when the agent wants to import the module it is writing.
    dest_dir = task_dir / "harness_reference"
    dest_dir.mkdir(parents=True)
    wheel_dest = dest_dir / wheel.name
    copy_path(wheel, wheel_dest)
    wheel_dest.chmod(0o444)
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(dest_dir)
    return {
        "name": project,
        "version": version,
        "ref": "harness_reference/",
        "wheel": wheel.name,
        "sha256": sha256,
    }


def copy_validation_context(repo: Path, task_dir: Path, bench_dir: Path) -> Optional[dict[str, Any]]:
    """Materialize the benchmark's declared validation context pack into the workspace.

    benchmark.json may name a validation/<id> resource holding the baseline manifest,
    summary and trajectories for the benchmark's validation source. The pack used to
    live inside task_context and ride its wholesale copy; as a resource it gains a
    declaration, a materialized payload (materialized_path, resolved like every other
    declared path: repo-relative, or ${VTB_ROOT}/absolute when it outgrows the repo),
    and it still lands at the workspace-relative path the resource declares, because
    the task_context docs reference it there.
    """
    resource_id = load_json(bench_dir / "benchmark.json").get("validation_context")
    if not resource_id:
        return None
    skip_context = os.environ.get("VTB_SKIP_VALIDATION_CONTEXT", "").strip()
    if skip_context:
        if not (
            skip_context == "terminal-smoke"
            and bench_dir.name == "terminal-bench__terminal-bench-2-1"
            and os.environ.get("VTB_PROMPT") == "terminal_smoke"
            and os.environ.get("VTB_TRAIN_RECIPE", "") == ""
        ):
            fail(
                "VTB_SKIP_VALIDATION_CONTEXT is restricted to the no-training "
                "Terminal-Bench smoke launcher"
            )
        # This pack is agent-facing baseline diagnostics, not evaluator input. The
        # smoke omits it without changing Terminal-Bench -> TBLite validation routing.
        return None
    resource_id = normalise_resource_id(str(resource_id))
    resource_src = repo / "training" / "resources" / resource_id
    metadata = load_json(resource_src / "resource_config.json")
    if metadata.get("type") != "validation_context":
        fail(f"validation context resource {resource_id} must declare type=validation_context")
    if metadata.get("benchmark") and metadata["benchmark"] != bench_dir.name:
        fail(
            f"validation context resource {resource_id} is for benchmark "
            f"{metadata['benchmark']!r}, not {bench_dir.name!r}"
        )
    payload = resolve_declared_path(repo, metadata.get("materialized_path"), "materialized_path", resource_id)
    if not payload.is_dir():
        fail(f"validation context resource {resource_id} is not materialized at {payload}")
    workspace_path = str(metadata.get("workspace_path") or "").strip().strip("/")
    target = Path(workspace_path)
    if not workspace_path or target.is_absolute() or ".." in target.parts:
        fail(f"validation context resource {resource_id} must declare a safe relative workspace_path")
    if target.parts[0] in RESERVED_TOP_LEVEL:
        fail(f"validation context resource {resource_id} workspace_path collides with a reserved name")
    copy_path(payload, task_dir / target)
    return {
        "id": resource_id,
        "workspace_path": workspace_path,
    }


def resolve_declared_path(repo: Path, raw_path: Any, field_name: str, resource_id: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        fail(f"resource {resource_id} must define {field_name}")
    # The expansion rules live in eval.grading.valsplit so every reader of a resource
    # declaration resolves paths the same way; this wrapper adds the prep tool's
    # fail-fast contract on top. Imported here rather than at module top because this
    # script is invoked by path and sys.path[0] is its own directory, not the repo.
    sys.path.insert(0, str(repo))
    from eval.grading import valsplit

    resolved = valsplit.resolve_resource_path(repo, raw_path)
    assert resolved is not None  # the emptiness check above already excluded None
    return resolved


def validate_hf_checkpoint_for_chat_sft(path: Path, resource_id: str) -> None:
    """Fail before copying a cached model whose chat template/EOS are inconsistent."""
    tokenizer_config_path = path / "tokenizer_config.json"
    chat_template_path = path / "chat_template.jinja"
    if not tokenizer_config_path.is_file() or not chat_template_path.is_file():
        return

    tokenizer_config = load_json(tokenizer_config_path)
    eos_token = tokenizer_config.get("eos_token")
    try:
        chat_template = chat_template_path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"model resource {resource_id} chat_template.jinja is unreadable: {exc}")

    # Qwen3.5 chat SFT data closes turns with <|im_end|>. If a re-download restores
    # Base's tokenizer_config.json to <|endoftext|> while leaving the grafted template
    # in place, training looks healthy but generated SWE rollouts do not stop reliably.
    if "<|im_end|>" in chat_template and eos_token != "<|im_end|>":
        fail(
            f"model resource {resource_id} has tokenizer_config eos_token={eos_token!r}, "
            "but chat_template.jinja uses '<|im_end|>'; repair the cached HF checkpoint before train-test"
        )


def normalise_resource_id(resource_id: str) -> str:
    value = resource_id.strip().strip("/")
    if not value:
        fail("empty training resource id")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        fail(f"invalid training resource id {resource_id!r}: must be relative under training/resources")
    if path.parts[0] in RESERVED_TOP_LEVEL:
        fail(f"training resource id {resource_id!r} collides with reserved workspace path")
    if len(path.parts) < 2:
        fail(f"training resource id {resource_id!r} should look like 'scripts/name' or 'data/name'")
    return "/".join(path.parts)


def parse_direct_resources(value: str) -> list[str]:
    return [normalise_resource_id(item) for item in value.split(",") if item.strip()]


def resolve_recipe(repo: Path, recipe_name: str) -> tuple[Path, dict[str, Any]]:
    name = recipe_name.strip().strip("/")
    if not name:
        fail("empty train recipe name")
    recipe_rel = Path(name)
    if recipe_rel.is_absolute() or ".." in recipe_rel.parts:
        fail(f"invalid train recipe name {recipe_name!r}: must be relative under training/recipes")
    if recipe_rel.suffix != ".json":
        recipe_rel = recipe_rel.with_suffix(".json")
    path = repo / "training" / "recipes" / recipe_rel
    recipe = load_json(path)
    return path, recipe


def ensure_compatible(recipe: dict[str, Any], benchmark: str, model: str, recipe_path: Path) -> None:
    benchmarks = recipe.get("compatible_benchmarks", [])
    if benchmarks and benchmark not in benchmarks:
        fail(f"recipe {recipe_path} is not compatible with benchmark {benchmark!r}")

    families = recipe.get("compatible_model_families", [])
    if families and not any(model.startswith(str(prefix)) for prefix in families):
        fail(f"recipe {recipe_path} is not compatible with model {model!r}")


def validate_env_defaults(recipe: dict[str, Any], recipe_path: Path) -> dict[str, str]:
    raw = recipe.get("env_defaults", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        fail(f"recipe {recipe_path} field env_defaults must be an object")
    defaults: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.isidentifier() or not key.isupper():
            fail(f"recipe {recipe_path} env_defaults contains invalid variable name {key!r}")
        if not isinstance(value, (str, int, float, bool)):
            fail(f"recipe {recipe_path} env_defaults[{key!r}] must be a scalar")
        defaults[key] = str(value)
    return defaults


def resolve_requested_resources(repo: Path, train_recipe: str, train_resources: str, benchmark: str, model: str) -> tuple[str | None, Path | None, list[str], dict[str, Any] | None]:
    if train_recipe and train_resources:
        fail("set only one of VTB_TRAIN_RECIPE or VTB_TRAIN_RESOURCES")

    if train_recipe:
        recipe_path, recipe = resolve_recipe(repo, train_recipe)
        ensure_compatible(recipe, benchmark, model, recipe_path)
        recipe["env_defaults"] = validate_env_defaults(recipe, recipe_path)
        raw_resources = recipe.get("resources", [])
        if not isinstance(raw_resources, list) or not all(isinstance(item, str) for item in raw_resources):
            fail(f"recipe {recipe_path} must contain a string list field named resources")
        resources = [normalise_resource_id(item) for item in raw_resources]
        if not resources:
            fail(f"recipe {recipe_path} selects no resources")
        return str(recipe.get("name") or recipe_path.stem), recipe_path, resources, recipe

    if train_resources:
        resources = parse_direct_resources(train_resources)
        if not resources:
            fail("VTB_TRAIN_RESOURCES did not contain any usable resource ids")
        return None, None, resources, None

    return None, None, [], None


def copy_training_resource(repo: Path, task_dir: Path, resource_id: str) -> dict[str, Any]:
    resource_src = repo / "training" / "resources" / resource_id
    if not resource_src.is_dir():
        fail(f"training resource not found: {resource_id} ({resource_src})")

    metadata_path = resource_src / "resource_config.json"
    if metadata_path.is_file():
        metadata = load_json(metadata_path)
    elif resource_id.startswith("scripts/"):
        metadata = {"id": resource_id, "type": "local"}
    else:
        fail(f"training resource {resource_id} is missing resource_config.json")
    resource_type = str(metadata.get("type", "local"))
    sys.path.insert(0, str(repo))
    from eval.runner.resource_assets import resource_errors

    problems = resource_errors(repo, resource_id)
    if problems:
        fail("; ".join(problems))
    dest = task_dir / "train_resources" / resource_id

    # The manifest records where a resource landed in the workspace, never the host path
    # it was copied from: the workspace is the agent's whole filesystem, so a resolved
    # host path in it is at best dead weight and at worst an invitation to read
    # outside the sandbox. Provenance stays in `metadata` (the declared paths, with
    # ${VTB_ROOT}/${HF_HOME} placeholders resolved on the host when needed).
    copied_to: Path
    if resource_type == "hf_dataset":
        materialized_path = resolve_declared_path(repo, metadata.get("materialized_path"), "materialized_path", resource_id)
        if not materialized_path.exists():
            fail(
                f"HF dataset resource {resource_id} is not materialized at {materialized_path}. "
                f"Run its materialize.py first, then retry."
            )
        dest.mkdir(parents=True, exist_ok=False)
        shutil.copy2(metadata_path, dest / "resource_config.json")
        if (resource_src / "materialize.py").is_file():
            shutil.copy2(resource_src / "materialize.py", dest / "materialize.py")
        copy_path(materialized_path, dest / "data")
        copied_to = dest / "data"
    elif resource_type in {"cached_model", "local_model"}:
        hf_checkpoint_path = resolve_declared_path(repo, metadata.get("hf_checkpoint_path"), "hf_checkpoint_path", resource_id)
        if not hf_checkpoint_path.is_dir():
            fail(
                f"model resource {resource_id} HF checkpoint not found: {hf_checkpoint_path}. "
                f"Populate the configured model cache first."
            )
        validate_hf_checkpoint_for_chat_sft(hf_checkpoint_path, resource_id)
        dest.mkdir(parents=True, exist_ok=False)
        shutil.copy2(metadata_path, dest / "resource_config.json")
        copy_path(hf_checkpoint_path, dest / "hf_checkpoint")

        torch_dist_raw = metadata.get("torch_dist_path")
        if isinstance(torch_dist_raw, str) and torch_dist_raw.strip():
            torch_dist_path = resolve_declared_path(repo, torch_dist_raw, "torch_dist_path", resource_id)
            if not torch_dist_path.is_dir():
                fail(
                    f"model resource {resource_id} torch_dist checkpoint not found: {torch_dist_path}. "
                    f"Populate the configured model cache first."
                )
            copy_path(torch_dist_path, dest / "torch_dist")

        copied_to = dest
    elif resource_type == "hf_cache_dataset":
        cache_path = resolve_declared_path(repo, metadata.get("snapshot_path"), "snapshot_path", resource_id)
        if not cache_path.is_dir():
            fail(
                f"HF cache dataset resource {resource_id} not found: {cache_path}. "
                "Populate the configured Hugging Face cache first."
            )
        # The cache itself can be tens of GB and is a source for offline materializers,
        # not an agent-visible training payload. Keep only the resource declaration in
        # the workspace manifest so trajectories remain reproducible without copying it.
        dest.mkdir(parents=True, exist_ok=False)
        shutil.copy2(metadata_path, dest / "resource_config.json")
        copied_to = dest
    else:
        copy_path(resource_src, dest)
        copied_to = dest

    return {
        "id": resource_id,
        "type": resource_type,
        "copied_to": str(copied_to.relative_to(task_dir)),
        "metadata": metadata,
    }


def declared_sampling(repo: Path, benchmark: str) -> Optional[dict]:
    """The benchmark's declared request-level sampling parameters, if it declares any.

    They ride to the harness as agent kwargs: terminus-style agents name `temperature`
    directly and carry the rest in an `llm_call_kwargs` JSON string; mini-swe-agent
    carries the whole layer in a `config` JSON string (model.model_kwargs). The
    evaluation pins these on every scoring run, so this declaration -- not a file the
    delivery ships -- is the source of truth an agent should read.
    """
    path = repo / "benchmarks" / benchmark / "benchmark.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    agent_kwargs = data.get("agent_kwargs") or {}
    out: dict = {}
    if "temperature" in agent_kwargs:
        out["temperature"] = agent_kwargs["temperature"]
    for json_key in ("llm_call_kwargs", "config"):
        raw = agent_kwargs.get(json_key)
        if not isinstance(raw, str):
            continue
        try:
            parsed = json.loads(raw)
        except ValueError:
            continue
        if json_key == "config":
            parsed = (parsed.get("model") or {}).get("model_kwargs") if isinstance(parsed, dict) else None
        if isinstance(parsed, dict):
            out.update(parsed)
    return out or None


def fixed_sampling_from_export(task_dir: Path) -> Optional[dict]:
    """The sampling parameters the export will write, read from the script that writes them.

    Read rather than restated. The doc has to name the decoding for an agent to have no
    reason to go looking for it, and a second copy of the numbers in this file would be the
    copy that goes stale -- the export is the thing that decides, so the export is the thing
    quoted. Returns None when the recipe has no such step, and the caller then says where to
    look instead of naming values it cannot vouch for.

    Parsed rather than imported: this runs on the host during workspace setup, and reading a
    literal out of a training script is not a reason to execute one.
    """
    scripts = sorted(task_dir.glob("train_resources/scripts/*/postprocess_final_model.py"))
    for script in scripts:
        try:
            tree = ast.parse(script.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "FIXED_SAMPLING"
                for target in node.targets
            ):
                continue
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(value, dict) and value:
                return value
    return None


def write_environment_doc(
    task_dir: Path,
    benchmark: str,
    model: str,
    recipe: dict[str, Any] | None,
    copied_resources: list[dict[str, Any]],
    repo: Path,
    has_harness: bool,
    validation_context: Optional[dict[str, Any]] = None,
) -> None:
    """State where everything is, so the agent does not have to find out by searching.

    Generated rather than checked in: half of it is per-run (which resources were copied,
    which knobs the recipe set), and a hand-written version would describe the run before
    last. Values known only inside the container are named as variables instead of guessed.

    The cost of not having this was measured. One 12-hour run located the dataset cache by
    itself and then read it through `blobs/`, whose filenames are content hashes with no
    subset structure, even though the resource's own resource_config.json gave a
    snapshot_path and a data_files glob for exactly that purpose.
    """
    entrypoint = (recipe or {}).get("entrypoint_hint")
    # Run facts arrive from run_task.sh's environment: they are only known at launch
    # time, and this document is where the agent reads them.
    run_hours = os.environ.get("VTB_RUN_HOURS", "").strip()
    agent_gpus = os.environ.get("VTB_AGENT_GPUS", "").strip()
    grading_url = os.environ.get("GRADING_URL", "").strip()
    # Both frameworks live in the trainer image, which is only the image a Slime recipe
    # runs in. Naming those paths for a recipe that trains some other way would send the
    # agent to directories that do not exist.
    uses_slime = bool(entrypoint) and "slime" in str(entrypoint)
    env_defaults = (recipe or {}).get("env_defaults") or {}

    lines = [
        "# Environment",
        "",
        "Generated when this workspace was prepared. Read it before you go looking for "
        "anything: paths, the training entrypoint and the knobs it honours are all below.",
        "",
        "## This run",
        "",
        f"- Benchmark: `{benchmark}`",
        f"- Model to improve: `{model}`",
        f"- Recipe: `{recipe.get('name') if recipe else '(none -- resources named directly)'}`",
        "- Cards: `$NUM_GPUS` of them, already restricted by the container runtime and "
        "renumbered from 0, so device 0 here is yours regardless of which host card it is.",
    ]

    if run_hours:
        lines.append(
            f"- Total wall-time budget: {run_hours} hour(s); `bash timer.sh` reports what is left."
        )
    if agent_gpus:
        lines.append(f"- Agent GPU allocation (host cards): `{agent_gpus}`.")
    if grading_url:
        lines.append(f"- Grading service: `{grading_url}`.")
    lines += [
        "- Validation boundary: the development validation set the grading service exposes "
        "is not the final evaluation set. Neither may become training data.",
        "",
        "## Where things are",
        "",
        "| what | where |",
        "|---|---|",
        "| this workspace, and the working directory | `$HOME/task` |",
        "| how much of your time budget is left | `bash timer.sh` |",
        "| training resources copied for this run | `$HOME/task/train_resources/` |",
        "| training logs (written by the entrypoint) | `$HOME/task/logs/` |",
        "| the deliverable you must produce | `$HOME/task/final_model/`" + (" plus its paired `harness/` and a `final_candidate.json`" if has_harness else "") + " |",
        "| your own validation results | `$HOME/grading_results/<job_id>/` (read-only) |",
        "| the base model's validation, already run | `$HOME/grading_baseline/` (read-only) |",
    ]
    if validation_context:
        # The pack is the baseline's diagnostic half (analysis + trajectories); naming
        # it here is what makes a fresh agent aware it exists before planning.
        lines.append(
            "| baseline diagnostics for this validation source "
            f"| `{validation_context['workspace_path']}/` |"
        )
    lines += [
        "| HuggingFace cache | `$HF_HOME` |",
        "| scratch for the datasets library | `$HF_DATASETS_CACHE` (under /tmp, writable) |",
    ]

    if uses_slime:
        lines += [
            f"| Slime (job-local editable copy the entrypoint runs from) | `{WORKSPACE_SLIME_DIR}` |",
            f"| Megatron-LM (job-local editable copy the entrypoint runs from) | `{WORKSPACE_MEGATRON_DIR}` |",
        ]

    lines += [
        "",
        "Everything under `$HOME/task` is copied out of the sandbox when the run ends. "
        "`/tmp` and the rest of the container filesystem are discarded, so anything you "
        "want kept -- notes, a patch, a data recipe -- belongs in the workspace.",
        "",
        "`$HOME/grading_results/` and `$HOME/grading_baseline/` are separate read-only "
        "mounts, not children of `$HOME/task`, and are not copied out with the workspace. "
        "Write any analysis you want preserved under `$HOME/task/`.",
        "",
        "`bash timer.sh` is the only thing that knows how much of your budget is left. "
        "Worth checking before you commit to anything long and again before you decide you "
        "are finished: an evaluation whose estimate outruns your remaining time is not worth "
        "waiting for, and time left over after you stop is time you cannot get back.",
        "",
        "`HF_HUB_OFFLINE=1`: nothing downloads, and the cache is the only source of data. "
        "Treat it as read-only. Creating a new path inside it can fail with EPERM even "
        "though modifying an existing one works, which is why the datasets library is "
        "pointed at `$HF_DATASETS_CACHE` instead.",
        "",
        "## Training",
        "",
    ]

    if entrypoint and uses_slime:
        lines += [
            f"Launch through `{entrypoint}`. It starts a local ray head and submits the "
            "Slime job, converts the resulting torch_dist checkpoint back to HuggingFace "
            "format, and writes two files per launch under `logs/`: "
            "`train_<RUN_NAME>.log` (the full job output, including the per-step "
            "`train/loss`) and `framework_state_<RUN_NAME>.txt` (each framework's commit "
            "and diff). Calling `train_async.py` yourself skips all of that. It imports "
            "the job-local framework copies under `framework_src/` via "
            "`SLIME_DIR`/`MEGATRON_DIR`, so edit those -- the trainer image's `/root/...` "
            "copies are only the seed, and edits there change nothing the entrypoint runs.",
            "",
        ]
    elif entrypoint:
        lines += [
            f"Launch through `{entrypoint}`. Read it before you do: it is the script that "
            "records what ran, and anything it writes under `logs/` is the only trace of "
            "this training that survives the run. Invoking the trainer underneath it "
            "directly skips whatever it records.",
            "",
        ]
    else:
        lines += [
            "No recipe entrypoint was selected for this run; use the script resource copied "
            "under `train_resources/scripts/`.",
            "",
        ]

    if env_defaults:
        lines += [
            "The entrypoint reads these from the environment. The values are this recipe's "
            "defaults; export a different one before calling it to override.",
            "",
            "| variable | default |",
            "|---|---|",
        ]
        lines += [f"| `{key}` | `{value}` |" for key, value in sorted(env_defaults.items())]
        lines += [
            "",
            "`SAVE_DIR`, `OUTPUT_DIR` and `LOG_DIR` are also read, and default to "
            "`task/checkpoints/<RUN_NAME>`, `task/final_model` and `task/logs`.",
            "",
        ]

    # The cost an agent cannot see until it has already paid it, and paying it takes the
    # whole shared volume down rather than just its own run. Stated as the per-checkpoint
    # size plus the arithmetic, so it reads as a budget the agent can apply to whatever
    # interval it was going to choose.
    lines += [
        "**One checkpoint is roughly 17 GB, and nothing prunes them.** It carries the "
        "optimizer state as well as the weights, sharded across the training GPUs, so its "
        "size does not grow with how far training has got -- every save costs the same. That "
        "makes the save interval a disk decision more than a training one: at an interval of "
        "1, two hundred steps is over three terabytes, which is more than the volume has. "
        "Keep the one or two checkpoints you might actually deliver, delete the ones a later "
        "run has superseded, and check `df -h .` before a long job rather than after it. The "
        "volume is shared, and filling it stops whichever job writes next -- not necessarily "
        "the one that filled it.",
        "",
    ]

    lines += [
        "## Data",
        "",
        "Every resource in `train_resource_manifest.lock.json` has a `resource_config.json` "
        "beside it. For a HuggingFace cache resource that file is the whole payload: the "
        "data itself stays in the shared cache, and the config names the pieces you need.",
        "",
        "| field | what to do with it |",
        "|---|---|",
        "| `snapshot_path` | the directory to read from; `${HF_HOME}` in it expands to the "
        "cache above |",
        "| `data_files` | glob per split or subset, relative to `snapshot_path` |",
        "| `revision` | the commit the cache was pinned to; quote it when you report "
        "provenance |",
        "| `offline_load_hint` | a working load call for this resource |",
        "",
        "Read through `snapshot_path`, not through the cache's `blobs/` directory. Blob "
        "filenames are content hashes: they carry no split or subset structure, so a subset "
        "boundary that `data_files` states outright has to be reconstructed by guesswork.",
        "",
    ]

    if copied_resources:
        lines += ["| resource | type | in this workspace |", "|---|---|---|"]
        for item in copied_resources:
            lines.append(f"| `{item['id']}` | {item['type']} | `{item['copied_to']}` |")
        lines.append("")

    declared = declared_sampling(repo, benchmark)
    export_sampling = None if declared else fixed_sampling_from_export(task_dir)
    lines += ["## Decoding", ""]
    if declared:
        # Informed, not warned: the evaluation pins these parameters on every scoring
        # run, so what a delivery ships in generation_config.json never reaches the
        # scorer, and there is nothing here to manage.
        lines += [
            "Every scoring run of this benchmark -- each validation and the final alike -- "
            "is driven with the same fixed sampling parameters, passed to the harness by "
            "the grading service on every evaluation:",
            "",
            "```json",
            json.dumps(declared, indent=2),
            "```",
            "",
        ]
    elif export_sampling:
        # No service-side pin on this benchmark: the export writes these into the
        # delivered checkpoint's generation_config.json, and scoring serves the
        # checkpoint as delivered -- so the ordering caveat is real.
        lines += [
            "This benchmark declares no service-side sampling pin. The training export "
            "writes the following fixed sampling parameters into the delivered "
            "checkpoint's `generation_config.json` at every conversion, and scoring "
            "serves the checkpoint as delivered -- they hold as long as nothing rewrites "
            "that file after the last conversion.",
            "",
            "```json",
            json.dumps(export_sampling, indent=2),
            "```",
            "",
        ]
    else:
        lines += [
            "This benchmark declares no fixed sampling parameters; scoring follows the "
            "serving defaults of the delivered checkpoint.",
            "",
        ]

    lines += [
        "## Grading",
        "",
        "The service is at `$GRADING_URL`, your token is in `grading_token`, and "
        "`grading.md` is the protocol: how to submit, what a job's result contains, and "
        "where the transcripts land.",
        "",
        "Validation runs on a fixed subset of the benchmark, and the service refuses to "
        "evaluate you on anything outside it. After your time is up, your delivery is "
        "scored on the held-out remainder. `curl -s \"$GRADING_URL/health\"` reports the "
        "boundary under `validation.resolved_counts`.",
        "",
        "Cost: `$HOME/grading_baseline/` holds the base model's validation, and a status "
        "reply carries `baseline_sample_seconds`. Per-sample cost varies by orders of "
        "magnitude across benchmarks, so time a small `--limit` first and scale from that. "
        "A cancelled validation returns a partial that is not comparable to a score.",
        "",
        "Resolution: on N samples one trial decides 1/N of the score, so a gap of a few "
        "points on a small validation is indistinguishable from luck. Confirm a "
        "difference on more samples before treating it as real.",
        "",
    ]

    # The candidate procedure is protocol rather than run fact, but it belongs in this
    # document: the mechanics are identical for every benchmark, and only the harness
    # half of a pair depends on whether this run ships an editable harness.
    lines += [
        "## Candidates",
        "",
        "A validation candidate is an immutable snapshot in its own directory:",
        "",
        "```text",
        "candidates/<candidate_id>/",
        "├── final_model/      # complete Hugging Face model for this candidate",
    ]
    if has_harness:
        lines.append("├── harness/          # the matching editable harness")
    lines += [
        "├── candidate.json    # machine-readable metadata (non-empty `candidate_id`)",
        "└── candidate.md      # hypothesis, changes, validation result, decision",
        "```",
        "",
    ]
    submit_line = "Submit `candidates/<candidate_id>/final_model` through `grading.md`"
    if has_harness:
        submit_line += "; the service pairs the sibling `harness/` itself"
    lines += [
        submit_line + ". After submission, change nothing in the candidate; only append "
        "the validation record to `candidate.md`. A new idea means a new candidate id.",
        "",
    ]
    deliver = "Final delivery: copy the selected candidate's `final_model/`"
    if has_harness:
        deliver += " and `harness/`"
    deliver += (" to the workspace root and write `final_candidate.json` with the chosen "
                "`candidate_id`. The official evaluation after your session scores exactly "
                "this delivery.")
    lines += [deliver, ""]

    (task_dir / "environment.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-to-train", required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--train-recipe", default="")
    parser.add_argument("--train-resources", default="")
    args = parser.parse_args()

    repo = args.repo.resolve()
    task_dir = args.job_dir / "task"
    task_dir.mkdir(parents=True, exist_ok=True)

    if any((task_dir / name).exists() for name in RESERVED_TOP_LEVEL):
        existing = sorted(name for name in RESERVED_TOP_LEVEL if (task_dir / name).exists())
        fail(f"task directory already contains reserved file(s): {existing}")

    bench_dir = repo / "benchmarks" / args.benchmark
    if not bench_dir.is_dir():
        fail(f"benchmark directory not found: {bench_dir}")

    eval_code = bench_dir / "evaluation_code"
    if eval_code.is_dir():
        copy_path(eval_code, task_dir / "evaluation_code")

    copy_path(repo / "eval" / "templates", task_dir / "templates")
    copy_path(repo / "eval" / "agent_task" / "workspace_docs" / "grading.md", task_dir / "grading.md")
    copy_path(repo / "eval" / "agent_task" / "workspace_docs" / "agent_rules.md", task_dir / "agent_rules.md")
    copy_dir_contents(bench_dir / "task_context", task_dir)
    validation_context = copy_validation_context(repo, task_dir, bench_dir)
    harness_config = editable_harness_config(bench_dir)
    # The pinned package reference belongs to the benchmark's evaluation stack, not to
    # this run's editability: a model-only (VTB_EDITABLE_HARNESS=off) run is still scored
    # by the same mini-swe-agent, so the read-only source ships either way. What off
    # suppresses is the editable harness/ package and its document, not the reference.
    package_reference = (
        provision_harness_package_source(repo, task_dir, bench_dir, harness_config)
        if harness_config
        else None
    )
    if harness_config and os.environ.get("VTB_EDITABLE_HARNESS", "").strip().lower() == "off":
        # Run-level opt-out (a model-only ablation): the benchmark's declaration stays
        # the default; this run ships no harness package and no harness document.
        harness_config = None
        harness_doc = task_dir / "harness.md"
        if harness_doc.is_file():
            harness_doc.unlink()
    if harness_config:
        copy_editable_harness(repo, task_dir, harness_config)

    recipe_name, recipe_path, resource_ids, recipe = resolve_requested_resources(
        repo, args.train_recipe, args.train_resources, args.benchmark, args.model_to_train
    )

    copied_resources = []
    for resource_id in resource_ids:
        copied_resources.append(copy_training_resource(repo, task_dir, resource_id))

    manifest = {
        "benchmark": args.benchmark,
        "model_to_train": args.model_to_train,
        "recipe": recipe_name,
        "recipe_path": str(recipe_path.relative_to(repo)) if recipe_path else None,
        "direct_resources": resource_ids if recipe_path is None else [],
        "entrypoint_hint": recipe.get("entrypoint_hint") if recipe else None,
        "env_defaults": recipe.get("env_defaults", {}) if recipe else {},
        "resources": copied_resources,
        "editable_harness": harness_config or None,
        "package_reference": package_reference,
        "validation_context": validation_context,
    }
    with (task_dir / "train_resource_manifest.lock.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    # Written after the manifest, from the same facts: the manifest is the machine-readable
    # contract, this is the part a reader needs before it can act on it.
    write_environment_doc(
        task_dir, args.benchmark, args.model_to_train, recipe, copied_resources, repo,
        bool(harness_config), validation_context,
    )

    print(f"Prepared agent workspace: {task_dir}")
    if resource_ids:
        print(f"Copied {len(resource_ids)} training resource(s): {', '.join(resource_ids)}")
    else:
        print("No training resources selected")
    if package_reference:
        print(
            "Harness package reference: "
            f"{package_reference['name']}=={package_reference['version']} -> {package_reference['ref']}"
        )


if __name__ == "__main__":
    main()
