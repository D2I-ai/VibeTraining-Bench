#!/usr/bin/env python3
"""Register a Harbor Hub dataset as a local vibetrainbench benchmark.

If VTB_HARBOR_REPO points at a Harbor checkout, this reads adapter metadata from
that checkout and records the adapter's intended harness in benchmark.json. The
metadata is advisory for registration, but the selected agent is used by the
runner, so evaluation follows Harbor's adapter intent when the metadata is
available.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

DATASET_ADAPTER_OVERRIDES = {
    "aime/aime": "aime",
    "bigcode/humanevalfix": "humanevalfix",
    "gorilla/bfcl_parity": "bfcl",
    "gpqa-diamond/gpqa-diamond": "gpqa-diamond",
    "sierra-research/tau3-bench": "tau3-bench",
    "swe-bench/swe-bench-verified": "swebench",
}

DATASET_TOTAL_OVERRIDES = {
    # The AIME adapter has no local adapter_metadata.json. Its generator builds the
    # registered dataset from AIME Preview 2024 plus 2025-I and 2025-II JSONLs,
    # matching the 60-example local AIME meta-agent eval/test mirrors.
    "aime/aime": 60,
}

AGENT_ENV_OVERRIDES = {
    "sierra-research/tau3-bench": "VTB_TAUBENCH_AGENT",
    "bigcode/humanevalfix": "VTB_HUMANEVALFIX_AGENT",
    "swe-bench/swe-bench-verified": "VTB_SWEBENCH_AGENT",
}

DEFAULT_HARBOR_AGENT = os.environ.get("VTB_HUB_DEFAULT_AGENT", "codex")

# Adapter metadata says what Harbor used for parity. By default, registration follows
# that parity agent when it exists, because Harbor task instructions are often written
# for that harness. If metadata has no parity/custom agent, registration falls back to
# `VTB_HUB_DEFAULT_AGENT` (default: codex).
USE_PARITY_AGENT_FOR_LLM = (os.environ.get("VTB_HUB_USE_PARITY_AGENT_FOR_LLM") or "1").lower() not in {
    "0",
    "false",
    "no",
}


def fail(message: str) -> None:
    raise SystemExit(f"FATAL: {message}")


def safe_env_name(alias: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", alias).strip("_").upper()
    return f"VTB_{token}_AGENT"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metadata(adapter_dir: Path) -> dict[str, Any] | None:
    path = adapter_dir / "adapter_metadata.json"
    if not path.is_file():
        return None
    payload = load_json(path)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        return payload
    return None


def candidate_adapter_names(dataset: str, alias: str) -> list[str]:
    owner, name = dataset.split("/", 1)
    values = [
        DATASET_ADAPTER_OVERRIDES.get(dataset),
        name,
        name.replace("_", "-"),
        name.replace("-", "_"),
        owner,
        alias,
        alias.replace("__", "-"),
    ]
    seen: set[str] = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def find_adapter(harbor_repo: Path | None, dataset: str, alias: str) -> tuple[Path | None, dict[str, Any] | None]:
    if harbor_repo is None:
        return None, None
    adapters_dir = harbor_repo / "adapters"
    if not adapters_dir.is_dir():
        return None, None

    for name in candidate_adapter_names(dataset, alias):
        adapter_dir = adapters_dir / name
        if adapter_dir.is_dir():
            return adapter_dir, read_metadata(adapter_dir)

    needle_parts = set(re.split(r"[^A-Za-z0-9]+", dataset.lower())) - {"", "bench", "benchmark"}
    for metadata_path in sorted(adapters_dir.glob("*/adapter_metadata.json")):
        metadata = read_metadata(metadata_path.parent)
        if not metadata:
            continue
        haystack = json.dumps(metadata, ensure_ascii=False).lower()
        if all(part in haystack for part in needle_parts):
            return metadata_path.parent, metadata
    return None, None


def first_adapter_record(metadata: dict[str, Any] | None) -> dict[str, Any]:
    records = (metadata or {}).get("harbor_adapter") or []
    return records[0] if records and isinstance(records[0], dict) else {}


def first_original_record(metadata: dict[str, Any] | None) -> dict[str, Any]:
    records = (metadata or {}).get("original_benchmark") or []
    return records[0] if records and isinstance(records[0], dict) else {}


def positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def dataset_total(dataset: str, alias: str, old: dict[str, Any], metadata: dict[str, Any] | None) -> int | None:
    old_total = positive_int(old.get("dataset_total"))
    if old_total is not None:
        return old_total

    adapter_records = [record for record in ((metadata or {}).get("harbor_adapter") or []) if isinstance(record, dict)]
    if adapter_records:
        selected = adapter_records[0]
        if "parity" in dataset.rsplit("/", 1)[-1] or "parity" in alias:
            selected = next(
                (record for record in adapter_records if "parity" in str(record.get("split") or "")),
                selected,
            )
        for key in ("registry_benchmark_size", "adapted_benchmark_size", "parity_benchmark_size"):
            value = positive_int(selected.get(key))
            if value is not None:
                return value

    original_size = positive_int(first_original_record(metadata).get("size"))
    if original_size is not None:
        return original_size
    return DATASET_TOTAL_OVERRIDES.get(dataset)


def parse_parity_agent(value: str) -> tuple[str, dict[str, str]]:
    token = value.split("+", 1)[0].strip()
    if "@" not in token:
        return token, {}
    name, version = token.split("@", 1)
    if not version:
        return name, {}
    return name, {"version": version}


def find_import_path(adapter_dir: Path, added_agents: list[str]) -> str | None:
    texts = []
    for path in sorted(adapter_dir.glob("*.yaml")) + sorted(adapter_dir.glob("*.yml")):
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    text = "\n".join(texts)
    if not text:
        return None

    import_paths = re.findall(r"import_path:\s*([^\s#]+)", text)
    if not import_paths:
        return None
    if not added_agents:
        return import_paths[0]

    normalized_agents = {re.sub(r"[^a-z0-9]+", "", item.lower()) for item in added_agents}
    for import_path in import_paths:
        normalized_path = re.sub(r"[^a-z0-9]+", "", import_path.lower())
        if any(agent in normalized_path for agent in normalized_agents):
            return import_path
    return import_paths[0]


def adapter_summary(
    adapter_dir: Path | None,
    metadata: dict[str, Any] | None,
    harbor_repo: Path | None,
) -> dict[str, Any] | None:
    if adapter_dir is None and metadata is None:
        return None
    original = first_original_record(metadata)
    adapter = first_adapter_record(metadata)
    adapter_path = None
    if adapter_dir is not None:
        if harbor_repo is not None:
            try:
                adapter_path = str(adapter_dir.relative_to(harbor_repo))
            except ValueError:
                adapter_path = adapter_dir.name
        else:
            adapter_path = adapter_dir.name
    return {
        "name": (metadata or {}).get("adapter_name") or (adapter_dir.name if adapter_dir else None),
        "path": adapter_path,
        "repo_env": "VTB_HARBOR_REPO" if adapter_dir else None,
        "original_harness": original.get("harness"),
        "supported_agents": original.get("supported_agents"),
        "added_agents": adapter.get("added_agents") or [],
        "parity_matching_agents": adapter.get("parity_matching_agents") or [],
        "metadata_source": "VTB_HARBOR_REPO" if adapter_dir else None,
    }


def choose_agent(dataset: str, alias: str, adapter_dir: Path | None, metadata: dict[str, Any] | None) -> tuple[Any, bool, dict[str, str]]:
    original = first_original_record(metadata)
    adapter = first_adapter_record(metadata)
    original_harness = original.get("harness")
    added_agents = [str(item) for item in (adapter.get("added_agents") or []) if item]
    parity_agents = [str(item) for item in (adapter.get("parity_matching_agents") or []) if item]

    if adapter_dir is not None and added_agents:
        import_path = find_import_path(adapter_dir, added_agents)
        if import_path:
            env = AGENT_ENV_OVERRIDES.get(dataset) or safe_env_name(alias)
            return {"default": import_path, "env": env}, True, {}

    if parity_agents and (original_harness != "llm" or USE_PARITY_AGENT_FOR_LLM):
        agent, kwargs = parse_parity_agent(parity_agents[0])
        return agent, original_harness != "llm", kwargs

    if original_harness == "agent":
        supported = original.get("supported_agents") or []
        if supported:
            return str(supported[0]).lower(), True, {}
        return "mini-swe-agent", True, {}

    return DEFAULT_HARBOR_AGENT, False, {}


def main() -> None:
    dataset = os.environ["DATASET"]
    alias = os.environ["ALIAS"]
    bench_dir = Path(os.environ["BENCH_DIR"])
    harbor_repo_env = os.environ.get("VTB_HARBOR_REPO")
    harbor_repo = Path(harbor_repo_env).resolve() if harbor_repo_env else None

    spec_path = bench_dir / "benchmark.json"
    shim_path = bench_dir / "evaluate.py"

    old: dict[str, Any] = {}
    if spec_path.exists():
        old = load_json(spec_path)
        if old.get("dataset") != dataset:
            fail(
                f"{spec_path} already names dataset {old.get('dataset')!r}, not {dataset!r}; "
                "set VTB_HUB_ALIAS to a different alias"
            )
        if "backend" not in old and "framework" in old:
            old["backend"] = old.pop("framework")

    adapter_dir, metadata = find_adapter(harbor_repo, dataset, alias)
    agent, agentic, agent_kwargs = choose_agent(dataset, alias, adapter_dir, metadata)
    summary = adapter_summary(adapter_dir, metadata, harbor_repo)
    total = dataset_total(dataset, alias, old, metadata)

    spec = {
        **old,
        "name": alias,
        "backend": "harbor",
        "agent": agent,
        "agentic": agentic,
        "default_limit": int(old.get("default_limit", -1)),
        "dataset": dataset,
        "environment": old.get("environment") or "docker",
        "notes": old.get("notes")
        or f"Generated Harbor Hub registration for {dataset}. Heavy task exports live under vtb-env/hub_task_specs/{alias}.",
    }
    if total is not None:
        spec["dataset_total"] = total
    if agent_kwargs:
        spec["agent_kwargs"] = {**dict(old.get("agent_kwargs") or {}), **agent_kwargs}
    elif "agent_kwargs" in old:
        spec["agent_kwargs"] = old["agent_kwargs"]
    if summary:
        spec["adapter"] = summary
    elif "adapter" in old:
        spec.pop("adapter", None)

    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not shim_path.exists():
        shim_path.write_text(
            "#!/usr/bin/env python3\n"
            "from __future__ import annotations\n\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "HERE = Path(__file__).resolve().parent\n"
            "sys.path.insert(0, str(HERE.parents[1]))\n\n"
            "from eval.runner.run import main  # noqa: E402\n\n"
            "if __name__ == \"__main__\":\n"
            f"    main(benchmark={alias!r}, task_dir=HERE)\n",
            encoding="utf-8",
        )
        shim_path.chmod(0o755)

    if summary:
        print(f"[hub] adapter metadata: {summary['name']} -> agent={agent!r}")
    elif harbor_repo_env:
        print(f"[hub] no local Harbor adapter metadata found for {dataset}; agent={agent!r}")


if __name__ == "__main__":
    main()
