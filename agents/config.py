"""Operator-selected adapter paths, model names, and credential boundaries."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
# Operator/grading authority must never be made available to researcher adapters.
RESERVED = {
    "VTB_ADMIN_TOKEN_FILE", "VTB_GRADING_STATE", "VTB_ROOT", "VTB_REPO",
    "VTB_AGENT_HOME", "HOME", "PATH", "PYTHONPATH", "LD_PRELOAD",
    "GRADING_URL", "PROMPT", "AGENT_CONFIG", "NVIDIA_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES", "VTB_MODEL_RESOURCE_ROOT", "HF_HOME",
    "VTB_AGENT_ADAPTERS_DIR", "VTB_LOCAL_ENV_FILE", "VTB_PYTHON", "PYTHONHOME",
}


def model_slug(model: str) -> str:
    if not isinstance(model, str) or not model.strip() or any(ord(c) < 32 or ord(c) == 127 or c == "'" for c in model):
        raise ValueError("RESEARCHER must be a nonempty model ID without shell quotes or control characters")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model)


def resolve_adapter(name: str, repo: Path = REPO) -> Path:
    if not NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError("unsafe adapter name")
    override = os.environ.get("VTB_AGENT_ADAPTERS_DIR")
    root = Path(override).expanduser().resolve() if override else (repo / "agents").resolve()
    adapter = (root / name).resolve()
    if adapter.parent != root or not (adapter / "solve.sh").is_file():
        raise ValueError(f"adapter entrypoint missing or outside adapter root: {root / name / 'solve.sh'}")
    return adapter


def read_config(name: str, repo: Path = REPO) -> tuple[Path, dict]:
    adapter = resolve_adapter(name, repo)
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("adapter_config.json must be an object")
    for field in ("required_env", "forward_env"):
        names = cfg.get(field)
        if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
            raise ValueError(f"adapter {name}: {field} must be a list of environment variable names")
        if len(set(names)) != len(names):
            raise ValueError(f"adapter {name}: duplicate environment variable in {field}")
        for item in names:
            if not ENV_NAME.fullmatch(item) or item in RESERVED or item.startswith("GRADING_") or "ADMIN" in item or "OFFICIAL" in item:
                raise ValueError(f"adapter {name}: forbidden environment variable: {item}")
    if not set(cfg["required_env"]).issubset(cfg["forward_env"]):
        raise ValueError(f"adapter {name}: required_env must be included in forward_env")
    return adapter, cfg


def validate(name: str, researcher: str, repo: Path = REPO) -> tuple[Path, dict]:
    model_slug(researcher)
    adapter, cfg = read_config(name, repo)
    missing = [key for key in cfg["required_env"] if not os.environ.get(key, "").strip()]
    if missing:
        raise ValueError(f"adapter {name}: missing required environment variables: {', '.join(missing)}")
    return adapter, cfg


def forwarded_environment(cfg: dict) -> dict[str, str]:
    # Preserve explicit empty strings: some CLI authentication modes require them.
    return {key: os.environ[key] for key in cfg["forward_env"] if key in os.environ}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "path", "env-names", "model-slug", "shell-inputs", "result-path", "record-result"))
    parser.add_argument("adapter", nargs="?")
    parser.add_argument("--researcher", default=os.environ.get("RESEARCHER", ""))
    parser.add_argument("--state", type=Path)
    parser.add_argument("--default-result-dir", default="")
    args = parser.parse_args()
    try:
        if args.action in {"result-path", "record-result"}:
            if args.state is None:
                raise ValueError("--state is required")
            manifest = args.state / "smoke_run.json"
            if args.action == "record-result":
                validate(args.adapter or "claude", args.researcher)
                record = {"adapter": args.adapter or "claude", "researcher": args.researcher,
                          "result_dir": str(Path(args.default_result_dir).resolve())}
                if manifest.exists() and json.loads(manifest.read_text()) != record:
                    raise ValueError("smoke state belongs to another researcher/result directory")
                manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            elif manifest.is_file():
                record = json.loads(manifest.read_text())
                if args.researcher and (record.get("researcher") != args.researcher or record.get("adapter") != args.adapter):
                    raise ValueError("smoke state belongs to another researcher; keep launch inputs consistent")
                print(record["result_dir"])
            else:
                print(args.default_result_dir)
        elif args.action == "model-slug":
            print(model_slug(args.researcher))
        elif args.action == "shell-inputs":
            adapter, _ = validate(args.adapter or "claude", args.researcher)
            print(f"AGENT_ADAPTER_DIR={shlex.quote(str(adapter))}")
            print(f"RESEARCHER_SAFE={shlex.quote(model_slug(args.researcher))}")
        elif args.action == "check":
            validate(args.adapter or "claude", args.researcher)
            print(f"AGENT_ADAPTER_OK adapter={args.adapter or 'claude'}")
        else:
            adapter, cfg = read_config(args.adapter or "claude")
            if args.action == "path":
                print(adapter)
            else:
                for name in cfg["forward_env"]:
                    print(name)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
