#!/usr/bin/env python3
"""Check or sanitize tracked publication metadata without touching local runtime files."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Generic task/container accounts are not developer identities.
GENERIC = {"agent", "user", "runner", "sandbox", "ubuntu", "root", "public", "runtime",
           "data", "cache", "models", "datasets", "workspace", "shared", "tmp"}
ACCOUNT_PATH = re.compile(r"(?P<base>/(?:home|Users|mnt)/)(?P<account>[A-Za-z0-9_.@-]+)(?=/|\b)")
WINDOWS_HOME = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+(?!user\b)[A-Za-z0-9_.@-]+")
LOGIN = re.compile(r"(?<![\w./])([A-Za-z0-9_.-]+)@([A-Za-z0-9_.-]+)(?=[:#]\s*[/~]|[#$]\s)")
# These fields are operator metadata, not benchmark customer identities or attribution.
IDENTITY_KEYS = {"hostname", "host_name", "host_user", "username", "operator", "created_by"}


def sanitize_text(text: str) -> str:
    def replace(match):
        if match["account"] in GENERIC:
            return match[0]
        return match["base"] + ("runtime" if match["base"] == "/mnt/" else "user")
    text = ACCOUNT_PATH.sub(replace, text)
    text = WINDOWS_HOME.sub("C:/Users/user", text)
    return LOGIN.sub(lambda match: (match[1] if match[1] in GENERIC else "user") + "@host", text)


def sanitize_document(value, *, artifact_metadata=False):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if artifact_metadata and key == "trial_uri" and isinstance(item, str) and item.startswith("file:///"):
                result[key] = "."
            elif artifact_metadata and key == "trials_dir" and isinstance(item, str) and item.startswith("/"):
                result[key] = ".."
            elif artifact_metadata and key in IDENTITY_KEYS and isinstance(item, str):
                result[key] = "redacted"
            else:
                result[key] = sanitize_document(item, artifact_metadata=artifact_metadata)
        return result
    if isinstance(value, list):
        return [sanitize_document(item, artifact_metadata=artifact_metadata) for item in value]
    return sanitize_text(value) if isinstance(value, str) else value


def cleaned_content(path: Path, relative: Path) -> tuple[bytes, bytes]:
    before = path.read_bytes()
    try:
        text = before.decode("utf-8")
    except UnicodeDecodeError:
        return before, before
    if relative.suffix == ".json":
        value = json.loads(text)
        metadata = ("baseline" in relative.parts or "validation_context" in relative.parts) and path.name in {"config.json", "result.json", "lock.json"}
        clean = sanitize_document(value, artifact_metadata=metadata)
        if clean == value:
            return before, before
        # Preserve the existing indent style; only changed documents are rewritten.
        indent_match = re.search(r"\n( +)\"", text)
        indent = len(indent_match[1]) if indent_match else 2
        after = (json.dumps(clean, indent=indent, ensure_ascii=False) + "\n").encode()
    else:
        cleaned = sanitize_text(text)
        if "baseline" in relative.parts or "validation_context" in relative.parts:
            # Redacted diagnostic lines must not introduce trailing-whitespace diffs.
            lines = cleaned.splitlines(keepends=True)
            for index, line in enumerate(lines):
                if any(marker in line for marker in ("/home/user", "/Users/user", "/mnt/runtime", "C:/Users/user", "@host:")):
                    newline = "\n" if line.endswith("\n") else ""
                    lines[index] = line.rstrip(" \t\r\n") + newline
            cleaned = "".join(lines)
        after = cleaned.encode()
    return before, after


def tracked_paths(repo: Path) -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=repo)
    return sorted({Path(name.decode()) for name in raw.split(b"\0") if name})


def process(repo: Path, *, apply=False) -> list[Path]:
    changed = []
    for relative in tracked_paths(repo):
        if relative.parts[0] in {".local", "vtb-env", ".git"} or relative.name in {"api_key.env", ".env"}:
            raise ValueError("local-only content was selected for publication")
        path = repo / relative
        if not path.is_file() or path.is_symlink():
            continue
        before, after = cleaned_content(path, relative)
        if before != after:
            changed.append(relative)
            if apply:
                path.write_bytes(after)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="sanitize tracked/unignored files; never local state or Git history")
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    changed = process(args.repo.resolve(), apply=args.apply)
    print(f"PUBLICATION_METADATA_{'CLEANED' if args.apply else 'CHECK'} files={len(changed)}")
    for path in changed:
        print(path)
    return 0 if args.apply or not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
