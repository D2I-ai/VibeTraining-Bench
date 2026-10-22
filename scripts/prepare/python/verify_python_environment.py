#!/usr/bin/env python3
"""Verify the selected Python minor and an optional exact dependency manifest."""
from __future__ import annotations

import argparse
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def errors(requirements: Path | None = None) -> list[str]:
    expected = (REPO / ".python-version").read_text().strip()
    failures = []
    if ".".join(map(str, sys.version_info[:2])) != expected:
        failures.append(f"Python {expected} is required")
    if requirements:
        for raw in requirements.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            match = re.fullmatch(r"([\w.-]+)(?:\[[\w,.-]+\])?==([^\s]+)", line)
            if not match:
                failures.append(f"requirement must be an exact package pin: {line}")
                continue
            name, wanted = match.groups()
            try:
                found = version(name)
            except PackageNotFoundError:
                found = "not installed"
            if found != wanted:
                failures.append(f"{name}: {found}; required {wanted}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path)
    args = parser.parse_args()
    failures = errors(args.requirements)
    if failures:
        parser.exit(1, "\n".join(failures) + "\n")
    print(f"PYTHON_ENVIRONMENT_OK version={sys.version.split()[0]}")


if __name__ == "__main__":
    main()
