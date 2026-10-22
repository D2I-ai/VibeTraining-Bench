#!/usr/bin/env python3
"""Assert an image's installed stack falls inside containers/model-stack.txt.

Run at build time in every image that touches model weights, and usable
afterwards as a preflight on a host or in a running container:

    docker run --rm vibetrainbench/trainer:latest python3 /opt/vtb/check_stack.py

A failure here is the whole point: an image whose transformers major version
disagrees with the trainer's reads a checkpoint written by the trainer without
complaining, and gets the wrong answer. Catching that at build time costs a
rebuild; catching it later costs a benchmark run nobody can trust.

Deliberately dependency-free -- not even `packaging`. It has to be runnable in
whatever interpreter is at hand, including a bare host python used to sanity
check the contract before any image exists.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

CONTRACT = Path(__file__).resolve().parent / "model-stack.txt"

_REQ = re.compile(r"^([A-Za-z0-9._-]+)\s*(.*)$")


def parse_version(text: str) -> tuple:
    """Leading numeric release segments, which is all this contract compares.

    A local segment (torch's `+cu129`) and any pre-release suffix are dropped:
    the contract is about which minor a stack is on, not about build metadata.
    """
    release = text.split("+", 1)[0]
    parts = []
    for chunk in release.split("."):
        match = re.match(r"^(\d+)", chunk)
        if match is None:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


def satisfies(found: str, clause: str) -> bool:
    match = re.match(r"^(==|!=|>=|<=|>|<)\s*(.+)$", clause)
    if match is None:
        raise SystemExit(f"check_stack: cannot parse contract clause {clause!r}")
    op, wanted = match.group(1), match.group(2).strip()

    if wanted.endswith(".*"):
        prefix = parse_version(wanted[:-2])
        got = parse_version(found)[: len(prefix)]
        return (got == prefix) if op == "==" else (got != prefix)

    got, want = parse_version(found), parse_version(wanted)
    width = max(len(got), len(want))
    got += (0,) * (width - len(got))
    want += (0,) * (width - len(want))
    return {
        "==": got == want,
        "!=": got != want,
        ">=": got >= want,
        "<=": got <= want,
        ">": got > want,
        "<": got < want,
    }[op]


def clauses(path: Path) -> list:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, spec = _REQ.match(line).groups()
        out.append((name, [c.strip() for c in spec.split(",") if c.strip()]))
    return out


def main() -> int:
    if not CONTRACT.is_file():
        print(f"check_stack: no contract at {CONTRACT}", file=sys.stderr)
        return 2

    failures = []
    python_contract = CONTRACT.parent / ".python-version"
    if not python_contract.is_file():
        python_contract = CONTRACT.parent.parent / ".python-version"
    if not python_contract.is_file():
        print("check_stack: Python contract missing", file=sys.stderr)
        return 2
    wanted_python = python_contract.read_text().strip()
    found_python = ".".join(map(str, sys.version_info[:2]))
    if found_python != wanted_python:
        failures.append(f"  Python {found_python}: contract requires {wanted_python}")
    for name, spec in clauses(CONTRACT):
        try:
            found = version(name)
        except PackageNotFoundError:
            failures.append(f"  {name}: not installed, contract wants {','.join(spec)}")
            continue
        broken = [c for c in spec if not satisfies(found, c)]
        if broken:
            failures.append(f"  {name}=={found} violates {','.join(broken)}")
        else:
            print(f"  ok   {name}=={found}")

    if failures:
        print(f"\ncheck_stack: installed stack is outside {CONTRACT.name}:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        print(
            "\nEither pick a base image inside the window, or -- if the whole fleet is "
            "moving together -- widen the contract and rebuild every image in containers/.",
            file=sys.stderr,
        )
        return 1

    print("check_stack: stack agrees with the contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
