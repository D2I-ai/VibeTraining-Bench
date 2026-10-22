"""Authoritative score-affecting backend pins parsed from their install manifests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HARBOR_REQUIREMENTS = REPO_ROOT / "containers" / "requirements.txt"
_HARBOR_PATTERN = re.compile(
    r"^harbor\s*@\s*git\+(?P<url>.+)@(?P<commit>[0-9a-fA-F]{40})$"
)


@dataclass(frozen=True)
class GitPin:
    url: str
    commit: str
    requirement: str


def harbor_pin(path: Path = HARBOR_REQUIREMENTS) -> GitPin:
    """Read the one operational Harbor pin from containers/requirements.txt."""
    matches: list[GitPin] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _HARBOR_PATTERN.fullmatch(line)
        if match:
            matches.append(
                GitPin(
                    url=match.group("url"),
                    commit=match.group("commit").lower(),
                    requirement=line,
                )
            )
        elif re.match(r"^harbor(?:\s|[<>=!~@])", line, re.IGNORECASE):
            raise RuntimeError(
                f"Harbor must use an exact 40-character git commit in {path}: {line!r}"
            )
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one Harbor git pin in {path}, found {len(matches)}")
    return matches[0]


HARBOR_PIN = harbor_pin()
HARBOR_REPO_URL = HARBOR_PIN.url
HARBOR_COMMIT = HARBOR_PIN.commit
