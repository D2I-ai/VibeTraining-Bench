#!/usr/bin/env python3
"""Direct AIME 2025 evaluator."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _direct_aime import main as _main  # noqa: E402


def main(argv=None) -> int:
    return _main(
        "aime__aime",
        "opencompass/AIME2025",
        "a6ad95f611d72cf628a80b58bd0432ef6638f958",
        "VTB_AIME_DATASET_DIR",
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
