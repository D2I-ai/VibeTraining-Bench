#!/usr/bin/env python3
"""Direct AIME 2026 evaluator."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _direct_aime import main as _main  # noqa: E402


def main(argv=None) -> int:
    return _main(
        "aime__26",
        "MathArena/aime_2026",
        "d2de22f3c656b4f56cf8981212186377d1e23bc3",
        "VTB_AIME26_DATASET_DIR",
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
