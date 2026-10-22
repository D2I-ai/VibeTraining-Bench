#!/usr/bin/env python3
"""swe-bench-verified: 500 SWE-bench tasks, resolved from the Harbor registry.

A shim. There is no task tree here: harbor downloads and caches the dataset by name,
so this directory holds only the entry point and whatever the agent is allowed to see.
Everything else is in benchmarks/registry.py (which benchmark) and eval/ (how to run one).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # the repo root, for the eval package

from eval.runner.run import main  # noqa: E402

if __name__ == "__main__":
    main(benchmark="swe-bench__swe-bench-verified", task_dir=HERE)
