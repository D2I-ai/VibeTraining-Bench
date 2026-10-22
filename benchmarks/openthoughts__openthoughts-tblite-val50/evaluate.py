#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from eval.runner.run import main  # noqa: E402

if __name__ == "__main__":
    main(benchmark="openthoughts__openthoughts-tblite-val50", task_dir=HERE)
