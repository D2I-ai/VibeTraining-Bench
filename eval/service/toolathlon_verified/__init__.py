"""Toolathlon-VERIFIED evaluation service (self-contained island).

app.py    FastAPI surface (default :8061) a benchmark adapter POSTs to.
runner.py per-task container orchestration (official phased flow:
          preprocess -> stash -> agent -> restore -> eval).
paths.py  resolves the external official verified checkout + applies this
          package's overlay and the per-host secrets onto a throwaway
          staged copy.
"""
