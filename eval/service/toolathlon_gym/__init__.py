"""Toolathlon-GYM evaluation service (self-contained island).

app.py    FastAPI surface (default :8060) a benchmark adapter POSTs to.
runner.py per-task container orchestration.
paths.py  resolves the external official gym checkout + applies this
          package's overlay onto a throwaway staged copy.
"""
