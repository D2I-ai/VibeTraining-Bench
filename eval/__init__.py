"""Evaluation: benchmark execution and the token-gated grading service.

A package so `python3 -m eval.grading.server` works and grading's relative imports
resolve. The name shadows the `eval` builtin only as a module name, never as a
callable in any module that does not import it.
"""
