"""Benchmark execution layer.

The benchmark definition lives in `benchmarks/<name>/benchmark.json`; this package only
executes that spec and converts the backend's trial tree into the metrics contract.

Module map:
- run.py: CLI/shim entry point and orchestration.
- task_backends.py: Harbor command builders and child process environment.
- model_server.py: serving the checkpoint locally (docker sglang, or a host server)
  when no --api-base is supplied. Also the one definition of the port arithmetic and the
  card/tensor-parallel resolution that eval.grading.executor uses when it serves a
  submission.
- trial_results.py: collect Harbor trial output and write metrics.json.
- runtime_paths.py: all vtb-env paths and docker-daemon selection.
"""
