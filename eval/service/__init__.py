"""Deployable evaluation-side services.

This is a functional layer under eval/, a sibling of grading/ and runner/. It
holds long-lived HTTP services that a benchmark adapter reaches over the
network to score a model -- the same shape as the resident grading service.

Each subpackage here is one deployable service instance (e.g. toolathlon_gym,
toolathlon_verified). By design a service under this package:

  * imports nothing from eval.runner or eval.grading -- it is a self-contained
    island that talks to the rest of the system only through the benchmark
    adapter's CLI contract and its own HTTP surface;
  * resolves the external, read-only upstream repo it drives from its own
    env var (VTB_TOOLATHLON_*_REPO), the same way eval/runner/task_backends.py
    resolves VTB_HARBOR_REPO -- the backend owns its own path resolution, so
    eval/runner/runtime_paths.py stays a generic path layer;
  * never writes into that upstream checkout: customisations live in the
    service's own overlay/ and are layered on top of a throwaway staged copy.
"""
