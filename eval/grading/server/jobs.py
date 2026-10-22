"""Queue-side value objects: the Job record, mode constants, loop metrics.

These types carry no service state; both the service core and the HTTP layer
use them, so they live below both.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional
from ..store import RunContext


# An agent submission is always scored on the validation split; only the
# operator-side token may ask for the whole benchmark.
VALID_MODES = ("val", "final")


class Conflict(Exception):
    """A request that is well formed but arrived too late to be carried out.

    Its own type because the handler maps exceptions to status codes: cancelling an
    evaluation that has already finished is not a bad request and not a missing one.
    """


# Share of trials that stopped making progress, above which an evaluation's duration is
# not kept as history. A healthy queue measured 0 of 147; the run this threshold was
# drawn from measured 53 of 149, or 0.36. The gap between those is wide enough that 0.2
# separates pathology from variance without discarding a merely unlucky evaluation.
NO_PROGRESS_DURATION_LIMIT = 0.2


def _no_progress_share(metrics: Optional[dict]) -> Optional[float]:
    """Fraction of graded trials that stopped making progress, or None if unmeasured.

    Benchmarks that do not carry per-trial transcripts report `no_progress_trials` as
    null, and a missing measurement must not read as zero -- that would let exactly the
    evaluations this guard exists for through.
    """
    if not isinstance(metrics, dict):
        return None
    stuck = metrics.get("no_progress_trials")
    graded = metrics.get("graded")
    if not isinstance(stuck, int) or not isinstance(graded, int) or graded <= 0:
        return None
    return stuck / graded


class Job:
    __slots__ = (
        "job_id",
        "token",
        "run_id",
        "task",
        "mode",
        "limit",
        "model_path",
        "raw_model_path",
        "write_metrics",
        "state",
        "phase",
        "wait_detail",
        "submitted_at",
        "started_at",
        "finished_at",
        "result",
        "attempt_phase",
        "attempt_index",
        "current_log",
        "target_samples",
        "role",
        "eval_task",
        "cancel_reason",
        "sample_ids",
        "service_split",
        "eval_ids",
        "harness_snapshot",
        "harness_import_path",
        "harness_fingerprint",
        "candidate_id",
    )

    def __init__(
        self,
        job_id: str,
        token: str,
        ctx: RunContext,
        mode: str,
        limit: int,
        model_path: Path,
        raw_model_path: str,
        write_metrics: bool,
        eval_task: Optional[str] = None,
        harness_submission: Optional[dict[str, Any]] = None,
    ) -> None:
        self.job_id = job_id
        self.token = token
        self.run_id = ctx.run_id
        self.task = ctx.task
        self.role = ctx.role
        # Which benchmark's evaluate.py runs. Differs from task only when
        # validation is configured to use a separate benchmark.
        self.eval_task = eval_task or ctx.task
        self.mode = mode
        self.limit = limit
        self.model_path = model_path
        self.raw_model_path = raw_model_path
        self.write_metrics = write_metrics
        self.state = "queued"
        self.phase: Optional[str] = None
        self.wait_detail: Optional[dict[str, Any]] = None
        self.submitted_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.result: Optional[dict] = None
        # Updated by the worker as the ladder advances, so /result can tail the
        # log of whichever attempt is currently running.
        self.attempt_phase: Optional[int] = None
        self.attempt_index: Optional[int] = None
        self.current_log: Optional[Path] = None
        # Learned once from inspect's log; see GradingService._target_samples.
        self.target_samples: Optional[int] = None
        # Why the run asked for this evaluation to stop, when it did. Kept because
        # when an agent gives up on a validation, and what it said it was going to do
        # instead, is part of the iteration being observed.
        self.cancel_reason: Optional[str] = None
        # The samples this submission named, when it named any. None means the
        # service's own choice of split prefix.
        self.sample_ids: Optional[list] = None
        # True when sample_ids came from the service's own file: split rather than
        # from the submission naming samples. Same --only plumbing, opposite
        # meaning downstream: a service-owned list is the whole split, so it is
        # comparable to a full evaluation and may write official metrics, while a
        # submission-chosen subset is neither.
        self.service_split: bool = False
        # Service-computed exact evaluation set for an official final: the held-out
        # remainder of the benchmark (everything after the validation prefix). Set at
        # submit; distinct from sample_ids, which is an agent-chosen subset with
        # not-a-real-score semantics. An eval_ids final IS the official measurement.
        self.eval_ids: Optional[list] = None
        self.harness_snapshot: Optional[Path] = None
        self.harness_import_path = harness_submission["import_path"] if harness_submission else None
        self.harness_fingerprint = harness_submission["fingerprint"] if harness_submission else None
        self.candidate_id = harness_submission["candidate_id"] if harness_submission else None

    @property
    def is_subset(self) -> bool:
        """Whether sample_ids were the submission's own pick.

        The service's file: split reaches the runner through the same --only
        plumbing, but it is the whole boundary rather than a chosen piece of one,
        so every gate that keeps a self-picked subset from posing as a full score
        lets it through.
        """
        return bool(self.sample_ids) and not self.service_split

    def harness_metadata(self) -> dict[str, Optional[str]]:
        return {
            "harness_import_path": self.harness_import_path,
            "harness_fingerprint": self.harness_fingerprint,
            "candidate_id": self.candidate_id,
        }

    def public(self) -> dict:
        """Everything except the token and the private harness snapshot path."""
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "task": self.task,
            "eval_task": self.eval_task,
            "role": self.role,
            "mode": self.mode,
            "limit": self.limit,
            "sample_ids": self.sample_ids,
            "service_split": self.service_split,
            "heldout_only": bool(self.eval_ids),
            "model_path": self.raw_model_path,
            "state": self.state,
            "phase": self.phase,
            "wait_detail": self.wait_detail,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancel_reason": self.cancel_reason,
            "result": self.result,
            **self.harness_metadata(),
        }
