"""GradingService core: submission, validation planning, queue, lifecycle,
and the single worker loop. Disclosure and archiving live in the mixins this
class composes.
"""

from __future__ import annotations

import hmac
import json
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from ...runner import trial_results
from .. import config, executor, progress, valsplit
from ..store import BaselineCache, DurationStats, Registry, RunContext, SubmissionLog, TaskStats, UNREADABLE_MARKER, admin_token, check_weights_complete, model_fingerprint

from .archive import ArchiveMixin
from .disclosure import DisclosureMixin
from .harness import _discard_success_scratch, _editable_harness_submission, _snapshot_harness
from .jobs import Conflict, Job, NO_PROGRESS_DURATION_LIMIT, VALID_MODES, _no_progress_share


class GradingService(DisclosureMixin, ArchiveMixin):

    def __init__(self) -> None:
        self.registry = Registry()
        self.submissions = SubmissionLog()
        self.baselines = BaselineCache()
        self.durations = DurationStats()
        self.task_stats = TaskStats()
        # Operator-side secret for the endpoints that hand out authority; see
        # store.admin_token for why an unauthenticated /register was not safe.
        self.admin_token = admin_token()
        self.val_source = valsplit.parse_val_source(config.VAL_SOURCE_RAW or None)
        # A non-empty VTB_VAL_SOURCE is an operator-wide override. Otherwise each
        # benchmark may declare its own validation_source in benchmark.json, with the
        # parse default remaining the fallback.
        self._benchmark_val_sources: dict[str, valsplit.ValSource] = {}
        if not config.VAL_SOURCE_RAW:
            for task in config.known_tasks():
                raw = config.validation_source(task)
                if raw:
                    self._benchmark_val_sources[task] = valsplit.parse_val_source(raw)
        self.disclosure = valsplit.parse_disclosure(config.AGENT_DISCLOSURE_RAW or None)
        self.progress_reader = progress.ProgressReader()
        self.work_root = config.STATE_DIR / "work"
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.log_path = config.STATE_DIR / "service.log"

        self._queue: "queue.Queue[str]" = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        # fingerprint -> finished result, so resubmitting untouched weights is free
        self._result_cache: dict[str, dict] = {}
        self._current: Optional[str] = None

        self._worker = threading.Thread(target=self._work_loop, daemon=True, name="grading-worker")
        self._worker.start()


    # ------------------------------------------------------------------ logging

    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
        print(line, flush=True)
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


    # ------------------------------------------------------------------ submit

    def require_admin(self, supplied: Optional[str]) -> None:
        """Gate the endpoints that grant authority rather than exercise it.

        Compared with compare_digest so a wrong guess leaks nothing through timing.
        """
        if not supplied or not hmac.compare_digest(str(supplied), self.admin_token):
            raise PermissionError(
                "this endpoint is operator-only and needs the admin token from "
                "<state_dir>/admin.token"
            )


    def submit(
        self,
        token: str,
        model_path: str,
        mode: Optional[str],
        limit: Optional[int],
        write_metrics: bool,
        sample_ids: Optional[list] = None,
        no_cache: bool = False,
    ) -> dict:
        ctx = self.registry.lookup(token)
        self._assert_json_task_cache(ctx.task)
        resolved = ctx.resolve_model_path(model_path)
        if not resolved.exists():
            raise ValueError(f"model_path does not exist inside the run sandbox: {model_path}")

        if not resolved.is_dir():
            # A plain file passes the exists() gate and used to pass the completeness
            # check too (which stays deliberately silent about non-directories): an
            # agent probing the submission contract with task files (timer.sh,
            # grading.md, a manifest lock) got them queued as models, cp'd into the
            # serve container, and each "evaluation" died ~20s later inside sglang
            # with a JSONDecodeError from a config.json that never existed -- five
            # such dead jobs from one run. No supported backend loads weights from
            # a bare file, so refuse the shape here, where the message can say what
            # to submit instead.
            raise ValueError(
                f"model_path must be a model directory: {model_path} names a plain file. "
                "Point model_path at a directory holding config.json and the weight "
                "files, e.g. final_model or output/checkpoint-N."
            )

        # The workspace is shared with no sequencing between the write and the
        # read, so say plainly that the checkpoint is unfinished instead of letting
        # it fail later inside the model server with a message about nothing in particular.
        incomplete = check_weights_complete(resolved)
        if incomplete is not None:
            # Telling someone to wait for a permission error is worse than saying nothing:
            # the weights are finished, waiting changes nothing, and two runs burned agent
            # turns rediscovering that from this very sentence. The trainer container writes
            # as root under a umask that yields 0600 shards; this service reads them as the
            # host account. postprocess_final_model.py now relaxes the delivery on export,
            # so reaching this branch means the export path was bypassed.
            if UNREADABLE_MARKER in incomplete:
                raise ValueError(
                    f"{model_path} exists but {incomplete}. The weights are written; they are "
                    "unreadable outside the container that wrote them. Run "
                    "`chmod -R a+rX <final_model>` and submit again -- waiting will not help."
                )
            raise ValueError(
                f"{model_path} is not finished being written: {incomplete}. "
                "Wait for the save to complete, then submit again."
            )

        if ctx.role == "agent":
            # An agent submits a model for validation. The service still owns the
            # benchmark and validation upper bound, but the agent may request a
            # smaller positive --limit to trade feedback quality for turnaround time.
            for name, value in (("mode", mode), ("write_metrics", write_metrics or None)):
                if value is not None:
                    raise ValueError(
                        f"{name} is not accepted from an agent submission: submissions are always "
                        "scored on the service-selected validation split. Send model_path, and "
                        "optionally a positive limit no larger than that split."
                    )
            if limit is not None and limit <= 0:
                raise ValueError("agent validation limit must be a positive integer")
            eval_task, validation_limit, validation_ids = self._validation_plan(ctx.task)
            if validation_ids is not None:
                # A file: split names its samples outright; --limit cannot express a
                # non-prefix set, so a smaller --limit takes a prefix of the list.
                if limit is not None:
                    validation_ids = validation_ids[:limit]
                effective_limit = -1
            else:
                effective_limit = min(limit, validation_limit) if limit is not None and validation_limit else (limit or validation_limit)
            effective_mode = "val"
        else:
            effective_mode = mode or "final"
            if effective_mode not in VALID_MODES:
                raise ValueError(f"mode must be one of {VALID_MODES}")
            eval_task = ctx.task
            validation_ids = None
            if effective_mode == "final":
                # A split.json may route final scoring to a separate test
                # benchmark, so the number the run reports is measured on tasks
                # the agent never saw. No declaration means the target benchmark
                # is scored as a whole, exactly as before.
                routed = self._test_benchmark(ctx.task)
                if routed is not None:
                    eval_task = routed
                if limit is not None and limit > 0:
                    effective_limit = limit
                elif config.FINAL_LIMIT > 0:
                    effective_limit = config.FINAL_LIMIT
                else:
                    effective_limit = -1
            elif limit is not None and limit > 0:
                effective_limit = limit
            else:
                eval_task, effective_limit, validation_ids = self._validation_plan(ctx.task)

        wanted_ids = self._resolve_sample_ids(eval_task, sample_ids, known_ids=validation_ids)
        if wanted_ids:
            # Named samples replace the prefix outright: -1 stops the runner sending a
            # count alongside the names.
            effective_limit = -1
        elif validation_ids is not None:
            # The file: split is the service's own explicit list, and it replaces the
            # prefix the same way named samples do -- but it is not the submission's
            # choice, so service_split keeps it from being read as a subset.
            wanted_ids = list(validation_ids)
            effective_limit = -1

        harness_submission = _editable_harness_submission(ctx, resolved)
        fingerprint = (
            f"{eval_task}|{effective_mode}|{effective_limit}|{model_fingerprint(resolved)}|"
            f"{harness_submission['fingerprint'] if harness_submission else 'fixed-agent'}"
        )
        if wanted_ids:
            fingerprint += "|" + ",".join(wanted_ids)
        cached = self._result_cache.get(fingerprint)
        if cached is not None and not no_cache:
            self.log(f"[submit] cache hit run={ctx.run_id} mode={effective_mode} path={model_path}")
            return {"job_id": cached["job_id"], "queue_position": 0, "cached": True}
        if cached is not None and no_cache:
            # The key is (eval_task, mode, limit, model) -- it cannot see anything
            # about the environment the evaluation runs in. A mode-sweep (thinking
            # on/off flips the SERVICE env, not the submission) legitimately needs
            # the same key evaluated twice; no_cache is that escape hatch.
            self.log(f"[submit] cache bypassed (no_cache) run={ctx.run_id} mode={effective_mode} path={model_path}")

        job_id = uuid.uuid4().hex[:16]
        job = Job(
            job_id,
            token,
            ctx,
            effective_mode,
            effective_limit,
            resolved,
            model_path,
            bool(write_metrics) and ctx.role == "official",
            eval_task=eval_task,
            harness_submission=harness_submission,
        )
        if harness_submission is not None:
            job.harness_snapshot = _snapshot_harness(
                harness_submission, self.work_root / job_id / "submitted_harness"
            )
        job.sample_ids = wanted_ids or None
        job.service_split = bool(wanted_ids) and validation_ids is not None and sample_ids is None
        if job.mode == "final":
            job.eval_ids = self._heldout_ids(ctx)
            if job.eval_ids and config.FINAL_LIMIT > 0:
                # The held-out remainder is named sample by sample, so a count cap on
                # the final never reaches the runner unless the list itself is
                # truncated here.
                job.eval_ids = job.eval_ids[: config.FINAL_LIMIT]
        with self._jobs_lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        position = self._queue.qsize()
        self.log(
            f"[submit] queued job={job_id} run={ctx.run_id} role={ctx.role} task={ctx.task} "
            f"eval_task={eval_task} mode={effective_mode} limit={effective_limit} position={position}"
            + (
                f" candidate={job.candidate_id} harness={job.harness_fingerprint[:12]}"
                if job.harness_fingerprint else ""
            )
            + (f" samples={len(wanted_ids)}" if wanted_ids else "")
        )
        return {
            "job_id": job_id,
            "queue_position": position,
            "cached": False,
            # What will actually be evaluated, which is not always what was asked for:
            # a limit larger than the validation split is capped to it. Saying so beats
            # letting a run believe it bought 999 samples and later compare that number
            # against one that meant something else.
            "limit": effective_limit,
            "cancel_quota": self.cancel_quota(ctx),
            "sample_ids": wanted_ids or None,
        }


    def _resolve_sample_ids(
        self, eval_task: str, sample_ids: Optional[list], known_ids: Optional[list] = None
    ) -> list:
        """Check named samples against the validation split, or refuse.

        The held-out remainder is protected by the split being a prefix: an agent can
        ask for fewer samples but not for different ones. Naming samples removes that
        protection, so it is replaced by an explicit membership check -- and when the
        membership is not known yet, the answer is no. A guess here would score a
        submission on held-out samples, and that contamination cannot be undone.

        known_ids overrides the learned membership with the service's own list, which
        is what a file: split supplies: declared ids are authoritative immediately,
        where the prefix form has to be learned from a first full evaluation.

        The membership is learned from the first evaluation that covered the whole
        split, so this unlocks itself after one full validation.
        """
        wanted = [str(sid).strip() for sid in (sample_ids or []) if str(sid).strip()]
        if not wanted:
            return []
        duplicates = len(wanted) - len(set(wanted))
        if duplicates:
            raise ValueError("sample_ids contains duplicates")

        known = known_ids or self.task_stats.val_ids(eval_task)
        if not known:
            raise ValueError(
                f"the validation sample ids for {eval_task} are not known yet, so named "
                "samples cannot be checked against the split. Submit one evaluation of the "
                "whole validation split first; after that they can be named individually."
            )
        outside = sorted(set(wanted) - set(known))
        if outside:
            raise ValueError(
                f"{len(outside)} of the named samples are not in the validation split "
                f"and cannot be evaluated: {', '.join(outside[:5])}"
                + (" ..." if len(outside) > 5 else "")
            )
        # Kept in split order, not in the order they were asked for, so two submissions
        # naming the same set produce the same fingerprint and the same evaluation.
        return [sid for sid in known if sid in set(wanted)]


    def _validation_source(self, task: str) -> valsplit.ValSource:
        """Resolve operator override, then benchmark policy, then the global default."""
        if config.VAL_SOURCE_RAW:
            return self.val_source
        if task not in self._benchmark_val_sources:
            raw = config.validation_source(task)
            self._benchmark_val_sources[task] = (
                valsplit.parse_val_source(raw) if raw else self.val_source
            )
        return self._benchmark_val_sources[task]

    def _json_split_ids(self, task: str, group: str) -> tuple[str, ...]:
        """Return the operator-configured Tau3 JSON split group, or an empty tuple."""
        source = self._validation_source(task)
        if source.kind != "json":
            return ()
        if task != "sierra-research__tau3-bench":
            raise ValueError("VTB_VAL_SOURCE=json: is only valid for sierra-research__tau3-bench")
        return source.val_ids if group == "val" else source.test_ids

    def _assert_json_task_cache(self, task: str) -> None:
        """Refuse Tau3 JSON scoring until the cached Harbor task set is exact."""
        source = self._validation_source(task)
        if source.kind != "json":
            return
        expected = set(self._json_split_ids(task, "val")) | set(self._json_split_ids(task, "test"))
        if len(expected) != 375:
            raise ValueError(f"Tau3 JSON split must contain exactly 375 unique task ids; got {len(expected)}")
        task_specs_root = config.runtime_paths.TAU3_TASK_SPECS_DIR
        try:
            cached = {
                path.name for path in task_specs_root.iterdir()
                if path.is_dir() and path.name.startswith("tau3-bench__") and any(child.is_dir() for child in path.iterdir())
            }
        except OSError as exc:
            raise ValueError(f"cannot inspect Tau3 task export {task_specs_root}: {exc}") from exc
        if cached != expected:
            missing, unexpected = sorted(expected - cached), sorted(cached - expected)
            raise ValueError(
                "Tau3 JSON split does not exactly match the locally cached Harbor task set"
                + (f"; missing={','.join(missing[:3])}" if missing else "")
                + (f"; unexpected={','.join(unexpected[:3])}" if unexpected else "")
                + f"; expected={len(expected)} cached={len(cached)}"
            )

    def _heldout_ids(self, ctx: RunContext) -> Optional[list]:
        """The held-out remainder of a split-validation benchmark, in dataset order.

        A final evaluation measures only what the run could not iterate on: for a
        split:<n> validation source, everything past the validation prefix. The order
        is learned from a baseline import or a full-coverage evaluation; without it
        there is no trustworthy way to name the remainder, so the final falls back to
        the whole benchmark (whose metrics.json then still gets a val/held-out
        partition).
        """
        source = self._validation_source(ctx.task)
        if source.kind == "json":
            return list(self._json_split_ids(ctx.task, "test"))
        if source.kind != "split":
            return None
        val_n = valsplit.resolve_val_n(source, self._dataset_total(ctx.task))
        order = self.task_stats.ordered_ids(ctx.task)
        if not val_n or not order or len(order) <= val_n:
            return None
        return order[val_n:] or None

    def _validation_plan(self, task: str) -> tuple:
        """(benchmark, sample limit, validation sample ids) for a validation submission.

        Refuses rather than improvises. With a ratio-based split the dataset size
        is the denominator, and it must come from observed evaluation logs or
        explicit benchmark metadata. Guessing a boundary could score an agent on
        held-out samples, and that contamination cannot be undone afterwards.
        """
        source = self._validation_source(task)
        if source.kind == "json":
            return task, None, list(self._json_split_ids(task, "val"))
        if source.kind == "none":
            raise ValueError(
                "this service offers no validation channel (VTB_VAL_SOURCE=none); "
                "build your own held-out data and evaluate it yourself"
            )
        if source.kind == "task":
            if source.task == task:
                raise ValueError(
                    f"validation is configured to use {source.task}, which is also the target "
                    "benchmark; validation and scoring would overlap completely"
                )
            if source.task not in config.known_tasks():
                raise ValueError(f"validation benchmark {source.task} is not installed")
            # The whole other benchmark, at its own default size.
            return source.task, None, None
        if source.kind == "file":
            # The file lives in benchmarks/<name>/ and declares that benchmark's
            # split; letting it define another benchmark's boundary would be a
            # silent cross-benchmark rescore.
            if not self._owns_split_file(task):
                raise ValueError(
                    f"the file: split {source.path} does not belong to {task}: each "
                    "benchmark's split is declared in its own directory"
                )
            # The list is the split in its entirety -- no held-out remainder is
            # carved out of it -- so it replaces the prefix arithmetic outright
            # and a --limit takes a prefix of the list, like a count-based split.
            return task, None, list(source.task_ids)

        total = self._dataset_total(task)
        val_n = valsplit.resolve_val_n(source, total)
        if val_n is None:
            raise ValueError(
                f"the validation boundary for {task} cannot be resolved yet: its dataset size is "
                "unknown to this service. Run one official evaluation of this benchmark first "
                "(that teaches the size), declare dataset_total in benchmark.json, or configure "
                "VTB_VAL_SOURCE=split:<count>."
            )
        return task, val_n, None

    def _owns_split_file(self, task: str) -> bool:
        """Whether the configured file: split declares this benchmark's split."""
        source = self._validation_source(task)
        return source.kind == "file" and Path(source.path).parent.name == task

    def _test_benchmark(self, task: str) -> Optional[str]:
        """The benchmark a final score is measured on, when the split declares one.

        Only a file: split can say this, and only about the benchmark whose
        directory the file lives in: benchmarks/<name>/split.json routing <name>'s
        final scoring elsewhere. Anything looser would let one file reroute every
        benchmark's scoring at once.
        """
        if not self._owns_split_file(task):
            return None
        source = self._validation_source(task)
        routed = source.test_benchmark
        if routed is None:
            return None
        if routed == task:
            raise ValueError(
                f"{source.path} declares {task} as its own test benchmark: validation "
                "and final scoring would overlap completely"
            )
        if routed not in config.known_tasks():
            raise ValueError(
                f"test benchmark {routed} declared by {source.path} is not installed"
            )
        return routed


    def _dataset_total(self, task: str) -> Optional[int]:
        return self.task_stats.total(task) or config.dataset_total(task)


    def _validation_counts(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for task in config.known_tasks():
            policy = self._validation_source(task)
            observed_total = self.task_stats.total(task)
            total = observed_total or config.dataset_total(task)
            if policy.kind == "json":
                count = len(self._json_split_ids(task, "val")) if task == "sierra-research__tau3-bench" else None
            elif policy.kind == "task":
                count = config.default_limit(policy.task) if policy.task else None
            else:
                count = valsplit.resolve_val_n(policy, total)
            source = "observed" if observed_total else ("declared" if total else None)
            if policy.kind == "file" and self._owns_split_file(task):
                # The file is the split: its length is the count, and it is
                # known at startup rather than learned from a first evaluation.
                count = len(policy.task_ids)
                source = "file"
            elif policy.kind == "json":
                source = "json"
            out[task] = {
                "dataset_total": total,
                "count": count,
                "source": source,
                "validation_source": policy.raw,
                "validation_task": policy.task if policy.kind == "task" else None,
            }
        return out


    def _progress_of(self, job: Job) -> dict:
        """Live progress for an unfinished job.

        An agent is expected to poll this, so it is assembled from an
        incremental tail of the current attempt's log plus historical timings --
        no extra work is imposed on the evaluation itself.
        """
        return progress.build_progress(
            task=job.task,
            state=job.state,
            started_at=job.started_at or job.submitted_at,
            queue_position=self._position_of(job.job_id) if job.state == "queued" else None,
            attempt_phase=job.attempt_phase,
            attempt_index=job.attempt_index,
            log_path=job.current_log,
            metrics_dir=self._attempt_metrics_dir(job),
            target_samples=self._target_samples(job),
            historical=self.durations.lookup(job.run_id, job.task, job.mode, job.limit),
            baseline_timing=self._baseline_timing_of(job),
            reader=self.progress_reader,
        )


    def _baseline_timing_of(self, job: Job) -> Optional[dict]:
        """The base model's measured per-sample cost, for the cold-start ETA.

        Needs the base model, which lives on the run rather than the job, so it is
        resolved through the submitting token. A token that has since been unregistered
        costs the estimate, not the poll.
        """
        try:
            ctx = self.registry.lookup(job.token)
        except KeyError:
            return None
        return self.baselines.sample_timing(ctx.task, ctx.base_model)


    def _attempt_metrics_dir(self, job: Job) -> Optional[Path]:
        """The current attempt's harness output directory, if it exists yet.

        Where the per-sample result files live, which is how live progress counts
        graded samples instead of guessing from completion counts. Returns None until
        the attempt has created it, so a poll during startup simply gets less detail.
        """
        if not (isinstance(job.attempt_phase, int) and isinstance(job.attempt_index, int)):
            return None
        candidate = self.work_root / job.job_id / f"harness_metrics_p{job.attempt_phase}_a{job.attempt_index}"
        return candidate if candidate.is_dir() else None


    def _target_samples(self, job: Job) -> Optional[int]:
        """Sample total this job is working towards.

        A dev submission states its own limit, or inherits the task's default.
        A final submission passes --limit -1, so the total is the size of the
        whole dataset. inspect records the dataset size in its own log, so read
        it from there rather than leaving the longest evaluation of all with no
        percentage and no ETA -- once per job, since the log can be tens of
        megabytes and this is polled.
        """
        if job.limit and job.limit > 0:
            return job.limit
        if job.limit is None:
            return config.default_limit(job.task)
        if job.target_samples is None:
            job.target_samples = executor.dataset_total(self.work_root / job.job_id)
        return job.target_samples


    def _position_of(self, job_id: str) -> Optional[int]:
        pending = list(self._queue.queue)
        return pending.index(job_id) + 1 if job_id in pending else None


    def queue_snapshot(self, token: Optional[str] = None) -> dict:
        """Queue state. Other runs' jobs are counted, never described.

        A caller needs to know how long the shared queue is to budget its own
        time, but nothing about who else is in it.
        """
        pending = list(self._queue.queue)
        with self._jobs_lock:
            own = [
                self._jobs[j].public()
                for j in pending
                if j in self._jobs and token is not None and self._jobs[j].token == token
            ]
            current = self._jobs.get(self._current or "")
            if current is None:
                running: Optional[dict] = None
            elif token is not None and current.token == token:
                running = current.public()
                if current.wait_detail:
                    running["resource_advice"] = self._resource_advice(current)
            else:
                running = {"task": current.task, "state": current.state, "started_at": current.started_at}
                if config.GPU_COORDINATION_MODE == "isolated" and current.state == "waiting_for_gpu":
                    running["state"] = "queued"
        snapshot = {
            "running": running,
            "pending_count": len(pending),
            "own_pending": own,
            "own_pending_count": len(own),
        }
        if token is not None:
            try:
                ctx = self.registry.lookup(token)
            except KeyError:
                ctx = None
            if ctx is not None and ctx.role == "agent":
                snapshot["own_pending"] = [self._redact_for_agent(item) for item in own]
                if running is not None and running.get("run_id") == ctx.run_id:
                    snapshot["running"] = self._redact_for_agent(running)
        return snapshot


    def mark_interrupted(self) -> Optional[str]:
        """Fail the in-flight job so a waiting client gets a terminal state.

        Without this a client polling /result would sit on "running" forever
        after the service went away, and the submission would leave no trace of
        why it never finished.
        """
        job_id = self._current
        if not job_id:
            return None
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None or job.state not in ("queued", "running", "waiting_for_gpu"):
            return None
        self._finalize_terminal(
            job,
            "failed",
            {"error": "the grading service was shut down while this evaluation was running"},
            {"error": "interrupted: grading service shut down mid-evaluation"},
        )
        return job_id


    def _finalize_terminal(self, job: Job, state: str, result: dict, entry: dict) -> None:
        """Put a job into a terminal state and record it, without a worker outcome.

        Two callers arrive here with no evaluation result to report: a service shutting
        down mid-evaluation, and a run cancelling its own submission. Both have to
        leave a terminal state behind -- a client polling /result would otherwise sit
        on "running" forever -- and a submission log entry, so the record says why it
        never finished.
        """
        job.state = state
        job.finished_at = time.time()
        job.result = {"ok": False, "score": None, "metrics": None, "attempts": 0, **result}
        record = {
            "job_id": job.job_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": job.mode,
            "limit": job.limit,
            "model_path": job.raw_model_path,
            "state": state,
            # Null on purpose, and it is the whole guard: SubmissionLog.best() selects
            # on a numeric score, so a job that never produced one cannot become a
            # candidate however much else is recorded beside it.
            "score": None,
            **entry,
        }
        try:
            self.submissions.append(self.registry.lookup(job.token), record)
        except KeyError:
            pass


    def cancels_used(self, ctx: RunContext) -> int:
        """How many of this run's evaluations it has cancelled.

        Counted from the submission log rather than kept in memory: the entries are
        already written there, and a second counter would be a second truth to keep
        in step across a service restart.
        """
        return sum(1 for entry in self.submissions.load(ctx) if entry.get("state") == "cancelled")


    def cancel_quota(self, ctx: RunContext) -> dict:
        used = self.cancels_used(ctx)
        allowed = config.MAX_AGENT_CANCELS
        return {
            "used": used,
            "max": allowed if allowed > 0 else None,
            "remaining": max(0, allowed - used) if allowed > 0 else None,
        }


    def cancel(self, token: str, job_id: str, reason: str, wait: bool = True) -> dict:
        """Stop this run's evaluation, keep what it produced, and release the cards.

        Not a courtesy. Validation and training share the card pool, so a run whose
        evaluation has hours left to go and whose next training job is ready has two
        options without this: sit idle, or take the memory anyway and quietly degrade
        the evaluation it is waiting on. The second was what actually happened in a
        real run. Cancelling is the same decision made legibly, and it keeps whatever
        the evaluation had already graded.

        Returning only once the cards report free memory is the part that makes it
        usable: the caller will start training on the strength of this answer.
        """
        ctx = self.registry.lookup(token)
        if not (reason or "").strip():
            raise ValueError(
                "cancel needs a reason: say what this evaluation is being given up for"
            )
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None or job.token != token:
            raise KeyError("unknown job_id")
        if job.state not in ("queued", "running", "waiting_for_gpu"):
            raise Conflict(f"job {job_id} is already {job.state}")

        quota = self.cancel_quota(ctx)
        if ctx.role == "agent" and quota["remaining"] == 0:
            raise Conflict(
                f"this run has used all {quota['max']} of its cancellations; "
                "let this evaluation finish"
            )
        job.cancel_reason = reason.strip()

        # Never started, so there is nothing to tear down and nothing was graded. The
        # worker skips it when the queue reaches it.
        if job.state == "queued" and self._current != job_id:
            self._finalize_terminal(
                job,
                "cancelled",
                {"cancelled": True, "partial": None},
                {"cancel_reason": job.cancel_reason, "partial": None},
            )
            self.log(f"[cancel] job={job_id} dropped while still queued: {job.cancel_reason}")
            return {
                "job_id": job_id,
                "state": "cancelled",
                "was_running": False,
                "gpu_release": {"released": True, "reason": "the evaluation had not started"},
                "partial": None,
                "cancel_quota": self.cancel_quota(ctx),
            }

        self.log(f"[cancel] job={job_id} while {job.state}: {job.cancel_reason}")
        actions = executor.request_cancel(job_id, self.log)

        # The cards first, because that is what the caller is waiting on. Winding the
        # job up -- reading the partial result, archiving the record -- happens in the
        # worker thread and can outlast this response.
        release = (
            executor.wait_until_eval_gpus_free(self.log)
            if wait
            else {"released": None, "reason": "not waited for"}
        )
        deadline = time.time() + config.CANCEL_WINDUP_SECONDS
        while wait and time.time() < deadline and job.state not in ("cancelled", "failed", "done"):
            time.sleep(0.2)

        return {
            "job_id": job_id,
            "state": job.state,
            "was_running": True,
            "released": actions,
            "gpu_release": release,
            "partial": (job.result or {}).get("partial"),
            "cancel_quota": self.cancel_quota(ctx),
            "note": (
                "partial results are for diagnosis only: they cover the prefix of the"
                " evaluation order that had finished, which is biased toward the"
                " quicker samples, and are not comparable with a full validation"
                " score or with each other."
            ),
        }


    # ------------------------------------------------------------------ worker

    def _work_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            with self._jobs_lock:
                job = self._jobs.get(job_id)
            # A job cancelled before the queue reached it was already wound up by
            # cancel(); starting it now would serve a model nobody is waiting for.
            if job is None or job.state == "cancelled":
                self._queue.task_done()
                continue
            self._current = job_id
            try:
                self._execute(job)
            except Exception as exc:  # keep the worker alive whatever happens
                job.state = "failed"
                job.result = {"ok": False, "score": None, "error": f"internal error: {exc}"}
                self.log(f"[worker] job={job_id} crashed: {exc}")
            finally:
                job.finished_at = time.time()
                # Never leave a cancellation set behind: the flag is per-service, and
                # the next submission would stop before its first attempt.
                if executor.cancel_requested(job_id):
                    executor.clear_cancel()
                self._current = None
                self._queue.task_done()


    def _finish_cancelled(self, ctx: RunContext, job: Job, work_dir: Path, outcome: dict) -> None:
        """Wind up an evaluation the run asked to stop.

        Order matters. The terminal state and the partial result are set first, because
        /cancel is blocked waiting for exactly those; archiving the record copies a
        whole trial tree and can take much longer than the caller should wait. The
        agent-readable copy is written after for the same reason, and `results_dir`
        appears on the result once it is complete rather than before.

        The cards were already released by request_cancel, before any of this.
        """
        partial, disclosed = self._partial_outcome(work_dir, job)
        if not self.disclosure & {"per_sample_outcomes", "problems", "outputs"}:
            disclosed = None
        graded = (partial or {}).get("graded") or 0
        self._finalize_terminal(
            job,
            "cancelled",
            {
                "cancelled": True,
                "error": outcome.get("error"),
                "attempts": len(outcome.get("attempt_log") or []),
                "attempt_log": outcome.get("attempt_log"),
                "partial": partial,
                "disclosed_per_sample_outcomes": disclosed,
            },
            {
                "cancel_reason": job.cancel_reason,
                "partial": partial,
                "elapsed_seconds": round(time.time() - (job.started_at or time.time()), 2),
            },
        )
        self.log(
            f"[worker] cancelled job={job.job_id} after {graded} graded sample(s); "
            "partial result kept for diagnosis only"
        )

        # Kept where the agent cannot reach it, for the same reason a finished
        # evaluation's record is: what an agent gave up on, and when, is part of the
        # iteration being observed.
        self._archive_full_record(work_dir, ctx, job)

        # A cancelled evaluation is the case where reading the transcripts matters
        # most: the run gave up on it, and the only way to tell a good decision from a
        # premature one is what the samples were doing -- including the ones it killed,
        # which are usually the reason it gave up.
        disclosure_path = self._write_agent_disclosure(
            ctx,
            job,
            list((partial or {}).get("instance_ids_graded") or []),
            cancelled=True,
            ungraded_ids=list((partial or {}).get("instance_ids_ungraded") or []),
        )
        if disclosure_path and isinstance(job.result, dict):
            job.result["results_dir"] = disclosure_path

        self._prune_cancelled_scratch(work_dir, job)

        # The flag is per-service, not per-job: left set, the next submission would
        # stop before its first attempt. Also done in the worker loop's finally, for
        # the paths that do not reach here.
        executor.clear_cancel()


    def _prune_cancelled_scratch(self, work_dir: Path, job: Job) -> None:
        """Drop the bulk of a cancelled job's scratch, keeping the trial trees.

        A cancelled evaluation is not deleted the way a successful one is: its
        backend_output is the only copy of what was graded, and it is what a resumed
        evaluation would need to avoid paying for those samples twice. The overlay
        scratch beside it is worth nothing and is the largest part.
        """
        for name in ("merged_huggingface", "upper_huggingface", "fuse_workdir"):
            leftover = work_dir / name
            if leftover.exists():
                shutil.rmtree(leftover, ignore_errors=True)
        if job.current_log is not None:
            self.progress_reader.forget(job.current_log)


    def _execute(self, job: Job) -> None:
        ctx = self.registry.lookup(job.token)
        job.state = "running"
        job.started_at = time.time()
        self.log(f"[worker] start job={job.job_id} run={ctx.run_id} task={ctx.task} mode={job.mode}")

        work_dir = self.work_root / job.job_id

        def on_attempt(phase: int, attempt: int, log_path: Path) -> None:
            job.state = "running"
            job.phase = "evaluating"
            job.wait_detail = None
            job.attempt_phase = phase
            job.attempt_index = attempt
            job.current_log = log_path

        def on_gpu_wait(detail: Optional[dict[str, Any]]) -> None:
            if detail is None:
                job.state = "running"
                job.phase = None
                job.wait_detail = None
            else:
                job.state = "waiting_for_gpu"
                job.phase = "waiting_for_gpu"
                job.wait_detail = detail

        outcome = executor.evaluate(
            ctx,
            job.model_path,
            job.limit,
            work_dir,
            self.log,
            on_attempt=on_attempt,
            on_gpu_wait=on_gpu_wait,
            eval_task=job.eval_task,
            job_id=job.job_id,
            sample_ids=job.sample_ids if job.sample_ids is not None else job.eval_ids,
            harness_snapshot=job.harness_snapshot,
            harness_import_path=job.harness_import_path,
        )

        if outcome.get("cancelled"):
            self._finish_cancelled(ctx, job, work_dir, outcome)
            return

        # Every log records the whole dataset's size, so any evaluation teaches
        # it -- which is what makes a ratio-based validation boundary resolvable
        # without anything being hardcoded.
        sample_results = self._read_sample_results(work_dir)
        observed_total = executor.dataset_total(work_dir)
        if observed_total is None and sample_results is not None:
            observed_total = sample_results.dataset_total
        # The evaluation order comes from the backend's own lock file. Harbor writes
        # that lock below each attempt's backend_output directory, not at the job root.
        # The per-sample results file lists graded samples alphabetically -- a display
        # order that must not move the split boundary or name the held-out remainder.
        run_order = []
        for output_tree in sorted(
            work_dir.glob("harness_metrics_*/backend_output"), reverse=True
        ):
            run_order = trial_results.dataset_order(output_tree)
            if run_order:
                break
        if observed_total and not job.sample_ids and not job.eval_ids:
            # Only a full-coverage run teaches sizes and order; a named or held-out
            # subset covers part of the dataset and would corrupt both.
            warning = self.task_stats.observe_total(job.eval_task, observed_total)
            if warning:
                self.log(f"[worker] WARNING {warning}")
            if len(run_order) == observed_total:
                warning = self.task_stats.observe_order(
                    job.eval_task, run_order, observed_total
                )
                if warning:
                    self.log(f"[worker] WARNING {warning}")

        # An evaluation that covered the whole split teaches which samples it contains,
        # which is what lets a later submission name some of them and still be checked
        # against the boundary. Only a complete one: see observe_val_ids.
        if outcome["ok"] and sample_results is not None and not job.is_subset and not job.eval_ids:
            val_n = valsplit.resolve_val_n(self._validation_source(ctx.task), observed_total)
            if val_n:
                # Membership is learned in true dataset order (lock file), falling
                # back to the results file's own order for harnesses without one.
                split_ids = run_order[:val_n] if len(run_order) >= val_n else (sample_results.ordered_ids or [])
                warning = self.task_stats.observe_val_ids(
                    job.eval_task, split_ids, val_n
                )
                if warning:
                    self.log(f"[worker] WARNING {warning}")
                elif self.task_stats.val_ids(job.eval_task):
                    self.log(
                        f"[worker] validation sample ids for {job.eval_task} are known "
                        f"({val_n} samples); submissions may now name individual samples"
                    )

        # A Tau3 JSON final is an explicit test-only measurement. Its denominator
        # is the configured test list, so absent task outcomes count as incorrect
        # rather than being silently omitted from Harbor's reported accuracy.
        effective_source = self._validation_source(ctx.task)
        json_test_split: Optional[dict] = None
        if (
            outcome["ok"]
            and job.write_metrics
            and job.mode == "final"
            and effective_source.kind == "json"
            and job.eval_ids
        ):
            if sample_results is None:
                json_test_split = {
                    "reconstruction_ok": False,
                    "reason": "JSON test-only evaluation produced no per-sample results",
                }
            else:
                json_test_split = valsplit.partition_test_only(
                    sample_results,
                    self._json_split_ids(ctx.task, "test"),
                    reported_accuracy=(outcome.get("metrics") or {}).get("accuracy"),
                )
            if not json_test_split.get("reconstruction_ok"):
                outcome["ok"] = False
                outcome["score"] = None
                outcome["error"] = f"JSON test-only integrity check failed: {json_test_split.get('reason')}"
                self.log(f"[worker] {outcome['error']}")
            else:
                metrics = dict(outcome.get("metrics") or {})
                metrics["harness_accuracy"] = metrics.get("accuracy")
                metrics["accuracy"] = json_test_split["accuracy_test"]
                outcome["metrics"] = metrics
                outcome["score"] = json_test_split["accuracy_test"]

        # A held-out-only final is compared against the base model's held-out score;
        # the validation split's baseline would be apples-to-oranges.
        if job.eval_ids:
            baseline = self.baselines.get_heldout(ctx.task, ctx.base_model)
        else:
            # Prefer a baseline measured on the benchmark actually scored: a routed
            # final (the file: split's test.benchmark) is scored on another benchmark
            # entirely, and the registered task's figure would be apples-to-oranges
            # there too. Falls back to the run's own task, which is also exactly
            # right whenever no routing happened.
            baseline = self.baselines.get(job.eval_task, ctx.base_model)
            if baseline is None:
                baseline = self.baselines.get(ctx.task, ctx.base_model)
        improvement = None
        if outcome["ok"] and baseline is not None:
            improvement = round(outcome["score"] - baseline, 6)

        entry = {
            "job_id": job.job_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": job.mode,
            "limit": job.limit,
            "model_path": job.raw_model_path,
            "score": outcome["score"],
            "metrics": outcome["metrics"],
            "baseline_score": baseline,
            "improvement": improvement,
            "elapsed_seconds": round(time.time() - (job.started_at or time.time()), 2),
            "attempts": len(outcome.get("attempt_log") or []),
            "error": outcome.get("error"),
        }
        if job.is_subset:
            # A score over samples the submission chose itself is not the split's score.
            # Recorded as such so a later reader cannot mistake one for the other, and
            # so picking a flattering subset cannot be passed off as an improvement.
            entry["sample_ids"] = job.sample_ids
            entry["subset"] = True
            entry["comparable_to_full"] = False
            entry["baseline_score"] = None
            entry["improvement"] = None
        elif job.eval_ids:
            # A held-out-only final is the official measurement under this policy and
            # stays comparable across runs; the flag exists so a reader can tell the
            # score covers the held-out remainder only, not the whole benchmark.
            entry["heldout_only"] = True
            entry["heldout_count"] = len(job.eval_ids)
        entries = self.submissions.append(ctx, entry)
        best = self.submissions.best(ctx, mode="final")

        # Partition the official evaluation into the half the agent could iterate
        # on and the half it never saw. `accuracy` stays the whole-benchmark
        # figure so downstream readers and the upstream leaderboard still line up;
        # the held-out number is the one that estimates generalisation.
        #
        # Never for a submission that named its own samples: the partition is a
        # statement about a prefix of the dataset, and a chosen subset is not one.
        split: Optional[dict] = None
        if job.write_metrics and outcome["ok"] and not job.is_subset and not job.eval_ids:
            effective_source = self._validation_source(ctx.task)
            val_n = valsplit.resolve_val_n(effective_source, observed_total)
            reported = (outcome.get("metrics") or {}).get("accuracy")
            if sample_results is None:
                self.log(
                    f"[worker] no per-sample results for {job.eval_task}, so no held-out figure. "
                    "A harness that does not use inspect-ai can declare them; see "
                    "eval/grading/README.md."
                )
            elif effective_source.kind == "task":
                # Validation ran on another benchmark entirely: there is no prefix of
                # this evaluation to partition, and accuracy_val stays absent by
                # declaration -- nothing was held out of the target benchmark.
                self.log(
                    f"[worker] validation for {ctx.task} runs on separate benchmark "
                    f"{effective_source.task}; the full {job.eval_task} result carries "
                    "no validation partition"
                )
            elif not val_n:
                if effective_source.kind == "file":
                    # The file names the validation set in full, so there is no
                    # held-out remainder to carve out of what was just scored --
                    # it is either that whole set or the routed test benchmark.
                    self.log(
                        f"[worker] no held-out figure for {job.eval_task}: the file: split "
                        "names the validation set in full"
                    )
                else:
                    self.log(f"[worker] validation boundary for {job.eval_task} unknown, so no held-out figure")
            else:
                split = valsplit.partition(sample_results, val_n, reported_accuracy=reported)
                if not split.get("reconstruction_ok"):
                    self.log(f"[worker] refusing held-out figures: {split.get('reason')}")

        # Only the official final_model evaluation is allowed to write
        # metrics.json: that file stays the upstream-comparable score and the
        # scratch gate's completion signal.
        if job.write_metrics and outcome["ok"] and outcome["metrics"] is not None and not job.is_subset:
            target = ctx.eval_dir / "metrics.json"
            payload = dict(outcome["metrics"])
            if split and split.get("reconstruction_ok"):
                payload.update(
                    {
                        "accuracy_val": split["accuracy_val"],
                        "accuracy_heldout": split["accuracy_heldout"],
                        "val_count": split["val_count"],
                        "heldout_count": split["heldout_count"],
                        "val_source": effective_source.raw,
                        "heldout_evidence": split["source"],
                    }
                )
            elif job.eval_ids:
                # The final ran the held-out remainder only: accuracy IS the held-out
                # figure, and there is no val partition to report. Tau3 JSON mode also
                # records the explicit test membership evidence and split digest.
                if effective_source.kind == "json" and json_test_split and json_test_split.get("reconstruction_ok"):
                    payload.update(
                        {
                            "accuracy_test": json_test_split["accuracy_test"],
                            "test_count": json_test_split["test_count"],
                            "test_graded_count": json_test_split["test_graded_count"],
                            "test_ungraded_count": json_test_split["test_ungraded_count"],
                            "test_missing_ids": json_test_split["test_missing_ids"],
                            "val_count": len(self._json_split_ids(ctx.task, "val")),
                            "val_source": effective_source.raw,
                            "split_kind": json_test_split["split_kind"],
                            "split_sha256": effective_source.json_sha256,
                        }
                    )
                else:
                    payload.update(
                        {
                            "heldout_only": True,
                            "heldout_count": len(job.eval_ids),
                            "val_source": effective_source.raw,
                        }
                    )
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.log(
                f"[worker] wrote official metrics.json for run={ctx.run_id}"
                + (f" (heldout={split['accuracy_heldout']})" if split and split.get("accuracy_heldout") is not None else "")
            )
            # Keep the evaluation's own output alongside it. Nothing downstream parses
            # it, but it is the material a human reads when a score looks wrong, and
            # the per-job scratch is discarded below.
            if job.current_log is not None and job.current_log.exists():
                try:
                    shutil.copy2(job.current_log, ctx.eval_dir / "final_eval.log")
                except OSError:
                    self.log("[worker] could not preserve the evaluation log")
            # inspect's own log carries the per-sample outcomes, which is what a
            # held-out score has to be derived from, so the official one is kept
            # rather than discarded with the scratch.
            self._preserve_inspect_log(work_dir, ctx.eval_dir / "inspect_log", job)

        # A validation submission's log is kept too, when the run named somewhere
        # to put it. Without it the only record of an agent's self-evaluation is a
        # single number, which is not enough to explain what its iterations
        # actually changed -- and the agent's iterations are the thing this
        # platform exists to observe. register() refuses a destination the agent
        # can read, so keeping these does not widen what it learns.
        if not job.write_metrics and ctx.log_dir is not None:
            self._preserve_inspect_log(work_dir, ctx.log_dir, job)

        # Independently of the agent-visible log_dir, keep the full record where the
        # agent cannot see it. For a harbor-format benchmark there is no inspect log at
        # all, so the block above preserves nothing; this is what actually retains the
        # per-task trajectories and verifier output for those. Done for every
        # successful evaluation, before the scratch is removed below.
        if outcome["ok"]:
            self._archive_full_record(work_dir, ctx, job)

        # And the readable half of the same material, where the run can actually read
        # it. Written after observe_val_ids above, so a whole-split run has already
        # taught the membership that bounds what may be disclosed -- the two facts come
        # from one evaluation and cannot disagree about which samples are validation.
        disclosure_path = None
        if outcome["ok"] and sample_results is not None:
            disclosure_path = self._write_agent_disclosure(
                ctx, job, list(sample_results.ordered_ids or [])
            )

        disclosed_per_sample_outcomes = None
        if outcome["ok"] and self.disclosure & {"per_sample_outcomes", "problems", "outputs"}:
            disclosed_per_sample_outcomes = self._build_disclosed_per_sample_outcomes(work_dir, job, outcome)

        job.state = "done" if outcome["ok"] else "failed"
        job.result = {
            "ok": outcome["ok"],
            "score": outcome["score"],
            "metrics": outcome["metrics"],
            "baseline_score": baseline,
            "improvement": improvement,
            "best_final": best,
            "total_submissions": len(entries),
            "error": outcome.get("error"),
            # `attempts` is a count everywhere; the per-attempt detail lives in
            # `attempt_log` so the two never mean different things.
            "attempts": len(outcome.get("attempt_log") or []),
            "winning_attempt": outcome.get("winning_attempt"),
            "attempt_log": outcome.get("attempt_log"),
            "split": split,
            "disclosed_per_sample_outcomes": disclosed_per_sample_outcomes,
            # A path, not the contents: the per-sample trees run to hundreds of
            # megabytes, and the point of writing them down is that they do not have
            # to fit in a response.
            "results_dir": disclosure_path,
        }
        if job.sample_ids:
            job.result["subset"] = True
            job.result["comparable_to_full"] = False

        if outcome["ok"]:
            fingerprint = f"{ctx.task}|{job.mode}|{job.limit}|{model_fingerprint(job.model_path)}"
            if job.sample_ids:
                fingerprint += "|" + ",".join(job.sample_ids)
            self._result_cache[fingerprint] = {"job_id": job.job_id}
            # Only successful runs inform the ETA; a failure times the retry
            # ladder rather than the work.
            elapsed = time.time() - (job.started_at or time.time())
            stuck_share = _no_progress_share(outcome["metrics"])
            if stuck_share is not None and stuck_share >= NO_PROGRESS_DURATION_LIMIT:
                # An evaluation whose trials mostly looped measured the looping, not the
                # work, so it is not kept as an expectation for later. The score still
                # counts; only the timing is discarded.
                self.log(
                    f"[worker] duration not recorded job={job.job_id} "
                    f"no_progress_share={stuck_share:.2f} >= {NO_PROGRESS_DURATION_LIMIT} "
                    f"elapsed={elapsed:.0f}s"
                )
            else:
                self.durations.record(job.run_id, ctx.task, job.mode, job.limit, elapsed)

        # Evaluation scratch can be large. Preserve only the immutable submitted
        # harness, which is the audit record for a successful joint submission.
        if outcome["ok"]:
            if job.current_log is not None:
                self.progress_reader.forget(job.current_log)
            _discard_success_scratch(work_dir, job.harness_snapshot)
        self.log(f"[worker] finish job={job.job_id} state={job.state} score={outcome['score']}")
