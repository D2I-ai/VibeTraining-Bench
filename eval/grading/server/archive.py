"""Result persistence for the grading service.

What happens to an evaluation once the executor returns: full-record archive,
the agent disclosure tree, baseline import, and scratch pruning. Mixin methods
of GradingService; no queue or lifecycle logic here.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional
from .. import config, executor, valsplit
from ...runner import trial_results
from ..store import RunContext

from .jobs import Job


class ArchiveMixin:

    # GradingService archiving methods; see class assembly in .service


    def _partial_outcome(self, work_dir: Path, job: Job) -> tuple:
        """What a cancelled evaluation had already graded, and its per-sample rows.

        Read from the backend's own trial tree, because the metrics file only exists
        once an evaluation has finished. trial_results does the counting -- the same
        function a complete run is scored with -- so a partial cannot be counted by
        different rules than a whole one.

        comparable_to_full is False and stays False. Evaluation follows the dataset's
        order at a fixed concurrency, so the trials that have finished are the ones the
        model got through quickly: any prefix is biased towards them, and biased
        upwards. This is a diagnosis, not a score, and nothing selects a candidate on
        it -- the null score in the submission log is what enforces that.
        """
        attempt_dir = None
        if isinstance(job.attempt_phase, int) and isinstance(job.attempt_index, int):
            candidate = work_dir / f"harness_metrics_p{job.attempt_phase}_a{job.attempt_index}"
            if candidate.is_dir():
                attempt_dir = candidate
        if attempt_dir is None:
            # No attempt ever started -- cancelled while queued or waiting for cards.
            attempt_dirs = sorted(work_dir.glob("harness_metrics_*"), reverse=True)
            attempt_dir = attempt_dirs[0] if attempt_dirs else None
        if attempt_dir is None:
            return None, None

        output_tree = attempt_dir / "backend_output"
        if not output_tree.is_dir():
            return None, None
        direct_rows = []
        direct_dir = output_tree / "direct_trajectories"
        if direct_dir.is_dir():
            # A direct task is complete only after every requested completion is
            # recorded. Interrupted half-tasks are omitted, never scored as zero.
            for path in sorted(direct_dir.glob("*.json")):
                payload = self._read_json_or_none(path)
                if not isinstance(payload, dict):
                    continue
                task_id = payload.get("task_id")
                samples = payload.get("samples_per_task")
                completions = payload.get("completions")
                score = payload.get("avg_at_n")
                if (
                    not isinstance(task_id, str)
                    or not isinstance(samples, int)
                    or samples < 1
                    or not isinstance(completions, list)
                    or len(completions) != samples
                    or isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not 0.0 <= score <= 1.0
                ):
                    continue
                direct_rows.append({"id": task_id, "score": float(score), "graded": True})
        if direct_rows:
            accuracy = sum(row["score"] for row in direct_rows) / len(direct_rows)
            partial = {
                "accuracy": accuracy,
                "mean_reward": accuracy,
                "graded": len(direct_rows),
                "attempted": len(direct_rows),
                "ungraded": 0,
                "target_samples": self._target_samples(job),
                "instance_ids_graded": [row["id"] for row in direct_rows],
                "instance_ids_ungraded": [],
                "comparable_to_full": False,
                "note": (
                    "completed direct-evaluator tasks preserved after cancellation; "
                    "incomplete tasks are omitted and the partial is diagnostic only"
                ),
            }
            return partial, self._disclose_rows(direct_rows, job, {})
        records = trial_results.collect(output_tree)
        if not records:
            return None, None

        counts = trial_results.score_records(records)
        graded_ids = [str(r.get("task_id")) for r in records if r.get("graded")]
        # The ids of the trials that were still going, named rather than only counted. These
        # are the error scene: a cancellation is usually decided *because* of what these
        # samples were doing, and a caller cannot ask /sample about a sample whose id it was
        # never told. The last progress block named at most the slowest five.
        ungraded_ids = [str(r.get("task_id")) for r in records if not r.get("graded")]
        partial = {
            **counts,
            "target_samples": self._target_samples(job),
            "instance_ids_graded": graded_ids,
            "instance_ids_ungraded": ungraded_ids,
            "comparable_to_full": False,
            "note": (
                "the prefix of the evaluation order that had finished when this was"
                " cancelled. Biased towards the samples the model got through quickly,"
                " so it reads high; for diagnosis only, and not comparable with a full"
                " validation score or with another partial. `graded` is the denominator"
                " of score_on_graded; trials counted in ungraded were killed part-way"
                " and scored nothing at all, not zero."
            ),
        }
        # Cross-check against the directories on disk. Teardown signals the evaluation
        # with SIGTERM first, so harbor normally gets to write a verdict for every trial
        # it started and these two agree -- `attempted` already covers everything. They
        # diverge only when the SIGKILL fallback fires, and then some trials died
        # leaving nothing at all behind: not a zero score, no record. Reported only in
        # that case, so the number is absent when it would be 0 and loud when it is not.
        started = trial_results.count_started_trials(output_tree)
        vanished = started - (counts.get("attempted") or 0)
        if vanished > 0:
            partial["started"] = started
            partial["vanished_without_a_verdict"] = vanished
            self.log(
                f"[worker] WARNING {vanished} of {started} trials for job={job.job_id} "
                "left no record at all; they were killed before the harness could write "
                "one, and are counted in neither graded nor ungraded"
            )
        trial_dirs = {
            str(record.get("task_id")): Path(record["trial_dir"])
            for record in records
            if isinstance(record.get("trial_dir"), str)
        }
        rows = [
            {
                "id": str(record.get("task_id")),
                "score": record.get("reward"),
                "partial": record.get("partial"),
                "graded": bool(record.get("graded")),
                "reason": record.get("reason"),
            }
            for record in records
        ]
        return partial, self._disclose_rows(rows, job, trial_dirs)


    def _read_sample_results(self, work_dir: Path):
        """Per-sample outcomes of the attempt that just succeeded, from either source.

        The declared file wins over inspect's log, so a harness can override how it
        is read without the service knowing anything about its internals -- which is
        what an SWE-style benchmark will need, since it has its own runner and will
        never write an inspect log.
        """
        for candidate in sorted(work_dir.glob("metrics_p*_a*.json"), reverse=True):
            declared = valsplit.read_per_sample_file(valsplit.per_sample_path(candidate))
            if declared is not None:
                return declared
        log = executor.newest_inspect_log(work_dir)
        return valsplit.read_inspect_log(log) if log is not None else None


    def _preserve_inspect_log(self, work_dir: Path, target_dir: Path, job: Job) -> None:
        """Copy this evaluation's inspect log out before the scratch is cleaned.

        Named after the job so several submissions from one run stay
        distinguishable, which is the whole point of keeping them.
        """
        inspect_log = executor.newest_inspect_log(work_dir)
        if inspect_log is None:
            return
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{job.mode}_{job.job_id}_{inspect_log.name}"
            shutil.copy2(inspect_log, target)
            self.log(f"[worker] preserved inspect log at {target}")
        except OSError as exc:
            self.log(f"[worker] could not preserve the inspect log: {exc}")


    def _archive_full_record(self, work_dir: Path, ctx: RunContext, job: Job) -> None:
        """Keep the whole evaluation record where the agent cannot reach it.

        This is the difference between the two destinations. `log_dir` is a place the
        run named, so whatever lands there is by definition something the agent may
        read -- for a validation submission that is deliberately just a scalar. The
        archive is under grading_state, which is on no sandbox_root and so is invisible
        to the agent; it holds the full backend_output (per-task trajectories and the
        verifier's own output), harbor.log, and the metrics files. That is the material
        for reconstructing what an agent's self-evaluation actually did, and the
        agent's iterations are the thing this platform exists to observe.

        Kept for every mode, validation included, and precisely because the per-job
        scratch is deleted immediately below.
        """
        dest = config.STATE_DIR / "archive" / ctx.run_id / f"{job.mode}_{job.job_id}"
        try:
            dest.mkdir(parents=True, exist_ok=True)
            kept = []
            # The backend's per-task tree: trajectories, verifier output, result.json.
            # attempt directories are named harness_metrics_p<phase>_a<attempt>.
            for attempt_dir in sorted(work_dir.glob("harness_metrics_*")):
                for output_name in ("backend_output", "framework_output"):
                    output_tree = attempt_dir / output_name
                    if output_tree.is_dir():
                        shutil.copytree(
                            output_tree,
                            dest / attempt_dir.name / output_name,
                            dirs_exist_ok=True,
                        )
                        kept.append(attempt_dir.name)
                        break
                for extra in ("harbor.log", "backend_command.txt", "framework_command.txt"):
                    src = attempt_dir / extra
                    if src.is_file():
                        (dest / attempt_dir.name).mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dest / attempt_dir.name / extra)
            # The metrics files sit directly under work_dir, one per attempt.
            for metrics in work_dir.glob("metrics_p*.json"):
                shutil.copy2(metrics, dest / metrics.name)
            for per_sample in work_dir.glob("metrics_p*.per_sample.json"):
                shutil.copy2(per_sample, dest / per_sample.name)
            if kept:
                self.log(f"[worker] archived full record for job={job.job_id} at {dest} ({', '.join(kept)})")
            else:
                self.log(f"[worker] archived metrics for job={job.job_id} at {dest} (no backend_output found)")
        except OSError as exc:
            self.log(f"[worker] could not archive the full record: {exc}")


    def _write_agent_disclosure(
        self,
        ctx: RunContext,
        job: Job,
        graded_ids: list,
        *,
        cancelled: bool = False,
        ungraded_ids: Optional[list] = None,
    ) -> Optional[str]:
        """Leave the run a full, readable copy of what this evaluation produced.

        The HTTP response cannot be the whole channel. A median trajectory on this
        benchmark is over a megabyte and the largest are five, so anything returned
        inline has to be cut, and a cut leaves the agent diagnosing the first percent
        of a transcript whose interesting part -- where a run stopped converging -- is
        at the end. Written to a file instead, the whole thing is there and the run
        greps it however it likes: no page size to choose, nothing silently withheld.

        A cancelled evaluation's unfinished trials are written too, and kept in a
        separate list. They are the error scene: the decision to cancel is usually taken
        *because* of what those samples were doing, so excluding them left the one channel
        that carries whole transcripts unable to show the thing being diagnosed -- and a
        cancellation with nothing graded yet published nothing at all. They carry no score
        and are named apart from the graded ids for that reason.

        Three things bound what lands here, and each is a different boundary:

          which trials   only ids inside the validation split. A final-mode evaluation
                         covers the whole benchmark including the held-out remainder, so
                         it writes nothing at all -- that copy would be the leak.
          which files    trial_results.copy_disclosable_trial's whitelist, which keeps
                         the hidden test names out.
          where          ctx.disclosure_dir, refused at registration if it sits inside
                         sandbox_root, and mounted read-only into the agent container.

        Returns the path handed back to the caller, or None when nothing was written.
        """
        unfinished = list(ungraded_ids or [])
        if ctx.disclosure_dir is None or not (graded_ids or unfinished):
            return None
        # The official whole-benchmark evaluation is not the agent's to read.
        if job.mode != "val":
            return None

        # Named samples were already checked against the split at submission; a whole
        # split run defines it. Either way, disclose the intersection and nothing more,
        # so an id that somehow arrived from outside the split cannot ride along.
        known = set(self.task_stats.val_ids(job.eval_task) or [])
        allowed = [i for i in graded_ids if i in known] if known else list(graded_ids)
        allowed_unfinished = [i for i in unfinished if i in known] if known else list(unfinished)
        dropped = (len(graded_ids) - len(allowed)) + (len(unfinished) - len(allowed_unfinished))
        if known and dropped:
            self.log(
                f"[worker] WARNING {dropped} ids for job={job.job_id} are outside the "
                "recorded validation split and were not disclosed"
            )
        if not (allowed or allowed_unfinished):
            return None

        dest = Path(ctx.disclosure_dir) / job.job_id
        wanted = set(allowed) | set(allowed_unfinished)
        written = 0
        summary = {
            "job_id": job.job_id,
            "eval_task": job.eval_task,
            "mode": job.mode,
            # Whether this is everything the evaluation was asked for, taken
            # from the caller rather than from job.state: this runs before
            # the terminal state is set, so reading job.state here recorded
            # "running" into a file describing a finished evaluation.
            "cancelled": cancelled,
            "comparable_to_full": not (cancelled or job.is_subset),
            "sample_ids": sorted(allowed),
            "withheld": (
                "verifier/report.json and verifier/test-stdout.txt name the "
                "hidden tests, which is the grading answer"
            ),
        }
        # Only when there are any: a field explaining unfinished trials reads as though some
        # were withheld when in fact the evaluation finished everything it started.
        if allowed_unfinished:
            summary["unfinished_sample_ids"] = sorted(allowed_unfinished)
            summary["unfinished_note"] = (
                "these trials were killed part-way and scored nothing at all, not zero. "
                "Their transcripts are here because what they were doing is usually why "
                "the evaluation was cancelled; they are not part of any score."
            )
        try:
            samples_dir = dest / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for task_id, trial_dir in self._trial_dirs(ctx, job).items():
                if task_id not in wanted:
                    continue
                trial_results.copy_disclosable_trial(trial_dir, samples_dir / task_id)
                written += 1
            # Direct evaluators have no Harbor trial tree. Copy only the disclosure-
            # filtered trajectory; answer files and unfiltered outputs stay archived.
            for task_id in sorted(wanted):
                direct = self._direct_trajectory(ctx, job, task_id)
                if direct is None:
                    continue
                filtered = self._direct_disclosure(direct)
                if len(filtered) <= 2:
                    continue
                target = samples_dir / task_id / "direct_trajectory.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(filtered, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                written += 1
            summary["samples_written"] = written
            (dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except OSError as exc:
            self.log(f"[worker] could not write the agent-readable results: {exc}")
            return None
        self.log(
            f"[worker] disclosed {written} sample(s) for job={job.job_id} at {dest} "
            f"({len(allowed)} graded, {len(allowed_unfinished)} unfinished)"
        )
        return str(dest)


    def import_baseline(self, ctx: RunContext, results_dir: str) -> dict:
        """Adopt a finished baseline validation: its score, its ids, and its transcripts.

        One artefact, three facts, recorded together. Kept together on purpose: a run
        that is handed the base model's per-sample results can read 150 sample ids out
        of them, and if the service had learned the score from one place and the split
        membership from another, it could be refusing to evaluate an id the agent is
        already looking at. Deriving all three from the same tree makes that
        disagreement impossible.

        The score is computed here rather than typed by the operator, for the same
        reason: a number entered by hand can disagree with the tree it came from.
        """
        self._assert_json_task_cache(ctx.task)
        root = Path(results_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"not a directory: {root}")
        records = {str(r.get("task_id")): r for r in trial_results.collect(root)}
        if not records:
            raise ValueError(f"no trials under {root}")

        effective_source = self._validation_source(ctx.task)
        declared = self.baselines.baseline_resource_config(ctx.task, ctx.base_model, effective_source.raw)
        if declared is not None:
            raise ValueError("This baseline is a published resource; restore/update its benchmark payload offline instead of importing over it.")
        order = [str(item) for item in trial_results.dataset_order(root)]
        if effective_source.kind == "json":
            # JSON membership is explicit and Tau3-only. It must never be coerced
            # into a prefix or recorded in the prefix-oriented TaskStats store.
            val_prefix = list(self._json_split_ids(ctx.task, "val"))
            heldout_ids = list(self._json_split_ids(ctx.task, "test"))
            if set(order) != set(val_prefix) | set(heldout_ids):
                raise ValueError(f"the evaluation under {root} does not cover exactly the configured JSON split")
            val_n, held_out = len(val_prefix), len(heldout_ids)
        else:
            # The split boundary is a ratio, so it needs the dataset size, and the size is
            # learned from evaluations rather than declared. Without it there is no way to
            # tell a complete validation from a truncated one.
            total = self._dataset_total(ctx.task)
            val_n = valsplit.resolve_val_n(effective_source, total)
            if not val_n:
                raise ValueError(
                    f"the dataset size for {ctx.task} is not known yet, so the validation split "
                    "cannot be told from the held-out remainder. Declare dataset_total in the "
                    "benchmark.json, run one evaluation through the service, or configure "
                    "VTB_VAL_SOURCE=split:<count>."
                )
            if len(order) < val_n:
                raise ValueError(
                    f"the evaluation under {root} covers {len(order) or 'an unknown number of'} "
                    f"samples in dataset order, fewer than the {val_n}-sample validation split"
                )
            val_prefix = order[:val_n]
            heldout_ids = order[val_n:]
            held_out = len(heldout_ids)

        graded_ids = [i for i in val_prefix if (records.get(i) or {}).get("graded")]
        if not graded_ids:
            raise ValueError(f"no graded trials in the validation prefix of {root}")
        prefix_counts = trial_results.score_records([records[i] for i in val_prefix if i in records])
        # The held-out remainder's baseline score is what a held-out-only final is
        # compared against; computed from the same tree so the two never disagree.
        heldout_counts = trial_results.score_records([records[i] for i in heldout_ids if i in records])
        # An empty or all-ungraded remainder scores 0.0 by convention, which is not a
        # measurement; None keeps a later held-out-only final from being compared
        # against a fabricated zero.
        heldout_score = heldout_counts.get("accuracy") if heldout_counts.get("graded") else None

        # Prefix policies must retain observed order and membership for a future
        # held-out-only final. JSON policies already carry authoritative membership.
        warning_order = None
        warning = None
        if effective_source.kind != "json":
            warning_order = self.task_stats.observe_order(ctx.task, order, len(order))
            if warning_order:
                self.log(f"[baseline] WARNING {warning_order}")
            warning = self.task_stats.observe_val_ids(ctx.task, val_prefix, val_n)
        # The score the agent is shown has to be measured on the same samples the agent
        # is scored on. On this dataset the validation prefix ran 5.7 points harder than
        # the full 500, so publishing the full-dataset number would have every run read
        # its first honest validation as a large regression.
        score = prefix_counts.get("accuracy")
        if isinstance(score, (int, float)):
            # The score is only interpretable against the val source it was measured
            # under; pin it into the cache record so a later service running another
            # source cannot silently pair the two.
            self.baselines.put(
                ctx.task, ctx.base_model, float(score),
                extra={
                    "val_source": effective_source.raw,
                    "split_members": val_n,
                    "heldout_score": heldout_score,
                },
            )

        dest = self.baselines.baseline_payload_dir(ctx.task, ctx.base_model, effective_source.raw)
        samples_dir = dest / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        wanted = set(graded_ids)
        for task_id, trial_dir in trial_results.index_trials(root).items():
            if task_id not in wanted:
                continue
            trial_results.copy_disclosable_trial(trial_dir, samples_dir / task_id)
            written += 1
        (dest / "summary.json").write_text(
            json.dumps(
                {
                    "task": ctx.task,
                    "base_model": ctx.base_model,
                    "baseline_score": score,
                    "measured_on": "the %d-sample validation split" % val_n,
                    "val_source": effective_source.raw,
                    "graded": prefix_counts.get("graded"),
                    "solved": prefix_counts.get("solved"),
                    "heldout_score": heldout_score,
                    "heldout_graded": heldout_counts.get("graded"),
                    "held_out_samples_skipped": held_out,
                    "sample_ids": sorted(graded_ids),
                    "samples_written": written,
                    "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "withheld": (
                        "verifier/report.json and verifier/test-stdout.txt name the "
                        "hidden tests, which is the grading answer"
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Promote the tree to a named validation resource (declaration in
        # training/resources/validation/<id>/, payload under VTB_ROOT), so a run
        # mounts it by resource id and a recipe's validation block resolves to it. The
        # legacy baseline_disclosure path stays as a symlink for in-flight readers.
        resource_id = valsplit.baseline_resource_id(ctx.task, ctx.base_model, effective_source.raw)
        resource_dir = Path(config.REPO_ROOT) / "training" / "resources" / resource_id
        resource_dir.mkdir(parents=True, exist_ok=True)
        (resource_dir / "resource_config.json").write_text(
            json.dumps(
                {
                    "id": resource_id,
                    "type": "validation_baseline",
                    "description": (
                        f"Baseline validation transcripts: {ctx.base_model} on "
                        f"{effective_source.describe()} of {ctx.task}."
                    ),
                    "benchmark": ctx.task,
                    "base_model": ctx.base_model,
                    "val_source": effective_source.raw,
                    "materialized_path": str(dest).replace(str(config.VTB_ROOT), "${VTB_ROOT}"),
                    "score": score,
                    "heldout_score": heldout_score,
                    "graded": prefix_counts.get("graded"),
                    "split_members": val_n,
                    "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "mount_hint": (
                        "Bind-mounted read-only at $HOME/grading_baseline by "
                        "run_task.sh when the run's recipe declares this val "
                        "source. Never copied into the workspace."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        legacy = self.baselines.disclosure_dir(ctx.task, ctx.base_model)
        if legacy != dest:
            legacy.parent.mkdir(parents=True, exist_ok=True)
            if legacy.is_symlink():
                legacy.unlink()
            elif legacy.is_dir():
                shutil.rmtree(str(legacy))  # a pre-promotion import's copy; canonical now
            legacy.symlink_to(dest)
        self.log(
            f"[baseline] imported {ctx.task} {ctx.base_model}: score={score} on the "
            f"{val_n}-sample split (graded={prefix_counts.get('graded')}), "
            f"samples={written}, held-out skipped={held_out}, at {dest}"
        )
        if warning:
            self.log(f"[baseline] WARNING {warning}")
        return {
            "task": ctx.task,
            "base_model": ctx.base_model,
            "baseline_score": score,
            "measured_on_samples": val_n,
            "graded": prefix_counts.get("graded"),
            "val_ids_recorded": len(val_prefix),
            "samples_disclosed": written,
            "held_out_samples_skipped": held_out,
            "disclosure_dir": str(dest),
            "warning": warning,
        }
