"""Agent-facing disclosure: redaction and per-sample outcome assembly.

Everything an agent is allowed to read back about its own validation runs is
shaped here. The mixin methods share GradingService state (registry, config,
task stats) but contain no queue or lifecycle logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from .. import config, progress
from ...runner import trial_results
from ..store import RunContext

from .jobs import Job


class DisclosureMixin:

    # GradingService disclosure methods; see class assembly in .service


    def job(self, token: str, job_id: str) -> dict:
        """Fetch a job, refusing tokens that do not own it."""
        ctx = self.registry.lookup(token)  # rejects unknown tokens outright
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError("unknown job_id")
        if job.token != token:
            raise KeyError("unknown job_id")
        payload = job.public()
        if job.state == "queued":
            payload["queue_position"] = self._position_of(job_id)
        if job.state in ("queued", "running", "waiting_for_gpu"):
            payload["progress"] = self._progress_of(job)
        if job.wait_detail:
            payload["resource_advice"] = self._resource_advice(job)
        if ctx.role == "agent":
            payload = self._redact_for_agent(payload)
        return payload


    def sample_detail(self, token: str, job_id: str, sample_id: str) -> dict:
        """One evaluated sample's problem statement, transcript and agent counts.

        Fetched per id rather than shipped with every result. A 150-sample split with
        a 12k-character excerpt each is a quarter of a megabyte in one response,
        which an agent reads once, spills to a file and never opens again. What it is
        actually diagnosing is a handful of ids -- and the ones worth reading are the
        slow failures at the back of the run, which is exactly what a bulk response
        truncated away.

        The samples this job really evaluated are the whole boundary, and a sample it
        is still working on counts: the caller is told that id is running, and will be
        allowed to read the same trial in full once the evaluation finishes, so serving
        its transcript early discloses nothing the finished job would not. Which is the
        case worth serving -- a sample still running after twenty minutes is the one a
        caller needs to distinguish a loop from slow honest work, and by the time it is
        graded the decision it informed has already been made.
        """
        ctx = self.registry.lookup(token)
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None or job.token != token:
            raise KeyError("unknown job_id")
        if not self.disclosure & {"problems", "outputs"}:
            raise PermissionError("this service does not disclose per-sample detail")
        graded = self._trial_dirs(ctx, job)
        # Graded first: a sample that finished between the two lookups is described by
        # its result rather than reported as still running.
        running = progress.live_trial_dirs(self._attempt_metrics_dir(job))
        trial_dir = graded.get(sample_id) or running.get(sample_id)
        direct = self._direct_trajectory(ctx, job, sample_id)
        if trial_dir is None and direct is None:
            known = len(graded) + len(running)
            if not known:
                raise KeyError(f"job {job_id} has no readable per-sample record")
            raise KeyError(
                f"{sample_id} is not one of the {known} samples this job has run"
            )

        out: dict = {"job_id": job_id, "sample_id": sample_id}
        # Stated rather than left to be inferred from a missing reward: a transcript that
        # ends mid-task looks identical to one that gave up, and the difference decides
        # whether the caller should wait or act.
        out["running"] = sample_id not in graded and direct is None
        if "problems" in self.disclosure:
            out["problem_statement"] = self._problem_statement(job.eval_task, sample_id)
        if direct is not None:
            if "per_sample_outcomes" in self.disclosure:
                for key in ("correct_completions", "avg_at_n"):
                    if key in direct:
                        out[key] = direct[key]
            if "outputs" in self.disclosure:
                out["direct_trajectory"] = self._direct_disclosure(direct)
            return out
        if "outputs" in self.disclosure:
            trajectory_file, trajectory = self._trajectory_excerpt(trial_dir)
            out["trajectory_file"] = trajectory_file
            out["trajectory_excerpt"] = trajectory
        out["agent_stats"] = trial_results.agent_stats(trial_dir)
        return out


    def _sample_record_roots(self, ctx: RunContext, job: Job) -> list:
        """This job's backend trees, wherever they can still be read.

        Two places, because the per-job scratch is deleted as soon as an evaluation
        succeeds while the archive under grading_state is written precisely so the
        record outlives it. A cancelled or failed job still has its scratch, so that
        is tried first.

        Both output directory names are accepted, matching what archive.py keeps:
        harness-shaped benchmarks write backend_output, the ones that drive an
        external framework write framework_output. Reading only the first name here
        is how a benchmark ends up archived but never disclosed.
        """
        roots = []
        for parent in (
            self.work_root / job.job_id,
            config.STATE_DIR / "archive" / ctx.run_id / f"{job.mode}_{job.job_id}",
        ):
            if not parent.is_dir():
                continue
            for attempt_dir in sorted(parent.glob("harness_metrics_*"), reverse=True):
                for output_name in ("backend_output", "framework_output"):
                    output_tree = attempt_dir / output_name
                    if output_tree.is_dir():
                        roots.append(output_tree)
                        break
        return roots


    def _trial_dirs(self, ctx: RunContext, job: Job) -> dict:
        """sample id -> trial directory, for every sample this job actually ran."""
        for root in self._sample_record_roots(ctx, job):
            found = trial_results.index_trials(root)
            if found:
                return found
        return {}


    def _direct_trajectory(
        self,
        ctx: RunContext,
        job: Job,
        sample_id: str,
    ) -> Optional[dict]:
        """Read one direct-evaluator trajectory without opening answer data."""
        for root in self._sample_record_roots(ctx, job):
            payload = self._read_json_or_none(
                root / "direct_trajectories" / f"{sample_id}.json"
            )
            if isinstance(payload, dict) and str(payload.get("task_id")) == sample_id:
                return payload
        return None


    def _direct_disclosure(self, trajectory: dict) -> dict:
        """Whitelist direct trajectory fields and strip scoring-only correctness."""
        out = {
            "task_id": trajectory.get("task_id"),
            "samples_per_task": trajectory.get("samples_per_task"),
        }
        if "per_sample_outcomes" in self.disclosure:
            for key in ("correct_completions", "avg_at_n"):
                if key in trajectory:
                    out[key] = trajectory[key]
        if "outputs" in self.disclosure:
            rows = []
            for item in trajectory.get("completions") or []:
                if not isinstance(item, dict):
                    continue
                row = {
                    key: item[key]
                    for key in (
                        "sample_index",
                        "seed",
                        "response",
                        "parsed_answer",
                        "parse_method",
                        "elapsed_seconds",
                    )
                    if key in item
                }
                if "response" in row:
                    row["response"] = self._truncate_disclosure(str(row["response"]))
                rows.append(row)
            out["completions"] = rows
        return out


    def _resource_advice(self, job: Job) -> dict:
        detail = job.wait_detail or {}
        if config.GPU_COORDINATION_MODE == "shared" and detail.get("reason") == "gpu_busy":
            return {
                "kind": "release_gpu",
                "message": (
                    "Validation is waiting for GPU memory. Stop training or serving GPU "
                    "processes, then do CPU-only log inspection or planning while waiting."
                ),
                "eval_gpus": detail.get("eval_gpus"),
                "free_mib": detail.get("free_mib"),
                "required_mib": detail.get("required_mib"),
                "remaining_seconds": detail.get("remaining_seconds"),
                "retry_after_seconds": detail.get("retry_after_seconds"),
            }
        return {
            "kind": "operator_contention",
            "message": "Validation is waiting for the grading GPU allocation.",
            "eval_gpus": detail.get("eval_gpus"),
            "free_mib": detail.get("free_mib"),
            "required_mib": detail.get("required_mib"),
            "remaining_seconds": detail.get("remaining_seconds"),
            "retry_after_seconds": detail.get("retry_after_seconds"),
        }


    def _redact_for_agent(self, payload: dict) -> dict:
        """Reduce a job view to what the configured disclosure level allows.

        An agent learns its validation score and nothing else by default. Each
        further level is also a contamination channel: per-sample outcomes let
        the split be mapped by elimination over enough submissions, and outputs
        carry the model's own reasoning back into its context. So the filter is
        a whitelist, and anything added to the job view later stays hidden until
        it is named here.
        """
        allowed = self.disclosure
        isolated_gpu_wait = payload.get("state") == "waiting_for_gpu" and config.GPU_COORDINATION_MODE == "isolated"
        out = {
            "job_id": payload["job_id"],
            "state": "queued" if isolated_gpu_wait else payload["state"],
            "mode": payload["mode"],
            "submitted_at": payload["submitted_at"],
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
        }
        if "queue_position" in payload:
            out["queue_position"] = payload["queue_position"]
        if payload.get("progress") and not isolated_gpu_wait:
            # Progress is counts and timings only. The log path is a host path and
            # is not the agent's business.
            out["progress"] = {k: v for k, v in payload["progress"].items() if k != "log"}
        if config.GPU_COORDINATION_MODE == "shared" and payload.get("resource_advice"):
            out["resource_advice"] = payload["resource_advice"]

        result = payload.get("result") or None
        if result:
            disclosed = {"ok": result.get("ok"), "error": result.get("error")}
            if "score" in allowed:
                disclosed["validation_score"] = result.get("score")
            per_sample_outcomes = result.get("disclosed_per_sample_outcomes")
            if per_sample_outcomes and allowed & {"per_sample_outcomes", "problems", "outputs"}:
                filtered = []
                for outcome in per_sample_outcomes:
                    item = {"id": outcome.get("id")}
                    if "per_sample_outcomes" in allowed:
                        for key in ("score", "partial", "correct", "reward", "graded", "ungraded_reason"):
                            if key in outcome:
                                item[key] = outcome.get(key)
                        # Counts, not content: turns, model calls and the harness's
                        # own exit status. This is the signal that separates a task
                        # the model gave up on from one it looped on, and it says
                        # nothing about the sample itself.
                        if "agent_stats" in outcome:
                            item["agent_stats"] = outcome.get("agent_stats")
                    if "problems" in allowed and "problem_statement" in outcome:
                        item["problem_statement"] = outcome.get("problem_statement")
                    if "outputs" in allowed:
                        for key in ("trajectory_excerpt", "output_excerpt", "trajectory_file"):
                            if key in outcome:
                                item[key] = outcome.get(key)
                    filtered.append(item)
                disclosed["per_sample_outcomes"] = filtered
            if result.get("partial") is not None:
                disclosed["partial"] = result.get("partial")
            # Said plainly rather than left to be inferred from the request: a score
            # over samples the submission named is not the validation score, and
            # comparing two of them is not an experiment.
            if result.get("subset"):
                disclosed["subset"] = True
                disclosed["comparable_to_full"] = False
            out["result"] = disclosed
        return out


    def _truncate_disclosure(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        limit = max(0, config.AGENT_DISCLOSURE_MAX_CHARS)
        if limit and len(text) > limit:
            return text[:limit] + f"\n... [truncated to {limit} characters]"
        return text


    def _read_json_or_none(self, path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None


    def _read_text_or_none(self, path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None


    def _successful_attempt_dir(self, work_dir: Path, outcome: dict) -> Optional[Path]:
        attempts = outcome.get("attempt_log") or []
        winning = outcome.get("winning_attempt")
        if not isinstance(winning, int) or winning < 1 or winning > len(attempts):
            return None
        entry = attempts[winning - 1]
        phase = entry.get("phase")
        attempt = entry.get("attempt")
        if not isinstance(phase, int) or not isinstance(attempt, int):
            return None
        candidate = work_dir / f"harness_metrics_p{phase}_a{attempt}"
        return candidate if candidate.is_dir() else None


    def _problem_statement(self, eval_task: str, sample_id: str) -> Optional[str]:
        roots = [
            config.runtime_paths.HUB_TASK_SPECS_DIR / eval_task / sample_id,
            config.runtime_paths.HUB_TASK_SPECS_DIR / eval_task / "tasks" / sample_id,
            config.task_dir(eval_task) / sample_id,
            config.task_dir(eval_task) / "tasks" / sample_id,
        ]
        for root in roots:
            for name in ("instruction.md", "prompt.md", "problem.md", "task.md"):
                text = self._read_text_or_none(root / name)
                if text:
                    return self._truncate_disclosure(text)
            task_toml = self._read_text_or_none(root / "task.toml")
            if task_toml:
                return self._truncate_disclosure(task_toml)
        return None


    def _trajectory_excerpt(self, trial_dir: Optional[Path]) -> tuple[Optional[str], Optional[str]]:
        """The agent transcript for one trial, for /sample to hand back by id."""
        if trial_dir is None:
            return None, None
        path = trial_results.agent_trajectory_path(trial_dir)
        if path is not None:
            limit = max(0, config.AGENT_DISCLOSURE_MAX_CHARS)
            # Steps first, raw tail only for a transcript shape that cannot be parsed into
            # them: a character tail of a looping trial is one huge tool result with no
            # step in it, which is the case a caller is asking about.
            stepwise = trial_results.trajectory_tail(trial_dir, limit)
            if stepwise is not None:
                return path.name, stepwise
            raw = self._truncate_trajectory(self._read_text_or_none(path))
            # A running trial is rewritten as it goes, so roughly one read in a hundred
            # catches a half-written file and lands here. Said plainly, because the raw
            # slice looks like an unreadable transcript rather than a transient miss, and
            # a caller that concludes the former stops asking instead of asking again.
            if raw is not None and path.suffix == ".json":
                return path.name, "[transcript was mid-write; ask again for steps]\n" + raw
            return path.name, raw
        result = self._read_json_or_none(trial_dir / "result.json")
        message = ((result or {}).get("exception_info") or {}).get("exception_message")
        if not isinstance(message, str) or not message:
            # Nothing written yet, which a sample that has just started has every right to
            # be. Naming a file that does not exist beside a null excerpt reads as a trial
            # that failed, and that is the opposite of what has happened.
            return None, None
        return "result.json", self._truncate_disclosure(message)


    def _truncate_trajectory(self, text: Optional[str]) -> Optional[str]:
        """Keep the end of a transcript, not the beginning.

        The other direction is what _truncate_disclosure does, and it is right for a
        problem statement, which starts by stating the problem. A transcript is the
        opposite: it starts with setup that every trial shares and ends where this one
        stopped converging, so a head-truncated megabyte returns the one part that
        cannot explain a slow or looping sample. This module's own reason for serving
        transcripts by id says as much -- "the ones worth reading are the slow failures
        at the back of the run" -- and truncating from the front withheld exactly that.

        The omission is counted in the text rather than reported in a separate field, so
        an excerpt cannot be read as a short trial by a caller that ignored the field.
        """
        if text is None:
            return None
        limit = max(0, config.AGENT_DISCLOSURE_MAX_CHARS)
        if limit and len(text) > limit:
            return f"... [earlier {len(text) - limit} characters omitted]\n" + text[-limit:]
        return text


    def _load_attempt_records(self, attempt_dir: Optional[Path]) -> dict[str, dict]:
        if attempt_dir is None:
            return {}
        records = self._read_json_or_none(attempt_dir / "records.json")
        if not isinstance(records, list):
            return {}
        return {str(record.get("task_id")): record for record in records if record.get("task_id") is not None}


    def _build_disclosed_per_sample_outcomes(self, work_dir: Path, job: Job, outcome: dict) -> Optional[list[dict]]:
        """Every sample this evaluation graded, with the counts that explain it.

        Complete on purpose. This used to stop at AGENT_DISCLOSURE_MAX_SAMPLES rows,
        which on a 150-sample split silently dropped 86 of them: the agent was left
        analysing a prefix of an evaluation it had paid for all of, with nothing in
        the response to say so. The cap now bounds what may be disclosed -- never
        more samples than the job was allowed to evaluate -- not how much of it comes
        back.

        The heavy material is deliberately absent; /sample serves it by id, which is
        how a diagnosis actually proceeds.
        """
        attempts = outcome.get("attempt_log") or []
        winning = outcome.get("winning_attempt")
        if not isinstance(winning, int) or winning < 1 or winning > len(attempts):
            return None
        entry = attempts[winning - 1]
        phase = entry.get("phase")
        attempt = entry.get("attempt")
        per_sample_path = work_dir / f"metrics_p{phase}_a{attempt}.per_sample.json"
        raw = self._read_json_or_none(per_sample_path)
        if isinstance(raw, dict):
            sample_ids = [str(item) for item in raw.get("sample_ids") or []]
            correct = raw.get("correct") if isinstance(raw.get("correct"), dict) else {}
            value_key = "score" if raw.get("score_mode") == "avg_at_n" else "correct"
            rows = [{"id": sid, value_key: correct.get(sid)} for sid in sample_ids]
        elif isinstance(raw, list):
            rows = [dict(item) for item in raw if isinstance(item, dict) and item.get("id") is not None]
        else:
            return None

        records = self._load_attempt_records(self._successful_attempt_dir(work_dir, outcome))
        trial_dirs = {
            sid: Path(record["trial_dir"])
            for sid, record in records.items()
            if isinstance(record.get("trial_dir"), str)
        }
        return self._disclose_rows(rows, job, trial_dirs)


    def _disclose_rows(self, rows: list, job: Job, trial_dirs: dict) -> list:
        """Per-sample rows for an agent: the verdict, and how the trial was spent.

        Shared by a finished evaluation and by a cancelled one. They reach their rows
        differently -- a metrics file that is only written at the end, versus the
        trial tree itself -- and disclose them identically.
        """
        # A bound, not a page size: the agent may see every sample it had evaluated
        # and no sample it did not. An agent's limit is already capped at the
        # validation split, so this cannot reach past that boundary either. An
        # operator may lower it further with VTB_AGENT_DISCLOSURE_MAX_SAMPLES.
        bounds = [n for n in (job.limit, config.AGENT_DISCLOSURE_MAX_SAMPLES) if isinstance(n, int) and n > 0]
        bound = min(bounds) if bounds else None
        if bound is not None and len(rows) > bound:
            self.log(
                f"[worker] job={job.job_id} produced {len(rows)} per-sample rows against a "
                f"disclosure bound of {bound}; disclosing the first {bound}"
            )
            rows = rows[:bound]

        disclosed = []
        for row in rows:
            sid = str(row.get("id"))
            item = {"id": sid}
            for key in ("score", "partial", "correct", "reward", "graded"):
                if key in row:
                    item[key] = row.get(key)
            if row.get("reason"):
                item["ungraded_reason"] = row.get("reason")
            trial_dir = trial_dirs.get(sid)
            if trial_dir is not None:
                stats = trial_results.agent_stats(trial_dir)
                if stats:
                    item["agent_stats"] = stats
            disclosed.append(item)
        return disclosed
