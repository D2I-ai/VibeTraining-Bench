"""Live progress for a running evaluation, read from what it leaves on disk.

`evaluate.py` must not be instrumented -- it is the immutable scoring authority and
the agent prompt forbids touching it -- so progress has to be recovered from what the
subprocess already writes.

The reliable signal is the per-sample result files. Harnesses that run one directory
per sample write `<instance_id>__<hash>/result.json` as each one is graded, so
counting them is a real "k of n done", and each file also carries what that sample
spent and how it ended. `read_sample_dirs` uses this, and where it works the
percentage, the ETA, the token distribution and the failure counts all come from it.
It also answers the question that matters at the end of a long evaluation -- *which*
samples are still outstanding, and for how long -- because a run sitting at 29 of 30
for forty minutes is one long-tail sample, not a wedged job, and the two call for
opposite decisions.

The stdout of the evaluation is a much weaker source, and on the agentic benchmarks it
is no source at all:

  * inspect-ai prints NO sample counter. evaluate.py configures
    init_display_type("plain"), score_display=False and log_realtime=False, so there
    is no "12/150" and no percentage to scrape.
  * the harness's own progress bar is written by rich, which detects a non-tty and
    prints one line at the end instead of updating in place -- a completed 150-sample
    log measured zero carriage returns. Nothing to tail while it runs.
  * where a vLLM APIServer access log is present it logs one line per generation:
        INFO: 127.0.0.1:35222 - "POST /v1/chat/completions HTTP/1.1" 200 OK
    which is countable. The agentic harnesses' attempt logs contain none of these,
    so `completions_observed` reads 0 for the whole evaluation there.
  * startup markers, where the log has them: a "Starting new server" warning,
    safetensors shard loading, CUDA graph capture, "Application startup complete".

So the completion count is a *proxy* and it is reported as `completions_observed`
rather than as a sample count. The two coincide only where one sample costs exactly
one completion; a multi-turn, judged or agentic benchmark spends several completions
per sample, so the count runs ahead of real progress -- or stays at 0. A percentage is
derived from it only for the benchmarks named in SINGLE_COMPLETION_TASKS below, and
everything else gets a `note` saying why not. Where the two disagree the sample files
win, including over the phase the log implies: "launching the container" printed
beside 18 graded samples is worse than printing nothing.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Optional

from . import config
from ..runner import trial_results

# Tasks where one sample corresponds to exactly one completion, so
# completions_observed can be turned into a percentage.
#
# Empty, because the agentic benchmarks spend many completions per task and a
# completion count says nothing about how many of their tasks are done. The
# non-agentic ones installed here look like one completion per sample and could be
# listed, but only after confirming each runs a single epoch -- with epochs>1 the
# count outruns the sample total and the percentage passes 100.
SINGLE_COMPLETION_TASKS: frozenset[str] = frozenset()

# Phases, in the order they occur. Later matches win.
_PHASE_MARKERS = (
    ("starting_server", re.compile(r"Starting new server for")),
    ("loading_weights", re.compile(r"Loading safetensors checkpoint shards")),
    ("warmup", re.compile(r"Capturing CUDA graphs")),
    ("generating", re.compile(r"Application startup complete|vLLM server started at")),
)

_COMPLETION_RE = re.compile(r'"POST /v1/chat/completions HTTP/1\.1" 200')

# Below this many records, a historical median is reported with a caveat rather than
# as an estimate. Two records make a median that is just their mean, so one outlier
# drags it halfway.
HISTORY_TRUSTED_RUNS = 3

# `<instance_id>__<suffix>` directories the per-sample harnesses create. The suffix
# is a per-trial nonce, so the id is everything before the last `__`.
_SAMPLE_DIR_RE = re.compile(r"^(?P<instance>.+)__[A-Za-z0-9]+$")


def read_sample_dirs(metrics_dir: Optional[Path], now: float) -> Optional[dict]:
    """Count finished samples by looking for the files the harness already writes.

    Harnesses that give each sample its own directory drop a `result.json` into it
    when that sample is graded, which is a true completion signal rather than the
    proxy a completion count gives. Returns None when the layout is not present, so
    a caller can fall back without having to know which harnesses do this.

    The result files also carry each sample's token spend, which is what separates
    the two ways an evaluation gets slow. A model that got uniformly more expensive
    moves the median; a model with a few trials looping until they run out of turns
    leaves the median alone and stretches the upper percentiles. On this benchmark a
    healthy queue measured p90/median 4.0 against 18.0 for one where a third of the
    trials looped, while the medians differed only 2x -- one number cannot tell those
    apart, so the percentiles are reported rather than a mean.

    Deliberately cheap: two directory listings, a stat per outstanding sample, and a
    small read per finished one. It runs on every poll, and polls are not rate-limited.
    Loop indicators cost more than this and are not gathered here -- see loop_indicators,
    which is applied only to the handful of samples actually reported.
    """
    if metrics_dir is None:
        return None
    try:
        roots = sorted(
            (p for p in (metrics_dir / "backend_output").iterdir() if p.is_dir()),
            key=lambda p: p.name,
        )
    except OSError:
        return None
    if not roots:
        return None

    finished = 0
    outstanding = []
    # Kept beside the ids rather than in them: an outstanding entry is disclosed verbatim,
    # and a host path is not something a run should learn from its own progress block.
    dirs: dict = {}
    spend: list = []
    errors: dict = {}
    for entry in _iter_sample_dirs(roots[-1]):
        match = _SAMPLE_DIR_RE.match(entry.name)
        instance = match.group("instance") if match else entry.name
        result = entry / "result.json"
        if result.is_file():
            finished += 1
            tokens, failure = _sample_facts(result)
            if tokens:
                spend.append(tokens)
            if failure:
                errors[failure] = errors.get(failure, 0) + 1
            continue
        try:
            started = entry.stat().st_mtime
        except OSError:
            started = now
        outstanding.append({"instance_id": instance, "seconds_running": int(max(now - started, 0))})
        dirs[instance] = entry

    if finished == 0 and not outstanding:
        return None
    outstanding.sort(key=lambda item: item["seconds_running"], reverse=True)
    return {
        "finished": finished,
        "outstanding": outstanding,
        "dirs": dirs,
        "spend": spend,
        "errors": errors,
    }


# How often one running sample's transcript is re-read. A loop that is worth acting on
# lasts many minutes, so a minute-old count is as useful as a fresh one, and rating the
# scan by the clock is what makes its cost independent of how fast the harness writes.
# A content-keyed cache would never hit here: a running transcript changes on every step.
LOOP_SCAN_INTERVAL = 60.0

# Above this, the field is omitted rather than paid for. Transcripts of twenty megabytes
# have been measured on this benchmark, and parsing one inside a poll is not free.
LOOP_SCAN_MAX_BYTES = 24 * 1024 * 1024

# Deliberately unlocked, unlike the other shared state in this module. build_progress runs
# on concurrent request threads, but nothing here iterates the dict and a single get or set
# is atomic, so the worst race is two threads parsing one transcript and agreeing. A lock
# would be worse than the race it prevents: it would serialise a parse that has been
# measured at 181ms behind every other caller's poll.
_loop_cache: dict = {}


def loop_indicators(trial_dir: Optional[Path], now: float) -> Optional[dict]:
    """Whether a running sample is still getting anywhere, from its own transcript.

    These are the counters a caller was otherwise fetching one request at a time. Over one
    twelve-hour session a researcher asked /sample about individual running samples
    nineteen times and stripped the transcript excerpt out of seventeen of those replies,
    keeping only `no_progress_repeats` -- so the number, not the transcript, was what it
    came for. Reporting it beside an id that is already disclosed removes the round trip
    and discloses nothing new: it describes the caller's own model.

    The field names match trial_results.agent_stats on purpose. Anyone comparing the
    aggregate here against the detail from /sample has to get the same number twice, or
    the two contradict each other and neither can be trusted.

    None when the transcript cannot be read cheaply, so a caller sees the field missing
    rather than a zero that would read as "making progress".
    """
    if trial_dir is None:
        return None
    key = str(trial_dir)
    cached = _loop_cache.get(key)
    if cached is not None and now - cached[0] < LOOP_SCAN_INTERVAL:
        return cached[1]
    indicators = None
    path = trial_results.agent_trajectory_path(trial_dir)
    if path is not None and path.suffix == ".json":
        try:
            small_enough = path.stat().st_size <= LOOP_SCAN_MAX_BYTES
        except OSError:
            small_enough = False
        if small_enough:
            stats = trial_results.agent_stats(trial_dir)
            repeats = stats.get("no_progress_repeats")
            if isinstance(repeats, int):
                indicators = {
                    "turns": stats.get("turns"),
                    "no_progress_repeats": repeats,
                    "stalled": repeats >= trial_results.NO_PROGRESS_REPEATS,
                }
    # Bounded rather than pruned: job directories are removed when an evaluation ends, so
    # entries here go stale in bulk and there is nothing to preserve across that.
    if len(_loop_cache) > 512:
        _loop_cache.clear()
    # Only an answer is cached. Reaching None costs a few path checks and no parse, while
    # caching it would hold "nothing to read yet" over a sample that started writing its
    # transcript a second later -- the rate limit exists to bound parsing, not to make a
    # sample's first minute of progress invisible.
    if indicators is not None:
        _loop_cache[key] = (now, indicators)
    return indicators


def live_trial_dirs(metrics_dir: Optional[Path]) -> dict:
    """sample id -> trial directory, for samples started but not yet graded.

    The same walk read_sample_dirs does, exposed by id so a caller can be served the
    detail of a sample while it is still running. Sharing the walk is the point rather
    than an economy: the ids reported as outstanding are the ids that can then be asked
    about, and two independent scans of this tree would eventually disagree about which
    those are -- leaving an id that is visibly running but answers 404.

    Graded samples are deliberately absent. trial_results.index_trials already indexes
    those, keyed on the result.json this skips, so a caller checks that first and falls
    back here; a sample that lands between the two lookups is found by the first.
    """
    if metrics_dir is None:
        return {}
    try:
        roots = sorted(
            (p for p in (metrics_dir / "backend_output").iterdir() if p.is_dir()),
            key=lambda p: p.name,
        )
    except OSError:
        return {}
    if not roots:
        return {}
    live: dict = {}
    for entry in _iter_sample_dirs(roots[-1]):
        if (entry / "result.json").is_file():
            continue
        match = _SAMPLE_DIR_RE.match(entry.name)
        live[match.group("instance") if match else entry.name] = entry
    return live


def _sample_facts(result: Path) -> tuple:
    """(input tokens, exception type) for one graded sample; either may be None.

    Both come out of one read because this runs per finished sample on every poll, and
    polls are not rate-limited.

    Input rather than output tokens: a looping trial re-sends its whole transcript
    every step, so the prompt side is where the cost of a loop actually shows up.

    The exception type is the harness's own classification of how the trial ended, and
    it is the least ambiguous loop signal available -- a trial that repeats one command
    until it runs out of turns ends as a timeout, and the count of those is a fact
    rather than a heuristic over the transcript.
    """
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (None, None)
    if not isinstance(payload, dict):
        return (None, None)
    tokens = (payload.get("agent_result") or {}).get("n_input_tokens")
    tokens = tokens if isinstance(tokens, int) and tokens > 0 else None
    failure = (payload.get("exception_info") or {}).get("exception_type")
    failure = failure if isinstance(failure, str) and failure else None
    return (tokens, failure)


def _append_note(progress: dict, sentence: str) -> None:
    """Add a caveat without dropping whichever one is already there.

    The counting note and the ETA note are decided independently, so the second one
    written must not silently replace the first.
    """
    existing = progress.get("note")
    progress["note"] = f"{existing}; {sentence}" if existing else sentence


def _percentile(ordered: list, fraction: float) -> int:
    """Nearest-rank percentile of an already sorted list."""
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return int(ordered[index])


def _iter_sample_dirs(root: Path) -> list:
    try:
        return [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return []

PHASE_LABELS = {
    "queued": "waiting in the shared evaluation queue",
    "starting": "launching the evaluation container",
    "starting_server": "starting vLLM",
    "loading_weights": "loading model weights",
    "warmup": "capturing CUDA graphs",
    "generating": "generating answers",
}


class _LogState:
    __slots__ = (
        "offset",
        "completions",
        "phase",
        "generating_since",
        "last_completion_at",
        "checked_at",
    )

    def __init__(self) -> None:
        self.offset = 0
        self.completions = 0
        self.phase = "starting"
        # When the vLLM server came up, i.e. when generation could first start.
        # This is the correct denominator for a completion rate: every counted
        # completion happened after it. Anchoring on "when we first *noticed*
        # completions" instead inflates the rate, because the completions
        # already counted at that moment get credited zero elapsed time.
        self.generating_since: Optional[float] = None
        self.last_completion_at: Optional[float] = None
        self.checked_at = 0.0


class ProgressReader:
    """Incrementally tails evaluation logs so polling stays cheap.

    An agent is expected to poll for progress, so each read only consumes the
    bytes appended since the previous read, and results are cached briefly.
    """

    def __init__(self, cache_ttl: float = 2.0) -> None:
        self._states: dict[str, _LogState] = {}
        self._lock = threading.Lock()
        self._cache_ttl = cache_ttl

    def read(self, log_path: Path) -> dict:
        key = str(log_path)
        now = time.time()
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = _LogState()
                self._states[key] = state
            elif now - state.checked_at < self._cache_ttl:
                return self._snapshot(state)

            try:
                size = log_path.stat().st_size
            except OSError:
                state.checked_at = now
                return self._snapshot(state)

            if size < state.offset:  # log rotated or replaced
                state.offset = 0
                state.completions = 0

            if size > state.offset:
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                        fh.seek(state.offset)
                        chunk = fh.read()
                        state.offset = fh.tell()
                except OSError:
                    state.checked_at = now
                    return self._snapshot(state)

                found = len(_COMPLETION_RE.findall(chunk))
                if found:
                    state.last_completion_at = now
                    state.completions += found

                for name, pattern in _PHASE_MARKERS:
                    if pattern.search(chunk):
                        state.phase = name
                if state.phase == "generating" and state.generating_since is None:
                    state.generating_since = now

            state.checked_at = now
            return self._snapshot(state)

    @staticmethod
    def _snapshot(state: _LogState) -> dict:
        return {
            "phase": state.phase,
            "completions_observed": state.completions,
            "generating_since": state.generating_since,
            "last_completion_at": state.last_completion_at,
        }

    def forget(self, log_path: Path) -> None:
        with self._lock:
            self._states.pop(str(log_path), None)


def build_progress(
    *,
    task: str,
    state: str,
    started_at: Optional[float],
    queue_position: Optional[int],
    attempt_phase: Optional[int],
    attempt_index: Optional[int],
    log_path: Optional[Path],
    target_samples: Optional[int],
    historical: Optional[tuple],
    reader: ProgressReader,
    # Defaulted: per-sample counting is an enrichment, and a caller that cannot
    # name the directory should still get the rest of the progress block.
    metrics_dir: Optional[Path] = None,
    # Defaulted for the same reason: a task whose baseline came from the published
    # table rather than a local evaluation has no per-sample timing to offer.
    baseline_timing: Optional[dict] = None,
) -> dict:
    """Assemble the progress block returned by /result.

    Several independent ETA sources, and the one used is always named so a caller
    can judge how much to trust it:

      observed_rate                  measured over this run's whole generation
                                     window; needs a known sample total.
      observed_sample_rate           measured from the per-sample result files,
                                     which count graded samples rather than
                                     completions. Preferred where available.
      historical_median              median duration of this run's own previous
                                     evaluations of the same task/mode/limit.
      historical_median_other_limit  same run, task and mode but a different
                                     sample limit, so only a rough hint. Startup
                                     cost dominates short evaluations, which is
                                     why this cannot simply be scaled by limit.
      baseline_sample_rate           the base model's measured per-sample cost on
                                     this split, scaled to this submission's
                                     sample count and concurrency. The cold-start
                                     source: a run's first submission has no
                                     history of its own, and another run's history
                                     describes another run's model.

    `historical` is (median_seconds, is_exact_key, n_records) or None.
    `baseline_timing` is {"mean_seconds": float, "samples": int} or None.
    """
    now = time.time()
    elapsed = int(now - started_at) if started_at else None

    progress: dict = {
        "phase": "queued" if state == "queued" else "starting",
        "phase_label": PHASE_LABELS["queued" if state == "queued" else "starting"],
        "elapsed_seconds": elapsed,
        "queue_position": queue_position,
        "attempt": None,
        "completions_observed": None,
        "target_samples": target_samples,
        "samples_finished": None,
        "samples_outstanding": None,
        "samples_unstarted": None,
        "outstanding_samples": None,
        "sample_input_tokens": None,
        "sample_errors": None,
        "percent": None,
        "eta_seconds": None,
        "eta_basis": None,
        "eta_from_runs": None,
        "baseline_sample_seconds": None,
        "note": None,
    }

    if attempt_phase is not None:
        progress["attempt"] = {
            "ladder_phase": attempt_phase,
            "attempt_in_phase": attempt_index,
            "ladder": "4/3/2 attempts across 3 phases",
        }

   # Counted independently of the log: the two signals fail in different ways, and
    # tying them together makes the reliable one inherit the fragile one's blind
    # spots. On the agentic harnesses the log carries no per-request lines at all --
    # rich detects a non-tty and prints its progress bar once, at the end -- so an
    # evaluation that is plainly running shows a completion count of 0 for hours.
    # The sample directories are there the whole time.
    samples = read_sample_dirs(metrics_dir, now)

    if log_path is not None:
        live = reader.read(log_path)
        progress["phase"] = live["phase"]
        progress["phase_label"] = PHASE_LABELS.get(live["phase"], live["phase"])
        progress["completions_observed"] = live["completions_observed"]
        progress["log"] = str(log_path)

        # How long since the last finished generation. An agent can use this to
        # tell "slow" from "wedged" without guessing from elapsed time alone.
        if live["last_completion_at"]:
            progress["seconds_since_last_completion"] = int(now - live["last_completion_at"])

        done = live["completions_observed"]
        single = task in SINGLE_COMPLETION_TASKS
    else:
        live = {"generating_since": None, "completions_observed": None}
        done = None
        single = False

    # A phase is a guess about a marker in a log; a sample directory is proof. Where they
    # disagree the phase is stale, and "launching the evaluation container" printed next
    # to samples that are already running is worse than printing nothing -- the label is
    # the part a reader believes. Applied whether or not there is a log to read, because
    # the default phase is just as wrong once samples have landed.
    #
    # An outstanding sample counts as proof, not just a graded one: the harness creates a
    # sample's directory when its trial starts, which is after the container is up and
    # the model is being served. Keying this on graded samples alone left the label
    # reading "launching the evaluation container" while 16 trials were already running.
    if samples and (samples["finished"] or samples["outstanding"]) and progress["phase"] != "generating":
        progress["phase"] = "generating"
        progress["phase_label"] = PHASE_LABELS.get("generating", "generating")

    if samples is not None:
        # Real per-sample counts beat both the completion proxy and the
        # historical median, so this branch wins over either.
        graded = samples["finished"]
        outstanding = samples["outstanding"]
        progress["samples_finished"] = graded
        progress["samples_outstanding"] = len(outstanding)
        if target_samples:
            # What has not started yet, which is what separates "early" from
            # "nearly done": both look like a handful of running samples.
            progress["samples_unstarted"] = max(target_samples - graded - len(outstanding), 0)
        # Capped: the id and age of the slowest few is what a decision needs,
        # and a queued benchmark can have hundreds outstanding at once.
        progress["outstanding_samples"] = outstanding[:5]
        # Loop counters, for the reported few only. That cap is what bounds the cost:
        # scanning every running sample on every poll would pay for transcripts nobody
        # asked about, while the slowest few are the ones a caller is deciding about.
        # Age alone cannot separate a loop from slow honest work, and that distinction
        # decides between cancelling and waiting.
        sample_dirs = samples.get("dirs") or {}
        stalled = 0
        for item in progress["outstanding_samples"]:
            indicators = loop_indicators(sample_dirs.get(item["instance_id"]), now)
            if not indicators:
                continue
            item.update(indicators)
            if indicators.get("stalled"):
                stalled += 1
        # Where the time is going, in the only form that answers the question.
        # A loop is not idle -- it re-sends its transcript every step -- so no
        # timing signal catches it; token spend does. The median says whether
        # every sample got more expensive, the upper percentiles say whether a
        # few are running away, and only both together distinguish the two.
        # Compare against the baseline disclosure, which carries the same field.
        spend = sorted(samples.get("spend") or [])
        if spend:
            progress["sample_input_tokens"] = {
                "median": _percentile(spend, 0.5),
                "p90": _percentile(spend, 0.9),
                "max": spend[-1],
                "counted": len(spend),
            }
        # How the failed trials failed, counted as they land. A timeout is what a trial
        # that keeps repeating one command finally ends as, so this rising while the
        # score does not is the loop showing itself mid-evaluation rather than in the
        # post-mortem.
        if samples.get("errors"):
            progress["sample_errors"] = dict(
                sorted(samples["errors"].items(), key=lambda kv: kv[1], reverse=True)
            )
        if target_samples:
            progress["percent"] = min(100.0, round(100.0 * graded / target_samples, 1))
        started_generating = live["generating_since"] or started_at
        if target_samples and graded and started_generating and now > started_generating:
            rate = graded / (now - started_generating)
            if rate > 0:
                remaining = max(target_samples - graded, 0)
                # A mean rate underestimates what is left when the remainder is
                # the slow tail: those samples have already outlived the average
                # by definition, or they would have finished. Never promise less
                # than the longest one has already been running.
                eta = int(remaining / rate)
                if outstanding:
                    eta = max(eta, outstanding[0]["seconds_running"])
                progress["eta_seconds"] = eta
                progress["eta_basis"] = "observed_sample_rate"
        # A long tail means nothing is left to start: every remaining sample is
        # already running. Judging it by the number running instead would call
        # any concurrency-limited run a tail -- 5 of 150 graded with 16 running
        # is not a tail, it is the beginning, and telling an agent to cancel
        # there would throw away the evaluation it is waiting for.
        unstarted = progress.get("samples_unstarted")
        is_tail = bool(target_samples) and unstarted == 0 and bool(outstanding)
        if is_tail and graded:
            slowest = outstanding[0]
            count = len(outstanding)
            progress["note"] = (
                f"{graded} of {target_samples} samples are graded and "
                f"{count} {'is' if count == 1 else 'are'} still running, with none left "
                f"to start; the slowest is {slowest['instance_id']} at "
                f"{slowest['seconds_running'] // 60} minutes. This is a long tail rather "
                "than a stalled evaluation: cancelling now returns everything already "
                "graded"
            )
        elif target_samples and unstarted:
            progress["note"] = (
                f"{graded} of {target_samples} samples are graded, {len(outstanding)} are "
                f"running and {unstarted} have not started yet, so this is still working "
                "through the split rather than waiting on a tail"
            )
        # Appended last, after the branches above have set whichever sentence describes the
        # shape of the run: those assign rather than append, so composing here is the only
        # order in which both survive. A tail made of looping samples is a different
        # decision from a tail of slow ones, and it takes both sentences to say which.
        if stalled:
            reported = len(progress["outstanding_samples"])
            if reported == 1:
                subject = "the running sample has"
            else:
                subject = (
                    f"{stalled} of the {reported} slowest running samples "
                    + ("has" if stalled == 1 else "have")
                )
            _append_note(
                progress,
                f"{subject} repeated one action without new output for "
                f"{trial_results.NO_PROGRESS_REPEATS}+ steps",
            )
    elif not single:
        progress["note"] = (
            "this task spends several completions per sample, so "
            "completions_observed runs ahead of the sample count and no "
            "percentage is derived"
        )
    elif target_samples and done:
        pct = min(100.0, round(100.0 * done / target_samples, 1))
        progress["percent"] = pct
        since = live["generating_since"]
        if since and now > since:
            # Rate over the whole generation window, not just since the
            # first poll that happened to notice completions.
            rate = done / (now - since)
            remaining = max(target_samples - done, 0)
            if rate > 0:
                progress["eta_seconds"] = int(remaining / rate)
                progress["eta_basis"] = "observed_rate"
    elif not target_samples:
        progress["note"] = (
            "total sample count is unknown for this submission, so only the "
            "raw completion count is reported"
        )

    if progress["eta_seconds"] is None and historical:
        median, exact, runs = historical
        remaining = median - (elapsed or 0)
        progress["eta_seconds"] = int(max(remaining, 0))
        progress["eta_basis"] = "historical_median" if exact else "historical_median_other_limit"
        progress["eta_from_runs"] = runs
        # A median of one or two records is a single observation wearing a statistic's
        # name, and quoting it as an ETA implies a sample size it does not have. Say
        # what the number rests on -- see DurationStats for what it cost not to.
        if runs < HISTORY_TRUSTED_RUNS:
            note = (
                f"this ETA is the median of {runs} previous "
                f"{'evaluation' if runs == 1 else 'evaluations'}, which is too few to "
                "be a promise; it is replaced by a measured rate once samples start "
                "landing"
            )
            _append_note(progress, note)

    # The base model's measured cost per sample, always reported once known: it is the
    # yardstick for the run's own rate. A trained model that costs several times the
    # base per sample is spending that time somewhere, and on this benchmark that
    # somewhere was trials repeating one command until they ran out of turns.
    base_mean = (baseline_timing or {}).get("mean_seconds")
    if isinstance(base_mean, (int, float)) and base_mean > 0:
        progress["baseline_sample_seconds"] = int(base_mean)

    # Cold start: no measured rate yet and no history of this run's own. Scale the base
    # model's per-sample cost by what is left and by how many run at once. Deliberately
    # last, so any measurement of the model actually being evaluated wins over an
    # assumption about a different model.
    if progress["eta_seconds"] is None and progress["baseline_sample_seconds"] and target_samples:
        graded_so_far = progress["samples_finished"] or 0
        remaining = max(target_samples - graded_so_far, 0)
        if remaining:
            progress["eta_seconds"] = int(remaining * base_mean / config.eval_concurrency())
            progress["eta_basis"] = "baseline_sample_rate"
            note = (
                "no measured rate yet, so this ETA assumes your model costs what the "
                "base model measured per sample; a model that loops will exceed it, and "
                "it is replaced by a measured rate once samples start landing"
            )
            _append_note(progress, note)

    return progress
