"""Reading an execution backend's trial tree and writing the metrics contract.

Generic middleware. Harbor writes the same trial layout for every benchmark -- a per-trial
directory holding `result.json` and `verifier/reward.json` -- so one collector serves
both, and a benchmark contributes nothing here.

Two shapes of input, because a benchmark's tasks are known up front only sometimes:

  in-repo task tree   the manifest is on disk, so a task that produced no trial at all
                      can be named and reported as ungraded
  registry dataset    the task list exists only after the backend has downloaded it,
                      so records are whatever trials are found

The distinction matters for honesty, not convenience: with a known list, silence about
a task is itself a finding.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

TRIAL_RESULT_NAME = "result.json"
JOB_LOCK_NAME = "lock.json"
VERIFIER_DIR = Path("verifier")
REWARD_JSON = VERIFIER_DIR / "reward.json"
REWARD_TXT = VERIFIER_DIR / "reward.txt"
# A machine-readable test report, in the Common Test Report Format. Not every verifier
# writes one, but the ones that do put the pass/fail counts there and nowhere else --
# which is the difference between "scored 0" and "scored 0 with 2 of 3 tests passing".
CTRF_JSON = VERIFIER_DIR / "ctrf.json"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _fraction(value):
    """A reward as a float in [0, 1], or None when it is not a score at all.

    Rewards arrive in more than one shape. Some verifiers write a binary 0/1 in
    reward.json; terminal-bench writes a float in reward.txt and no reward.json at
    all; a partial
    grader can write anything in between. `reward in (0, 1)` accepted only the binary
    ones -- and by accident the floats 0.0 and 1.0, since Python compares them equal --
    so a task scored 0.5 was reported as *ungraded*, which reads as infrastructure
    trouble rather than as a partial pass.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # -1 is the crash sentinel a task's own verifier writes when it could not run. It is
    # not a score and must not be averaged into one.
    if number < 0.0 or number > 1.0:
        return None
    return number


def _test_counts(trial_dir: Path) -> dict:
    """Passed/total from a CTRF report, when the verifier wrote one.

    Recorded alongside the reward because a binary 0 hides how close a run came, and
    that is the difference between a model that did nothing and one that failed a single
    assertion.
    """
    payload = _read_json(trial_dir / CTRF_JSON)
    summary = ((payload or {}).get("results") or {}).get("summary") or {}
    total = summary.get("tests")
    passed = summary.get("passed")
    if not isinstance(total, int) or not isinstance(passed, int) or total <= 0:
        return {}
    return {"tests_passed": passed, "tests_total": total, "partial": passed / total}


def _latest_job_dir(output_dir: Path) -> Path:
    """The backend job directory to collect from when a jobs-dir has prior runs.

    Harbor writes each invocation under a timestamped child of --jobs-dir. The
    persistent scratch directory keeps older children, so scanning the whole jobs-dir can
    mix this run with a previous run of the same task id. If the supplied directory is
    already a job directory, keep it; otherwise choose the newest immediate child that
    looks like a job directory.
    """
    own_result = _read_json(output_dir / TRIAL_RESULT_NAME)
    if isinstance(own_result, dict) and own_result.get("stats"):
        return output_dir

    candidates = []
    for child in output_dir.iterdir() if output_dir.is_dir() else []:
        if not child.is_dir():
            continue
        payload = _read_json(child / TRIAL_RESULT_NAME)
        if isinstance(payload, dict) and payload.get("stats"):
            candidates.append(child)
    if not candidates:
        return output_dir
    return sorted(candidates, key=lambda path: path.name)[-1]


def index_trials(output_dir: Path) -> dict:
    """Map task id -> trial directory.

    Matches on the location of the backend's own output rather than guessing its
    directory naming, which carries per-attempt suffixes. A trial that crashed before
    writing result.json may still have a verifier directory, so that is a second pass
    rather than a fallback for the same path.

    The job directory also contains a run-level result.json with no task_name.
    Require a task name or verifier directory so the run-level summary is not
    mistaken for an ungraded trial.
    """
    output_dir = _latest_job_dir(output_dir)
    found: dict = {}
    if not output_dir.is_dir():
        return found
    for candidate in sorted(output_dir.rglob(TRIAL_RESULT_NAME)):
        trial_dir = candidate.parent
        payload = _read_json(candidate)
        if payload is None:
            continue
        name = payload.get("task_name") or payload.get("trial_name")
        if not name:
            # The run-level summary, or a file that is not a trial result at all.
            if not (trial_dir / REWARD_JSON.parent).is_dir():
                continue
            name = trial_dir.name
        # task_name can carry an owner prefix ("swe-bench/django__django-13925",
        # "terminal-bench/fix-git"); the dataset's own ids do not.
        found.setdefault(str(name).rsplit("/", 1)[-1], trial_dir)
    for candidate in sorted(output_dir.rglob(str(REWARD_JSON))):
        found.setdefault(candidate.parent.parent.name.rsplit("/", 1)[-1], candidate.parent.parent)
    return found


def dataset_order(output_dir: Path) -> list:
    """The task ids of a finished evaluation, in the order the dataset presents them.

    Read from the backend's own lock file, which it writes when it plans the run and
    before it executes anything. That ordering is what `--n-tasks N` truncates, so the
    first N entries here are exactly the samples an N-limited run would evaluate --
    verified against a 500-task lock and the 4- and 16-task locks of runs over the same
    dataset, both of which were exact prefixes of it.

    Nothing else in a finished tree carries the order. Directory names do not, and
    completion times only approximate it because trials run concurrently. Returns an
    empty list when the lock is absent or unreadable, because a caller drawing the
    validation boundary must be able to tell "unknown" from "empty" -- guessing the
    boundary would score a submission on held-out samples.
    """
    payload = _read_json(_latest_job_dir(output_dir) / JOB_LOCK_NAME)
    if not isinstance(payload, dict):
        return []
    trials = payload.get("trials")
    if isinstance(trials, dict):
        trials = list(trials.values())
    if not isinstance(trials, list):
        return []
    order = []
    for entry in trials:
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            task = entry.get("task")
            name = (task.get("name") if isinstance(task, dict) else None) or entry.get("trial_name") or ""
        else:
            continue
        # "swe-bench/astropy__astropy-12907" -> "astropy__astropy-12907", matching the
        # task ids collect() reports.
        name = str(name).split("/")[-1].strip()
        if name:
            order.append(name)
    return order


def count_started_trials(output_dir: Path) -> int:
    """How many trials the backend created a directory for, finished or not.

    index_trials only sees a trial once it has written result.json or a verifier
    directory, so on a tree that was stopped mid-run the trials still executing are
    invisible to it. They exist on disk, though, and the difference between the two
    counts is exactly "how many were in flight when this stopped" -- which is the
    number that keeps a cancelled evaluation's `ungraded: 0` from reading as "nothing
    was left unfinished".
    """
    job_dir = _latest_job_dir(output_dir)
    if not job_dir.is_dir():
        return 0
    return sum(1 for child in job_dir.iterdir() if child.is_dir())


def read_trial(trial_dir: Path, *, task_id: str, language: Optional[str] = None) -> dict:
    """One record from one trial directory.

    Three outcomes are kept apart, and the third is why this is not a one-liner: a score,
    no score, and a verifier that could not reach a verdict -- an image that would not
    build, a verifier that hit its own timeout, a trial the backend never reached.
    Folding that third kind into 0 lets infrastructure trouble read as a weaker model,
    which is the most expensive kind of wrong number because nothing about it looks wrong.

    The reward is looked for in three places because different verifiers write different
    ones, and only ever one of them:

      verifier/reward.json   a binary reward plus, for some verifiers, f2p/p2p counts
      verifier/reward.txt    terminal-bench: a bare number, and no reward.json at all
      result.json            the backend's own copy, under verifier_result.rewards

    Missing any of the three cost a real run: terminal-bench graded fine but was
    reported through the result.json fallback only, and would have been ungraded if that
    had changed shape.
    """
    record = {"task_id": task_id, "language": language, "trial_dir": str(trial_dir)}

    detail = _read_json(trial_dir / REWARD_JSON) or {}
    reward = _fraction(detail.get("reward"))

    if reward is None:
        # A bare number in reward.txt. Also where the -1 crash sentinel appears, which
        # _fraction rejects, so a crashed verifier does not become a zero here.
        raw = _read_text(trial_dir / REWARD_TXT)
        if raw is not None:
            reward = _fraction(raw.strip())

    if reward is None:
        result = _read_json(trial_dir / TRIAL_RESULT_NAME) or {}
        rewards = ((result.get("verifier_result") or {}) or {}).get("rewards")
        if isinstance(rewards, dict):
            detail = detail or dict(rewards)
            reward = _fraction(rewards.get("reward"))

    if reward is not None:
        record.update(
            {
                "graded": True,
                "reward": reward,
                # A pass fraction, from whichever source has one. reward.json's own
                # `partial` wins; otherwise the test counts, which most verifiers report
                # even when the reward is binary.
                "partial": _fraction(detail.get("partial")),
                "f2p_passed": detail.get("f2p_passed"),
                "f2p_total": detail.get("f2p_total"),
                "p2p_passed": detail.get("p2p_passed"),
                "p2p_total": detail.get("p2p_total"),
                "apply_failed": bool(detail.get("apply_failed")),
                "reason": None,
            }
        )
        counts = _test_counts(trial_dir)
        if counts:
            record["tests_passed"] = counts["tests_passed"]
            record["tests_total"] = counts["tests_total"]
            if record.get("partial") is None:
                record["partial"] = counts["partial"]
        return record

    sentinel = _read_text(trial_dir / REWARD_TXT)
    if sentinel and sentinel.strip().startswith("-1"):
        reason = "the task's verifier wrote its -1 crash sentinel, which says nothing about the model"
    elif detail:
        reason = f"reward present but not a score in [0,1]: {detail.get('reward')!r}"
    else:
        reason = "no reward was recorded for this trial"
    record.update({"graded": False, "reward": None, "partial": None, "reason": reason})
    return record


# Agent artefacts inside a trial, in the order they are preferred: the structured
# trajectory first, then the plain-text transcript some harnesses write instead.
AGENT_TRAJECTORY_NAMES = (
    "mini-swe-agent.trajectory.json",
    "trajectory.json",
    "mini-swe-agent.txt",
)

# How much of a failing trial's exception text belongs in a one-line-per-sample index.
EXCEPTION_MESSAGE_CHARS = 2000

# Diagnostic threshold for repeated identical (action, observation) pairs.
# Twenty repetitions indicate a likely no-progress loop; this is a heuristic,
# not a percentile or a benchmark scoring rule.
NO_PROGRESS_REPEATS = 20


def _action_steps(messages: list) -> list:
    """(action, observation) for each step the agent took, in order.

    One extraction rather than two, because two would drift: the repeat count and the
    tail an agent reads to see *what* repeated have to agree about what an action is,
    or the count reports four hundred repeats while the excerpt beside it shows steps
    that do not look alike.
    """
    steps: list = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        actions = [
            call.get("function", {}).get("arguments")
            for call in (calls if isinstance(calls, list) else [])
            if isinstance(call, dict)
        ]
        # A harness that puts the command in the text instead of a tool call still has a
        # comparable action: the message body it sent.
        if not actions:
            actions = [message.get("content")]
        observation = None
        for follower in messages[index + 1:index + 3]:
            if isinstance(follower, dict) and follower.get("role") in ("tool", "user"):
                observation = follower.get("content")
                break
        for action in actions:
            steps.append((action, observation))
    return steps


def _no_progress_repeats(messages: list) -> int:
    """The longest run of consecutive steps that repeated an action *and* got the same
    output back.

    This is the one thing a task's difficulty cannot excuse. A hard sample may legitimately
    need hundreds of steps, and repeating a command is often the right move -- polling a
    build every minute, re-running a test after an edit, listing a directory again after
    writing to it. All of those change the output, so none of them count here. What counts
    is the same command returning the same bytes over and over: no information is arriving,
    and the next step is conditioned on exactly what the last one was, so the loop cannot
    end on its own.

    Hence no step ceiling: a ceiling cuts the hard samples, which is where the difference
    between two models lives. This number separates "spent a lot of steps" from "stopped
    getting anywhere" without touching either.
    """
    steps = _action_steps(messages)

    longest = current = 1 if steps else 0
    for previous, following in zip(steps, steps[1:]):
        current = current + 1 if previous == following else 1
        longest = max(longest, current)
    return longest


def agent_trajectory_path(trial_dir: Path) -> Optional[Path]:
    """The trial's agent transcript, whichever shape its harness wrote.

    One list, in one place: the grading service reads the same artefact to show an
    agent what its model did, and two copies of these names would drift apart.
    """
    for name in AGENT_TRAJECTORY_NAMES:
        candidate = trial_dir / "agent" / name
        if candidate.is_file():
            return candidate
    return None


# How many steps a transcript tail shows. Enough that a repeat is unmistakable rather than
# a coincidence: three identical steps can be a poll waiting on a build, a dozen cannot.
TRAJECTORY_TAIL_STEPS = 12


def trajectory_tail(trial_dir: Path, budget_chars: int) -> Optional[str]:
    """The last few steps of this trial's transcript, rendered for reading.

    Counted in steps rather than characters, because the two disagree exactly where the
    answer matters. A trial that has stopped getting anywhere produces enormous
    observations, so a fixed character budget taken from the end of the file lands inside
    a single tool result and contains no complete step at all -- measured over one
    evaluation's twenty-eight transcripts, the two worst loopers (468 and 323 repeats)
    were precisely the two whose character tail held zero steps, while the healthy short
    ones held nine. The budget and the pathology were anticorrelated.

    Clipping each field instead keeps the commands, which is where a repeat is visible,
    and drops the output bulk, which is the part a repeat makes redundant.

    None when the transcript is not a shape this can read, so a caller can fall back to
    the raw text rather than be told a readable trial has nothing in it.
    """
    path = agent_trajectory_path(trial_dir)
    if path is None or path.suffix != ".json":
        return None
    payload = _read_json(path)
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list) or not messages:
        return None
    steps = _action_steps(messages)
    if not steps:
        return None
    tail = steps[-TRAJECTORY_TAIL_STEPS:]
    # Each step gets an equal share of the same budget the raw excerpt would have used, so
    # showing more steps costs no more to send. The floor keeps a command readable when a
    # caller has configured the budget down.
    field_chars = max(200, budget_chars // max(1, len(tail) * 2))

    def clip(value) -> str:
        text = value if isinstance(value, str) else ("" if value is None else str(value))
        text = text.strip()
        if len(text) > field_chars:
            return text[:field_chars] + f" ... [+{len(text) - field_chars} chars]"
        return text

    first = len(steps) - len(tail) + 1
    lines = [f"last {len(tail)} of {len(steps)} steps, each field clipped to {field_chars} chars"]
    for offset, (action, observation) in enumerate(tail, start=first):
        lines.append(f"--- step {offset}")
        lines.append(f"action: {clip(action)}")
        lines.append(f"output: {clip(observation)}")
    return "\n".join(lines)


def agent_stats(trial_dir: Path) -> dict:
    """How the agent spent this trial, beside whether it was scored.

    A reward says a task was not solved; it does not say whether the model gave up
    after nineteen turns or looped for a thousand. Those are different failures --
    the second points at the turn distribution of the training data rather than at
    the score -- and the counts are the only place that distinction lives.

    Best-effort by design: a field is omitted rather than guessed when the harness
    that wrote this trial does not record it.
    """
    stats: dict = {}

    path = agent_trajectory_path(trial_dir)
    if path is not None and path.suffix == ".json":
        payload = _read_json(path)
        payload = payload if isinstance(payload, dict) else {}
        info = payload.get("info")
        info = info if isinstance(info, dict) else {}
        model_stats = info.get("model_stats")
        model_stats = model_stats if isinstance(model_stats, dict) else {}
        if isinstance(model_stats.get("api_calls"), int):
            stats["api_calls"] = model_stats["api_calls"]
        if isinstance(payload.get("messages"), list):
            stats["turns"] = len(payload["messages"])
            # Recorded for every trial, not only the flagged ones: the distribution is what
            # makes a threshold defensible later, and a healthy trial's value is 1 or 2.
            stats["no_progress_repeats"] = _no_progress_repeats(payload["messages"])
        # Why the loop ended, in the harness's own words: "Submitted" against a
        # step or cost limit is the difference between a decision and a timeout.
        if isinstance(info.get("exit_status"), str) and info["exit_status"]:
            stats["exit_status"] = info["exit_status"]

    result = _read_json(trial_dir / TRIAL_RESULT_NAME)
    result = result if isinstance(result, dict) else {}
    agent_result = result.get("agent_result")
    agent_result = agent_result if isinstance(agent_result, dict) else {}
    for key in ("n_input_tokens", "n_output_tokens"):
        if isinstance(agent_result.get(key), int):
            stats[key] = agent_result[key]
    exception_info = result.get("exception_info")
    message = (exception_info or {}).get("exception_message") if isinstance(exception_info, dict) else None
    # The class, not just the text. It is the harness's own verdict on how the trial
    # ended, so counting it needs no interpretation, and it is the crispest loop signal
    # there is: a trial repeating one command until its turns run out ends as a timeout.
    failure = (exception_info or {}).get("exception_type") if isinstance(exception_info, dict) else None
    if isinstance(failure, str) and failure:
        stats["exception_type"] = failure
    if isinstance(message, str) and message:
        # Kept short on purpose. These rows are an index over an evaluation -- one line
        # per sample -- and a nine-kilobyte traceback in an index row buries the counts
        # beside it. The whole message stays in the trial's own result.json, which the
        # grading service copies where the run can read it, so nothing is lost by
        # summarising here.
        head = message.strip()
        if len(head) > EXCEPTION_MESSAGE_CHARS:
            head = head[:EXCEPTION_MESSAGE_CHARS] + "... [full text in the trial's result.json]"
        stats["exception_message"] = head
    return stats


# What of a trial an agent may read, and what it may never.
#
# A whitelist, because the withheld half is the grading answer: verifier/report.json
# carries tests_status.FAIL_TO_PASS and PASS_TO_PASS -- the names of the hidden tests --
# and verifier/test-stdout.txt names them again in the test runner's output. An agent
# holding those does not need to fix the bug; it needs to make those test names pass.
#
# reward.txt is two bytes and only repeats the score the agent is told anyway, so it
# stays. Everything outside verifier/ is the agent's own model's work or the harness's
# record of running it, which is the material a run is supposed to learn from.
DISCLOSABLE_TRIAL_ENTRIES = (
    "agent",
    "artifacts",
    "config.json",
    "exception.txt",
    "lock.json",
    TRIAL_RESULT_NAME,
    "trial.log",
)
DISCLOSABLE_VERIFIER_FILES = ("reward.txt",)


def copy_disclosable_trial(trial_dir: Path, dest: Path) -> list:
    """Copy the parts of one trial an agent may read, and nothing else.

    Walked entry by entry from a whitelist rather than copied whole and then pruned.
    The order matters: a copy-then-delete leaves the hidden test names readable under
    the agent's own path for as long as the delete takes, and a crash in between
    leaves them there permanently. Anything this file does not name is not copied, so
    a future harness adding a new artefact defaults to withheld rather than disclosed.

    Returns the entry names actually copied, for the caller to log.
    """
    copied = []
    dest.mkdir(parents=True, exist_ok=True)
    for name in DISCLOSABLE_TRIAL_ENTRIES:
        source = trial_dir / name
        if source.is_dir():
            shutil.copytree(source, dest / name, dirs_exist_ok=True)
            copied.append(name + "/")
        elif source.is_file():
            shutil.copy2(source, dest / name)
            copied.append(name)
    for name in DISCLOSABLE_VERIFIER_FILES:
        source = trial_dir / VERIFIER_DIR / name
        if source.is_file():
            (dest / VERIFIER_DIR).mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest / VERIFIER_DIR / name)
            copied.append(str(VERIFIER_DIR / name))
    return copied


def collect(output_dir: Path, *, expected: Optional[list] = None) -> list:
    """Records for a finished run.

    `expected` is the task set when it is known in advance: objects with `.task_id`
    (and optionally `.language`), or plain task-id strings when that is all the caller
    has -- a registry dataset selected by name resolves to ids without task objects.
    Given one, a task with no trial directory is reported as ungraded with a reason
    rather than quietly dropped -- the backend having produced nothing for a task is
    exactly the kind of failure that otherwise shrinks the denominator and flatters the
    score.
    """
    found = index_trials(output_dir)
    if expected is None:
        return [read_trial(trial, task_id=task_id) for task_id, trial in sorted(found.items())]

    records = []
    for task in expected:
        task_id = task if isinstance(task, str) else task.task_id
        trial = found.get(task_id)
        language = getattr(task, "language", None)
        if trial is None:
            records.append(
                {
                    "task_id": task_id,
                    "language": language,
                    "graded": False,
                    "reward": None,
                    "partial": None,
                    "reason": "the backend produced no trial directory for this task",
                }
            )
            continue
        records.append(read_trial(trial, task_id=task_id, language=language))
    return records


def _record_failure(record: dict) -> Optional[str]:
    """One record's harness-level failure class, or None if it did not fail that way.

    Same two sources as _record_repeats: the grading service attaches agent_stats to the
    rows it shows a run, while a record straight out of collect() carries only its trial
    directory. A trial that simply scored 0 contributes nothing -- a graded failure is
    not a harness error, and conflating them would make every hard sample look broken.
    """
    stats = record.get("agent_stats")
    if isinstance(stats, dict) and isinstance(stats.get("exception_type"), str):
        return stats["exception_type"] or None
    trial_dir = record.get("trial_dir")
    if not trial_dir:
        return None
    value = agent_stats(Path(trial_dir)).get("exception_type")
    return value if isinstance(value, str) and value else None


def _record_repeats(record: dict) -> Optional[int]:
    """One record's no-progress run length, from wherever it can be had.

    The grading service already attaches agent_stats to the rows it shows a run, so that is
    used when present; a record straight out of collect() carries only its trial directory,
    so the trajectory is read. Best-effort: a harness that wrote no trajectory contributes
    nothing rather than a zero, which would read as "healthy".
    """
    stats = record.get("agent_stats")
    if isinstance(stats, dict) and isinstance(stats.get("no_progress_repeats"), int):
        return stats["no_progress_repeats"]
    trial_dir = record.get("trial_dir")
    if not trial_dir:
        return None
    value = agent_stats(Path(trial_dir)).get("no_progress_repeats")
    return value if isinstance(value, int) else None


def score_records(records: list) -> dict:
    """The score half of the metrics contract, without the run's own metadata.

    Split out so an evaluation stopped part-way -- only some of its trials on disk --
    is counted by exactly the same rules as a complete one and cannot drift from them.
    The grading service reads a cancelled evaluation this way.

    accuracy is over *graded* tasks only. Ungraded tasks are counted separately and
    never as zeros, which is what makes a partial count honest rather than flattering:
    trials still in flight when the run stopped are missing, not failed.
    """
    graded = [r for r in records if r.get("graded")]
    solved = [r for r in graded if r.get("reward") == 1]
    rewards = [r["reward"] for r in graded if isinstance(r.get("reward"), (int, float))]
    partials = [r["partial"] for r in graded if isinstance(r.get("partial"), float)]
    # Report no-progress loops beside scores to explain repeated actions and wall
    # time. Aggregate them here so cancelled evaluations retain these diagnostics.
    repeats = [n for n in (_record_repeats(r) for r in records) if isinstance(n, int)]
    stuck = [n for n in repeats if n >= NO_PROGRESS_REPEATS]
    # How the failed trials failed, by the harness's own classification. Beside
    # no_progress_trials rather than instead of it: the repeat count says a trial stopped
    # getting anywhere, this says what finally ended it, and a queue that is mostly
    # timeouts is a different problem from one that is mostly crashes.
    failures: dict = {}
    for record in records:
        kind = _record_failure(record)
        if kind:
            failures[kind] = failures.get(kind, 0) + 1
    return {
        "accuracy": (len(solved) / len(graded)) if graded else 0.0,
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "solved": len(solved),
        "graded": len(graded),
        "attempted": len(records),
        "ungraded": len(records) - len(graded),
        "partial_mean": (sum(partials) / len(partials)) if partials else None,
        "no_progress_trials": len(stuck) if repeats else None,
        "no_progress_threshold": NO_PROGRESS_REPEATS if repeats else None,
        "no_progress_max": max(repeats) if repeats else None,
        "trial_errors": dict(sorted(failures.items(), key=lambda kv: kv[1], reverse=True)) or None,
    }


def summarise(records: list, *, benchmark: str, agent: str, model: str, backend: str,
              backend_version: str, environment: str) -> dict:
    """The metrics contract every benchmark in this repo reports.

    accuracy is over *graded* tasks only, and the per-sample file lists exactly those,
    so the grading service's reconstruction check compares like with like. Ungraded
    tasks are counted separately and never as zeros.

    accuracy counts a full pass, because that is what these benchmarks are compared on
    upstream. mean_reward is reported next to it because a verifier may award a fraction,
    and accuracy alone would throw that away -- a model scoring 0.9 on every task would
    read as 0.0, which is true to the leaderboard and useless for telling whether
    training helped.
    """
    return {
        **score_records(records),
        "benchmark": benchmark,
        "runner": f"{backend} {backend_version}",
        "backend": backend,
        "backend_version": backend_version,
        "environment_type": environment,
        "agent": agent,
        "model": model,
        "records": records,
    }


def write_outputs(result: dict, *, metrics_path: Optional[Path], work_dir: Path,
                  dataset_total: Optional[int] = None) -> None:
    """metrics.json, its per-sample companion, and the raw records.

    The per-sample file sits beside the metrics file because that is where the grading
    service looks for it, and it lists one entry per *graded* task so a validation
    split can be reconstructed and checked.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "records.json").write_text(
        json.dumps(result["records"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = {k: v for k, v in result.items() if k != "records"}
    if dataset_total is not None:
        summary["dataset_total"] = dataset_total

    per_sample = [
        {
            "id": r["task_id"],
            "score": r.get("reward"),
            "partial": r.get("partial"),
        }
        for r in result["records"]
        if r.get("graded")
    ]

    for target in filter(None, (metrics_path, work_dir / "metrics.json")):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        companion = target.with_name(target.stem + ".per_sample.json")
        companion.write_text(json.dumps(per_sample, indent=2, ensure_ascii=False), encoding="utf-8")
