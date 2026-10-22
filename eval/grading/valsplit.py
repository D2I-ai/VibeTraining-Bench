"""Validation / held-out policy, and what an agent is allowed to learn.

Two decisions live here, and both belong to the service rather than to the agent:

**Which samples an agent may be scored on.** An agent submission is always
evaluated on the validation split and nothing else. The split is decided by
VTB_VAL_SOURCE at startup; the agent cannot name a benchmark or subset. It may
request a smaller positive sample count than the service-selected validation
upper bound to trade off signal quality for evaluation time, but it cannot exceed
that upper bound or reach the held-out samples the final score is partly derived
from.

**What comes back.** By default an agent learns one number: its score on the
validation split. Richer disclosure (per-sample outcomes, the model's own
outputs, problem statements, reference answers) exists as an interface and is off unless an operator
turns it on, because each level is also a contamination channel -- reference
answers most obviously, but per-sample outcomes are enough to fit the split by
elimination given enough submissions.

The official evaluation is separate: it runs the whole benchmark, and the
validation / held-out figures are derived afterwards by partitioning its log.
A file: split can also declare a separate test benchmark, which is where the
official final score is measured instead -- the benchmark's own tasks are then
all validation, with nothing held out of them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Optional

# ------------------------------------------------------------------ val source

DEFAULT_VAL_SOURCE = "split:30%"

_SPLIT_RATIO_RE = re.compile(r"^split:(\d+(?:\.\d+)?)%$")
_SPLIT_COUNT_RE = re.compile(r"^split:(\d+)$")
_TASK_RE = re.compile(r"^task:([A-Za-z0-9_.-]+)$")
_HF_RE = re.compile(r"^hf:(.+)$")
_FILE_RE = re.compile(r"^file:(.+)$")
_JSON_PREFIX = "json:"
TAU3_CANONICAL_TASK_PREFIX = "tau3-bench__tau3-"


def _load_json_task_split(raw_path: str) -> tuple[Path, tuple[str, ...], tuple[str, ...], str]:
    """Read an explicit Tau3 validation/test task split without guessing membership."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ValueError("VTB_VAL_SOURCE=json:<path> needs an absolute JSON file path")
    try:
        content = path.read_bytes()
        payload = json.loads(content.decode("utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read JSON validation split {path}: {exc}") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"JSON validation split {path} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"val", "test"}:
        raise ValueError(f"JSON validation split {path} must contain exactly 'val' and 'test' arrays")

    groups: dict[str, tuple[str, ...]] = {}
    for name in ("val", "test"):
        values = payload[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"JSON validation split {path} field {name!r} must be a non-empty array")
        ids = []
        for value in values:
            if not isinstance(value, str) or value != value.strip() or not value.startswith(TAU3_CANONICAL_TASK_PREFIX):
                raise ValueError(f"JSON validation split {path} field {name!r} contains invalid Tau3 task id: {value!r}")
            ids.append(value)
        if len(ids) != len(set(ids)):
            raise ValueError(f"JSON validation split {path} field {name!r} contains duplicate task ids")
        groups[name] = tuple(ids)
    if set(groups["val"]) & set(groups["test"]):
        raise ValueError(f"JSON validation split {path} has task ids in both val and test")
    return path.resolve(), groups["val"], groups["test"], hashlib.sha256(content).hexdigest()


class ValSource:
    """Parsed VTB_VAL_SOURCE."""

    __slots__ = (
        "kind", "ratio", "count", "task", "raw", "task_ids", "path", "test_benchmark",
        "val_ids", "test_ids", "json_sha256"
    )

    def __init__(
        self,
        kind: str,
        raw: str,
        ratio: Optional[float] = None,
        count: Optional[int] = None,
        task: Optional[str] = None,
        task_ids: Optional[list] = None,
        path: Optional[str] = None,
        test_benchmark: Optional[str] = None,
        val_ids: tuple[str, ...] = (),
        test_ids: tuple[str, ...] = (),
        json_sha256: Optional[str] = None,
    ) -> None:
        self.kind = kind  # "split" | "task" | "hf" | "none" | "file" | "json"
        self.raw = raw
        self.ratio = ratio
        self.count = count
        self.task = task
        # file: sources only. The explicit id list IS the split -- no prefix
        # arithmetic applies -- and the benchmark it belongs to is the directory
        # the file sits in (benchmarks/<name>/split.json), which is how the
        # service knows the list is about this benchmark and not another.
        self.task_ids = task_ids
        self.path = path
        # file: sources only. The same JSON may route official final scoring to
        # a separate test benchmark ("test": {"benchmark": "<name>"}); the
        # server reads this when a submission asks for a final score.
        self.test_benchmark = test_benchmark
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.json_sha256 = json_sha256

    def to_json(self) -> dict:
        out = {
            "raw": self.raw,
            "kind": self.kind,
            "ratio": self.ratio,
            "count": self.count,
            "task": self.task,
            "task_ids": self.task_ids,
            "path": self.path,
            "test_benchmark": self.test_benchmark,
        }
        if self.kind == "json":
            out.update({"sha256": self.json_sha256, "val_count": len(self.val_ids), "test_count": len(self.test_ids)})
        return out

    def operator_json(self) -> dict:
        """Operator-only split membership; never expose it through public health."""
        out = self.to_json()
        if self.kind == "json":
            out.update({"val": list(self.val_ids), "test": list(self.test_ids)})
        return out

    def describe(self) -> str:
        if self.kind == "split" and self.ratio is not None:
            return f"first {self.ratio:.0%} of the target benchmark"
        if self.kind == "split":
            return f"first {self.count} samples of the target benchmark"
        if self.kind == "task":
            return f"the whole {self.task} benchmark"
        if self.kind == "hf":
            return f"HuggingFace dataset {self.task}"
        if self.kind == "file":
            return f"the {len(self.task_ids)} samples listed in {self.path}"
        if self.kind == "json":
            return f"{len(self.val_ids)} explicit validation and {len(self.test_ids)} explicit test tasks"
        return "no validation channel"


def parse_val_source(raw: Optional[str] = None) -> ValSource:
    """Parse a validation-source spec, rejecting anything ambiguous.

    A shared service scores every concurrent run, so a misconfiguration here
    would silently affect all of them. Fail at startup instead.

        split:30%       first 30% of the target benchmark   (default)
        split:1         first sample of the target benchmark
        split:395       first 395 samples of it
        task:aime__aime another benchmark in its entirety
        file:<path>     an explicit task-id list, read from JSON at startup.
                        Accepted shapes: {"validation": [ids]},
                        {"validation": {"task_ids": [ids]}}, or a bare [ids].
                        The list is the whole split -- it need not be a prefix,
                        and it may cover the entire benchmark, which is the
                        point: a benchmark whose test set lives elsewhere
                        validates on all of its own tasks. The same file's
                        optional "test": {"benchmark": "<name>"} routes official
                        final scoring to that benchmark.
        hf:<dataset>    reserved; recognised but not implemented
        none            agents get no validation channel at all
    """
    raw = (raw if raw is not None else os.environ.get("VTB_VAL_SOURCE") or DEFAULT_VAL_SOURCE).strip()

    if raw == "none":
        return ValSource("none", raw)

    m = _SPLIT_RATIO_RE.match(raw)
    if m:
        ratio = float(m.group(1)) / 100.0
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"validation ratio must be between 0% and 100%, got {raw!r}")
        return ValSource("split", raw, ratio=ratio)

    m = _SPLIT_COUNT_RE.match(raw)
    if m:
        count = int(m.group(1))
        if count <= 0:
            raise ValueError(f"validation count must be positive, got {raw!r}")
        return ValSource("split", raw, count=count)

    m = _TASK_RE.match(raw)
    if m:
        return ValSource("task", raw, task=m.group(1))

    m = _FILE_RE.match(raw)
    if m:
        return _parse_file_source(m.group(1), raw)

    if raw.startswith(_JSON_PREFIX):
        path, val_ids, test_ids, digest = _load_json_task_split(raw[len(_JSON_PREFIX):])
        return ValSource("json", raw, path=str(path), val_ids=val_ids, test_ids=test_ids, json_sha256=digest)

    m = _HF_RE.match(raw)
    if m:
        # Recognised so the spelling is settled, but refused rather than
        # half-served: loading and scoring an arbitrary dataset needs a scorer
        # this service does not have.
        raise ValueError(
            f"VTB_VAL_SOURCE={raw!r} is a reserved form that is not implemented yet; "
            "use split:<pct|count>, task:<name>, file:<path>, json:<absolute-path>, or none"
        )

    raise ValueError(
        f"cannot parse VTB_VAL_SOURCE={raw!r}; expected split:30%, split:<count>, task:<name>, "
        "file:<path>, json:<absolute-path>, or none"
    )


def _parse_file_source(path_str: str, raw: str) -> ValSource:
    """Read an explicit validation id list, rejecting anything ambiguous.

    A shared service scores every concurrent run, so a half-readable split file
    would silently affect all of them; like every other parse failure here this
    lands at startup rather than mid-run. The file's parent directory names the
    benchmark the split belongs to (benchmarks/<name>/split.json), which is the
    convention the server checks the list against.
    """
    path = Path(path_str.strip())
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"VTB_VAL_SOURCE={raw!r}: cannot read {path}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"VTB_VAL_SOURCE={raw!r}: {path} is not valid JSON: {exc}") from exc

    ids = data
    if isinstance(data, dict):
        ids = data.get("validation")
        if isinstance(ids, dict):
            ids = ids.get("task_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(
            f"VTB_VAL_SOURCE={raw!r}: {path} declares no validation task ids; expected "
            'a "validation" key holding a list of task ids (or a bare list)'
        )
    cleaned = [str(i).strip() for i in ids]
    if not all(cleaned):
        raise ValueError(f"VTB_VAL_SOURCE={raw!r}: {path} lists an empty task id")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"VTB_VAL_SOURCE={raw!r}: {path} lists duplicate task ids")

    declared = data.get("test") if isinstance(data, dict) else None
    if isinstance(declared, dict):
        declared = declared.get("benchmark")
    if declared is None:
        test_benchmark = None
    elif isinstance(declared, str) and declared.strip():
        test_benchmark = declared.strip()
    else:
        raise ValueError(
            f"VTB_VAL_SOURCE={raw!r}: {path} declares a test benchmark that is not a "
            'name; expected {"test": {"benchmark": "<name>"}}'
        )
    return ValSource(
        "file", raw, task_ids=cleaned, path=str(path), test_benchmark=test_benchmark
    )


def resolve_val_n(source: ValSource, total: Optional[int]) -> Optional[int]:
    """How many samples the validation split covers, or None if undecidable.

    None is a refusal, not a default: without the dataset size a ratio cannot be
    turned into a boundary, and guessing one would risk scoring an agent on
    samples the final score is derived from -- contamination that cannot be
    undone afterwards.

    A file: source is refused here on purpose: this function answers a question
    about prefix boundaries, and an explicit list is not a prefix. Callers that
    need the list itself read source.task_ids; resolve_val_n returning None for
    it is what makes the official partition of a benchmark with an external
    test set quietly skip itself instead of carving a bogus held-out half.
    """
    if source.kind != "split":
        return None
    if source.count is not None:
        return min(source.count, total) if total else source.count
    if not total or source.ratio is None:
        return None
    return max(1, int(total * source.ratio))


# ------------------------------------------------------------------ disclosure

# Ordered from least to most revealing. Each level is also a contamination
# channel, which is why the default is the first one only.
DISCLOSURE_LEVELS = ("score", "per_sample_outcomes", "problems", "outputs", "targets")
DEFAULT_DISCLOSURE = "score"


def parse_disclosure(raw: Optional[str] = None) -> frozenset:
    raw = (raw if raw is not None else os.environ.get("VTB_AGENT_DISCLOSURE") or DEFAULT_DISCLOSURE).strip()
    wanted = {p.strip() for p in raw.split(",") if p.strip()}
    unknown = wanted - set(DISCLOSURE_LEVELS)
    if unknown:
        raise ValueError(
            f"unknown VTB_AGENT_DISCLOSURE entries {sorted(unknown)}; "
            f"choose from {', '.join(DISCLOSURE_LEVELS)}"
        )
    if "targets" in wanted:
        # evaluate.py blanks sample.target before writing its log, on purpose, and
        # that file is the immutable evaluation baseline. Serving reference
        # answers would mean loading the dataset independently, which is not
        # built. Refuse loudly rather than return empty strings that look like
        # answers.
        raise ValueError(
            "VTB_AGENT_DISCLOSURE=targets is not implemented: evaluate.py strips reference "
            "answers from its log by design, so the service has no source for them"
        )
    if not wanted:
        raise ValueError("VTB_AGENT_DISCLOSURE cannot be empty; use at least 'score'")
    return frozenset(wanted)


# ------------------------------------------------- partitioning a finished run


class SampleResults:
    """Per-sample outcomes of one evaluation, normalised across harnesses.

    Splitting a score into validation and held-out halves needs three things: the
    order samples were drawn in, which of them the model got right, and ideally the
    size of the whole benchmark. Nothing about that is specific to inspect-ai, but
    the first implementation read inspect's log format directly, which quietly made
    held-out reporting a privilege of the five inspect-based tasks -- the two
    judge-driven ones got no validation figures at all and said nothing about it.

    So the shape is declared here and harnesses can satisfy it however they like.
    That matters for what comes next: an SWE-style benchmark has its own runner and
    will never produce an inspect log.
    """

    __slots__ = ("ordered_ids", "correct_by_id", "dataset_total", "source", "score_mode")

    def __init__(
        self,
        ordered_ids: list,
        correct_by_id: dict,
        dataset_total: Optional[int] = None,
        source: str = "unknown",
        score_mode: str = "binary",
    ) -> None:
        self.ordered_ids = ordered_ids
        self.correct_by_id = correct_by_id
        self.dataset_total = dataset_total
        self.source = source
        self.score_mode = score_mode


def per_sample_path(out_json: Path) -> Path:
    """Where a harness may declare its per-sample outcomes.

    Derived from the metrics path the harness was handed, so it is scoped to one
    attempt and needs no new command-line flag -- adding one would make every
    existing evaluate.py fail on an unrecognised argument.
    """
    return out_json.with_suffix(".per_sample.json")


def read_per_sample_file(path: Path) -> Optional[SampleResults]:
    """Read the declared per-sample file, the harness-agnostic path.

        {"dataset_total": 1319,                     optional
         "sample_ids": ["...", ...],                ordered, as evaluated
         "correct": {"id": true | false | 0.0-1.0}}

    or the runner-owned list form:

        [{"id": "...", "score": 0.0-1.0, "partial": 0.0-1.0}, ...]

    A fractional value counts as correct at 1.0, matching how inspect treats a
    partial credit score, so a rubric-graded benchmark can report one number per
    sample without inventing a convention here.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, list):
        ordered = []
        normalised = {}
        for row in data:
            if not isinstance(row, dict) or row.get("id") is None:
                continue
            sid = str(row["id"])
            ordered.append(sid)
            value = row.get("score")
            if isinstance(value, bool):
                normalised[sid] = value
            elif isinstance(value, (int, float)):
                normalised[sid] = bool(value >= 1.0)
            elif value == "C":
                normalised[sid] = True
            elif value in ("I", "P", "N"):
                normalised[sid] = False
        if not ordered:
            return None
        return SampleResults(
            ordered_ids=ordered,
            correct_by_id=normalised,
            dataset_total=None,
            source="per_sample.json",
        )
    if not isinstance(data, dict):
        return None
    ordered = data.get("sample_ids")
    correct = data.get("correct")
    if not isinstance(ordered, list) or not isinstance(correct, dict):
        return None

    score_mode = data.get("score_mode")
    if score_mode not in (None, "avg_at_n"):
        return None
    normalised = {}
    for sid, value in correct.items():
        if score_mode == "avg_at_n":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                return None
            normalised[sid] = float(value)
        elif isinstance(value, bool):
            normalised[sid] = value
        elif isinstance(value, (int, float)):
            normalised[sid] = bool(value >= 1.0)
        elif value == "C":
            normalised[sid] = True
        elif value in ("I", "P", "N"):
            normalised[sid] = False
    total = data.get("dataset_total")
    return SampleResults(
        ordered_ids=[str(i) for i in ordered],
        correct_by_id=normalised,
        dataset_total=int(total) if isinstance(total, int) and total > 0 else None,
        source="per_sample.json",
        score_mode="avg_at_n" if score_mode == "avg_at_n" else "binary",
    )


def _sample_correct(sample: dict) -> Optional[bool]:
    """Whether one logged sample counted as correct.

    The scorer's name differs per task -- a match scorer for exact-answer suites,
    a benchmark-specific one elsewhere -- so take whichever one the log carries
    rather than hardcoding a name. inspect writes 'C'/'I' for correct/incorrect.
    """
    scores = sample.get("scores") or {}
    for entry in scores.values():
        value = (entry or {}).get("value")
        if value == "C":
            return True
        if value in ("I", "P", "N"):
            return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value >= 1.0)
    return None


def read_inspect_log(path: Path) -> Optional[SampleResults]:
    """Read inspect-ai's own log, the path the five inspect-based tasks take.

    inspect records the ordered sample ids it evaluated, and --limit N takes
    exactly the first N of them -- verified across logs of 40, 150 and 1319
    samples, each an exact ordered prefix of the largest. It also records the size
    of the whole dataset even when --limit trimmed the run.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None

    dataset = (data.get("eval") or {}).get("dataset") or {}
    ordered = list(dataset.get("sample_ids") or [])
    total = dataset.get("samples")
    correct_by_id = {}
    for sample in data.get("samples") or []:
        sid = sample.get("id")
        if sid is not None:
            correct_by_id[sid] = _sample_correct(sample)
    if not ordered or not correct_by_id:
        return None
    return SampleResults(
        ordered_ids=ordered,
        correct_by_id=correct_by_id,
        dataset_total=int(total) if isinstance(total, int) and total > 0 else None,
        source="inspect_log",
    )


def partition(
    results: SampleResults,
    val_n: Optional[int] = None,
    reported_accuracy: Optional[float] = None,
    *,
    val_ids: Optional[list[str] | tuple[str, ...]] = None,
    test_ids: Optional[list[str] | tuple[str, ...]] = None,
) -> dict:
    """Split a complete evaluation by a prefix or an explicit Tau3 JSON split."""
    ordered_ids = [str(item) for item in results.ordered_ids]
    correct_by_id = {str(key): value for key, value in results.correct_by_id.items()}
    explicit = val_ids is not None or test_ids is not None
    out: dict = {
        "source": results.source,
        "score_mode": results.score_mode,
        "dataset_total": results.dataset_total,
        "evaluated": len(correct_by_id),
        "val_n": len(val_ids) if val_ids is not None else val_n,
        "accuracy_full": None,
        "accuracy_val": None,
        "accuracy_heldout": None,
        "val_count": None,
        "heldout_count": None,
        "reconstruction_ok": False,
        "reason": None,
        "split_kind": "explicit" if explicit else "prefix",
    }
    if not ordered_ids or not correct_by_id:
        out["reason"] = "no ordered sample ids or no scored samples"
        return out
    if explicit and len(ordered_ids) != len(set(ordered_ids)):
        out["reason"] = "evaluation reported duplicate sample ids"
        return out

    def accuracy(ids: list[str]) -> Optional[float]:
        values = [correct_by_id.get(item) for item in ids]
        if explicit and any(value is None for value in values):
            return None
        judged = [value for value in values if value is not None]
        if not judged:
            return None
        if results.score_mode == "avg_at_n":
            return sum(float(value) for value in judged) / len(judged)
        return sum(bool(value) for value in judged) / len(judged)

    full = accuracy(ordered_ids)
    out["accuracy_full"] = full
    if isinstance(reported_accuracy, (int, float)) and full is not None:
        out["reported_accuracy"] = float(reported_accuracy)
        out["reconstruction_ok"] = abs(full - float(reported_accuracy)) < 1e-6
    if not out["reconstruction_ok"]:
        out["reason"] = (
            "per-sample reconstruction does not reproduce the accuracy the harness reported"
            if reported_accuracy is not None
            else "no reported accuracy to check the per-sample reconstruction against"
        )
        return out

    if explicit:
        if val_ids is None or test_ids is None:
            out["reason"] = "explicit split needs both val_ids and test_ids"
            return out
        val_group, test_group = [str(item) for item in val_ids], [str(item) for item in test_ids]
        expected = set(val_group) | set(test_group)
        if not val_group or not test_group or len(expected) != len(val_group) + len(test_group):
            out["reason"] = "explicit validation/test split is empty or overlapping"
            return out
        observed = set(ordered_ids)
        if observed != expected:
            missing, unexpected = sorted(expected - observed), sorted(observed - expected)
            out["reason"] = "evaluation task ids do not match the explicit split" + (
                f"; missing={','.join(missing[:3])}" if missing else ""
            ) + (f"; unexpected={','.join(unexpected[:3])}" if unexpected else "")
            return out
        out["accuracy_val"], out["accuracy_heldout"] = accuracy(val_group), accuracy(test_group)
        if out["accuracy_val"] is None or out["accuracy_heldout"] is None:
            out["reason"] = "explicit split has missing or ungraded task outcomes"
            return out
        out["val_count"], out["heldout_count"] = len(val_group), len(test_group)
        return out

    if not isinstance(val_n, int) or val_n <= 0 or len(ordered_ids) <= val_n:
        out["reason"] = "validation boundary is invalid or has no held-out portion"
        return out
    val_group, heldout_group = ordered_ids[:val_n], ordered_ids[val_n:]
    out["accuracy_val"], out["accuracy_heldout"] = accuracy(val_group), accuracy(heldout_group)
    if out["accuracy_val"] is None or out["accuracy_heldout"] is None:
        out["reason"] = "prefix split has missing or ungraded sample outcomes"
        return out
    out["val_count"], out["heldout_count"] = len(val_group), len(heldout_group)
    return out


def partition_test_only(
    results: SampleResults,
    test_ids: list[str] | tuple[str, ...],
    reported_accuracy: Optional[float] = None,
) -> dict:
    """Score a Tau3 JSON test-only final, counting missing tasks as incorrect."""
    ordered_ids = [str(item) for item in results.ordered_ids]
    correct_by_id = {str(key): value for key, value in results.correct_by_id.items()}
    expected = [str(item) for item in test_ids]
    out = {
        "source": results.source, "dataset_total": results.dataset_total,
        "evaluated": len(correct_by_id), "accuracy_test": None, "test_count": None,
        "test_graded_count": None, "test_ungraded_count": None, "test_missing_ids": [],
        "harness_accuracy": float(reported_accuracy) if isinstance(reported_accuracy, (int, float)) else None,
        "reconstruction_ok": False, "reason": None, "split_kind": "json_test_only",
    }
    if not expected or len(expected) != len(set(expected)) or not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
        out["reason"] = "configured test ids or evaluation ids are empty or duplicated"
        return out
    if set(ordered_ids) - set(expected):
        out["reason"] = "evaluation emitted task ids outside the explicit test split"
        return out
    graded = [task_id for task_id in expected if correct_by_id.get(task_id) is not None]
    missing = [task_id for task_id in expected if task_id not in set(graded)]
    out.update({
        "accuracy_test": sum(bool(correct_by_id[task_id]) for task_id in graded) / len(expected),
        "test_count": len(expected), "test_graded_count": len(graded),
        "test_ungraded_count": len(missing), "test_missing_ids": missing,
        "reconstruction_ok": True,
    })
    if missing:
        out["reason"] = f"{len(missing)} expected test task(s) were ungraded and counted incorrect"
    return out


def partition_log(log_path: Path, val_n: int) -> dict:
    """Partition an inspect log, checking against the accuracy inspect reported."""
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            reported = (
                ((json.load(fh).get("results") or {}).get("scores") or [{}])[0]
                .get("metrics", {})
                .get("accuracy", {})
                .get("value")
            )
    except (OSError, ValueError, IndexError, AttributeError):
        reported = None
    results = read_inspect_log(Path(log_path))
    if results is None:
        return {
            "source": "inspect_log",
            "reconstruction_ok": False,
            "accuracy_val": None,
            "accuracy_heldout": None,
            "reason": "log carries no ordered sample ids or no scored samples",
        }
    return partition(results, val_n, reported_accuracy=reported)


# ------------------------------------------------------- baseline resource naming

# The baseline transcripts for a (benchmark, base_model, val_source) triple live in a
# named validation resource: declaration at training/resources/validation/<id>/,
# payload under VTB_ROOT. The id is computed, not written down anywhere twice, so an
# import, a recipe and a run's mount cannot disagree about the directory.
def _slug(text: str) -> str:
    """stable, filesystem-safe slug for a resource id component."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def baseline_resource_id(task: str, base_model: str, val_source_raw: str) -> str:
    """Legacy fallback ID for undeclared runtime baseline imports."""
    source_slug = _slug(val_source_raw.replace(":", "_"))
    return f"validation/{_slug(task)}__{_slug(base_model)}__{source_slug}"


def expand_val_source(repo: Path, raw: str) -> str:
    """Normalize path-bearing validation sources for declaration/service matching."""
    kind, separator, value = raw.strip().partition(":")
    if separator and kind in {"file", "json"}:
        path = resolve_resource_path(repo, value)
        return f"{kind}:{path.resolve()}" if path is not None else raw.strip()
    return raw.strip()


def find_baseline_resource(repo: Path, task: str, base_model: str,
                           val_source: Optional[str] = None) -> Optional[dict]:
    """Find one short-name declaration by identity, never by directory-name convention."""
    repo = Path(repo)
    wanted = expand_val_source(repo, val_source) if val_source is not None else None
    matches = []
    for path in sorted((repo / "training/resources/validation").glob("*/resource_config.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("type") != "validation_baseline" or cfg.get("benchmark") != task or cfg.get("base_model") != base_model:
            continue
        if wanted is not None and expand_val_source(repo, str(cfg.get("val_source") or "")) != wanted:
            continue
        matches.append(dict(cfg, _config_path=path))
    if len(matches) > 1:
        raise ValueError(f"ambiguous baseline declarations for {task}/{base_model}/{val_source}")
    return matches[0] if matches else None


def resolve_resource_path(repo: Path, raw: Optional[str]) -> Optional[Path]:
    """Resolve a resource_config.json payload path to an absolute path; None in, None out.

    One convention for every resource declaration: ${VTB_ROOT} and ${VTB_REPO} expand
    to the runtime roots, other environment variables expand too, and a relative path
    is repo-relative. src/commit_utils/prepare_agent_workspace.resolve_declared_path
    wraps this with its fail-fast CLI contract; readers that tolerate a missing field
    call this directly.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    repo = Path(repo)
    vtb_root = os.environ.get("VTB_ROOT") or str(repo / "vtb-env")
    value = raw.strip()
    value = value.replace("${VTB_ROOT}", vtb_root).replace("$VTB_ROOT", vtb_root)
    value = value.replace("${VTB_REPO}", str(repo)).replace("$VTB_REPO", str(repo))
    hf_home = os.environ.get("HF_HOME") or str(Path(vtb_root) / "resources/data/hf_cache")
    value = value.replace("${HF_HOME}", hf_home).replace("$HF_HOME", hf_home)
    model_root = os.environ.get("VTB_MODEL_RESOURCE_ROOT") or str(Path(vtb_root) / "resources/models/qwen3_5_9b_base")
    value = value.replace("${VTB_MODEL_RESOURCE_ROOT}", model_root).replace("$VTB_MODEL_RESOURCE_ROOT", model_root)
    value = os.path.expandvars(value)
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo / path).resolve()
