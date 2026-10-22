"""Run registry, path sandbox, submission log and baseline cache.

The registry is what makes one grading service safely serve many concurrent
runs: a run registers once and receives an opaque token. Every later request
carries that token, and the token -- never anything the caller sends -- decides
which task is evaluated and which directory a submitted model may live in.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from . import config


def _safe(name: str) -> str:
    return re.sub(r'[/\\:*?"<>|]', "_", name)


def _sample_span_seconds(result: Path) -> Optional[float]:
    """Wall-clock seconds one graded sample took, from the timestamps it records.

    The harness writes `started_at` and `finished_at` into every sample's result file,
    so a completed evaluation carries its own per-sample timing without the service
    having had to watch it happen.
    """
    try:
        data = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    started, finished = data.get("started_at"), data.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        begin = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end = datetime.fromisoformat(finished.replace("Z", "+00:00"))
    except ValueError:
        return None
    span = (end - begin).total_seconds()
    return span if span > 0 else None


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def admin_token(state_dir: Optional[Path] = None) -> str:
    """The operator-side secret, kept in a file only the operator can reach.

    Issuing a run token is an operator action: it fixes which directory the run
    may submit from, whether it is allowed to write the official metrics, and how
    much of an evaluation it gets to see. Without a secret on /register any caller
    that can open the port could mint itself a token declaring role=official --
    and the agent container shares the host network namespace, so it can open the
    port. That was verified to work before this existed.

    The secret lives in the service state directory, which is never bound into an
    agent container (only <scratch>/job_dir, <scratch>/tmp and the HuggingFace
    overlay are), so an agent cannot read it. Persisted rather than regenerated so
    that restarting the service does not orphan a run that is about to register.
    """
    directory = Path(state_dir) if state_dir else config.STATE_DIR
    path = directory / "admin.token"
    if path.exists():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            os.chmod(path, 0o600)
            return existing
    token = secrets.token_urlsafe(24)
    _atomic_write(path, token)
    os.chmod(path, 0o600)
    return token


class RunContext:
    """Everything the service knows about one registered run."""

    __slots__ = (
        "run_id", "task", "base_model", "sandbox_root", "eval_dir", "role",
        "log_dir", "disclosure_dir", "registered_at", "validation_source",
        "editable_harness",
    )

    # An agent's token and the official token differ in what they are allowed to
    # ask for, not just in which directory they can reach. Carry the role
    # explicitly rather than inferring it from a run_id suffix, which would make
    # a naming convention load-bearing for a privilege decision.
    ROLES = ("agent", "official")

    def __init__(
        self,
        run_id: str,
        task: str,
        base_model: str,
        sandbox_root: Path,
        eval_dir: Path,
        role: str = "official",
        log_dir: Optional[Path] = None,
        disclosure_dir: Optional[Path] = None,
        registered_at: Optional[float] = None,
        validation_source: Optional[str] = None,
        editable_harness: Optional[str] = None,
    ) -> None:
        if role not in self.ROLES:
            raise ValueError(f"role must be one of {self.ROLES}, got {role!r}")
        self.run_id = run_id
        self.task = task
        self.base_model = base_model
        self.sandbox_root = sandbox_root
        self.eval_dir = eval_dir
        self.role = role
        self.log_dir = log_dir
        # Where this run's evaluations may leave a full, agent-readable copy of what
        # they graded. Distinct from log_dir, which holds one harness log: this holds
        # the per-sample trees the agent diagnoses from, and register() refuses a
        # destination inside sandbox_root so the agent cannot rewrite its own record.
        self.disclosure_dir = disclosure_dir
        self.registered_at = registered_at if registered_at is not None else time.time()
        # The validation source the run declared (its recipe's validation block).
        # Recorded for audit; the service refused registration if it disagreed with
        # the service's own val source, so this is the source the run's numbers are
        # interpretable under.
        self.validation_source = validation_source
        # "off" when the run opted out of its benchmark's editable harness at
        # registration (a model-only ablation); submissions then skip harness pairing.
        self.editable_harness = editable_harness

    def to_json(self) -> dict:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "base_model": self.base_model,
            "sandbox_root": str(self.sandbox_root),
            "eval_dir": str(self.eval_dir),
            "role": self.role,
            "log_dir": str(self.log_dir) if self.log_dir else None,
            "disclosure_dir": str(self.disclosure_dir) if self.disclosure_dir else None,
            "registered_at": self.registered_at,
            "validation_source": self.validation_source,
            "editable_harness": self.editable_harness,
        }

    @classmethod
    def from_json(cls, d: dict) -> "RunContext":
        return cls(
            run_id=d["run_id"],
            task=d["task"],
            base_model=d["base_model"],
            sandbox_root=Path(d["sandbox_root"]),
            eval_dir=Path(d["eval_dir"]),
            # Entries written before roles existed were all operator-side.
            role=d.get("role", "official"),
            log_dir=Path(d["log_dir"]) if d.get("log_dir") else None,
            disclosure_dir=Path(d["disclosure_dir"]) if d.get("disclosure_dir") else None,
            registered_at=d.get("registered_at"),
            validation_source=d.get("validation_source"),
            editable_harness=d.get("editable_harness"),
        )

    def submissions_file(self) -> Path:
        return self.eval_dir / "submissions.json"

    def resolve_model_path(self, model_path: str) -> Path:
        """Confine a submitted path to this run's sandbox root.

        Adapted from autorl_bench's GradingServer.resolve_model_path, with the
        root taken from the run's token instead of a single global workspace.
        This is what stops run A submitting run B's artifacts, and what stops
        anyone submitting a model out of the shared HuggingFace cache (the
        model-substitution reward hack recorded in the project README).
        """
        if not model_path or "\x00" in model_path:
            raise ValueError("invalid model_path")

        root = self.sandbox_root.expanduser().resolve()
        normalized = os.path.normpath(model_path)
        if os.path.splitdrive(normalized)[0]:
            raise ValueError("invalid model_path")

        candidate = normalized if os.path.isabs(normalized) else os.path.join(str(root), normalized)
        resolved = Path(candidate).expanduser().resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("model_path escapes the run sandbox") from exc
        return resolved


class Registry:
    """Token -> RunContext, persisted so a service restart keeps live runs."""

    def __init__(self, state_dir: Optional[Path] = None) -> None:
        self.state_dir = Path(state_dir) if state_dir else config.STATE_DIR
        self.runs_file = self.state_dir / "runs.json"
        self._lock = threading.Lock()
        self._by_token: dict[str, RunContext] = {}
        self._load()

    def _load(self) -> None:
        if not self.runs_file.exists():
            return
        try:
            raw = json.loads(self.runs_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for token, payload in raw.items():
            try:
                self._by_token[token] = RunContext.from_json(payload)
            except (KeyError, TypeError):
                continue

    def _flush_locked(self) -> None:
        payload = {t: c.to_json() for t, c in self._by_token.items()}
        _atomic_write(self.runs_file, json.dumps(payload, indent=2, ensure_ascii=False))

    def register(
        self,
        run_id: str,
        task: str,
        base_model: str,
        sandbox_root: Path,
        eval_dir: Path,
        role: str = "official",
        log_dir: Optional[Path] = None,
        disclosure_dir: Optional[Path] = None,
        validation_source: Optional[str] = None,
        editable_harness: Optional[str] = None,
    ) -> str:
        if task not in config.known_tasks():
            raise ValueError(f"unknown task: {task}")
        sandbox_root = Path(sandbox_root).expanduser().resolve()
        if not sandbox_root.is_dir():
            raise ValueError(f"sandbox_root is not a directory: {sandbox_root}")
        eval_dir = Path(eval_dir).expanduser().resolve()
        eval_dir.mkdir(parents=True, exist_ok=True)

        if log_dir is not None:
            log_dir = Path(log_dir).expanduser().resolve()
            # Evaluation logs carry per-sample outcomes and the model's own
            # outputs on the validation split -- exactly what the disclosure
            # policy withholds. Writing them anywhere the agent can read would
            # hand all of it over through the filesystem instead, so refuse a
            # destination inside its sandbox rather than trusting the caller.
            if log_dir == sandbox_root or sandbox_root in log_dir.parents:
                raise ValueError(
                    f"log_dir {log_dir} is inside the run sandbox {sandbox_root}; "
                    "evaluation logs would become readable by the agent"
                )
            log_dir.mkdir(parents=True, exist_ok=True)

        if disclosure_dir is not None:
            disclosure_dir = Path(disclosure_dir).expanduser().resolve()
            # Inside the sandbox the agent could edit what its own evaluation recorded,
            # and then "what it was told" would no longer be a fact anyone can check
            # afterwards. The run harness mounts this directory into the agent's
            # container read-only instead, so it is readable there and writable only
            # here.
            if disclosure_dir == sandbox_root or sandbox_root in disclosure_dir.parents:
                raise ValueError(
                    f"disclosure_dir {disclosure_dir} is inside the run sandbox {sandbox_root}; "
                    "mount it read-only instead so the agent cannot rewrite its own results"
                )
            disclosure_dir.mkdir(parents=True, exist_ok=True)

        token = secrets.token_urlsafe(24)
        ctx = RunContext(
            run_id, task, base_model, sandbox_root, eval_dir,
            role=role, log_dir=log_dir, disclosure_dir=disclosure_dir,
            validation_source=validation_source, editable_harness=editable_harness,
        )
        with self._lock:
            self._by_token[token] = ctx
            self._flush_locked()
        return token

    def lookup(self, token: str) -> RunContext:
        with self._lock:
            ctx = self._by_token.get(token or "")
        if ctx is None:
            raise KeyError("unknown or expired token")
        return ctx

    def unregister(self, token: str) -> None:
        with self._lock:
            self._by_token.pop(token, None)
            self._flush_locked()

    def active(self) -> list[dict]:
        with self._lock:
            return [c.to_json() for c in self._by_token.values()]


class SubmissionLog:
    """Per-run submission history, written next to the run's other artifacts.

    Deliberately a separate file from metrics.json: metrics.json stays the
    official final_model score so results remain comparable with upstream,
    while submissions.json carries the in-run submission sequence.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def load(self, ctx: RunContext) -> list[dict]:
        path = ctx.submissions_file()
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def append(self, ctx: RunContext, entry: dict) -> list[dict]:
        with self._lock:
            entries = self.load(ctx)
            entries.append(entry)
            _atomic_write(ctx.submissions_file(), json.dumps(entries, indent=2, ensure_ascii=False))
            return entries

    def best(self, ctx: RunContext, mode: str = "final") -> Optional[dict]:
        scored = [
            e
            for e in self.load(ctx)
            if e.get("mode") == mode and isinstance(e.get("score"), (int, float)) and not e.get("error")
        ]
        return max(scored, key=lambda e: e["score"]) if scored else None


class BaselineCache:
    """Baseline score per (task, base_model), shared across every run.

    Sharing is the point: the base model's score does not depend on which run
    asks for it, so concurrent runs on the same task/model stop paying for the
    same baseline evaluation over and over. A miss reports no baseline rather
    than inventing one.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else config.BASELINE_CACHE_DIR
        self._lock = threading.Lock()
        self._timing: dict = {}

    def _path(self, task: str, base_model: str) -> Path:
        return self.cache_dir / f"{task}_{_safe(base_model)}.json"

    def disclosure_dir(self, task: str, base_model: str) -> Path:
        """Legacy location of the base model's validation transcripts for this pair.

        Baselines imported through the resource convention live as named resources
        under VTB_ROOT/resources/validation/ (see baseline_payload_dir); this dir is
        kept as the compatibility alias older runs and in-flight services resolve.
        Derived from the same (task, base_model) key as the score, and here rather than
        at the call site so the two cannot end up naming different directories for the
        same base model.
        """
        return config.BASELINE_DISCLOSURE_DIR / f"{task}_{_safe(base_model)}"

    def baseline_payload_dir(self, task: str, base_model: str, val_source: str) -> Path:
        """Canonical location of a baseline's transcript tree: the validation resource.

        The resource id is computed by valsplit.baseline_resource_id from the same
        three facts, so an import and a run's mount name the same directory without
        consulting each other.
        """
        from . import valsplit  # local import: valsplit must stay importable alone

        cfg = self.baseline_resource_config(task, base_model, val_source)
        if cfg is not None:
            payload = valsplit.resolve_resource_path(config.REPO_ROOT, cfg.get("materialized_path"))
            if payload is None:
                raise ValueError(f"baseline {cfg['id']} has no materialized_path")
            return payload
        return Path(config.VTB_ROOT) / "resources" / valsplit.baseline_resource_id(task, base_model, val_source)

    def baseline_resource_config(self, task: str, base_model: str, val_source: Optional[str] = None) -> Optional[dict]:
        """The validation_baseline resource declared for this pair, if one exists.

        Read from the repo's training/resources/validation/*/resource_config.json, so
        a baseline that was promoted to a resource is discoverable without knowing its
        val source in advance.
        """
        from . import valsplit

        source = val_source or config.VAL_SOURCE_RAW or config.validation_source(task)
        return valsplit.find_baseline_resource(config.REPO_ROOT, task, base_model, source)

    def sample_timing(self, task: str, base_model: str) -> Optional[dict]:
        """What one sample cost the base model, measured from its own validation.

        This is the cold-start ETA source. A run's first submission has no history of
        its own, and another run's history describes another run's model, so the only
        honest reference left is the base model's measured timing on the same split --
        which every run has, because every run is scored against it.

        The mean is the right statistic here even though the median is the right one for
        reporting spend. Total time is total work over throughput, so a long tail has to
        be counted in: on this benchmark the mean per-sample cost reproduces the base
        model's measured wall clock to 3.5% (3.02h predicted against 3.13h actual) while
        the median underestimates it by 24%.

        Memoised because the caller is a polled endpoint and the answer cannot change:
        the base model's transcripts are written once, before any run starts. Without
        the cache this would re-read 147 files on every poll.

        Returns None when the base model's transcripts are not on disk, which is normal
        for a task whose baseline came from the published table rather than a local run.
        """
        cache_key = f"{task}|{base_model}"
        with self._lock:
            if cache_key in self._timing:
                return self._timing[cache_key]
        timing = self._measure_sample_timing(task, base_model)
        with self._lock:
            self._timing[cache_key] = timing
        return timing

    def _measure_sample_timing(self, task: str, base_model: str) -> Optional[dict]:
        root = None
        cfg = self.baseline_resource_config(task, base_model)
        if cfg is not None:
            from . import valsplit  # local import: valsplit must stay importable alone

            payload = valsplit.resolve_resource_path(config.REPO_ROOT, cfg.get("materialized_path"))
            if payload is not None:
                root = payload / "samples"
        if root is None:
            root = self.disclosure_dir(task, base_model) / "samples"
        if not root.is_dir():
            return None
        try:
            entries = list(root.iterdir())
        except OSError:
            return None
        spans = []
        for entry in entries:
            span = _sample_span_seconds(entry / "result.json")
            if span is not None:
                spans.append(span)
        if not spans:
            return None
        return {
            "mean_seconds": round(sum(spans) / len(spans), 1),
            "samples": len(spans),
        }

    def get(self, task: str, base_model: str) -> Optional[float]:
        path = self._path(task, base_model)
        if path.exists():
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                from . import valsplit
                source = config.VAL_SOURCE_RAW or config.validation_source(task)
                recorded = entry.get("val_source")
                compatible = not source or (isinstance(recorded, str) and valsplit.expand_val_source(config.REPO_ROOT, recorded) == valsplit.expand_val_source(config.REPO_ROOT, source))
                value = entry.get("score") if compatible else None
            except (OSError, ValueError):
                value = None
            if isinstance(value, (int, float)):
                return float(value)
        cfg = self.baseline_resource_config(task, base_model)
        if cfg is not None and isinstance(cfg.get("score"), (int, float)):
            return float(cfg["score"])
        return None

    def get_heldout(self, task: str, base_model: str) -> Optional[float]:
        """The base model's score on the held-out remainder, if one was measured.

        This is the reference a held-out-only final is compared against; the split's
        own baseline would be apples-to-oranges. None when the baseline predates
        held-out recording -- never silently substituted with the split score.
        """
        path = self._path(task, base_model)
        if path.exists():
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                from . import valsplit
                source = config.VAL_SOURCE_RAW or config.validation_source(task)
                recorded = entry.get("val_source")
                compatible = not source or (isinstance(recorded, str) and valsplit.expand_val_source(config.REPO_ROOT, recorded) == valsplit.expand_val_source(config.REPO_ROOT, source))
                value = entry.get("heldout_score") if compatible else None
            except (OSError, ValueError):
                value = None
            if isinstance(value, (int, float)):
                return float(value)
        cfg = self.baseline_resource_config(task, base_model)
        if cfg is not None and isinstance(cfg.get("heldout_score"), (int, float)):
            return float(cfg["heldout_score"])
        return None

    def put(self, task: str, base_model: str, score: float, extra: Optional[dict] = None) -> None:
        payload: dict[str, Any] = {
            "task": task,
            "base_model": base_model,
            "score": float(score),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        source = config.VAL_SOURCE_RAW or config.validation_source(task)
        if source:
            payload["val_source"] = source
        if extra:
            payload.update(extra)
        with self._lock:
            _atomic_write(self._path(task, base_model), json.dumps(payload, indent=2, ensure_ascii=False))


# Substring tagging the reasons that are permission failures rather than half-written
# files. check_weights_complete returns prose for a human; the caller matches on this to
# pick the right advice, so it has to stay a substring of every such message.
UNREADABLE_MARKER = "cannot be read by this service"


def check_weights_complete(model_dir: Path) -> Optional[str]:
    """Reject a checkpoint that is still being written; returns a reason or None.

    The agent's workspace is shared: it writes the checkpoint and the evaluation
    reads the same bytes off disk, with nothing sequencing the two. Without this
    check a mid-write submission fails deep inside the model server, where the message
    says nothing about the actual cause.

    safetensors makes truncation decidable without reading the weights. The file
    opens with a little-endian u64 header length, then that many bytes of JSON in
    which every tensor carries `data_offsets` relative to the end of the header. So
    the file has to be at least 8 + header + max(offset) bytes long, and a short
    file is a file still being written.

    Deliberately narrow. It says nothing about a directory holding no safetensors
    at all -- .bin checkpoints and adapter-only directories are legitimate layouts,
    and an over-strict check here would repeat delivery_audit's habit of calling a
    sharded model "missing model.safetensors".

    A file that exists but cannot be opened is not a file still being written, and the
    two need opposite advice: one says wait, the other says waiting will never help.
    Permission errors are therefore tagged with UNREADABLE_MARKER so the caller can
    say which it is without parsing an errno back out of prose.
    """
    if not model_dir.is_dir():
        return None

    # Probing a child needs execute on the directory and listing it needs read, so a
    # restrictive umask can deny either one independently. Both are guarded: a traceback
    # here would reach the agent as a 500 with no reason attached.
    index = model_dir / "model.safetensors.index.json"
    try:
        have_index = index.is_file()
    except PermissionError as exc:
        return f"the directory {UNREADABLE_MARKER} ({exc})"

    if have_index:
        try:
            weight_map = (json.loads(index.read_text(encoding="utf-8")) or {}).get("weight_map") or {}
        except PermissionError as exc:
            return f"model.safetensors.index.json {UNREADABLE_MARKER} ({exc})"
        except (OSError, ValueError) as exc:
            return f"model.safetensors.index.json is not readable yet ({exc})"
        for shard in sorted(set(weight_map.values())):
            try:
                present = (model_dir / shard).is_file()
            except PermissionError as exc:
                return f"the directory {UNREADABLE_MARKER} ({exc})"
            if not present:
                return f"the shard index references {shard}, which is not on disk yet"

    try:
        shards = sorted(model_dir.glob("*.safetensors"))
    except PermissionError as exc:
        return f"the directory {UNREADABLE_MARKER} ({exc})"

    for shard in shards:
        try:
            size = shard.stat().st_size
            with open(shard, "rb") as fh:
                raw = fh.read(8)
                if len(raw) < 8:
                    return f"{shard.name} is {size} bytes, too short to hold a header"
                header_len = int.from_bytes(raw, "little")
                if header_len <= 0 or 8 + header_len > size:
                    return f"{shard.name} declares a {header_len}-byte header but is only {size} bytes"
                header = json.loads(fh.read(header_len).decode("utf-8"))
        except PermissionError as exc:
            return f"{shard.name} {UNREADABLE_MARKER} ({exc})"
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            return f"{shard.name} has no readable safetensors header yet ({exc})"

        needed = 0
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            offsets = meta.get("data_offsets")
            if isinstance(offsets, list) and len(offsets) == 2:
                needed = max(needed, int(offsets[1]))
        expected = 8 + header_len + needed
        if size < expected:
            return (
                f"{shard.name} is {size} bytes but its header describes {expected}; "
                "the file is still being written"
            )
    return None


class TaskStats:
    """Observed dataset size per task, which is the denominator a ratio needs.

    Learned rather than declared. inspect records the whole dataset's size in
    every evaluation log, even one that --limit trimmed, so the first evaluation
    of a task teaches the service its size and nothing has to be hardcoded --
    which also means the figure cannot drift away from the data actually loaded.

    Until a task's size is known, a ratio-based validation boundary is
    undecidable and agent submissions for it are refused. That is deliberate:
    guessing a boundary risks scoring an agent on held-out samples, and that
    contamination cannot be undone after the fact.
    """

    def __init__(self, state_dir: Optional[Path] = None) -> None:
        self.path = (Path(state_dir) if state_dir else config.STATE_DIR) / "task_stats.json"
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def total(self, task: str) -> Optional[int]:
        entry = self._load().get(task) or {}
        value = entry.get("dataset_total")
        return int(value) if isinstance(value, int) and value > 0 else None

    def observe_total(self, task: str, total: int) -> Optional[str]:
        """Record an observed dataset size; returns a warning on disagreement.

        A changed size means the underlying dataset changed, which silently
        moves the validation boundary and makes old and new scores incomparable.
        Worth saying out loud rather than overwriting quietly.
        """
        if not isinstance(total, int) or total <= 0:
            return None
        warning = None
        with self._lock:
            data = self._load()
            entry = data.setdefault(task, {})
            previous = entry.get("dataset_total")
            if isinstance(previous, int) and previous != total:
                warning = (
                    f"dataset size for {task} changed from {previous} to {total}; "
                    "the validation boundary moves with it and scores across the change "
                    "are not comparable"
                )
            entry["dataset_total"] = total
            entry["observed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            _atomic_write(self.path, json.dumps(data, indent=2, ensure_ascii=False))
        return warning

    def snapshot(self) -> dict:
        return {t: e.get("dataset_total") for t, e in self._load().items()}

    def val_ids(self, task: str) -> Optional[list]:
        """The sample ids that make up this task's validation split, once known.

        Kept beside the dataset size because it is the same kind of fact, learned the
        same way: an evaluation teaches it, and nothing is hardcoded.
        """
        entry = self._load().get(task) or {}
        ids = entry.get("val_ids")
        return [str(i) for i in ids] if isinstance(ids, list) and ids else None

    def ordered_ids(self, task: str) -> Optional[list]:
        """The whole benchmark's sample ids in dataset order, once observed.

        This is what a held-out-only final is computed from: held-out = ordered_ids
        past the validation prefix. Learned from a baseline import or any
        full-coverage evaluation; never from a subset run.
        """
        entry = self._load().get(task) or {}
        ids = entry.get("ordered_ids")
        return [str(i) for i in ids] if isinstance(ids, list) and ids else None

    def observe_order(self, task: str, ordered_ids: list, total: int) -> Optional[str]:
        """Record the benchmark's full sample order, learned from a complete coverage.

        Only a run that covered every sample teaches the order: a prefix or subset
        says nothing about what comes after it. A disagreement means the dataset
        changed under the service, which moves both the validation boundary and the
        held-out remainder -- worth saying out loud.
        """
        wanted = [str(i) for i in ordered_ids or []]
        if not wanted or (isinstance(total, int) and total > 0 and len(wanted) != total):
            return None
        warning = None
        with self._lock:
            data = self._load()
            entry = data.setdefault(task, {})
            previous = entry.get("ordered_ids")
            if isinstance(previous, list) and [str(i) for i in previous] != wanted:
                warning = (
                    f"dataset order for {task} changed; the validation boundary and the "
                    "held-out remainder both move with it, and scores across the change "
                    "are not comparable"
                )
            entry["ordered_ids"] = wanted
            entry["ordered_ids_observed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            _atomic_write(self.path, json.dumps(data, indent=2, ensure_ascii=False))
        return warning

    def tasks_with_val_ids(self) -> list:
        """Tasks whose split membership is known, so named samples can be resolved.

        Reported by /health because the alternative way to discover it is to submit
        named ids and read the refusal.
        """
        data = self._load()
        if not isinstance(data, dict):
            return []
        return [
            task
            for task, entry in data.items()
            if isinstance(entry, dict) and entry.get("val_ids")
        ]

    def observe_val_ids(self, task: str, ids: list, val_n: int) -> Optional[str]:
        """Record the validation split's membership, learned from a complete run of it.

        Recorded only when the evaluation covered exactly val_n samples. A run that
        left some ungraded produces a shorter list, and a short list here would later
        refuse legitimate validation samples as out of bounds -- an incomplete set is
        worse than none, so none is kept.

        This is what makes it possible to accept named samples at all: without the
        membership, an id cannot be checked against the boundary, and an unchecked id
        could point into the held-out remainder.
        """
        wanted = [str(i) for i in ids or []]
        if not isinstance(val_n, int) or val_n <= 0 or len(wanted) != val_n:
            return None
        warning = None
        with self._lock:
            data = self._load()
            entry = data.setdefault(task, {})
            previous = entry.get("val_ids")
            if isinstance(previous, list) and {str(i) for i in previous} != set(wanted):
                warning = (
                    f"validation split membership for {task} changed; scores across the "
                    "change are not comparable and named-sample submissions were checked "
                    "against the old set"
                )
            entry["val_ids"] = wanted
            entry["val_ids_observed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            _atomic_write(self.path, json.dumps(data, indent=2, ensure_ascii=False))
        return warning


class DurationStats:
    """How long past evaluations took, keyed by (run, task, mode, limit, concurrency).

    Gives an ETA that is available immediately -- including while a submission
    is still queued, when there is no log to measure a rate from. Only
    successful evaluations are recorded; a failed one measures the retry ladder,
    not the work. Nor is one whose trials mostly made no progress: that measures the
    looping, and the server rejects it before it gets here.

    Scope timing history to the submitting run so differences in model behavior
    do not distort estimates for other runs sharing the service. A run's first
    submission uses the base model's measured per-sample timing instead of
    unrelated run records; see BaselineCache.sample_timing.

    Concurrency belongs in the key for the same reason: records from different
    concurrencies pool into a median that describes neither. Keys written before the
    run and concurrency were part of the key are left in place but never read.
    """

    KEEP_PER_KEY = 20

    def __init__(self, state_dir: Optional[Path] = None) -> None:
        self.path = (Path(state_dir) if state_dir else config.STATE_DIR) / "durations.json"
        self._lock = threading.Lock()

    @staticmethod
    def _concurrency() -> int:
        return config.eval_concurrency()

    @staticmethod
    def _run_scope(run_id: str) -> str:
        """One run, both of its tokens.

        A run registers twice, `...#agent` and `...#official`, and they evaluate the
        same model lineage on the same hardware, so they share one pool. The role
        suffix is dropped rather than kept because splitting them would leave the
        official evaluation with no history at all.
        """
        return (run_id or "_unknown").rsplit("#", 1)[0]

    @classmethod
    def _key(cls, run_id: str, task: str, mode: str, limit: Optional[int]) -> str:
        return f"{cls._run_scope(run_id)}|{task}|{mode}|{limit}|c{cls._concurrency()}"

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def record(self, run_id: str, task: str, mode: str, limit: Optional[int], elapsed: float) -> None:
        if elapsed <= 0:
            return
        with self._lock:
            data = self._load()
            samples = data.setdefault(self._key(run_id, task, mode, limit), [])
            samples.append(round(float(elapsed), 1))
            del samples[: max(0, len(samples) - self.KEEP_PER_KEY)]
            _atomic_write(self.path, json.dumps(data, indent=2))

    def lookup(self, run_id: str, task: str, mode: str, limit: Optional[int]) -> Optional[tuple]:
        """(median_seconds, is_exact_key, n), preferring an exact task/mode/limit match.

        Keying on the limit as well is correct -- 40 samples and 150 samples are
        not the same job -- but it also means a first-of-its-kind limit has no
        history at all. Rather than offering nothing, fall back to any run of the
        same task and mode and flag the result as inexact. It cannot simply be
        scaled by limit: a measured evaluation spends roughly 60s on container
        start, model load and CUDA graph capture before the first sample, so
        duration is startup plus per-sample cost, not proportional.

        `n` is how many records the median came from, and it is returned because a
        median does not carry its own sample size while a caller quoting it as an
        estimate implies one.
        """
        data = self._load()
        exact = data.get(self._key(run_id, task, mode, limit)) or []
        median = self._median_of(exact)
        if median is not None:
            return (median, True, self._count_of(exact))

        # This run, same task, mode and concurrency, any limit. A different limit makes
        # the estimate rough; a different concurrency or a different run makes it wrong.
        prefix = f"{self._run_scope(run_id)}|{task}|{mode}|"
        suffix = f"|c{self._concurrency()}"
        pooled: list[float] = []
        for key, samples in data.items():
            if key.startswith(prefix) and key.endswith(suffix) and isinstance(samples, list):
                pooled.extend(samples)
        approx = self._median_of(pooled)
        return (approx, False, self._count_of(pooled)) if approx is not None else None

    @staticmethod
    def _count_of(samples: list) -> int:
        return sum(1 for s in samples if isinstance(s, (int, float)))

    @staticmethod
    def _median_of(samples: list) -> Optional[float]:
        values = [float(s) for s in samples if isinstance(s, (int, float))]
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2


def model_fingerprint(path: Path) -> str:
    """Cache key for an evaluation result: path plus newest weight mtime.

    Same idea as autorl_bench's _make_cache_key -- re-submitting an unchanged
    directory is free, while overwriting the weights invalidates the entry.
    """
    newest = 0.0
    if path.is_dir():
        for f in path.rglob("*"):
            if f.suffix in (".safetensors", ".bin", ".json") and f.is_file():
                try:
                    newest = max(newest, f.stat().st_mtime)
                except OSError:
                    continue
    elif path.is_file():
        try:
            newest = path.stat().st_mtime
        except OSError:
            newest = 0.0
    return f"{path}@{newest}"
