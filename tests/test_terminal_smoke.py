from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFY = REPO / "scripts/run_smoke/python/verify_smoke_terminal.py"
SPEC = importlib.util.spec_from_file_location("verify_terminal_smoke", VERIFY)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class TerminalSmokeVerificationTest(unittest.TestCase):
    def _tree(self, root: Path, *, val_limit: int = 1) -> tuple[Path, Path]:
        state = root / "state"
        results = root / "results"
        state.mkdir()
        results.mkdir()
        val_job, final_job = "val123", "final456"
        lines = [
            "[submit] queued job="
            f"{val_job} run=run#agent role=agent "
            "task=terminal-bench__terminal-bench-2-1 "
            "eval_task=openthoughts__openthoughts-tblite-val50 "
            f"mode=val limit={val_limit} position=1",
            f"[worker] finish job={val_job} state=done score=0.0",
            "[submit] queued job="
            f"{final_job} run=run#official role=official "
            "task=terminal-bench__terminal-bench-2-1 "
            "eval_task=terminal-bench__terminal-bench-2-1 "
            "mode=final limit=1 position=1",
            f"[worker] finish job={final_job} state=done score=0.0",
        ]
        (state / "service.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        metrics = {
            "accuracy": 0.0,
            "graded": 1,
            "attempted": 1,
            "ungraded": 0,
            "backend_returncode": 0,
        }
        for mode, job, dataset in (
            ("val", val_job, "openthoughts/openthoughts-tblite"),
            ("final", final_job, "terminal-bench/terminal-bench-2-1"),
        ):
            archive = state / "archive/run" / f"{mode}_{job}"
            backend = archive / "harness_metrics_p1_a1/backend_output/job"
            backend.mkdir(parents=True)
            (archive / "metrics_p1_a1.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            (backend / "backend_command.txt").write_text(
                f"harbor run --dataset {dataset}\n", encoding="utf-8"
            )
        (results / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return state, results

    def test_accepts_exactly_one_clean_val_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, results = self._tree(Path(tmp))
            scores = verifier.verify(state, results)
        self.assertEqual(scores, {"agent": 0.0, "official": 0.0})

    def test_rejects_limit_ten_as_not_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, results = self._tree(Path(tmp), val_limit=10)
            with self.assertRaises(verifier.VerificationError):
                verifier.verify(state, results)

    def test_rejects_extra_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, results = self._tree(Path(tmp))
            with (state / "service.log").open("a", encoding="utf-8") as handle:
                handle.write(
                    "[submit] queued job=extra role=agent "
                    "task=terminal-bench__terminal-bench-2-1 "
                    "eval_task=openthoughts__openthoughts-tblite-val50 "
                    "mode=val limit=1 position=1\n"
                )
            with self.assertRaises(verifier.VerificationError):
                verifier.verify(state, results)


if __name__ == "__main__":
    unittest.main()
