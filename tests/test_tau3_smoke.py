from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFY = REPO / "scripts/run_smoke/python/verify_smoke_tau3.py"
SPEC = importlib.util.spec_from_file_location("verify_tau3_smoke", VERIFY)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class Tau3SmokeVerificationTest(unittest.TestCase):
    def _tree(self, root: Path, *, include_final: bool = True) -> tuple[Path, Path]:
        state = root / "state"
        results = root / "results"
        state.mkdir()
        results.mkdir()
        jobs = [("val123", "agent", "val")]
        if include_final:
            jobs.append(("final456", "official", "final"))
        lines = []
        metrics = {
            "accuracy": 0.0,
            "graded": 1,
            "attempted": 1,
            "ungraded": 0,
            "backend_returncode": 0,
        }
        for job, role, mode in jobs:
            lines.extend([
                f"[submit] queued job={job} run=run#{role} role={role} "
                "task=sierra-research__tau3-bench "
                "eval_task=sierra-research__tau3-bench "
                f"mode={mode} limit=1 position=1",
                f"[worker] finish job={job} state=done score=0.0",
            ])
            archive = state / "archive/run" / f"{mode}_{job}"
            archive.mkdir(parents=True)
            (archive / "metrics_p1_a1.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
        (state / "service.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (results / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return state, results

    def test_accepts_exactly_one_clean_val_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, results = self._tree(Path(tmp))
            scores = verifier.verify(state, results)
        self.assertEqual(scores, {"agent": 0.0, "official": 0.0})

    def test_rejects_missing_final_connectivity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, results = self._tree(Path(tmp), include_final=False)
            with self.assertRaisesRegex(ValueError, "expected two Tau3 submissions"):
                verifier.verify(state, results)


class Tau3StartupWaitTest(unittest.TestCase):
    def _startup(self, mode):
        launcher = (REPO / "scripts/run_smoke/run_smoke_tau3.sh").read_text()
        function = launcher[launcher.index("start_service() {"):launcher.index("start_run() {")]
        stubs = """
set -euo pipefail
calls=0
preflight() { :; }
sleep() { :; }
health() {
    calls=$((calls + 1))
    [ "$MODE" = delayed ] && [ "$calls" -ge 3 ]
}
tmux() {
    case "$1" in
        new-session) printf 'mock startup diagnostic\\n' > "$STARTUP_LOG" ;;
        has-session) [ "$MODE" != crashed ] ;;
        capture-pane) return 1 ;;
        *) return 1 ;;
    esac
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            env = {"PATH": os.environ["PATH"], "MODE": mode, "STATE": tmp + "/state",
                   "STARTUP_LOG": tmp + "/state/startup.log", "SERVICE_SESSION": "test-service",
                   "REPO": str(REPO), "VTB_ROOT": tmp, "HF_HOME": tmp, "RESULTS_DIR": tmp,
                   "HUB_TASK_SPECS": tmp, "VTB_HARBOR_REPO": tmp, "PORT": "9999",
                   "VTB_PYTHON": sys.executable,
                   "VTB_LOCAL_ENV_FILE_Q": "''", "VTB_AGENT_ADAPTERS_DIR_Q": "''"}
            result = subprocess.run(["bash", "-c", stubs + function + '\nstart_service\nprintf "polls=%s\\n" "$calls"\n'],
                                    env=env, capture_output=True, text=True, timeout=10)
        return result

    def test_service_waits_for_delayed_health(self):
        result = self._startup("delayed")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("polls=3", result.stdout)
        self.assertIn("Tau2 smoke grading:", result.stdout)

    def test_service_reports_crash_instead_of_success(self):
        result = self._startup("crashed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exited during startup", result.stderr)
        self.assertIn("mock startup diagnostic", result.stderr)
        self.assertNotIn("Tau2 smoke grading:", result.stdout)

    def test_service_times_out_without_starting_research(self):
        result = self._startup("timeout")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not become healthy", result.stderr)
        self.assertNotIn("Tau2 smoke grading:", result.stdout)


if __name__ == "__main__":
    unittest.main()
