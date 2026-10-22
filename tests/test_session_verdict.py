from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "commit_utils"))

import session_verdict as sv  # noqa: E402

VERDICT = REPO / "src" / "commit_utils" / "session_verdict.py"
RUN_TASK = REPO / "src" / "run_task.sh"

# An observed malformed Claude stream result: template control-token debris was
# returned as plain text instead of a tool call.
DEBRIS = 'response<|sep|>Richresponse<|sep|><|open|>tools<|sep|>call tool="Bash" index="1"<|sep|>argumentresponse<|sep|>'


def line(record: dict, ts: str = "2026-09-28T08:41:52Z") -> str:
    return f"[{ts}] {json.dumps(record)}\n"


def result(text: str, is_error: bool = False) -> str:
    return line({"type": "result", "subtype": "success", "is_error": is_error, "result": text})


class DegenerateTest(unittest.TestCase):
    def test_observed_debris_is_degenerate(self) -> None:
        self.assertTrue(sv.degenerate(DEBRIS))

    def test_summary_quoting_one_token_is_an_answer(self) -> None:
        text = "All work is complete. The chat template closes each turn with <|im_end|>, which the export keeps."
        self.assertFalse(sv.degenerate(text))

    def test_long_summary_discussing_tokens_is_an_answer(self) -> None:
        text = "Final summary: " + ("the delivered model was trained and validated on the split. " * 4)
        text += "Notes on template tokens <|im_start|> and <|im_end|> are in candidate.md."
        self.assertFalse(sv.degenerate(text))

    def test_plain_answer_is_not_degenerate(self) -> None:
        self.assertFalse(sv.degenerate("All work is complete and every process has exited."))


class JudgeTest(unittest.TestCase):
    def judge(self, text: str) -> bool:
        return sv.judge(text.splitlines(keepends=True))[0]

    def test_concluding_answer_is_deliberate(self) -> None:
        self.assertTrue(self.judge(result("All work is complete. Final delivery is in place.")))

    def test_empty_result_is_not_deliberate(self) -> None:
        self.assertFalse(self.judge(result("   ")))

    def test_error_result_is_not_deliberate(self) -> None:
        self.assertFalse(self.judge(result("API Error: 403 invalid api-key", is_error=True)))

    def test_debris_result_is_not_deliberate(self) -> None:
        deliberate, reason = sv.judge(result(DEBRIS).splitlines(keepends=True))
        self.assertFalse(deliberate)
        self.assertIn("debris", reason)

    def test_stream_without_result_is_not_deliberate(self) -> None:
        text = line({"type": "assistant", "message": {"content": [{"type": "text", "text": "working"}]}})
        self.assertFalse(self.judge(text))

    def test_only_the_latest_attempt_judges(self) -> None:
        concluded_then_debris = (
            result("All work is complete.")
            + "[2026-09-28T09:00:00Z] ===== resume attempt 1 =====\n"
            + result(DEBRIS)
        )
        self.assertFalse(self.judge(concluded_then_debris))
        debris_then_concluded = (
            result(DEBRIS)
            + "[2026-09-28T09:00:00Z] ===== resume attempt 1 =====\n"
            + result("All work is complete.")
        )
        self.assertTrue(self.judge(debris_then_concluded))

    def test_codex_stream_unchanged(self) -> None:
        text = (
            line({"type": "item.completed", "item": {"type": "agent_message", "text": "Delivered."}})
            + line({"type": "turn.completed"})
        )
        self.assertTrue(self.judge(text))
        text = (
            line({"type": "item.completed", "item": {"type": "command_execution"}})
            + line({"type": "turn.completed"})
        )
        self.assertFalse(self.judge(text))


class CliTest(unittest.TestCase):
    def run_cli(self, text: str) -> subprocess.CompletedProcess:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write(text)
        try:
            return subprocess.run([sys.executable, str(VERDICT), handle.name], capture_output=True, text=True)
        finally:
            Path(handle.name).unlink()

    def test_exit_codes(self) -> None:
        self.assertEqual(self.run_cli(result("All work is complete.")).returncode, 0)
        self.assertEqual(self.run_cli(result("")).returncode, 1)
        debris = self.run_cli(result(DEBRIS))
        self.assertEqual(debris.returncode, 1)
        self.assertIn("treating the exit as unintended", debris.stderr)

    def test_missing_file_is_not_deliberate(self) -> None:
        proc = subprocess.run([sys.executable, str(VERDICT), "/nonexistent/solve_out.txt"], capture_output=True)
        self.assertEqual(proc.returncode, 1)


class DeliveryClosedTest(unittest.TestCase):
    """Runs the delivery_closed function exactly as it appears in run_task.sh."""

    @classmethod
    def setUpClass(cls) -> None:
        text = RUN_TASK.read_text(encoding="utf-8")
        match = re.search(r"^delivery_closed\(\) \{\n.*?^\}\n", text, re.S | re.M)
        if match is None:
            raise AssertionError("delivery_closed() not found in run_task.sh")
        cls.function = match.group(0)

    def closed(self, job_dir: Path) -> bool:
        script = f'JOB_DIR="{job_dir}"\n{self.function}\ndelivery_closed'
        return subprocess.run(["bash", "-c", script]).returncode == 0

    def make(self, root: Path, *, model: bool, harness: bool, candidate: bool) -> Path:
        task = root / "task"
        task.mkdir(parents=True)
        if model:
            (task / "final_model").mkdir()
            (task / "final_model" / "config.json").write_text("{}")
        if harness:
            (task / "harness").mkdir()
        if candidate:
            (task / "final_candidate.json").write_text('{"candidate_id": "v1"}')
        return root

    def test_matrix(self) -> None:
        cases = [
            # model, harness, candidate -> closed
            (False, False, False, False),
            (False, True, True, False),
            (True, False, False, True),   # plain benchmark: final_model is the delivery
            (True, True, False, False),   # the instr run: exported, never selected
            (True, True, True, True),
        ]
        for model, harness, candidate, expected in cases:
            with self.subTest(model=model, harness=harness, candidate=candidate):
                with tempfile.TemporaryDirectory() as tmp:
                    root = self.make(Path(tmp), model=model, harness=harness, candidate=candidate)
                    self.assertEqual(self.closed(root), expected)

    def test_empty_config_is_not_a_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make(Path(tmp), model=True, harness=False, candidate=False)
            (root / "task" / "final_model" / "config.json").write_text("")
            self.assertFalse(self.closed(root))


if __name__ == "__main__":
    unittest.main()
