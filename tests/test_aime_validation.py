from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO / "scripts/run_smoke/python"))

import _direct_aime  # noqa: E402
import registry  # noqa: E402
from aime__26 import materialize_dataset as materialize_2026  # noqa: E402
from aime__aime import materialize_dataset as materialize_2025  # noqa: E402
from eval.grading import config, valsplit  # noqa: E402
import verify_smoke_aime  # noqa: E402
from eval.grading.server import GradingService  # noqa: E402
from eval.grading.server.archive import ArchiveMixin  # noqa: E402
from eval.grading.server.disclosure import DisclosureMixin  # noqa: E402


class _ArchiveProbe(ArchiveMixin):
    def _read_json_or_none(self, path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _target_samples(self, _job):
        return 30

    def _disclose_rows(self, rows, _job, _trial_dirs):
        return rows


def _dataset(root: Path, *, answer: int = 1) -> Path:
    tasks = []
    for index in range(30):
        task_id = f"aime_{index:02d}"
        relative = Path("tasks") / task_id
        task_dir = root / relative
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text(f"question {index}", encoding="utf-8")
        (task_dir / "answer.json").write_text(
            json.dumps({"answer": answer}), encoding="utf-8"
        )
        tasks.append({"task_id": task_id, "path": str(relative)})
    (root / "manifest.json").write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    return root


class AimeValidationTest(unittest.TestCase):
    def test_registry_routes_aime26_to_aime2025(self) -> None:
        final = registry.get("aime__26")
        validation = registry.get("aime__aime")
        self.assertEqual(final.dataset, "MathArena/aime_2026")
        self.assertEqual(final.dataset_total, 30)
        self.assertEqual(final.validation_source, "task:aime__aime")
        self.assertEqual(final.backend, "customed")
        self.assertEqual(validation.dataset, "opencompass/AIME2025")
        self.assertEqual(validation.dataset_total, 30)

    def test_cross_benchmark_validation_plan_and_count(self) -> None:
        service = object.__new__(GradingService)
        service.val_source = valsplit.parse_val_source("split:30%")
        service._benchmark_val_sources = {}
        service.task_stats = SimpleNamespace(total=lambda _task: None)
        old = config.VAL_SOURCE_RAW
        try:
            config.VAL_SOURCE_RAW = ""
            self.assertEqual(
                service._validation_plan("aime__26"),
                ("aime__aime", None, None),
            )
            with mock.patch.object(
                config, "known_tasks", return_value=["aime__aime", "aime__26"]
            ):
                self.assertEqual(service._validation_counts()["aime__26"]["count"], 30)
        finally:
            config.VAL_SOURCE_RAW = old

    def test_avg_at_n_scores_remain_fractional_for_prefix_and_explicit_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.per_sample.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset_total": 2,
                        "score_mode": "avg_at_n",
                        "sample_ids": ["one", "two"],
                        "correct": {"one": 0.25, "two": 0.75},
                    }
                ),
                encoding="utf-8",
            )
            results = valsplit.read_per_sample_file(path)
        self.assertIsNotNone(results)
        assert results is not None
        self.assertEqual(results.score_mode, "avg_at_n")
        prefix = valsplit.partition(results, 1, reported_accuracy=0.5)
        self.assertEqual(prefix["accuracy_val"], 0.25)
        self.assertEqual(prefix["accuracy_heldout"], 0.75)
        explicit = valsplit.partition(
            results,
            reported_accuracy=0.5,
            val_ids=["two"],
            test_ids=["one"],
        )
        self.assertEqual(explicit["accuracy_val"], 0.75)
        self.assertEqual(explicit["accuracy_heldout"], 0.25)

    def test_only_selection_is_ordered_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _dataset(Path(tmp))
            selected, total = _direct_aime.load_tasks(
                root, -1, ["aime_17", "aime_03"]
            )
            self.assertEqual(total, 30)
            self.assertEqual(
                [task.task_id for task in selected], ["aime_03", "aime_17"]
            )
            with self.assertRaisesRegex(_direct_aime.EvaluationError, "duplicate"):
                _direct_aime.load_tasks(root, -1, ["aime_03", "aime_03"])

    def test_direct_evaluator_writes_task_level_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root / "dataset")
            metrics = root / "metrics.json"
            work = root / "work"
            argv = [
                "--model-path",
                str(root / "final_model"),
                "--api-base",
                "http://127.0.0.1:9999/v1",
                "--work-dir",
                str(work),
                "--json-output-file",
                str(metrics),
                "--dataset-dir",
                str(dataset),
                "--served-model-name",
                "served-aime",
                "--limit",
                "1",
                "--samples-per-task",
                "2",
            ]
            with mock.patch.object(
                _direct_aime, "request_completion", return_value="reasoning \\boxed{1}"
            ), mock.patch("sys.stdout", new=io.StringIO()):
                rc = _direct_aime.main(
                    "aime__aime",
                    "opencompass/AIME2025",
                    "revision",
                    "VTB_AIME_DATASET_DIR",
                    argv,
                )
            payload = json.loads(metrics.read_text(encoding="utf-8"))
            per_sample = json.loads(
                metrics.with_suffix(".per_sample.json").read_text(encoding="utf-8")
            )
        self.assertEqual(rc, 0)
        self.assertEqual(payload["model"], "served-aime")
        self.assertEqual(payload["graded"], 1)
        self.assertEqual(payload["total_completions"], 2)
        self.assertEqual(payload["accuracy"], 1.0)
        self.assertEqual(per_sample["score_mode"], "avg_at_n")
        self.assertEqual(per_sample["correct"], {"aime_00": 1.0})

    def test_cancelled_direct_evaluation_keeps_complete_tasks_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            direct = (
                Path(tmp)
                / "harness_metrics_p1_a1/backend_output/direct_trajectories"
            )
            direct.mkdir(parents=True)
            (direct / "done.json").write_text(
                json.dumps(
                    {
                        "task_id": "done",
                        "samples_per_task": 4,
                        "avg_at_n": 0.5,
                        "completions": [{}, {}, {}, {}],
                    }
                ),
                encoding="utf-8",
            )
            (direct / "incomplete.json").write_text(
                json.dumps(
                    {
                        "task_id": "incomplete",
                        "samples_per_task": 4,
                        "avg_at_n": 0.5,
                        "completions": [{}, {}],
                    }
                ),
                encoding="utf-8",
            )
            job = SimpleNamespace(attempt_phase=1, attempt_index=1)
            partial, rows = _ArchiveProbe()._partial_outcome(Path(tmp), job)
        self.assertEqual(partial["graded"], 1)
        self.assertEqual(partial["instance_ids_graded"], ["done"])
        self.assertFalse(partial["comparable_to_full"])
        self.assertEqual(rows, [{"id": "done", "score": 0.5, "graded": True}])

    def test_direct_disclosure_never_returns_scoring_answer_or_correctness(self) -> None:
        probe = object.__new__(DisclosureMixin)
        probe.disclosure = frozenset({"per_sample_outcomes", "outputs"})
        probe._truncate_disclosure = lambda text: text
        result = probe._direct_disclosure(
            {
                "task_id": "aime_01",
                "samples_per_task": 4,
                "correct_completions": 2,
                "avg_at_n": 0.5,
                "answer": 999,
                "completions": [
                    {
                        "response": "answer text",
                        "parsed_answer": 17,
                        "correct": True,
                    }
                ],
            }
        )
        self.assertNotIn("answer", result)
        self.assertNotIn("correct", result["completions"][0])
        self.assertEqual(result["avg_at_n"], 0.5)

    def test_aime_launcher_uses_separate_validation_benchmark(self) -> None:
        launcher = (REPO / "scripts/run_exp/run_exp_aime.sh").read_text(encoding="utf-8")
        self.assertIn('BENCHMARK="aime__26"', launcher)
        self.assertIn('VAL_SOURCE="${VAL_SOURCE:-task:aime__aime}"', launcher)
        self.assertIn('RECIPE="${RECIPE:-qwen3_5_9b/slime_sft_aime}"', launcher)
        self.assertIn("prepare_aime.sh preflight", launcher)
        self.assertNotIn("--formal", launcher)
        self.assertIn('HOURS="${HOURS:-16}"', launcher)
        self.assertIn("VTB_EDITABLE_HARNESS=off", launcher)

    def _smoke_tree(self, root: Path, *, val_limit: int = 1) -> tuple[Path, Path]:
        state, results = root / "state", root / "results"
        state.mkdir()
        results.mkdir()
        val_job, final_job = "val123", "final456"
        lines = [
            f"[submit] queued job={val_job} run=run#agent role=agent task=aime__26 "
            f"eval_task=aime__aime mode=val limit={val_limit} position=1",
            f"[worker] finish job={val_job} state=done score=0.0",
            f"[submit] queued job={final_job} run=run#official role=official task=aime__26 "
            "eval_task=aime__26 mode=final limit=1 position=1",
            f"[worker] finish job={final_job} state=done score=0.0",
        ]
        (state / "service.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for role, mode, job, benchmark, dataset in (
            ("agent", "val", val_job, "aime__aime", "opencompass/AIME2025"),
            ("official", "final", final_job, "aime__26", "MathArena/aime_2026"),
        ):
            archive = state / "archive/run" / f"{mode}_{job}"
            trajectories = archive / "harness_metrics_p1_a1/backend_output/direct_trajectories"
            trajectories.mkdir(parents=True)
            metrics = {
                "accuracy": 0.0,
                "benchmark": benchmark,
                "dataset": dataset,
                "backend": "direct",
                "samples_per_task": 1,
                "graded": 1,
                "attempted": 1,
                "ungraded": 0,
                "backend_returncode": 0,
            }
            (archive / "metrics_p1_a1.json").write_text(json.dumps(metrics), encoding="utf-8")
            (trajectories / f"{role}.json").write_text(
                json.dumps({"task_id": role, "samples_per_task": 1, "completions": [{}]}),
                encoding="utf-8",
            )
            if role == "official":
                (results / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return state, results

    def test_aime_smoke_verifies_one_val_and_final_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, results = self._smoke_tree(Path(tmp))
            scores = verify_smoke_aime.verify(state, results)
        self.assertEqual(scores, {"agent": 0.0, "official": 0.0})

    def test_aime_smoke_rejects_non_one_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, results = self._smoke_tree(Path(tmp), val_limit=10)
            with self.assertRaises(verify_smoke_aime.VerificationError):
                verify_smoke_aime.verify(state, results)

    def test_aime_smoke_launcher_is_bounded_and_no_training(self) -> None:
        launcher = (REPO / "scripts/run_smoke/run_smoke_aime.sh").read_text(encoding="utf-8")
        prompt = (REPO / "eval/agent_task/aime_smoke.md").read_text(encoding="utf-8")
        self.assertIn("VTB_FINAL_LIMIT=1", launcher)
        self.assertIn("VTB_AIME_DIRECT_SAMPLES_PER_TASK=1", launcher)
        self.assertIn("VTB_TRAIN_RECIPE=", launcher)
        self.assertIn('Include `"limit": 1`', prompt)

    def test_materializers_publish_pinned_30_task_packages(self) -> None:
        rows_2025 = [
            {
                "exam": "I" if index < 15 else "II",
                "problem_number": index + 1 if index < 15 else index - 14,
                "question": f"problem {index}",
                "answer": str(index % 1000),
            }
            for index in range(30)
        ]
        rows_2026 = {
            "rows": [
                {
                    "row": {
                        "problem_idx": index,
                        "problem": f"problem {index}",
                        "answer": index,
                    }
                }
                for index in range(30)
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_2025 = materialize_2025.materialize(
                rows_2025, root / "aime__aime", False, "fixture"
            )
            result_2026 = materialize_2026.materialize(
                rows_2026, root / "aime__26", False, "fixture"
            )
            tasks_2025, _ = _direct_aime.load_tasks(root / "aime__aime", -1, None)
            tasks_2026, _ = _direct_aime.load_tasks(root / "aime__26", -1, None)
            env = dict(os.environ)
            env["VTB_HUB_TASK_SPECS"] = str(root)
            env["VTB_PYTHON"] = sys.executable
            preflight = subprocess.check_output(
                ["bash", "scripts/prepare/prepare_aime.sh", "preflight"],
                cwd=REPO,
                env=env,
                text=True,
            )
        self.assertIn("AIME_PREFLIGHT_OK", preflight)
        self.assertEqual(result_2025["dataset_total"], 30)
        self.assertEqual(result_2026["dataset_total"], 30)
        self.assertEqual(len(tasks_2025), 30)
        self.assertEqual(len(tasks_2026), 30)


if __name__ == "__main__":
    unittest.main()
