from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

import registry  # noqa: E402
from eval.grading import config, valsplit  # noqa: E402
from eval.grading.server import GradingService  # noqa: E402
from eval.runner import run, task_backends  # noqa: E402

TARGET = "terminal-bench__terminal-bench-2-1"
VAL50 = "openthoughts__openthoughts-tblite-val50"
_CONTEXT_CONFIG = (
    REPO / "training" / "resources" / "validation" / "tblite_val50_context" / "resource_config.json"
)
# The pack's payload location is the config's business, not the test's: resolve it the
# same way production does, so the test follows a redirected materialized_path.
CONTEXT = valsplit.resolve_resource_path(
    REPO, json.loads(_CONTEXT_CONFIG.read_text(encoding="utf-8"))["materialized_path"]
)


class FixedTBLiteValidationTest(unittest.TestCase):
    def test_registry_has_exact_fixed_subset(self) -> None:
        spec = registry.get(VAL50)
        self.assertEqual(spec.dataset, "openthoughts/openthoughts-tblite")
        self.assertEqual(spec.default_limit, 50)
        self.assertEqual(spec.dataset_total, 50)
        self.assertEqual(config.default_limit(VAL50), 50)
        self.assertEqual(config.dataset_total(VAL50), 50)
        self.assertEqual(len(spec.task_ids), 50)
        self.assertEqual(len(set(spec.task_ids)), 50)
        self.assertEqual(spec.task_ids[0], "application-debug")
        self.assertEqual(spec.task_ids[-1], "scan-linux-persistence-artifacts")

    def test_registry_rejects_invalid_curated_ids(self) -> None:
        common = {
            "name": "invalid",
            "backend": "harbor",
            "agent": "terminus-2",
            "default_limit": 2,
            "dataset": "owner/dataset",
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            registry.BenchmarkSpec(**common, task_ids=("same", "same"))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            registry.BenchmarkSpec(**common, task_ids=("valid", ""))

    def test_curated_subset_limit_is_a_stable_prefix(self) -> None:
        spec = registry.get(VAL50)
        args = SimpleNamespace(only=None, limit=3)
        tasks, task_ids, total = run._expected_tasks(spec, REPO / "benchmarks" / VAL50, args)
        self.assertEqual(total, 50)
        self.assertEqual(task_ids, list(spec.task_ids[:3]))
        self.assertEqual([task.task_id for task in tasks], task_ids)

    def test_harbor_uses_explicit_filters_without_sampling(self) -> None:
        spec = registry.get(VAL50)
        command = task_backends.build_harbor(
            spec,
            task_source=None,
            task_ids=list(spec.task_ids),
            output_dir=Path("/tmp/val50"),
            model="openai/test-model",
            n_concurrent=1,
            limit=50,
        )
        self.assertNotIn("--n-tasks", command)
        included = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--include-task-name"
        ]
        # Each curated id is passed in two forms: bare (trial-tree naming) and
        # org-qualified (harbor's PackageTaskId.get_name() shape) -- the filter is a
        # union, so the task still runs once.
        self.assertEqual(len(included), 2 * len(spec.task_ids))
        self.assertEqual(included[0::2], list(spec.task_ids))
        self.assertEqual(
            included[1::2], [f"*/{task_id}" for task_id in spec.task_ids]
        )

    def test_terminal_bench_uses_local_validation_source(self) -> None:
        service = object.__new__(GradingService)
        service.val_source = valsplit.parse_val_source("split:30%")
        service._benchmark_val_sources = {}
        service.task_stats = SimpleNamespace(total=lambda _task: None)
        old_override = config.VAL_SOURCE_RAW
        try:
            config.VAL_SOURCE_RAW = ""
            self.assertEqual(config.validation_source(TARGET), f"task:{VAL50}")
            self.assertEqual(service._validation_plan(TARGET), (VAL50, None, None))

            config.VAL_SOURCE_RAW = "split:5"
            service.val_source = valsplit.parse_val_source(config.VAL_SOURCE_RAW)
            self.assertEqual(service._validation_plan(TARGET), (TARGET, 5, None))
        finally:
            config.VAL_SOURCE_RAW = old_override

    def test_context_pack_is_complete_and_excludes_verifier_files(self) -> None:
        metadata = json.loads(_CONTEXT_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(metadata["type"], "validation_context")
        self.assertEqual(metadata["benchmark"], TARGET)
        self.assertEqual(metadata["val_source"], f"task:{VAL50}")
        self.assertEqual(metadata["workspace_path"], "tblite_validation")
        self.assertTrue(metadata.get("materialized_path"))
        manifest = json.loads((CONTEXT / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((CONTEXT / "baseline_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["task_count"], 50)
        self.assertEqual(len(manifest["tasks"]), 50)
        self.assertEqual(len(summary["tasks"]), 50)
        self.assertEqual(summary["task_count"], 50)
        self.assertEqual(summary["graded"], 49)
        self.assertEqual(summary["solved"], 6)
        self.assertAlmostEqual(summary["accuracy"], 0.12)
        self.assertAlmostEqual(summary["mean_reward"], 0.16269, places=6)
        self.assertEqual(sum(summary["counts"].values()), 50)
        self.assertNotIn("missing_trajectory", summary["counts"])
        for record in manifest["tasks"]:
            self.assertTrue((CONTEXT / record["instruction_path"]).is_file())
            trajectory = CONTEXT / record["trajectory_path"]
            self.assertTrue(trajectory.is_file())
            self.assertIsInstance(json.loads(trajectory.read_text(encoding="utf-8")), dict)
        forbidden = {"verifier", "tests", "solutions", "patches"}
        for path in CONTEXT.rglob("*"):
            parts = {part.lower() for part in path.relative_to(CONTEXT).parts}
            self.assertFalse(parts & forbidden, path)

    def test_workspace_materialization_matches_declared_workspace_path(self) -> None:
        """The pack reaches the agent workspace where the task_context docs point."""
        import tempfile

        sys.path.insert(0, str(REPO / "src" / "commit_utils"))
        from prepare_agent_workspace import copy_validation_context

        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            task_dir.mkdir()
            result = copy_validation_context(REPO, task_dir, REPO / "benchmarks" / TARGET)
            assert result is not None
            self.assertEqual(result["workspace_path"], "tblite_validation")
            self.assertTrue((task_dir / "tblite_validation" / "manifest.json").is_file())
            self.assertTrue((task_dir / "tblite_validation" / "baseline_summary.json").is_file())
            # The benchmark declaring nothing materializes nothing.
            other = copy_validation_context(REPO, task_dir, REPO / "benchmarks" / VAL50)
            self.assertIsNone(other)


if __name__ == "__main__":
    unittest.main()
